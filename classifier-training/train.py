import os
import gc
import json
import time
from os import path
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import librosa as lbr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from absl import flags, app
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from scipy import signal
from tqdm import tqdm
from reservoirpy import ESN, mat_gen
from reservoirpy import activationsfunc as F

sns.set(context="paper", style="dark")
FLAGS = flags.FLAGS

# path to your data root folder. 
# Here, it would be "./data_example" for instance.
DATA = "training/data_examples/audio"

# Number of folds
TRIALS = 1

# Number of random instances per fold
INSTANCES = 5

# Name of the model (important; needs to be unique at every launch)
MODEL = "classifier-pre"

# Seeds for the different instances
SEEDS = [55448, 2651, 55541, 58964, 4100, 1000, 2000, 4000, 5000]

# Activate cross validation
FOLD = True

# Seed for Kfold
FOLD_SEED = 5555

# Export complete model at the end, trained on all training data
EXPORT = False


# Training configuration (can be loaded in a JSON file)
CONF = {
    "esn": {
        "N": 1000, # nb of units in the reservoir
        "ridge": 1e-8, # regularization param
        "lr": 5e-2, # leak rate
        "sr": 0.5, # spectral radius
        "input_bias": True, # quite clear by itself
        "mfcc_scaling": 0.0, # input scaling of MFCC (no MFCC in our case)
        "delta_scaling": 1.0, # same for 1st derivative
        "delta2_scaling": 0.7, # same for 2nd
        "feedback": False, # no feedback connections
        "feedback_scaling": 0.0,
        "fbfunc": "softmax",
        "wash_nr_time_step": 0, # no transient states (no warmup)
        "seed": 42
    },
    "data": { # everything librosa related
        "sampling_rate": 16000,
        "hop_length": int(16000 * 0.01), 
        "n_fft": int(16000 * 0.02), # this is a bit wrong. we shouldnt change the default value, but the "window_length" parameter instead. 
        "fmax": 8000,
        "fmin": 500,
        "n_mfcc": 13,
        "padding": "nearest", # mode of the derivation of MFCC (for border effects)
        "trim_silence": False,
        "continuous": False,
        "mfcc": False,
        "d": True,
        "d2": True,
        "highpass": None,
        "order": None,
        "lifter": 0
    }
}


flags.DEFINE_string('conf', f'reports/{MODEL}/training-conf.json', "Path to the JSON config file.")
flags.DEFINE_string('data', f'data/{DATA}', 'Path to dataset with train and test directories.')
flags.DEFINE_string('report', f'reports/{MODEL}', 'Results directory.')
flags.DEFINE_string('save', f'models/esn-{MODEL}', 'Directory where the trained model will be saved.')
flags.DEFINE_integer('workers', 8, 'Number of parallel threads to launch for computation.', lower_bound=-1)
flags.DEFINE_integer('trials', TRIALS, 'Number of trials to perform to compute metrics', lower_bound=1)
flags.DEFINE_boolean('fold', FOLD, 'If set, will perform cross validation on the number of folds set by trials parameter.')
flags.DEFINE_boolean('export', EXPORT, 'If set, model is exported at the end (training with all available data).')
flags.DEFINE_integer('instances', INSTANCES, 'Number of instances to train.')


def _get_trials():
    return FLAGS.trials


def _get_fold():
    return FLAGS.fold

def _get_instances():
    return FLAGS.instances

def _get_workers():
    return FLAGS.workers


def _get_export():
    return FLAGS.export


def _get_conf_path():
    return path.join(FLAGS.conf)


def _get_save_dir():
    savedir = path.join(FLAGS.save)
    return savedir


def _get_report_dir():
    reportdir = path.join(FLAGS.report)
    if not(path.isdir(reportdir)):
        os.mkdir(reportdir)
    return reportdir


def _get_dataset_dirs():
    train_path = path.join(FLAGS.data, "train")
    test_path = path.join(FLAGS.data, "test")
    if os.path.isdir(train_path) and os.path.isdir(test_path):
        return train_path, test_path
    else:
        return path.join(FLAGS.data), path.join(FLAGS.data)


def _get_conf_from_json():
    config = {}
    _get_report_dir()
    conf_path = Path(_get_conf_path())
    if conf_path.exists():
        with conf_path.open("r") as f:
            config = json.load(f)
    else:
        config = CONF
        with conf_path.open('w+') as f:
            json.dump(CONF, f)
    return config


def _get_dataset_summaries():
    train_dataset = pd.read_csv(path.join(FLAGS.data, "train_dataset.csv"))
    test_dataset = pd.read_csv(path.join(FLAGS.data, "test_dataset.csv"))
    return train_dataset, test_dataset


def _check_paths():
    confpath = _get_conf_path()
    traindir, testdir = _get_dataset_dirs()

    if not(path.isdir(traindir)):
        raise NotADirectoryError(f"Dataset '{traindir}' not found.")
    elif not(path.isdir(testdir)):
        raise NotADirectoryError(f"Dataset '{testdir}' not found.")

    if not(path.isfile(confpath)):
        raise FileNotFoundError(f"Training conf '{confpath}' not found.")


# --------- PREPROCESSING --------------

def load_wave(x_path, y, sr):
    x, _ = lbr.load(x_path, sr=sr)
    return x, y


def apply_filter(wave, sr, freq=500, order=5):

    nyq = 0.5 * sr
    freq = freq / nyq
    sos = signal.butter(order, freq, btype='highpass', output='sos')
    y = signal.sosfilt(sos, wave)

    return y


def compute_features(wave, y, sr, hop_length, n_fft, padding, trim_silence, highpass, order, lifter):

    w = wave
    if trim_silence:
        w, _ = lbr.effects.trim(w, top_db=trim_silence)

    if highpass:
        w = apply_filter(w, sr, highpass, order)

    mfcc = lbr.feature.mfcc(w, sr, hop_length=hop_length, n_fft=n_fft, win_length=512, lifter=lifter)
    d = lbr.feature.delta(mfcc, mode=padding)
    d2 = lbr.feature.delta(mfcc, order=2, mode=padding)
    teacher = np.tile(y, (mfcc.shape[1], 1))

    return (mfcc.T, d.T, d2.T), teacher


def preprocess_all_waves(all_waves, config):

    loop = joblib.Parallel(n_jobs=_get_workers(), backend="threading")
    delayed_feat = joblib.delayed(compute_features)

    hop_length = config["data"]["hop_length"]
    n_fft = config["data"]["n_fft"]
    mode = config["data"]["padding"]
    sr = config["data"]["sampling_rate"]
    trim = config["data"]["trim_silence"]
    highpass = config["data"]["highpass"]
    order = config["data"]["order"]
    lifter = config["data"]["lifter"]

    all_data = loop(delayed_feat(w, y, sr, hop_length, n_fft, mode, trim, highpass, order, lifter)
                    for w, y in tqdm(all_waves, "Preprocessing", total=len(all_waves)))

    return all_data


def retrieve_all_waves(datadir, dataset, config):

    loop = joblib.Parallel(n_jobs=_get_workers(), backend='threading')
    sr = config["data"]["sampling_rate"]

    delayed_load = joblib.delayed(load_wave)

    path_and_labels = zip([path.join(datadir, x) for x in dataset.x], dataset.y)

    all_waves = loop(delayed_load(x, y, sr)
                     for x, y in tqdm(path_and_labels, "Loading waves", total=len(dataset)))

    return all_waves

# --------- ESN stuff ---------------

def build_matrices(dim_mfcc, dim_out, config, seed=None):

    sr = config["esn"]["sr"]
    N = config["esn"]["N"]
    input_bias = config["esn"]["input_bias"]
    iss = config["esn"]["mfcc_scaling"]
    isd = config["esn"]["delta_scaling"]
    isd2 = config["esn"]["delta2_scaling"]
    feedback = config["esn"]["feedback"]

    if seed is None:
        seed = config["esn"].get("seed")

    if seed is None:
        seed = np.random.randint(0, 99999)

    Wfb = None
    if feedback:
        isfb = config["esn"]["feedback_scaling"]
        Wfb = mat_gen.generate_input_weights(N=N, dim_input=dim_out,
                                            input_scaling=isfb, proba=0.1, input_bias=False,
                                            seed=seed, typefloat=np.float64)

    W = mat_gen.generate_internal_weights(N=N, spectral_radius=sr,
                                            seed=seed, typefloat=np.float64)

    input_matrices = []
    bias_added = False
    if config["data"]["mfcc"]:
        input_matrices.append(mat_gen.generate_input_weights(N=N, dim_input=dim_mfcc,
                                            input_bias=input_bias, input_scaling=iss,
                                            seed=seed, typefloat=np.float64))
        bias_added = True

    if config["data"]["d"]:
        input_matrices.append(mat_gen.generate_input_weights(N=N, dim_input=dim_mfcc,
                                            input_bias=(not(bias_added) and input_bias), input_scaling=isd,
                                            seed=seed+1, typefloat=np.float64))
        bias_added = True

    if config["data"]["d2"]:
        input_matrices.append(mat_gen.generate_input_weights(N=N, dim_input=dim_mfcc,
                                        input_bias=(not(bias_added) and input_bias), input_scaling=isd2,
                                        seed=seed+2, typefloat=np.float64))

    Win = np.concatenate(input_matrices, axis=1)

    return Win, W, Wfb, seed


# ---------- PLOTTING stuff (not used actually) ------------

def plot_train_summary(states, inputs):

    fig = plt.figure(figsize=(10, 10))

    ax0 = fig.add_subplot((411), title="Training summary", ylabel="States")
    ax0.plot(states[0][500:1000, :30])

    ax1 = fig.add_subplot((412), ylabel="MFCC")
    ax1.imshow(inputs[0].T[:20, 500:1000], aspect="auto")

    ax3 = fig.add_subplot((413), ylabel="Delta")
    ax3.imshow(inputs[0].T[20:40, 500:1000], aspect="auto")

    ax3 = fig.add_subplot((414), ylabel="Delta2", xlabel="Timestep")
    ax3.imshow(inputs[0].T[40:, 500:1000], aspect="auto")

    fig.savefig(path.join(_get_report_dir(), "training-summary"))


def plot_test_summary(outputs, truths, inputs):

    fig = plt.figure(figsize=(20, 10))

    ax0 = fig.add_subplot((211), title="Output summary", ylabel="Output")
    ax0.imshow(outputs[0].T[:, 500:1000], aspect="auto")
    ax0.scatter(range(500), truths[0].T[:, 500:1000].argmax(axis=0), marker='o', s=5, color="g", label="Truths")
    ax0.legend()

    ax2 = fig.add_subplot((212), ylabel="MFCC", sharex=ax0)
    ax2.imshow(inputs[0].T[1:20, 500:1000], aspect="auto")

    fig.savefig(path.join(_get_report_dir(), "runing-summary"))


def plot_confusion(y_test, y_preds, syllables, oh_encoder, cm=None):

    if cm is None:
        cm = metrics.confusion_matrix(
            oh_encoder.inverse_transform(np.vstack(y_test)),
            oh_encoder.inverse_transform(np.vstack(y_preds)),
            labels=syllables,
            normalize='true')

    fig = plt.figure(figsize=(20,20))

    ax = fig.add_subplot((111))
    plt.xlabel("Predictions", fontsize=20)
    plt.ylabel("Truths", fontsize=20)
    ax.xaxis.set_label_position('top')
    im = ax.matshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1.))

    ax.set_xticks(np.arange(len(syllables)))
    ax.set_yticks(np.arange(len(syllables)))

    ax.set_xticklabels(syllables)
    ax.set_yticklabels(syllables)
    ax.tick_params('x', top=True, bottom=False, labeltop=True, labelbottom=False, rotation=45)

    for i in range(len(syllables)):
        for j in range(len(syllables)):
            _ = ax.text(j, i, round(cm[i, j], 2),
                        ha="center", va="center", color="gray")

    plt.colorbar(im)
    fig.tight_layout()

    fig.savefig(Path(_get_report_dir()) / "confusion.png")


def train(args):
    """
    Train the model once on all training data. Test on testing data. 
    """

    config = _get_conf_from_json()
    trainset, testset = _get_dataset_summaries()
    traindir, testdir = _get_dataset_dirs()

    print("Training set")
    train_waves = retrieve_all_waves(traindir, trainset, config)
    print("Testing set")
    test_waves = retrieve_all_waves(testdir, testset, config)

    syllables = unique_labels(pd.concat([trainset, testset]).y).tolist()
    oh_encoder = OneHotEncoder(categories=[syllables], sparse=False)

    y_train_encoded = oh_encoder.fit_transform(np.array([y for _, y in train_waves]).reshape(-1, 1))
    y_test_encoded = oh_encoder.fit_transform(np.array([y for _, y in test_waves]).reshape(-1, 1))

    train_waves = [(x, y) for (x, _), y in zip(train_waves, y_train_encoded)]
    test_waves = [(x, y) for (x, _), y in zip(test_waves, y_test_encoded)]

    print('Training set')
    train_data = preprocess_all_waves(train_waves, config)
    print('Testing set')
    test_data = preprocess_all_waves(test_waves, config)

    dim_mfcc = train_data[0][0][0].shape[1]
    dim_out = len(syllables)

    selected_feat = np.array([config["data"]["mfcc"], config["data"]["d"], config["data"]["d2"]])

    x_train = [np.hstack(np.array(feat)[selected_feat]) for feat, _ in train_data]
    y_train = [y for _, y in train_data]

    x_test = [np.hstack(np.array(feat)[selected_feat]) for feat, _ in test_data]
    y_test = [y for _, y in test_data]

    if config["data"]["continuous"]:
        x_train = [np.vstack(x_train)]
        y_train = [np.vstack(y_train)]

        x_test = [np.vstack(x_test)]
        y_test = [np.vstack(y_test)]

    del(train_waves)
    del(test_waves)
    del(train_data)
    del(test_data)
    del(trainset)
    del(testset)
    gc.collect()

    Win, W, Wfb, seed = build_matrices(dim_mfcc, dim_out, config)

    lr = config["esn"]["lr"]
    input_bias = config["esn"]["input_bias"]
    ridge = config["esn"]["ridge"]
    feedback = config["esn"]["feedback"]
    fbfunc = None
    if feedback:
        fbfunc = F.get_function(config["esn"]["fbfunc"])

    reservoir = ESN(lr=lr, input_bias=input_bias, W=W, Win=Win,
                    Wfb=Wfb, fbfunc=fbfunc, ridge=ridge, typefloat=np.float64)

    warmup = config["esn"]["wash_nr_time_step"]
    _ = reservoir.train(x_train, y_train, wash_nr_time_step=warmup, verbose=True,
                        use_memmap=True, backend="loky", workers=-1)

    print(f"Running tests")
    outputs, _ = reservoir.run(x_test, verbose=True, backend="loky", workers=-1)

    top_1 = [np.array([syllables[t]
                       for t in outputs[i].argmax(axis=1)]).reshape(-1, 1)
             for i in range(len(outputs))]

    y_preds = [oh_encoder.transform(t) for t in top_1]

    met = {
        "f1": metrics.f1_score(np.vstack(y_preds), np.vstack(y_test), average='macro'),
        "accuracy": metrics.accuracy_score(np.vstack(y_preds), np.vstack(y_test)),
        "loss": metrics.log_loss(np.vstack(y_preds), np.vstack(y_test)),
        "seed": seed
    }

    curr_time = time.time()
    with open(f"{_get_report_dir()}/scores-{curr_time}.json", "w+") as f:
        json.dump(met, f)

    print(f"Scores (top 1):\n Cross-entropy: {met['loss']:.3f}, F1: {met['f1']:.3f}, Accuracy: {met['accuracy']:.3f}")
    plot_confusion(np.vstack(y_test), np.vstack(y_preds), syllables, oh_encoder)

    del(outputs)
    del(top_1)
    del(y_preds)
    del(met)

    if _get_export():
        x_train.extend(x_test)
        y_train.extend(y_test)
        train_export(x_train, y_train, dim_mfcc, syllables)

    return 0


def train_several_times(args):
    """
    Train the model on several instances (as much as TRIALS parameter) (and save a    posteriori scores for all)
    Not really usefull, I think it was to test memory problems.
    """

    config = _get_conf_from_json()
    trainset, testset = _get_dataset_summaries()
    traindir, testdir = _get_dataset_dirs()

    print("Training set")
    train_waves = retrieve_all_waves(traindir, trainset, config)
    print("Testing set")
    test_waves = retrieve_all_waves(testdir, testset, config)

    syllables = unique_labels(pd.concat([trainset, testset]).y).tolist()
    oh_encoder = OneHotEncoder(categories=[syllables], sparse=False)

    y_train_encoded = oh_encoder.fit_transform(np.array([y for _, y in train_waves]).reshape(-1, 1))
    y_test_encoded = oh_encoder.fit_transform(np.array([y for _, y in test_waves]).reshape(-1, 1))

    train_waves = [(x, y) for (x, _), y in zip(train_waves, y_train_encoded)]
    test_waves = [(x, y) for (x, _), y in zip(test_waves, y_test_encoded)]

    print('Training set')
    train_data = preprocess_all_waves(train_waves, config)
    print('Testing set')
    test_data = preprocess_all_waves(test_waves, config)

    dim_mfcc = train_data[0][0][0].shape[1]
    dim_out = len(syllables)

    selected_feat = np.array([config["data"]["mfcc"], config["data"]["d"], config["data"]["d2"]])

    x_train = [np.hstack(np.array(feat)[selected_feat]) for feat, _ in train_data]
    y_train = [y for _, y in train_data]

    x_test = [np.hstack(np.array(feat)[selected_feat]) for feat, _ in test_data]
    y_test = [y for _, y in test_data]

    if config["data"]["continuous"]:
        x_train = [np.vstack(x_train)]
        y_train = [np.vstack(y_train)]

        x_test = [np.vstack(x_test)]
        y_test = [np.vstack(y_test)]

    del(train_waves)
    del(test_waves)
    del(train_data)
    del(test_data)
    del(trainset)
    del(testset)
    gc.collect()

    met = {i: {} for i in range(FLAGS.trials)}
    for i in range(FLAGS.trials):

        print(f"---TRIAL No {i}---")

        Win, W, Wfb, seed = build_matrices(dim_mfcc, dim_out, config)

        lr = config["esn"]["lr"]
        input_bias = config["esn"]["input_bias"]
        ridge = config["esn"]["ridge"]
        feedback = config["esn"]["feedback"]
        fbfunc = None
        if feedback:
            fbfunc = F.get_function(config["esn"]["fbfunc"])

        reservoir = ESN(lr=lr, input_bias=input_bias, W=W, Win=Win,
                        Wfb=Wfb, fbfunc=fbfunc, ridge=ridge, typefloat=np.float64)

        warmup = config["esn"]["wash_nr_time_step"]
        _ = reservoir.train(x_train, y_train, wash_nr_time_step=warmup, verbose=True,
                            use_memmap=True, backend="loky", workers=-1)

        print(f"Running tests")
        outputs, _ = reservoir.run(x_test, verbose=True, backend="loky", workers=-1)

        top_1 = [np.array([syllables[t]
                        for t in outputs[i].argmax(axis=1)]).reshape(-1, 1)
                for i in range(len(outputs))]

        y_preds = [oh_encoder.transform(t) for t in top_1]

        met[i] = {
            "f1": metrics.f1_score(np.vstack(y_preds), np.vstack(y_test), average='macro'),
            "accuracy": metrics.accuracy_score(np.vstack(y_preds), np.vstack(y_test)),
            "loss": metrics.log_loss(np.vstack(y_preds), np.vstack(y_test)),
            "seed": seed
        }

        print(f"Scores (top 1):\n Cross-entropy: {met[i]['loss']:.3f}, F1: {met[i]['f1']:.3f}, Accuracy: {met[i]['accuracy']:.3f}")

        gc.collect()

    print("Average scores :")
    met["average"] = {
        "f1": np.mean([m["f1"] for m in met.values()]),
        "accuracy": np.mean([m["accuracy"] for m in met.values()]),
        "loss": np.mean([m["loss"] for m in met.values()])
    }

    print(met["average"])

    print("Std deviation :")

    met["std"] = {
        "f1": np.std([m["f1"] for m in met.values()]),
        "accuracy": np.std([m["accuracy"] for m in met.values()]),
        "loss": np.std([m["loss"] for m in met.values()])
    }

    print(met["std"])

    curr_time = time.time()
    with open(f"{_get_report_dir()}/scores-{curr_time}.json", "w+") as f:
        json.dump(met, f)

    del(outputs)
    del(top_1)
    del(y_preds)
    del(met)

    if _get_export():
        x_train.extend(x_test)
        y_train.extend(y_test)
        train_export(x_train, y_train, dim_mfcc, syllables)

    return 0


def cross_validation(args):
    """Performs cross validation and saves the results"""

    config = _get_conf_from_json()
    trainset, testset = _get_dataset_summaries()
    traindir, testdir = _get_dataset_dirs()

    dataset = pd.concat([trainset, testset])

    print("Loading dataset")
    waves = retrieve_all_waves(traindir, dataset, config)

    syllables = unique_labels(dataset.y).tolist()
    oh_encoder = OneHotEncoder(categories=[syllables], sparse=False)

    y_labels = np.array([y for _, y in waves])
    y_encoded = oh_encoder.fit_transform(np.array([y for _, y in waves]).reshape(-1, 1))

    Xy = [(x, y) for (x, _), y in zip(waves, y_encoded)]

    data = preprocess_all_waves(Xy, config)

    dim_mfcc = data[0][0][0].shape[1]
    dim_out = len(syllables)

    selected_feat = np.array([config["data"]["mfcc"], config["data"]["d"], config["data"]["d2"]])

    X = [np.hstack(np.array(feat)[selected_feat]) for feat, _ in data]
    X_indexes = np.arange(len(X)).reshape(-1, 1)
    y = [teacher for _, teacher in data]

    del(waves)
    del(data)
    del(dataset)
    del(Xy)
    del(y_encoded)
    gc.collect()

    met = {}
    cms = []
    seed = config["esn"]["seed"]

    skf = StratifiedKFold(n_splits=FLAGS.trials, shuffle=True, random_state=seed)
    trial = 1
    for train_index, test_index in skf.split(X_indexes, y_labels):

        X_train, y_train = [X[i] for i in train_index], [y[i] for i in train_index]
        X_test, y_test = [X[i] for i in test_index], [y[i] for i in test_index]

        print(f"---FOLD No {trial}---")

        Win, W, Wfb, seed = build_matrices(dim_mfcc, dim_out, config)

        lr = config["esn"]["lr"]
        input_bias = config["esn"]["input_bias"]
        ridge = config["esn"]["ridge"]
        feedback = config["esn"]["feedback"]
        fbfunc = None
        if feedback:
            fbfunc = F.get_function(config["esn"]["fbfunc"])

        reservoir = ESN(lr=lr, input_bias=input_bias, W=W, Win=Win,
                        Wfb=Wfb, fbfunc=fbfunc, ridge=ridge, typefloat=np.float64)

        warmup = config["esn"]["wash_nr_time_step"]
        _ = reservoir.train(X_train, y_train, wash_nr_time_step=warmup, verbose=True,
                            use_memmap=True, backend="loky", workers=-1)

        outputs, _ = reservoir.run(X_train, verbose=True, backend="loky", workers=-1)

        top_1 = [np.array([syllables[t]
                        for t in outputs[i].argmax(axis=1)]).reshape(-1, 1)
                for i in range(len(outputs))]

        y_preds = [oh_encoder.transform(t) for t in top_1]

        met[f'train-{trial}'] = {
            "f1": metrics.f1_score(np.vstack(y_preds), np.vstack(y_train), average='macro'),
            "accuracy": metrics.accuracy_score(np.vstack(y_preds), np.vstack(y_train)),
            "loss": metrics.log_loss(np.vstack(y_preds), np.vstack(y_train)),
            "seed": seed
        }

        print(f"Training scores (top 1):\
    \n Cross-entropy: {met[f'train-{trial}']['loss']:.3f}, F1: {met[f'train-{trial}']['f1']:.3f}, Accuracy: {met[f'train-{trial}']['accuracy']:.3f}")

        print(f"Running validation")
        outputs, _ = reservoir.run(X_test, verbose=True)

        top_1 = [np.array([syllables[t]
                        for t in outputs[i].argmax(axis=1)]).reshape(-1, 1)
                for i in range(len(outputs))]

        y_preds = [oh_encoder.transform(t) for t in top_1]

        met[trial] = {
            "f1": metrics.f1_score(np.vstack(y_preds), np.vstack(y_test), average='macro'),
            "accuracy": metrics.accuracy_score(np.vstack(y_preds), np.vstack(y_test)),
            "loss": metrics.log_loss(np.vstack(y_preds), np.vstack(y_test)),
            "seed": seed
        }

        cms.append(metrics.confusion_matrix(
            oh_encoder.inverse_transform(np.vstack(y_test)),
            oh_encoder.inverse_transform(np.vstack(y_preds)),
            labels=syllables).tolist())

        print(f"Validation scores (top 1):\n Cross-entropy: {met[trial]['loss']:.3f}, F1: {met[trial]['f1']:.3f}, Accuracy: {met[trial]['accuracy']:.3f}")

        gc.collect()

        trial += 1

    print("\n\nAverage scores :")

    met["average"] = {
        "f1": np.mean([m["f1"] for k, m in met.items() if "train" not in str(k)]),
        "accuracy": np.mean([m["accuracy"] for k, m in met.items() if "train" not in str(k)]),
        "loss": np.mean([m["loss"] for k, m in met.items() if "train" not in str(k)])
    }


    met["train-average"] = {
        "f1": np.mean([m["f1"] for k, m in met.items() if "train" in str(k)]),
        "accuracy": np.mean([m["accuracy"] for k, m in met.items() if "train" in str(k)]),
        "loss": np.mean([m["loss"] for k, m in met.items() if "train" in str(k)])
    }

    print("\nTRAIN : ")

    print(met["train-average"])

    print("\nVALIDATION : ")

    print(met["average"])

    print("\nStd deviation :")

    met["std"] = {
        "f1": np.std([m["f1"] for k, m in met.items() if "train" not in str(k)]),
        "accuracy": np.std([m["accuracy"] for k, m in met.items() if "train" not in str(k)]),
        "loss": np.std([m["loss"] for k, m in met.items() if "train" not in str(k)])
    }


    met["train-std"] = {
        "f1": np.std([m["f1"] for k, m in met.items() if "train" in str(k)]),
        "accuracy": np.std([m["accuracy"] for k, m in met.items() if "train" in str(k)]),
        "loss": np.std([m["loss"] for k, m in met.items() if "train" in str(k)])
    }

    print("\nTRAIN : ")

    print(met["train-std"])

    print("\nVALIDATION : ")

    print(met["std"])

    met["confusion"] = cms

    cms = [np.array(c, dtype=np.float64) for c in cms]
    cm = np.zeros_like(cms[0])
    for c in cms:
        cm += c
    cm = cm / cm.sum(axis=1)[:, np.newaxis]

    plot_confusion(None, None, syllables, oh_encoder, cm=cm)

    curr_time = time.time()
    with open(f"{_get_report_dir()}/scores-{curr_time}.json", "w+") as f:
        json.dump(met, f)

    del(cms)
    del(outputs)
    del(top_1)
    del(y_preds)

    if _get_export():
        train_export(X, y, dim_mfcc, syllables)

    return 0


def cross_validation_several_times(args, seed=None, fold_seed=None):
    """ Performs cross validation on several instances"""

    config = _get_conf_from_json()
    trainset, testset = _get_dataset_summaries()
    traindir, testdir = _get_dataset_dirs()

    dataset = pd.concat([trainset, testset])

    print("Loading dataset")
    waves = retrieve_all_waves(traindir, dataset, config)

    syllables = unique_labels(dataset.y).tolist()
    oh_encoder = OneHotEncoder(categories=[syllables], sparse=False)

    y_labels = np.array([y for _, y in waves])
    y_encoded = oh_encoder.fit_transform(np.array([y for _, y in waves]).reshape(-1, 1))

    Xy = [(x, y) for (x, _), y in zip(waves, y_encoded)]

    data = preprocess_all_waves(Xy, config)

    dim_mfcc = data[0][0][0].shape[1]
    dim_out = len(syllables)

    selected_feat = np.array([config["data"]["mfcc"], config["data"]["d"], config["data"]["d2"]])

    X = [np.hstack(np.array(feat)[selected_feat]) for feat, _ in data]
    X_indexes = np.arange(len(X)).reshape(-1, 1)
    y = [teacher for _, teacher in data]

    del(waves)
    del(data)
    del(dataset)
    del(Xy)
    del(y_encoded)
    gc.collect()

    met = {}
    cms = []

    skf = StratifiedKFold(n_splits=FLAGS.trials, shuffle=True, random_state=fold_seed)
    trial = 1
    for train_index, test_index in skf.split(X_indexes, y_labels):

        X_train, y_train = [X[i] for i in train_index], [y[i] for i in train_index]
        X_test, y_test = [X[i] for i in test_index], [y[i] for i in test_index]

        print(f"---FOLD No {trial}---")

        Win, W, Wfb, seed = build_matrices(dim_mfcc, dim_out, config, seed=seed)

        lr = config["esn"]["lr"]
        input_bias = config["esn"]["input_bias"]
        ridge = config["esn"]["ridge"]
        feedback = config["esn"]["feedback"]
        fbfunc = None
        if feedback:
            fbfunc = F.get_function(config["esn"]["fbfunc"])

        reservoir = ESN(lr=lr, input_bias=input_bias, W=W, Win=Win,
                        Wfb=Wfb, fbfunc=fbfunc, ridge=ridge, typefloat=np.float64)

        warmup = config["esn"]["wash_nr_time_step"]
        _ = reservoir.train(X_train, y_train, wash_nr_time_step=warmup,
                            verbose=True, use_memmap=True, backend="loky", workers=-1)

        outputs, _ = reservoir.run(X_train, verbose=True, backend="loky", workers=-1)

        top_1 = [np.array([syllables[t]
                        for t in outputs[i].argmax(axis=1)]).reshape(-1, 1)
                for i in range(len(outputs))]

        y_preds = [oh_encoder.transform(t) for t in top_1]

        met[f'train-{trial}'] = {
            "f1": metrics.f1_score(np.vstack(y_preds), np.vstack(y_train), average='macro'),
            "accuracy": metrics.accuracy_score(np.vstack(y_preds), np.vstack(y_train)),
            "loss": metrics.log_loss(np.vstack(y_preds), np.vstack(y_train)),
            "seed": seed
        }

        print(f"Training scores (top 1):\
    \n Cross-entropy: {met[f'train-{trial}']['loss']:.3f}, F1: {met[f'train-{trial}']['f1']:.3f}, Accuracy: {met[f'train-{trial}']['accuracy']:.3f}")

        print(f"Running validation")
        outputs, _ = reservoir.run(X_test, verbose=True)

        top_1 = [np.array([syllables[t]
                        for t in outputs[i].argmax(axis=1)]).reshape(-1, 1)
                for i in range(len(outputs))]

        y_preds = [oh_encoder.transform(t) for t in top_1]

        met[trial] = {
            "f1": metrics.f1_score(np.vstack(y_preds), np.vstack(y_test), average='macro'),
            "accuracy": metrics.accuracy_score(np.vstack(y_preds), np.vstack(y_test)),
            "loss": metrics.log_loss(np.vstack(y_preds), np.vstack(y_test)),
            "seed": seed
        }

        cms.append(metrics.confusion_matrix(
            oh_encoder.inverse_transform(np.vstack(y_test)),
            oh_encoder.inverse_transform(np.vstack(y_preds)),
            labels=syllables).tolist())

        print(f"Validation scores (top 1):\n Cross-entropy: {met[trial]['loss']:.3f}, F1: {met[trial]['f1']:.3f}, Accuracy: {met[trial]['accuracy']:.3f}")

        gc.collect()

        trial += 1

    print("\n\nAverage scores :")

    met["average"] = {
        "f1": np.mean([m["f1"] for k, m in met.items() if "train" not in str(k)]),
        "accuracy": np.mean([m["accuracy"] for k, m in met.items() if "train" not in str(k)]),
        "loss": np.mean([m["loss"] for k, m in met.items() if "train" not in str(k)])
    }


    met["train-average"] = {
        "f1": np.mean([m["f1"] for k, m in met.items() if "train" in str(k)]),
        "accuracy": np.mean([m["accuracy"] for k, m in met.items() if "train" in str(k)]),
        "loss": np.mean([m["loss"] for k, m in met.items() if "train" in str(k)])
    }

    print("\nTRAIN : ")

    print(met["train-average"])

    print("\nVALIDATION : ")

    print(met["average"])

    print("\nStd deviation :")

    met["std"] = {
        "f1": np.std([m["f1"] for k, m in met.items() if "train" not in str(k)]),
        "accuracy": np.std([m["accuracy"] for k, m in met.items() if "train" not in str(k)]),
        "loss": np.std([m["loss"] for k, m in met.items() if "train" not in str(k)])
    }


    met["train-std"] = {
        "f1": np.std([m["f1"] for k, m in met.items() if "train" in str(k)]),
        "accuracy": np.std([m["accuracy"] for k, m in met.items() if "train" in str(k)]),
        "loss": np.std([m["loss"] for k, m in met.items() if "train" in str(k)])
    }

    print("\nTRAIN : ")

    print(met["train-std"])

    print("\nVALIDATION : ")

    print(met["std"])

    met["confusion"] = cms

    cms = [np.array(c, dtype=np.float64) for c in cms]
    cm = np.zeros_like(cms[0])
    for c in cms:
        cm += c
    cm = cm / cm.sum(axis=1)[:, np.newaxis]

    plot_confusion(None, None, syllables, oh_encoder, cm=cm)

    curr_time = time.time()
    with open(f"{_get_report_dir()}/scores-{curr_time}.json", "w+") as f:
        json.dump(met, f)

    del(cms)
    del(outputs)
    del(top_1)
    del(y_preds)

    if _get_export():
        train_export(X, y, dim_mfcc, syllables)

    return 0


def train_export(X, y, dim_mfcc, syllables):
    """Train on all data an export the model"""

    gc.collect()

    print("---Training final model---")

    config = _get_conf_from_json()

    dim_out = len(syllables)

    Win, W, Wfb, seed = build_matrices(dim_mfcc, dim_out, config)

    lr = config["esn"]["lr"]
    input_bias = config["esn"]["input_bias"]
    ridge = config["esn"]["ridge"]
    feedback = config["esn"]["feedback"]
    fbfunc = None

    if feedback:
        fbfunc = F.get_function(config["esn"]["fbfunc"])

    reservoir = ESN(lr=lr, input_bias=input_bias, W=W, Win=Win,
                    Wfb=Wfb, fbfunc=fbfunc, ridge=ridge, typefloat=np.float64)

    warmup = config["esn"]["wash_nr_time_step"]
    _ = reservoir.train(X, y, wash_nr_time_step=warmup, verbose=True,
                        use_memmap=True, backend="loky", workers=-1)

    print(f"Saving model to {_get_save_dir()}")
    reservoir.save(_get_save_dir())

    np.save(Path(_get_save_dir()) / "vocab.npy", syllables)

    with Path(_get_save_dir(), "config.json").open("w+") as f:
        json.dump(config, f)

    return 0


def main(args):

    if _get_trials() == 1:
        response = train(args)
    else:
        if _get_fold():
            if _get_instances() > 1:

                print(f"-----TRAINING {_get_instances()}-----")
                for i in range(_get_instances()):
                    print(f"############{i+1}")
                    response = cross_validation_several_times(*args, seed=SEEDS[i], fold_seed=FOLD_SEED)
            else:
                response = cross_validation(args)
        else:
            response = train_several_times(args)

    return response

if __name__ == "__main__":

    app.run(main)

import json
import os
import time
import pickle
from os import path
from pathlib import Path

import joblib
import librosa as lbr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags
from reservoirpy import activationsfunc as F
from reservoirpy import mat_gen
from reservoirpy.nodes import ESN, Reservoir, Ridge
from scipy import signal
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm

sns.set(context="paper", style="dark")
FLAGS = flags.FLAGS

# path to your data root folder.
# Here, it would be "./data_example" for instance.
DATA = "../data/syllables-16-ot-e-noise"

# Number of folds
FOLDS = 1

# Number of random instances per fold
INSTANCES = 1

# Name of the model (important; needs to be unique at every launch)
MODEL = "./models/classifier-pre"

# Seeds for the different instances
SEEDS = [55448, 2651, 55541, 58964, 4100, 1000, 2000, 4000, 5000]

# Activate cross validation
FOLD = True

# Seed for Kfold
FOLD_SEED = 5555

# Export complete model at the end, trained on all training data
EXPORT = True

# Training configuration (can be loaded in a JSON file)
CONF = {
    "esn": {
        "N": 1000,  # nb of units in the reservoir
        "ridge": 1e-8,  # regularization param
        "lr": 5e-2,  # leak rate
        "sr": 0.5,  # spectral radius
        "input_bias": True,  # quite clear by itself
        "mfcc_scaling": 0.0,  # input scaling of MFCC (no MFCC in our case)
        "delta_scaling": 1.0,  # same for 1st derivative
        "delta2_scaling": 0.7,  # same for 2nd
        "feedback": False,  # no feedback connections
        "feedback_scaling": 0.0,
        "fbfunc": "softmax",
        "input_connectivity": 0.1,
        "rc_connectivity": 0.1,
        "fb_connectivity": 0.1,
        "wash_nr_time_step": 0,  # no transient states (no warmup)
        "seed": 42
        },
    "data": {  # everything librosa related
        "sampling_rate": 16000,
        "hop_length": int(16000 * 0.01),
        # this is a bit wrong. we shouldnt change the default "n_fft" value,
        # but the "window_length" parameter instead.
        "n_fft": int(16000 * 0.02),
        "fmax": 8000,
        "fmin": 500,
        "n_mfcc": 20,
        "padding": "nearest",
        # mode of the derivation of MFCC (for border effects)
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

flags.DEFINE_string('conf', f'reports/{MODEL}/training-conf.json',
                    "Path to the JSON config file.")
flags.DEFINE_string('data', f'{DATA}',
                    'Path to dataset with train and test directories.')
flags.DEFINE_string('report', f'./reports/{MODEL}', 'Results directory.')
flags.DEFINE_string('save', f'{MODEL}',
                    'Directory where the trained model will be saved.')
flags.DEFINE_integer('workers', -1,
                     'Number of parallel processes to launch for computation.',
                     lower_bound=-1)
flags.DEFINE_integer('folds', FOLDS,
                     'Number of trials to perform to compute metrics',
                     lower_bound=1)
flags.DEFINE_boolean('fold', FOLD,
                     'If set, will perform cross validation on the number of '
                     'folds set by folds parameter.')
flags.DEFINE_boolean('export', EXPORT,
                     'If set, model is exported at the end (training with '
                     'all available data).')
flags.DEFINE_integer('instances', INSTANCES, 'Number of instances to train.')


def _get_folds():
    return FLAGS.folds


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
    savedir = Path(FLAGS.save)
    if not savedir.exists():
        savedir.mkdir(parents=True)
    return savedir


def _get_report_dir():
    reportdir = path.join(FLAGS.report)
    if not (path.isdir(reportdir)):
        os.makedirs(reportdir)
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

    if not (path.isdir(traindir)):
        raise NotADirectoryError(f"Dataset '{traindir}' not found.")
    elif not (path.isdir(testdir)):
        raise NotADirectoryError(f"Dataset '{testdir}' not found.")

    if not (path.isfile(confpath)):
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


def compute_features(wave, y, sr, hop_length, n_fft, padding, trim_silence,
                     highpass, order, lifter, n_mfcc):
    w = wave
    if trim_silence:
        w, _ = lbr.effects.trim(w, top_db=trim_silence)

    if highpass:
        w = apply_filter(w, sr, highpass, order)

    mfcc = lbr.feature.mfcc(w, sr, hop_length=hop_length, n_fft=n_fft,
                            win_length=n_fft, lifter=lifter, n_mfcc=n_mfcc)
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
    n_mfcc = config["data"]["n_mfcc"]

    all_data = loop(
        delayed_feat(w, y, sr, hop_length, n_fft, mode, trim, highpass, order,
                     lifter, n_mfcc)
        for w, y in tqdm(all_waves, "Preprocessing", total=len(all_waves)))

    return all_data


def retrieve_all_waves(datadir, dataset, config):
    loop = joblib.Parallel(n_jobs=_get_workers(), backend='threading')
    sr = config["data"]["sampling_rate"]

    delayed_load = joblib.delayed(load_wave)

    path_and_labels = zip([path.join(datadir, x) for x in dataset.x],
                          dataset.y)

    all_waves = loop(delayed_load(x, y, sr)
                     for x, y in tqdm(path_and_labels, "Loading waves",
                                      total=len(dataset)))

    return all_waves


# --------- Model creation ---------------
def create_esn(dim_mfcc, config, seed=None):
    sr = config["esn"]["sr"]
    lr = config["esn"]["lr"]
    N = config["esn"]["N"]
    ridge = config["esn"]["ridge"]
    warmup = config["esn"]["wash_nr_time_step"]
    input_bias = config["esn"]["input_bias"]
    iss = config["esn"]["mfcc_scaling"]
    isd = config["esn"]["delta_scaling"]
    isd2 = config["esn"]["delta2_scaling"]
    isfb = config["esn"].get("feedaback_scaling", 0.0)
    feedback = config["esn"]["feedback"]
    input_connectivity = config["esn"].get("input_connectivity", 0.1)
    rc_connectivity = config["esn"].get("rc_connectivity", 0.1)
    fb_connectivity = config["esn"].get("fb_connectivity", 0.1)
    fb_func = config["esn"].get("fbfunc", "identity")

    if seed is None:
        seed = config["esn"].get("seed")

    if seed is None:
        seed = np.random.randint(0, 99999)

    rng = np.random.default_rng(seed)

    input_matrices = []
    bias_added = False
    if config["data"]["mfcc"]:
        input_matrices.append(mat_gen.generate_input_weights(N=N,
                                                             dim_input=dim_mfcc,
                                                             input_bias=input_bias,
                                                             input_scaling=iss,
                                                             seed=rng,
                                                             typefloat=np.float64))
        bias_added = True

    if config["data"]["d"]:
        input_matrices.append(
            mat_gen.generate_input_weights(N=N, dim_input=dim_mfcc,
                                           input_bias=(not (
                                               bias_added) and input_bias),
                                           input_scaling=isd,
                                           seed=rng, typefloat=np.float64))
        bias_added = True

    if config["data"]["d2"]:
        input_matrices.append(
            mat_gen.generate_input_weights(N=N, dim_input=dim_mfcc,
                                           input_bias=(not (
                                               bias_added) and input_bias),
                                           input_scaling=isd2,
                                           seed=rng, typefloat=np.float64))

    Win = np.concatenate(input_matrices, axis=1)

    reservoir = Reservoir(N, lr=lr, sr=sr, input_bias=input_bias, Win=Win,
                          input_connectivity=input_connectivity,
                          rc_connectivity=rc_connectivity,
                          fb_connectivity=fb_connectivity, fb_scaling=isfb,
                          seed=rng, fb_activation=F.get_function(fb_func))

    readout = Ridge(ridge=ridge, transient=warmup, input_bias=True)

    esn = ESN(reservoir=reservoir, readout=readout, workers=_get_workers(),
              feedback=feedback)

    return esn, seed


# ---------- PLOTTING ------------
def plot_confusion(y_test, y_preds, syllables, oh_encoder, cm=None):
    if cm is None:
        cm = metrics.confusion_matrix(
            oh_encoder.inverse_transform(np.vstack(y_test)),
            oh_encoder.inverse_transform(np.vstack(y_preds)),
            labels=syllables,
            normalize='true')

    fig = plt.figure(figsize=(20, 20))

    ax = fig.add_subplot((111))
    plt.xlabel("Predictions", fontsize=20)
    plt.ylabel("Truths", fontsize=20)
    ax.xaxis.set_label_position('top')
    im = ax.matshow(cm, cmap="Blues",
                    norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1.))

    ax.set_xticks(np.arange(len(syllables)))
    ax.set_yticks(np.arange(len(syllables)))

    ax.set_xticklabels(syllables)
    ax.set_yticklabels(syllables)
    ax.tick_params('x', top=True, bottom=False, labeltop=True,
                   labelbottom=False, rotation=45)

    for i in range(len(syllables)):
        for j in range(len(syllables)):
            _ = ax.text(j, i, round(cm[i, j], 2),
                        ha="center", va="center", color="gray")

    plt.colorbar(im)
    fig.tight_layout()

    plt.show()

    fig.savefig(Path(_get_report_dir()) / "confusion.png")


def load_and_preprocess():
    config = _get_conf_from_json()
    trainset, testset = _get_dataset_summaries()
    traindir, testdir = _get_dataset_dirs()

    print("Loading training set")
    train_waves = retrieve_all_waves(traindir, trainset, config)
    print("Loading testing set")
    test_waves = retrieve_all_waves(testdir, testset, config)

    syllables = unique_labels(pd.concat([trainset, testset]).y).tolist()
    oh_encoder = OneHotEncoder(categories=[syllables], sparse=False)

    y_train_encoded = oh_encoder.fit_transform(
        np.array([y for _, y in train_waves]).reshape(-1, 1))
    y_test_encoded = oh_encoder.fit_transform(
        np.array([y for _, y in test_waves]).reshape(-1, 1))

    train_waves = [(x, y) for (x, _), y in zip(train_waves, y_train_encoded)]
    test_waves = [(x, y) for (x, _), y in zip(test_waves, y_test_encoded)]

    print('Preprocessing training set')
    train_data = preprocess_all_waves(train_waves, config)
    print('Preprocessing testing set')
    test_data = preprocess_all_waves(test_waves, config)

    selected_feat = np.array(
        [config["data"]["mfcc"], config["data"]["d"], config["data"]["d2"]])

    x_train = [np.hstack(np.array(feat)[selected_feat]) for feat, _ in
               train_data]
    y_train = [y for _, y in train_data]

    x_test = [np.hstack(np.array(feat)[selected_feat]) for feat, _ in
              test_data]
    y_test = [y for _, y in test_data]

    if config["data"]["continuous"]:
        x_train = [np.vstack(x_train)]
        y_train = [np.vstack(y_train)]

        x_test = [np.vstack(x_test)]
        y_test = [np.vstack(y_test)]

    return config, x_train, y_train, x_test, y_test, oh_encoder, syllables


def train(*args, seed=None, data=None, save=True, eval=True):
    """
    Train the model once on all training data. Test on testing data.
    """
    if data is None:
        (config, x_train, y_train,
         x_test, y_test, oh_encoder, syllables) = load_and_preprocess()
    else:
        (config, x_train, y_train,
         x_test, y_test, oh_encoder, syllables) = data

    dim_mfcc = config["data"]["n_mfcc"]

    esn, seed = create_esn(dim_mfcc, config, seed=seed)
    esn.fit(x_train, y_train)

    print(f"Running tests")
    outputs = esn.run(x_test)

    top_1 = [np.array([syllables[t]
                       for t in outputs[i].argmax(axis=1)]).reshape(-1, 1)
             for i in range(len(outputs))]

    y_preds = [oh_encoder.transform(t) for t in top_1]

    if eval:
        met = {
            "f1": metrics.f1_score(np.vstack(y_test), np.vstack(y_preds),
                                   average='macro'),
            "accuracy": metrics.accuracy_score(np.vstack(y_test),
                                               np.vstack(y_preds)),
            "loss": metrics.log_loss(np.vstack(y_test), np.vstack(y_preds)),
            "seed": seed
            }

        if save:
            curr_time = time.time()
            with open(f"{_get_report_dir()}/scores-{curr_time}.json", "w+") as f:
                json.dump(met, f)

            print(
                f"Scores (top 1):\n Cross-entropy: {met['loss']:.3f}, F1: "
                f"{met['f1']:.3f}, Accuracy: {met['accuracy']:.3f}")

            plot_confusion(np.vstack(y_test), np.vstack(y_preds), syllables,
                           oh_encoder)

    return y_preds, esn


def train_several_times(*args, data=None):
    """
    Train the model on several instances (as much as INSTANCES parameter) (and
    save a posteriori scores for all)
    """
    if data is None:
        (config, x_train, y_train,
         x_test, y_test, oh_encoder, syllables) = load_and_preprocess()
    else:
        (config, x_train, y_train,
         x_test, y_test, oh_encoder, syllables) = data

    dim_mfcc = config["data"]["n_mfcc"]

    met = {i: {} for i in range(FLAGS.folds)}
    for i in range(FLAGS.folds):

        print(f"---INSTANCE No {i}---")

        seed = SEEDS[i]

        y_preds, esn = train(args, seed=seed, data=(config, x_train, y_train,
                                                    x_test, y_test, oh_encoder,
                                                    syllables), eval=False)

        met[i] = {
            "f1": metrics.f1_score(np.vstack(y_test), np.vstack(y_preds),
                                   average='macro'),
            "accuracy": metrics.accuracy_score(np.vstack(y_test),
                                               np.vstack(y_preds)),
            "loss": metrics.log_loss(np.vstack(y_test), np.vstack(y_preds)),
            "seed": seed
            }

        print(
            f"Scores (top 1):\n Cross-entropy: {met[i]['loss']:.3f}, "
            f"F1: {met[i]['f1']:.3f}, Accuracy: {met[i]['accuracy']:.3f}")

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


def cross_validation(*args, best_metrics=0, best_seed=0, seed=None, data=None):
    """Performs cross validation and saves the results"""

    if data is None:
        (config, x_train, y_train,
         x_test, y_test, oh_encoder, syllables) = load_and_preprocess()
    else:
        (config, x_train, y_train,
         x_test, y_test, oh_encoder, syllables) = data

    dim_mfcc = config["data"]["n_mfcc"]

    met = {}
    cms = []
    cms_test = []
    seed = seed if seed is not None else config["esn"]["seed"]

    skf = StratifiedKFold(n_splits=FLAGS.folds, shuffle=True,
                          random_state=FOLD_SEED)

    y_label = [np.array([syllables[t]
                         for t in y_train[i].argmax(axis=1)])[0]
               for i in range(len(y_train))]

    trial = 1
    for train_index, val_index in skf.split(np.zeros(len(x_train)), y_label):

        x, y = [x_train[i] for i in train_index], \
               [y_train[i] for i in train_index]
        x_val, y_val = [x_train[i] for i in val_index], \
                       [y_train[i] for i in val_index]

        print(f"---FOLD No {trial}---")

        print("Training")
        y_preds, esn = train(args, seed=seed, data=(config, x, y,
                                                    x, y, oh_encoder,
                                                    syllables), eval=False)

        met[f'train-{trial}'] = {
            "f1": metrics.f1_score(np.vstack(y), np.vstack(y_preds),
                                   average='macro'),
            "accuracy": metrics.accuracy_score(np.vstack(y),
                                               np.vstack(y_preds)),
            "loss": metrics.log_loss(np.vstack(y), np.vstack(y_preds)),
            "seed": seed
            }

        print(f"Training scores (top 1):\
    \n Cross-entropy: {met[f'train-{trial}']['loss']:.3f}, F1: "
              f"{met[f'train-{trial}']['f1']:.3f}, "
              f"Accuracy: {met[f'train-{trial}']['accuracy']:.3f}")

        print("Running validation")
        outputs = esn.run(x_val)

        top_1 = [np.array([syllables[t]
                           for t in outputs[i].argmax(axis=1)]).reshape(-1, 1)
                 for i in range(len(outputs))]

        y_preds = [oh_encoder.transform(t) for t in top_1]

        met[trial] = {
            "f1": metrics.f1_score(np.vstack(y_val), np.vstack(y_preds),
                                   average='macro'),
            "accuracy": metrics.accuracy_score(np.vstack(y_val),
                                               np.vstack(y_preds)),
            "loss": metrics.log_loss(np.vstack(y_val), np.vstack(y_preds)),
            "seed": seed
            }

        cms.append(metrics.confusion_matrix(
            oh_encoder.inverse_transform(np.vstack(y_val)),
            oh_encoder.inverse_transform(np.vstack(y_preds)),
            labels=syllables).tolist())

        print(
            f"Validation scores (top 1):\n Cross-entropy: "
            f"{met[trial]['loss']:.3f}, F1: {met[trial]['f1']:.3f}, "
            f"Accuracy: {met[trial]['accuracy']:.3f}")

        print("Running test")
        outputs = esn.run(x_test)

        top_1 = [np.array([syllables[t]
                           for t in outputs[i].argmax(axis=1)]).reshape(-1, 1)
                 for i in range(len(outputs))]

        y_preds = [oh_encoder.transform(t) for t in top_1]

        met[f'test-{trial}'] = {
            "f1": metrics.f1_score(np.vstack(y_test), np.vstack(y_preds),
                                   average='macro'),
            "accuracy": metrics.accuracy_score(np.vstack(y_test),
                                               np.vstack(y_preds)),
            "loss": metrics.log_loss(np.vstack(y_test), np.vstack(y_preds)),
            "seed": seed
            }

        cms_test.append(metrics.confusion_matrix(
            oh_encoder.inverse_transform(np.vstack(y_test)),
            oh_encoder.inverse_transform(np.vstack(y_preds)),
            labels=syllables).tolist())

        print(
            f"Test scores (top 1):\n Cross-entropy: "
            f"{met[f'test-{trial}']['loss']:.3f}, "
            f"F1: {met[f'test-{trial}']['f1']:.3f}, "
            f"Accuracy: {met[f'test-{trial}']['accuracy']:.3f}")

        trial += 1

    print("\n\nAverage scores :")

    met["average"] = {
        "f1": np.mean(
            [m["f1"] for k, m in met.items() if "train" not in str(k)]),
        "accuracy": np.mean(
            [m["accuracy"] for k, m in met.items() if "train" not in str(k)]),
        "loss": np.mean(
            [m["loss"] for k, m in met.items() if "train" not in str(k)])
        }

    met["train-average"] = {
        "f1": np.mean([m["f1"] for k, m in met.items() if "train" in str(k)]),
        "accuracy": np.mean(
            [m["accuracy"] for k, m in met.items() if "train" in str(k)]),
        "loss": np.mean(
            [m["loss"] for k, m in met.items() if "train" in str(k)])
        }

    met["test-average"] = {
        "f1": np.mean([m["f1"] for k, m in met.items() if "test" in str(k)]),
        "accuracy": np.mean(
            [m["accuracy"] for k, m in met.items() if "test" in str(k)]),
        "loss": np.mean(
            [m["loss"] for k, m in met.items() if "test" in str(k)])
        }

    if met['test-average']["accuracy"] > best_metrics:
        best_metrics = met[f'test-average']["accuracy"]
        best_seed = seed

    print("\nTRAIN : ")

    print(met["train-average"])

    print("\nVALIDATION : ")

    print(met["average"])

    print("\nTEST : ")

    print(met["test-average"])

    print("\nStd deviation :")

    met["std"] = {
        "f1": np.std(
            [m["f1"] for k, m in met.items() if "train" not in str(k)]),
        "accuracy": np.std(
            [m["accuracy"] for k, m in met.items() if "train" not in str(k)]),
        "loss": np.std(
            [m["loss"] for k, m in met.items() if "train" not in str(k)])
        }

    met["train-std"] = {
        "f1": np.std([m["f1"] for k, m in met.items() if "train" in str(k)]),
        "accuracy": np.std(
            [m["accuracy"] for k, m in met.items() if "train" in str(k)]),
        "loss": np.std(
            [m["loss"] for k, m in met.items() if "train" in str(k)])
        }

    met["test-std"] = {
        "f1": np.std([m["f1"] for k, m in met.items() if "test" in str(k)]),
        "accuracy": np.std(
            [m["accuracy"] for k, m in met.items() if "test" in str(k)]),
        "loss": np.std(
            [m["loss"] for k, m in met.items() if "test" in str(k)])
        }

    print("\nTRAIN : ")

    print(met["train-std"])

    print("\nVALIDATION : ")

    print(met["std"])

    print("\nTESt : ")

    print(met["test-std"])

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

    return best_metrics, best_seed


def train_export(best_seed, data=None):
    """Train on all available data an export the model."""

    print("---Training final model---")

    if data is None:
        (config, x_train, y_train,
         x_test, y_test, oh_encoder, syllables) = load_and_preprocess()
    else:
        (config, x_train, y_train,
         x_test, y_test, oh_encoder, syllables) = data

    x_train.extend(x_test)
    y_train.extend(y_test)

    dim_mfcc = config["data"]["n_mfcc"]

    esn, seed = create_esn(dim_mfcc, config, seed=best_seed)

    esn.fit(x_train, y_train)

    print(f"Saving model to {_get_save_dir()}")

    with Path(_get_save_dir(), "esn.pkl").open("w+b") as fp:
        pickle.dump(esn, fp)

    np.save(str(Path(_get_save_dir()) / "vocab.npy"), syllables)

    with Path(_get_save_dir(), "config.json").open("w+") as f:
        json.dump(config, f)


def main(args):
    best_metrics = 0
    best_seed = _get_conf_from_json()['esn']['seed']

    data = load_and_preprocess()

    if _get_folds() == 1 and _get_instances() == 1:
        train(args, data=data)
    else:
        if _get_folds() > 1 and _get_fold():
            if _get_instances() > 1:
                print(f"----- TRAINING {_get_instances()} instances -----")

                for i in range(_get_instances()):
                    print(f"############ Run {i + 1}")
                    best_metrics, best_seed = cross_validation(seed=SEEDS[i],
                                                               data=data,
                                                               best_metrics=best_metrics,
                                                               best_seed=best_seed)
            else:
                cross_validation(args, data=data)
        else:
            train_several_times(args, data=data)

    if FLAGS.export:
        train_export(best_seed, data=data)

    return 0


if __name__ == "__main__":
    app.run(main)

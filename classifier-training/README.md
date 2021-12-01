# Classifier model training

## Requirements

Decoder requirements can be found in the `requirements.txt` file, and installed via:

```bash
pip install -r requirements.txt
```

## Training

Training can be performed using command-line interface to launch the `train.py` script:

```bash
python classifier-training/train.py --help

       USAGE: classifier-training/train.py [flags]
flags:

classifier-training/train.py:
  --conf: Path to the JSON config file.
  --data: Path to dataset with train and test directories.
  --[no]export: If set, model is exported at the end (training with all available data).
    (default: 'true')
  --[no]fold: If set, will perform cross validation on the number of folds set by folds parameter.
    (default: 'true')
  --folds: Number of trials to perform to compute metrics
    (default: '1')
    (a positive integer)
  --instances: Number of instances to train.
    (default: '1')
    (an integer)
  --report: Results directory.
  --save: Directory where the trained model will be saved.
  --workers: Number of parallel processes to launch for computation.
    (default: '-1')
    (integer >= -1)
```

For instance, to perform 5-fold cross validation over 5 different random
initializations of the classifier, and to save the best final model, 
run:

```bash
python train.py --conf reports/conf.json --data data/classifier-data --export --fold --folds 5 --instances 5 --report reports/ --save models/classifier-model
```

## Data format

Data should be provided as two .csv files, `train_dataset.csv` and `test_dataset.csv`.
These .csv files have two columns, `x` and `y`, where `x` is the path
to the syllable audio sample in .wav format and `y` is the corresponding syllable label:

| x                   | y |
|---------------------|---|
| ./data/sample_1.wav | A |
| ./data/sample_2.wav | B |
| ./data/sample_3.wav | C |

`train_dataset.csv` contains all samples used during training and
cross-validation. `test_dataset.csv` contains all samples used during
testing.

## Configuration file

Configuration should be provided as a JSON file:

```json
{
  "esn": {
    "N": 1000,
    "ridge": 1e-8,
    "lr": 5e-2,
    "sr": 0.5,
    "input_bias": true,
    "mfcc_scaling": 0.0,
    "delta_scaling": 1.0,
    "delta2_scaling": 0.7,
    "feedback": false,
    "feedback_scaling": 0.0,
    "fbfunc": "softmax",
    "input_connectivity": 0.1,
    "rc_connectivity": 0.1,
    "fb_connectivity": 0.1,
    "wash_nr_time_step": 0,
    "seed": 42
  },
  "data": {
    "sampling_rate": 16000,
    "hop_length": 160,
    "n_fft": 320,
    "fmax": 8000,
    "fmin": 500,
    "n_mfcc": 20,
    "padding": "nearest",
    "trim_silence": false,
    "continuous": false,
    "mfcc": false,
    "d": true,
    "d2": true,
    "highpass": null,
    "order": null,
    "lifter": 0
  }
}
```

Parameters under `esn` key are hyperparameters of the Echo State Network classifier.

| param              | default | comment                                                 |
|--------------------|---------|---------------------------------------------------------|
| N                  | 1000    | Number of units in the Echo State Network reservoir.    |
| ridge              | 1e-8    | Regularization coefficient for learning.                |
| lr                 | 5e-2    | Leak rate for Echo State Network.                       |
| sr                 | 0.5     | Spectral radius of the ESN recurrent matrix.            |
| input_bias         | true    | Add bias parameter to inputs.                           |
| mfcc_scaling       | 0.0     | Scaling coefficient applied to MFCCs.                   |
| delta_scaling      | 1.0     | Scaling coefficient applied to MFCC 1st derivatives.    |
| delta2_scaling     | 0.7     | Scaling coefficient applied to MFCC 2nd derivatives.    |
| feedback           | false   | Activate feedback from readout to reservoir in the ESN. |
| feedback_scaling   | 0.0     | Scaling coefficient of feedback vector.                 |
| fbfunc             | softmax | Activation function for readout values.                 |
| input_connectivity | 0.1     | Input matrix connectivity.                              |
| rc_connectivity    | 0.1     | Recurrent matrix connectivity.                          |
| fb_connectivity    | 0.1     | Feedback matrix connectivity.                           |
| wash_nr_time_step  | 0       | Input timesteps to consider as warmup during training.  |
| seed               | 42      | Random generator seed for model initialization.         |

Parameters under `data` key are data preprocessing parameters:

| sampling_rate | 16000         | Audio file sampling rate.                                                                                            |
|---------------|---------------|----------------------------------------------------------------------------------------------------------------------|
| hop_length    | 160000 x 0.01 | Strides between FFT analysis windows.                                                                                |
| n_fft         | 160000 x 0.02 | Number of FFT components (size of the window).                                                                       |
| fmax          | 8000          | Maximum authorized frequency for samples.                                                                            |
| fmin          | 500           | Minimum authorized frequency for samples.                                                                            |
| n_mfcc        | 20            | Number of MFCC coefficients to extract.                                                                              |
| padding       | nearest       | Padding method used to compute derivatives at signal edges.                                                          |
| trim_silence  | false         | If is set to a number > 0, will use `librosa.trim_silence` method to remove silent parts under this power threshold. |
| continuous    | false         | If true, concatenate all samples for training.                                                                       |
| mfcc          | false         | Whether to use MFCC as features.                                                                                     |
| d             | true          | Wether to use MFCC 1st derivatives as features.                                                                      |
| d2            | true          | Wether to use MFCC 2nd derivatives as features.                                                                      |
| highpass      | null          | If >0, will apply a highpass filter using this value as cut frequency.                                               |
| order         | null          | Order of the highpass filter.                                                                                        |
| lifter        | 0             | Liftering coefficient of MFCC features.                                                                              |

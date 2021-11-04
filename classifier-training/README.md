# Canary decoder (alpha version 0.0.a1)

## Installation

Download the source files of `reservoirpy` (https://github.com/neuronalX/reservoirpy.git) and `canarydecoder`.

From within a virtual environment, run:

```bash
pip install <path/to/reservoirpy/root>
pip install <path/to/canarydecoder/root>
```

where the root directory contains the `setup.py` file and the `canarydecoder/` directory (same for `reservoirpy`).

## Get started

### Load a `Decoder`

In the following example, `canarydecoder` is used to produce annotations of samples of canary songs.

The input data is stored in a directory. Each input is a 1s sample of song.

To produce annotations, we first load a pre-trained `Decoder`, named `canary16-deltas`.

```python
from canarydecoder import load

decoder = load('canary16-deltas')
```

The `Decoder` object stores procedures to decode the input data. It can be seen as a pipeline objects, that manage the flow of data through
two main components, the `Processor` and the model.

`Processors` are objects in charge with preprocessing the input data and extracting the relevant features. The `Processor` object is instanciated following a configuration file stored in the same directory as the model data. This configuration
tells the `Processor` how to extract the required features in the input data.

The model is in this case a `reservoirpy` ESN object, loaded from checkpoint data stored in the model directory.

### Produce annotations

We can now call the `Decoder` on a list of arrays, on a directory containing .wav files or on a single .wav file :

```python
annotations = decoder('./samples/of/canary/songs')
```

The `annotations` object will then be a list of `Annotation` instances.

### Work with `Annotation`

`Annotation` objects store the input and output of a model. Through them, you can access:
- `audio`: the original audio data;
- `feat`: the features extracted by the `Processor`, for example MFCC;
- `vect`: the raw logits output of the model used;
- `lbl`: the top 1 class predicted by the model for each timestep;
- `vocab`: the vocabulary used for annotations;
- `id`: an identification for inputs, by default the name of the wave file decoded or an integer.

`Annotation` objects behave like iterable Python object, meaning you can access a particular timestep by its index, or slice, or iterate on them. You will have the corresponding slice of annotation displayed as a new `Annotation` instance.


```python
annotation.vect # output logits
annotation.vocab # class name attributed to each logit

one_annotation = annotation[0] # -> an Annotation of only one timestep, with corresponding feature, audio...
sliced_annotation_audio = annotation[0:5].audio # -> 5 first Annotations audio data

# Most of Annotation attributes are simple NumPy arrays or lists of arrays.
mean_vects = np.mean(annotation.vect)
```

## Tune parameters

Well please don't tune them... It doesn't work yet (or you would have to retrain everything, but the training process is being mean with me and still doesn't want to work)

But theoretically, you will soon be able to:
- plug a custom model
- train a custom model
- tune everything 

For now, it is technically possible to change parameters by changing the configuration files inside the `canarydecoder/decoding/models/` directory (or inside any trained model directory). But the ESN won't respond well to variations in preprocessing parameters (and even worse with model hyperparameters, of course).

## Available models

- `canary16`: ESN canary decoder with a 16 syllables repertory, trained over MFCC + deltas1 + deltas2.
- `canary16-deltas`: ESN canary decoder with 16 syllables repertory, trained over only deltas1 + deltas2 of MFCC.

Both models contains a `Processor` configuration to load and preprocess audio data and a pre-trained ESN models.
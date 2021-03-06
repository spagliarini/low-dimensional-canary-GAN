
SHORT ABSTRACT

Altogether, our results show that a latent space
of dimension 3 is enough to produce a varied repertoire of sounds
of quality often indistinguishable from real canary ones, spanning
all the types of syllables of the dataset. Importantly, we show
that the 3-dimensional GAN generalizes by interpolating between
the various syllable types. We rely on UMAP representations to
qualitatively show the similarities between the training data and
the generated data, and between the generated syllables and the
interpolations produced. Exploring the latent representations of
syllable types, we show that they form well identifiable subspaces
of the latent space. This study provides tools to train simple
sensorimotor models, as inverse models, from perceived sounds
to motor representations of the same sounds. Both the RNNbased classifier and the small dimensional GAN provide a way
to learn the mappings of perceived and produced sounds.


In the directory "examples" it is possible to find some examples of the generated data:
1) a spectrogram shows the comparison between one example from the real dataset, and two sequences obtained padding together several syllables.
2) the .wav sequences 

# REQUIREMENTS
1) WaveGAN training requirements
The requirements are the same as in the original WaveGAN project.

2) Pre-processing, classifier training and analysis requirements
To generate the sound, train the classifier, and perform the analysis the requirements are contained in the file requirements txt.



# REPOSITORY ORGANIZATION
## data-preprocessing
This directory contains the scripts useful to pre-process the data and prepare different types of datasets. ReadME available inside.

## WaveGAN training
This directory contains the scripts needed to train the generative adversarial network, mainly taken from the original WaveGAN project (reference below, and inside the directory). ReadME available inside.

## classifier-training
This directory contains the scripts to train the classifier once the training dataset is ready. Moreover, it contains the references to the necessary tools that one needs to run the experiments.

## classifier-analysis
This directory contains all the scripts to analyze the training and generated data: syllable features, classifier evaluation, utils needed to plot and perform the analysis. ReadME available inside.

## References to other projects 
### WaveGAN
Our version of WaveGAN is the same as the original one developed by Donahue et. Al, we only modified the latent space dimension parameter, and we added scripts for a straightforward generation after training in train_wavegan.py (see ReadME in wavegan-training directory).
- link to git original project: https://github.com/chrisdonahue/wavegan

### ReservoirPy
A simple and flexible code for Reservoir Computing architectures like Echo State Networks (ESN). We used this to build the syllable classifier. The version of reservoirPy needed for this project is version v0.2.0.
- link to git project: https://github.com/reservoirpy/reservoirpy
- link to ICANN paper: https://github.com/neuronalX/Trouvain2020_ICANN


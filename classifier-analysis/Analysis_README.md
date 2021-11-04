**INDEX OF THE README**

quantitative analysis

qualitative analysis

space representation (UMAP and latent space structure)

space representation (latent space exploration)

data analysis (features/mean spectrogram for dataset and generated syllables)

other codes in the project

----

**IMPORTANT EQUIVALENCE**

epoch = ckpt_n *5*64 / #dataset 

where #dataset = dataset size and ckpt_n = checkpoint

When GENERATION is involved usually the checkpoint is required, for ANALYSIS A POSTERIORI (e.g., UMAP) usually the epoch is required. 

---

# Quantitative analysis

## Requirements
This project needs to have the last version of reservoirpy and canarydecoder.

Using the function `decoder_analysis.py` it's possible to classify and quantitative analyse the training and the generated syllables. 

Virtual  environment: the same used for the project InverseModelGAN

## Create annotations
The function `create_annotations` creates a dictionary containing for each .wav file the following elements:
- name (path of the recording)
- vocab: the list of the whole vocabulary
- raw: this entry stores the raw outputs produced by the ESN. The ESN produce one annotation vector per timestep;
       the raw output does the same thing but for each timestep of input audio

The input directory should contain one or more audio files (.wav).
The output is saved in the same directory of the data (outside to be able to use the function somewhere else): it is in pickle format.

To change the input directory name, or the way to save the annotations file, go to line 2672.

To change the classifier model, go to line 38. 
In particular:
- to obtain CLASSIFIER -REAL (only the 16 syllables of the repertoire) use: `decoder = load('canary16-filtered-notrim')`
- to obtain CLASSIFIER - EXT (16 syllables of the repertoire plus 5 alternative classes EARLY15, EARLY30, EARLY45, OT, WN) use: `decoder = load('canarygan-f-3e-ot-noise-notrim') `


        python datasets.py --option annotations --data_dir INPUT_DIR

## Quantitative analysis training dataset
This function `analysis_dataset` is meant to analyse the datasets used to train the GAN and see how the decoder works on a known set of data.
The input directory contains one or more annotations. Moreover, a legend list is required as input (to deal with multiple dataset analysis).
In the output directory an histogram representing the distribution of the dataset is saved.
Moreover, in INPUT_DIR it is saved a dictionary containg:
- 'File_name': list_of_recordings (list, path to each .wav file)
- 'Real_name': original_list_of_recordings (list, the real name. Ex:'A')
- 'Decoder_name': decoder_list_of_recordings (list, the guess of the classifier. Ex:'A')
- 'Annotations': raw distribution of the dataset (vector, the total amount of syllables per class)
- 'Dataset_distr': soft-max distribution of the dataset obtained from the annotations (vector)
- 'Dataset_real_distr': soft-max distribution of the dataset obtained from the real dataset distribution (real amount of syllables per class) (vector)
- 'IS': Inception Score (float value)

To change the input directory name and the legend list, go to line 2688.
    
    python analysis_dataset.py --data_dir --data_dir INPUT_DIR --output_dir OUTPUT_DIR
    
[SAVED] Looking at the saved summaries:
 - 'Dataset_summary1.npy' : summary obtained from the 16000 complete dataset and using only the repertoire as vocabulary
 - 'Dataset_summary4.npy' : summary obtained from the 16000 complete dataset and using only the repertoire + 5 GAN classes as vocabulary

[PLUS] The function `analysis_error` to help the analysis of the errors in the dataset if of any interest. It works as the main one but has a special way to navigate to the input directory.
In the output directory an histogram representing the distribution of the dataset is saved.
Moreover, in INPUT_DIR it is saved a .npy dictionary containg:
- 'File_name': list_of_recordings (list, path to each .wav file)
- 'Real_name': original_list_of_recordings (list, the real name. Ex:'A')
- 'Decoder_name': decoder_list_of_recordings (list, the guess of the classifier. Ex:'A')
- 'Annotations': raw distribution of the dataset (vector, the total amount of syllables per class)

To change the input directory name and the legend list, go to line 2669. The legend list is very personal and depends on the type of errors arised
 in the dataset.
 
I used this function to detect errors in the dataser while building it, using the classifier to help me find the errors.
This is why it is not "general" but adapted to my paths, needs, etc. Function at line 2698.

## Quantitative analysis of the generated data
To quantitative analyse the generated data first one needs to create a suitable object (not directly the annotations, which are also heavier). 
Then, depending on the type of comparison one wants to see, different functions help the analysis.

### First step (common to any of the following)
The function `analysis_generation` takes as input the list of the annotations (pickle file, one per epoch, it can be only one epoch),
the dictionary of the training data (the one created using the function `analysis_dataset`), and all the parameters. 

To change the name of the input files, go to line 2718.
To see the parameters, go to line 2589. The important parameters for this function are:
- wavegan_latent_dim: it has to be the one used for training
- dataset_dim: it has to be the same size as the size of the training dataset
- GAN_classes: they have to be the same as the one defined for the classifier one has used to create the annotations. In particular, 
it would be an `empty` list if CLASSIFIER -REAL, or `EARLY15, EARLY30, EARLY45, OT, WN` if CLASSIFIER - EXT.

The function returns a .npy dictionary for each epoch containing
- 'File_name': GAN_list_of_recordings (list, path to each of the .wav file contained in the directory)
- 'Decoder_name': GAN_decoder_list_of_recordings (list, the guess of the classifier. Ex:'A')
- 'Annotations': raw_sum_distr (vector, the total amount of syllables per class)
- 'Decoder_index': raw_max_indices (vector, where the total amount of syllables per class is maximum)
- 'Latent_dim': ld_generation (integer, latent space dimension)
- 'Epoch': epoch (integer, epoch)
- 'IS': IS (integer, inception score)

Moreover, the function returns a cumulative .npy dictionary that summarizes the information for all the epochs. It contains
- 'mean_across_time': mean_across_time 
- 'std_across_time': std_across_time 
- 'var_across_time': var_across_time
- 'mean_across_time_noGAN': mean_across_time_noGAN 
- 'std_across_time_noGAN': std_across_time_noGAN
- 'median_across_time_noGAN': median_across_time_noGAN
- 'var_across_time_noGAN': var_across_time_noGAN
- 'percentile5_across_time_noGAN': percentile5_across_time_noGAN
- 'percentile95_across_time_noGAN': percentile95_across_time_noGAN 
- 'max_across_time_noGAN': max_across_time_noGAN
- 'min_across_time_noGAN': min_across_time_noGAN
- 'rep_classes_time': rep_classes_time
- 'classes_time': classes_time
- 'GAN_distr': GAN_distr
- 'GAN_distr_noGAN': GAN_distr_noGAN
- 'cross_entropy': classic_cross_entropy
- 'cross_entropy_noGAN':classic_cross_entropy_noGAN
- 'cross_entropy_real': classic_cross_entropy_real
- 'cross_entropy_noGAN_real':classic_cross_entropy_noGAN_real
- 'classes': classes
- 'classes_no_GAN': classes_noGAN

where `noGAN` means that the GAN classes have not been considered for this analysis, `std` stands for standard deviation, `var` stands for variance,
`distr` stands for distribution, `rep` stands for representation.

[NOTE] If one wants to compare several instances, it is required to have the same number of epochs per instance.


    python decoder_analysis.py --option analysis_gen --data_dir INPUT_DIR --wavegan_latent_dim X --dataset_size Y --output_dir OUTPUT_DIR

### Second step
* The function `several_instances` allows to compare several instances between them and to visualize average plots across different latent space dimensions.

    It takes as input
    - generation_data: summary of the generations (one per instance of the training, i.e. the general .npy dictionary created using `analysis_gen` on each instance)
    - summary_dataset: to get the distribution of the training data (i.e., the .npy dictionary created using `analysis_dataset`)
    - legend_list_avg: name of latent space dim if multiple latent space conditions
    - legend_list_instances: name of the instances, if multiple instances per latent space condition
    - colors_list : to have uniform plots across different analysis (for the instances)
    - classes_colors: to have uniform plots across different analysis (for the classes)
    
    To change any of these, and especially to specify the names of the generation summaries/colors/legends, go to line 2763. 
    
    Moreover, to change any parameter go to line 2628. The important parameters for this function are:
    - dataset_dim: it has to be the same size as the size of the training dataset
    - n_ld_dim: list of all the different dimensions for the analysis one wants to compare.
    - GAN_classes: they have to be the same as the one defined for the classifier one has used to create the annotations. In particular, 
      it would be an `empty` list if CLASSIFIER -REAL, or `EARLY15, EARLY30, EARLY45, OT, WN` if CLASSIFIER - EXT.

    OUTPUT_DIR will contain all the plots relative to the statistical analysis (mean, median, etc.) but not the Inception Score plot. This is created with an additional funtion, `plot_inception`.


    python decoder_analysis.py --option instances --data_dir INPUT_DIR --output_dir OUTPUT_DIR --dataset_size Y 

* The function `analysis_dim` allows to compare several instances between them and to visualize average plots across different dataset sizes.

    It takes as input
    - generation_data: summary of the generations (one per instance of the training, i.e. the general .npy dictionary created using `analysis_gen` on each instance)
    - summary_dataset: to get the distribution of the training data (i.e., the .npy dictionary created using `analysis_dataset`, one per each condition since we are comparing different dataset sizes. E.g., three in the case of our paper)
    - legend_list: name of the dataset size conditions
    - legend_list_instances: name of the instances, if multiple instances per condition
    - colors_list : to have uniform plots across different analysis (for the dataset size conditions)
    
    To change any of these, and especially to specify the names of the generation summaries/colors/legends, go to line 2741.
    In particular, at line 2745 it is important to specify the dataset size conditions:
    - NEW dataset: [16000, 8000, 4000] 
    - OLD dataset: [23456, 3600, 1600] 
    
    Moreover, to change any parameter go to line 2628. The important parameters for this function are:
    - n_ld_dim: list of all the different dataset size conditions.
    - GAN_classes: they have to be the same as the one defined for the classifier one has used to create the annotations. In particular, 
      it would be an `empty` list if CLASSIFIER -REAL, or `EARLY15, EARLY30, EARLY45, OT, WN` if CLASSIFIER - EXT.
    - wavegan_latent_dim: the same used for training, needs to be the same for all the instances.

    OUTPUT_DIR will contain all the plots relative to the statistical analysis (mean, median, etc.) but not the Inception Score plot. This is created with an additional function, `plot_inception`.


    python decoder_analysis.py --option analysis_dim --data_dir INPUT_DIR --output_dir OUTPUT_DIR --wavegan_latent_dim LD


* The function `analysis_latent` 

### Plot the Inception Score
To compute and/or plot the inception score one can use the function `plot_inception` (also works to compare different instances).

It takes as input
- [OPTIONAL] generation_summary_list of all the epochs (created using `analysis_gen`): if empty, it just skip the computation of IS (most likely all the time, since one should have it from `analysis_gen`)
- dataset_summary (created using `analysis_dataset`): to retrieve the inception score of the training dataset
- all_IS : list of all the inception scores (one, or multiple instances).
- colors_list: to have uniform plots across different analysis (for the instances)
- legend_list: names of the instances

To change any of these files' name, and especially to specify the names of the inception score files (which is not general), go to line 2792.
Be careful to change the legend list and the color list depending on the conditions.
 
    python decoder_analysis.py --option IS --data_dir INPUT_DIR --output_dir OUTPUT_DIR 

## Utils
* The function `open_pkl` (present in several .py) is used to deal with the opening of the annotations (pickle files)
* The function `statistics.py` contains different functions useful for the analysis. In particular:
    - `cross_entropy` 
    - `KL_divergence` and `KL_cross_entropy`
    - `cross_entropy_for_class_labels`
    - `inception_score`
    
* The function `plots.py` contains the following (and some more) functions that helps the plotting.
    - `plot_distribution_classes` : it realizes the histogram of the distribution of the classifier (ex. in analysis_dataset or analysis_gen). Be careful at line 21: if you need a nicer represenation (i.e., with the GAN classes at the end of the histogram) this might need to be changed.
    - `plot_spectro_sp` : spectrogram plot using scipy library 
    - `plot_spectro_librosa` : spectrogram plot using librosa library
    
* At any time, the code saves the figures as .png files, but one can add an option `--format` and specify an additional format (i.e., .png is always saved plus the chosen one.)

--- 
# Qualitative analysis
The function `qualitative_analysis.py` allows to build tables to compare different human judgements.
It is quite basic as function.

* The function `qualitative_test` builds the dataset for the test. It selects N syllables in a directory and copy to a new one.

    It takes as input
    - songs: the list of all the .wav files available for the test
    - all the other arguments
    
    The important parameters (to change, go to line 184) for this function are:
    - n_template: how many syllables are included in the test
    
    OUTPUT: a .npy file containing the name of the .wav files and the additional directory with the .wav files selected for the test.
    
* The function `quantitative_table` helps to build an excel table to be used later from the judges (one per each judge). Go to line 78 and through the function to make it more suitable for the test.

* The function `quantitative_analysis` helps to build an excel table to summarize all the answers from the judges. It creates a table with names of the files, and real/decoder names (check which one, it might need to be manually selected or an option might be needed here). The answer of the judges need to be add manually (for now).

* The function `Cohen_kappa` reads the cumulative table (after all the judges have filled their column) and compute the Cohen's kappa coefficient for each couple of judges.

---
# Space representation A
To represent the data space (spectrograms) and the latent space.

Function `space_representation`.

## UMAP representation
For all the representations obtained for UMAP the important parameters are (can be changed at line 1243 and below):
- n_neigh: How much local is the topology. Smaller means more local
- min_d: how tightly UMAP is allowed to pack points together. Higher values provide more details. Range 0.0 to 0.99
- spread: Additional parameter to change when min_d is changed, it has to be >= min_d
- seed: to obtain the same representation multiple times

Along with this, since UMAP is the representation of the spectrograms all the parameters to create the spectrograms are important:
- window: Type of window for the visualization of the spectrogram
- overlap: Overlap for the visualization of the spectrogram
- nperseg: Nperseg for the visualization of the spectrogram
- N: Nftt spectrogram librosa
- H: Hop length spectrogram librosa

Moreover, we need to choose:
- all_garbage: Do we want to group all the garbage classes together or not (i.e., whether or not we have class X)
- balanced: Do we want to consider a balanced genetated set? How many per class? This is the number of elements to be considered.

Finally, at the beginning of each function, there are the colors lists defined ad hoc. This might need to be changed if the number of classes changes for instance.
In that case, add a new one, do not delete the existing ones.

* The function `syll_UMAP` provides the UMAP representation of the training dataset with ONLY the repertoire classes. No garbage class, this is based on the real names (semi-authomatic labels) of the syllables.

    It takes as input:
    - summary_dataset: to get the spectrogram and the names of the training data (i.e., the .npy dictionary created using `analysis_dataset`)
    
    
    python space_representation.py --option syll_UMAP --data_dir INPUT_DIR --output_dir OUTPUT_DIR 

* The function `latent_UMAP` provides the UMAP representation of the generated data. 

    It takes as input:
    - gen_summary: to get the spectrogram and the classifier names of the generated data (i.e., the .npy dictionary created using `analysis_gen`). For this the epoch needs to be changed manually at line 1288.
    
    
    python space_representation.py --option latent_UMAP --data_dir INPUT_DIR --output_dir OUTPUT_DIR 
    
* The function `cfr_UMAP` provides the comparison between training and generated data. From line 1293 one can choose:
    - the classes of the vocabulary
    - if iterate over several epochs (if multiple) or just one, changing the name of the generation_summary
    
    It takes as input:
    - summary_dataset: to get the spectrogram and the classifier names of the training data (i.e., the .npy dictionary created using `analysis_dataset`)
    - generation_summary: to get the spectrogram and the classifier names of the generated data (i.e., the .npy dictionary created using `analysis_gen`). 
    - classes: vocabulary of the classifier.
    
    
    python space_representation.py --option cfr_UMAP --data_dir INPUT_DIR --output_dir OUTPUT_DIR 
    
* The function `gen_expl` 

    It takes as input:
    - summary_dataset: actually, in this case it is the generation_summary of the whole generated data
    - generation_summary: summary of the new explored data (obtained exploring linearly the latent space)
    - classes: vocabulary of the classifier

    
    python space_representation.py --option gen_expl --data_dir INPUT_DIR --output_dir OUTPUT_DIR 
  

* The function `phrase` allows to build a sequence of syllables of the same type with an arbitrary gap between them, and plots the spectrogram versus a template phrase.
    
    It takes as input:
    - generation_summary_list: to get the spectrogram, the classifier names, etc. of the generated data (i.e., the .npy dictionary created using `analysis_gen`). For this the epoch needs to be changed manually at line 1318.
    - classes: vocabulary of the classifier
    - template_list: .wav files used as template phrase. List of .wav files. 
    
    Important parameters:
    - _template_dir_: where the template phrases are stored, one per syllable type.
    - _n_template_: how many syllables in the sequence
    
    
    python space_representation.py --option phrases --data_dir INPUT_DIR --output_dir OUTPUT_DIR --template_dir TEMPLATE_DIR --n_template N

## Latent space analysis
* The function `xyz` provides the 3D scatter plot of the latent space in terms of the correspondent syllable type (can be compared with the spectrogram representation obtained using UMAP). Also, it provides slices of the 3D cube.

    It takes as input: 
    - generation_summary: to get the spectrogram, the classifier names, etc. of the generated data (i.e., the .npy dictionary created using `analysis_gen`). For this the epoch needs to be changed manually at line 1329.
      The path may be not correct, it depends on where it is saved the file. Check in the same line.
    - classes: vocabulary of the classifier
    
    
    python space_representation.py --option xyz --data_dir INPUT_DIR --output_dir OUTPUT_DIR 

[PLUS]
The functions `spect_plot` and `preview_plot` provide the plot of a sequence. Used to plot the exploration of the latent space or a complete song.

The function `convex` is not complete. I started to check the convexity of the 3-dimensional latent space but never finished.

---
# Space representation B
Function `exploration_latent`.

* The function `focus` takes two latent vectors and go from one to the other component by component (meaning only one component is moved, the other two components are fixed).

    It takes ad input:
    - _variation_step_: Variation step applied to each component of the latent vector
    - _variation_dir_: where to find the latent vectors
    - _data_dir_: where there is the train (checkpoint, model, infer etc)
    - _comp_: which component (e.g., 0)
    - _id1_:Number to identify the first element
    - _id2_:Number to identify the second element
    -_sign_: go towards pos or neg value
    -_ckpt_n_: at which checkpoint
    
    python exploration_latent.py --option focus --data_dir INPUT_DIR --output_dir OUTPUT_DIR 
    
    
* The function `plot_exploration` provides the plot for the variation. To be used after the function `focus`.

    
    python exploration_latent.py --option plot --data_dir INPUT_DIR --output_dir OUTPUT_DIR 
    

* The function `bridge` takes two latent vectors as input and linearly moves from one to the other with a step which depends on the distance between the components of the vectors. Each new vector and its correspondent generated syllables are saved.
    
    It takes as input:
    - _syll_0_: starting syllable
    - _syll_1_: ending syllable
    - _steps_: how many steps to go from one syllable to another
    - _data_dir_: where there is the train (checkpoint, model, infer etc)
    - _bridge_dir_: where to save 
    -_ckpt_n_: at which checkpoint
    
    Paramters are from line 437.
    
    
    python exploration_latent.py --option bridge --data_dir INPUT_DIR --output_dir BRIDGE_DIR --syll_0 A --syll_1 M --steps N

* The function `evolution` provides an example for each selected latent vector at some example epochs (pre-selected). At line 490 one can change the repertoire,
and the epochs. 
    
    I takes as input:
    - _data_dir_ we need to have the generative model saved (where the train is)
    - _output_dir_ the selected latent vectors (and we use it to save too)
    
    
    python exploration_latent.py --option evolution --data_dir INPUT_DIR --output_dir OUTPUT_DIR 

[PLUS] 

The function `padwav` allows to pad together several .wav files. To be used after having applied the function to generate variations in the latent space (in train_wavegan.py), this function merges the samples generated to obtain a easier visualization.

    python exploration_latent.py --option pad --data_dir INPUT_DIR

The function `syllable` helps analyzing the activity of the decoder and what we call 'rw_sum'. I used it just a couple of times to print the name of the syllable generated when varying the latent space, not sure about a general use for it.
In the _data_dir_ one needs to have the annotations. It returns the name of the syllables in the sequence and a probability plot. This function is thought for a manual inspection of a limited sequence of syllables.    

    python exploration_latent.py --option syll --data_dir INPUT_DIR
    
---

# Data Analysis Birdsong
## DataAnalysis
   The function features.py provides:

   * A test function to select the correct parameter
   to have a good selection of the syllables. It saves an args.txt file with the
   parameter choice. 

	python features --option test --data_dir DIRECTORY_NAME --min_syl_dur X --min_silent_dur X --threshold X


   * A function to visualize the repertoire of the 16 syllables (one example per syllable type).

	python --option repertoire --data_dir DIRECTORY_NAME


   * Several functions to extract features from .wav recordings
   
   	- There is a function to extract from the song (or phrases).

  	    --> TODO check if this function is working properly
   	    since I have used it long time back. In the test function I can find 
   	    all the informations to make this work properly. 
	    
	    python --option song_features --data_dir DIRECTORY_NAME
   
   	- There is a function working on single syllables. 
	   
	    python --option syll_features --data_dir DIRECTORY_NAME --output_dir OUTPUT_DIRECTORY_NAME

  	- There is a function working on syllale + silence, where the sound is
            cut and then the features are computed. Useful to compute the features 
            of generated sounds (from the GAN for example). 
	    
            Be careful that in the data_dir there will be a directory called "Repertoire" 
            containing one example syllable per class. 
            
            T line 1383 specify the classes.
            At line 1386 specify the directoy containing the syllables.
            
            At line 722/723 change the parameters depending on the dataset. 
            NEW dataset: line 722 is working fine.
            OLD dataset: if the mean spectrogram plot is not showing N (totally black) try to change for line 723. Still, sometimes it does not work properly.            

	    python --option syll_silence_features --data_dir DIRECTORY_NAME --output_dir OUTPUT_DIRECTORY_NAME

   
   * A function to plot the results.
   This function is to plot the results within the same type (for example
   the features of the dataset, or the features of the GAN results), and not
   to compare different sets. It provides also the mean spectrogram per each class.
  
   	python --option plot --data_dir DIRECTORY_NAME
   	
   * A function to plot only the mean spectrogram
   This function is to plot the mean spectrogram within the same type (for example
   the features of the dataset, or the features of the GAN results), and not
   to compare different sets.
  
    python --option plot_silence --data_dir DIRECTORY_NAME


## Useful visualization and mean spectrogram
To visualize a template phrase for each syllable type and then 100 examples of that particular syllable, after step 4.2.A,
to look at a particular syllable, run

	python datasets.py --option pad --data_dir DIRECTORY_NAME --syll_name SYLLABLE_NAME


To obtain the mean spectrogram matrix, after having run all the previous step on all the syllables (so that you have cut all single syllables
and correct eventual errors), choose which class you want to analyse and update the code at line 1093(first case below) or 1071(second case below) run 

	* if you want to use the syllables with the silence and if applied to generated data
		python features.py --option syll_silence_features --data_dir DIRECTORY_NAME --output_dir DIRECTORY_NAME --template_dir DIRECTORY_NAME 
		python features.py --option plot --data_dir DIRECTORY_NAME
	
	* if you want to use the syllables without the silence: filter + compute features + plot
		python datasets.py --option filter --data_dir DIRECTORY_NAME
		python features.py --option syll_features --data_dir DIRECTORY_NAME --output_dir DIRECTORY_NAME --template_dir DIRECTORY_NAME 
		python features.py --option plot --data_dir DIRECTORY_NAME 


---
# Others
In `cross_correlation` there are some attempts to compute spectrogram correlation. I didn't check them properly lately.
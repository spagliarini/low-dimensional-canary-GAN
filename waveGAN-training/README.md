# WaveGAN

## Requirements
### For training (GPU virtual environment)
pip install tensorflow-gpu==1.12.0
pip install scipy==1.0.0
pip install matplotlib==3.0.2
pip install librosa==0.6.2

### For generating (CPU virtual environment)
pip install tensorflow==1.12.0
pip install scipy==1.0.0
pip install matplotlib==3.0.2
pip install librosa==0.6.2

## To change the parameters
- go at the end of the code train_wavegan.py 

Important/useful parameters are:
* _data_sample_rate_ = sample rate of the training data
* _wavegan_latent_dim_ = dimension of the latent space
* [ONLY DURING TRAINING] _max_to_keep_ (line 202) = how many checkpoint one wants to keep (once the value is reached the oldest one is removed automatically)
* [ONLY DURING TRAINING] _train_save_secs_ and _train_summary_secs_ = saving step (every how many steps one wants to save). If thet coincide both the loss and the checkpoint are saved at the same time.
* [ONLY AFTER TRAINING] generation_n = how many generations 
* [ONLY AFTER TRAINING] ckpt_n + at which checkpoit one wants to generate from

## To run the model 
python version: 3.6

### For training or resuming (GPU)
- if UBUNTU
   
    `export CUDA_VISIBLE_DEVICES="0"   `

- if WINDOWS

    `bash -c "export" CUDA_VISIBLE_DEVICES="0"   `

- for both (EXAMPLE)  
        
    `python train_wavegan.py train ./FILTtrainPLAFRIM_16s_Marron1_ld5 --data_dir Train --data_fast_wav --data_first_slice --data_pad_end --wavegan_latent_dim 5 --train_save_secs 1200 --train_summary_secs 1200`                                   
   
### For preview in parallel during training 
- open a new terminal and activate the virtual env

- in ubuntu and if same virtualenv (to do not use the GPU)


    export CUDA_VISIBLE_DEVICES="-1" 

    python train_wavegan.py preview ./output_directory_name --preview_n 50

- if CPU virtualenv


    python train_wavegan.py preview ./output_directory_name --preview_n 50

### For preview after training (to see how the same latent vector evolves in terms of its spectrogram content)
- in ubuntu and if same virtualenv (to do not use the GPU)


    export CUDA_VISIBLE_DEVICES="-1" 

    python train_wavegan.py preview ./output_directory_name --preview_after 50 (generates 50 latent vectors per each saved checkpoint)

- if CPU virtualenv (ubuntu and windows)

    
    python train_wavegan.py preview ./output_directory_name --preview_after 50 (generates 50 latent vectors per each saved checkpoint)

### To generate
- generate HOW_MANY generation_n sounds AT_GIVEN_CHECKPOINT ckpt_n (only if not the latest, in that case do not give the value and it will
be automatically the latest checkpoint. In this particular case the directory will be named "generation_False"). If you want
to generate at a given chekpoint check that it is in the saved chekpoints. 
Use the function generate_S as follows (example):


    python train_wavegan.py variation ./output_directory_name --wavegan_latent_dim 1 --generation_n 1000 --ckpt_n 0 

###  To explore the latent space varying the latent vector
- generate the sound and the variation of the sound obtained varying one by one the components of the latent vector, AT_GIVEN_CHECKPOINT

    
    python train_wavegan.py variation ./output_directory_name --variation_step 0.10 --wavegan_latent_dim 1 --ckpt_n GIVEN_CHECKPOINT

## Other functions
* Structural functions

`wavegan.py, score.py, loader.py`

* Loss function plot from the data available on tensorbard (download as .csv): `plot_loss.py`
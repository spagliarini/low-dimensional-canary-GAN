# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:13:08 2019

@author: Mnemosyne

Function that I used to build the dataset from the raw data

Parameters such as data_path_in (location of the data), sampling rate and threshold need to be adjusted depending on the experiment.

Requires Python 3.7.3 and other packages.
"""

import os
import random
import shutil
import glob
import sys
import numpy as np
import scipy as sp
from scipy.fftpack import fft, rfft
import re
import audiosegment
from pydub import AudioSegment
import pydub.scipy_effects
from pydub.silence import split_on_silence
import scipy.io.wavfile as wav
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.signal as signal
import librosa.display
import librosa.feature
import librosa.effects
from threading import Thread
from songbird_data_analysis import Song_functions

def cut(rawsong, sr, threshold, min_syl_dur, min_silent_dur, f_cut_min, f_cut_max):
    """
    This function is meant to be used on a single recording, create an external loop 
    to apply on several recordings (see function select_syll)
    
    VARIABLES: 
        - rawsong: the wav file a song
        - sr: sampling rate
                
    OUTPUT:
        - npy/txt file which contains onset and offset of each syllable of the song.
          So for syllable 1 of a song, its onset is onsets[0] and its offset is offsets[0].
          To get that segment of the spectrogram, you'd take spect[:,onsets[0]:offsets[0]]
    """
    # parameters that might be adjusted dependending on the bird
    rawsong = rawsong.astype(float)
    rawsong = rawsong.flatten()
    
    amp = Song_functions.smooth_data(rawsong,sr,freq_cutoffs=(f_cut_min, f_cut_max))

    (onsets, offsets) = Song_functions.segment_song(amp,segment_params={'threshold': threshold, 'min_syl_dur': min_syl_dur, 'min_silent_dur': min_silent_dur},samp_freq=sr)

    return onsets, offsets

def select_syll_general(songfile, output_dir, args):
    """
    INPUT
    List of the recordings.
                
    OUTPUT:
    Single syllables: wav file of each syllable
    """

    # loop over all the recordings
    for i in range(0, np.size(songfile)):
        sr, samples = wav.read(songfile[i])
        #The sampling frequency or sampling rate, fs, is the average number of samples obtained in one second (samples per second), thus fs = 1/T.
        
        # select the syllables
        onset = cut(samples, sr, args.threshold, args.min_syl_dur, args.min_silent_dur, args.f_cut_min, args.f_cut_max)[0]
        offset = cut(samples, sr, args.threshold, args.min_syl_dur, args.min_silent_dur, args.f_cut_min, args.f_cut_max)[1]

        ns = 0
        while ns < np.size(onset):
            wav.write(args.data_dir + '/' + output_dir + '/' + songfile[i][len(args.data_dir):-4] + '_syllable_' + str(ns) +'.wav', sr, samples[int(onset[ns]-args.add_before_selection):int(offset[ns]+args.add_after_selection)])
            ns = ns + 1
                
    print('Done')

def high_pass_filter(songfile, args):
    import soundfile
    """
    INPUT
    List of the recordings containing single syllables (previously selected).
    Make a copy of the directory, since this code is going to overwrite the recordings.

    OUTPUT
    Recordings containing filtered single syllables.
    """

    for i in range(0, np.size(songfile)):
        # read the syllable
        samples, sr = librosa.load(songfile[i], sr=16000)

        if np.size(samples)>0:
            nyq = 0.5 * sr
            freq = args.filter_freq / nyq
            sos = signal.butter(args.filter_order, freq, btype='highpass', output='sos')
            y = signal.sosfilt(sos, samples)

            wav.write(args.data_dir + '/' + songfile[i][len(args.data_dir):-4] + '.wav', args.sr, y)

        else:
            os.remove(songfile[i])

    print('Done')

def uniform_dur_syllable(songfile, args):
    """
    INPUT
    List of the recordings containing single syllables (previously selected).
    Make a copy of the directory, since this code is going to overwrite the recordings.
    
    OUTPUT 
    Recordings containing filtered single syllables + silence for a total duration equal to one second.
    """

    duration = np.zeros((np.size(songfile),))

    for i in range(0,np.size(songfile)):
        #read the syllable
        sr, samples = wav.read(songfile[i])

        if samples.size == 0:
            os.remove(songfile[i])

    songfile = sorted(glob.glob(args.data_dir + '/' + '*wav'))

    for i in range(0,np.size(songfile)):
        #read the syllable
        or_rec = AudioSegment.from_wav(songfile[i])
        or_rec = or_rec.high_pass_filter(700, order=5)

        duration[i] = np.size(or_rec.get_array_of_samples())
        if np.int(1000*duration[i]/args.sr)<args.sd:
            # compute silence and add to the recording
            silence = AudioSegment.silent(duration=args.sd-np.round(1000*np.size(or_rec.get_array_of_samples())/args.sr))

            sound = or_rec + silence
            # export the audio (overwriting the old one)
            sound.export(songfile[i], format ="wav")

    np.save(args.data_dir + '/' + 'duration_in_samples.npy', duration)
            
    print('Done')

def uniform_partition(songfile, output_dir, args):
    """
    INPUT
    List of raw recordings
    
    OUTPUT 
    Recordings containing one second of the song.
    Shorter (than partition) recordings are not taken into account.
    """
    # lenght of selection (random part of the song) and size of the shift in term of the sample vector
    partition_samples = int(args.sr*args.partition/1000)
    overlap_samples = int(args.sr*args.overlap/1000)
    
    for i in range(0,np.size(songfile)):
        ar_rec = AudioSegment.from_wav(songfile[i])
        
        # high-pass filter to exclude <500Hz signals 
        filtered_rec = np.array(ar_rec.high_pass_filter(700).get_array_of_samples())
        # cut the silence at the beginning and at the end (excluding 4 seconds and 3 seconds)
        filtered_rec = filtered_rec[args.sr*4:-args.sr*3]
        dur_ms = int((filtered_rec.size*1000)/ args.sr)
        
        counter = 0
        shift =0 
        if dur_ms>10000:
            while shift<(dur_ms-args.overlap):
                new_song = filtered_rec[shift:(shift+partition_samples)]

                wav.write(output_dir + '/' + 'rand' + str(counter) + songfile[i][len(args.data_dir):-4] + '.wav', args.sr, new_song)
                
                counter = counter + 1
                shift = shift + overlap_samples

    print('Done')

def control_1sec(songfile, args):
    """
    First control = long recordings are cut if they are longer than 1s. This operations returns the wav files cutted
    if they are too long which is the case (to fix a bug coming from AudioSegment I had to
    make them ~10 samples longer).

    INPUT
    List or recordings (wav file)

    OUTPUT
    New waves
    """
    counter = 0
    for i in range(0, np.size(songfile)):
        sr, samples = wav.read(songfile[i])

        if samples.size > args.sr:
            os.remove(songfile[i])
            wav.write(args.data_dir + '/' + songfile[i][len(args.data_dir):-4] + '.wav', args.sr, samples[0:args.sr])
            counter = counter + 1

    print('Done')

def control_dur(songfile, args):
    """
    Then, a semi automatic control based on the duration of the syllable. It requires manual inspections of the syllables.
    1 Check: see the duration vector to check the different durations
    2 Investigate specific elements based on a threshold that depends on the duration vector
    3 Manually inspect the recordings that have been moved to a specific directory (args.error_dir) and either confirm
      the movement or put them back to the original directory

    INPUT
    List or recordings (wav file)
    """

    dur_syll = np.zeros((np.size(songfile),))
    for i in range(0, np.size(songfile)):
        sr, samples = wav.read(songfile[i])
        dur_syll[i] = int(samples.size/16)

    # Semi automatic control
    #1 Check: see the duration vector to check the different durations
    print(sorted(dur_syll))
    input()
    #2 Investigate specific elements
    a = np.where(dur_syll>180)[0]
    print(np.size(a))
    input()
    for i in range(0, np.size(a)):
        shutil.move(songfile[int(a[i])], args.data_dir + '/' + args.error_dir)

    print('Please go to the error directory')

def pad_syllables(template, syll_ex, args):
    """
    A function to explore the dataset and compare it with the original phrases.

    :param syll_ex: list of syllables to plot

    :return: a wav file with all the syllables pad together and a silence between them
             a spectrogram plot to compare this sequence to a real phrase example of the same syllable type
    """
    # BUILD the sequence and adapt the template using audiosegment
    # sequence
    max_size = np.size(syll_ex)
    silence = AudioSegment.silent(duration=100)
    sequence = silence

    # if we are using generated data we need to put a max size (to cut the silence)
    size_aux = 400
    for s in range(0, int(max_size / 2)):
        aux = AudioSegment.from_wav(syll_ex[s])
        aux = aux[0:size_aux]
        sequence = sequence + aux
        sequence = sequence + silence
    sequence.export(args.data_dir + '/' + args.output_dir + '/' + 'Sequence1_' + str(args.syll_name) + '.wav',
                    format="wav")

    sequence = silence
    for s in range(int(max_size / 2), max_size):
        aux = AudioSegment.from_wav(syll_ex[s])
        aux = aux[0:size_aux]
        sequence = sequence + aux
        sequence = sequence + silence
    sequence.export(args.data_dir + '/' + args.output_dir + '/' + 'Sequence2_' + str(args.syll_name) + '.wav',
                    format="wav")

    # template
    aux = AudioSegment.from_wav(template[0])
    new_template = aux + AudioSegment.silent(duration=size_aux * int(max_size / 2) - np.size(aux))
    new_template.export(args.data_dir + '/' + args.output_dir + '/' + 'Template_' + args.syll_name + '.wav',
                        format="wav")

    # PLOT using librosa
    # template
    y_template, sr_template = librosa.load(
        args.data_dir + '/' + args.output_dir + '/' + 'Template_' + args.syll_name + '.wav', sr=16000)

    X_template = librosa.stft(y_template, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant',
                              center=True)
    Y_template = np.log(1 + 100 * np.abs(X_template) ** 2)
    T_coef_template = np.arange(X_template.shape[1]) * args.H / sr_template
    K_template = args.N // 2
    F_coef_template = np.arange(K_template + 1) * sr_template / args.N

    # sequence
    y, sr = librosa.load(args.data_dir + '/' + args.output_dir + '/' + 'Sequence1_' + args.syll_name + '.wav', sr=16000)

    X = librosa.stft(y, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
    Y_1 = np.log(1 + 100 * np.abs(X) ** 2)
    T_coef = np.arange(X.shape[1]) * args.H / sr
    K = args.N // 2
    F_coef = np.arange(K + 1) * sr / args.N

    y, sr = librosa.load(args.data_dir + '/' + args.output_dir + '/' + 'Sequence2_' + args.syll_name + '.wav', sr=16000)

    X = librosa.stft(y, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
    Y_2 = np.log(1 + 100 * np.abs(X) ** 2)
    T_coef = np.arange(X.shape[1]) * args.H / sr
    K = args.N // 2
    F_coef = np.arange(K + 1) * sr / args.N

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(22, 5))
    extent_template = [T_coef_template[0], T_coef_template[-1], F_coef_template[0], 8000]
    ax1.imshow(Y_template, cmap=args.color, aspect='auto', origin='lower', extent=extent_template,
               norm=colors.PowerNorm(gamma=0.2))
    ax1.title.set_text('Template ' + args.syll_name)
    ax1.title.set_fontsize(15)

    extent = [T_coef[0], T_coef[-1], F_coef[0], 8000]
    ax2.imshow(Y_1, cmap=args.color, aspect='auto', origin='lower', extent=extent, norm=colors.PowerNorm(gamma=0.2))
    ax2.set_ylabel('Frequency (Hz)', fontsize=15)
    ax2.title.set_text('Selected ' + args.syll_name)
    ax2.title.set_fontsize(15)

    ax3.imshow(Y_2, cmap=args.color, aspect='auto', origin='lower', extent=extent, norm=colors.PowerNorm(gamma=0.2))

    plt.xlabel('Time (seconds)', fontsize=15)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'spectogram_' + args.syll_name + '.' + args.format)

    print('Done')

if __name__=="__main__":
    """
    Example to run 
    >python datasets.py --option MODE --data_dir DATA --output_dir OUTPUT
    """
    import argparse
    import glob
    import sys
    
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str,
                        choices=['selection', 'uniform_dur_syllable',
                                 'uniform_partition', 'control', 'control_dur', 'filter', 'pad'],
                        help="which simulation type should be chosen")
    parser.add_argument('--data_dir', type=str,
                        help='Data directory, if this is changed one needs to change how to save in the code,'
                             'unless in the mode control, where no savings are involved', default = 'Data')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory', default = None)
    parser.add_argument('--template_dir', type=str,
                        help='Template directory to use the pad function', default=None)

    data_args = parser.add_argument_group('Data')
    data_args.add_argument('--sr', type=int,
                           help='Sampling rate', default = 16000)
    data_args.add_argument('--threshold', type=float,
                           help='Threshold to select syllables', default=1000000)
    data_args.add_argument('--min_syl_dur', type=float,
                           help='Minimal duration for a syllable to be selected', default=0.03)
    data_args.add_argument('--min_silent_dur', type=float,
                           help='Minimal silence in between two syllables, to cut the selection there', default=0.01)

    selection_args = parser.add_argument_group('Selection')
    selection_args.add_argument('--add_before_selection', type=int, help='How many samples to add before the selcted syllable',
                                default=200)
    selection_args.add_argument('--add_after_selection', type=int,
                                help='How many samples to add after the selcted syllable',
                                default=2000)
    data_args.add_argument('--f_cut_min', type = int,
                           help='Minimal cut off in frequency when cutting the syllables, greater than zero',
                           default=500)
    data_args.add_argument('--f_cut_max', type = int,
                           help='Maximal cut off in frequency when cutting the syllables, less than sampling rate - 1',
                           default=7999)
    selection_args.add_argument('--low_pass', type=int, help='frequency value to use to apply a low-pass filter',
                                default=500)

    filter_args = parser.add_argument_group('Filter')
    filter_args.add_argument('--filter_freq', type=int, help='Frequency cut off for the filter, in Hz', default=700)
    filter_args.add_argument('--filter_order', type=int, help='Order of the filter', default=5)

    uniform_duration_args = parser.add_argument_group('Uniform')
    uniform_duration_args.add_argument('--sd', type=int,
                                       help='Expected sound duration in milliseconds', default = 1001)

    uniform_partition_args = parser.add_argument_group('Partition')
    uniform_partition_args.add_argument('--partition', type=int,
                                        help='Expected partition duration in milliseconds', default = 1000)

    control_args = parser.add_argument_group('Control')
    control_args.add_argument('--error_dir', type=str, help='Where to save the error files',
                              default='Errors')
                              #default='D:\PhD_Bordeaux\DATA\Dataset_GAN\Datasets_Marron\Dataset_2\Errors')
    control_args.add_argument('--n_template', type=int, help='How many examples to plot', default=30)

    plot_args = parser.add_argument_group('Plot')
    plot_args.add_argument('--window', type=str,
                           help='Type of window for the visualization of the spectrogram', default='hanning')
    plot_args.add_argument('--overlap', type=int,
                           help='Overlap for the visualization of the spectrogram', default=64)
    plot_args.add_argument('--nperseg', type=int,
                           help='Nperseg for the visualization of the spectrogram', default=1024)
    plot_args.add_argument('--format', type=str, help='Saving format' , default='png')
    plot_args.add_argument('--syll_name', type = str, help='Name of the syllable')
    plot_args.add_argument('--color', type = str, help='Colormap', default='inferno')
    plot_args.add_argument('--N', type = int, help='Nftt spectrogram librosa', default=256)
    plot_args.add_argument('--H', type = int, help='Hop length spectrogram librosa', default=64)

    args = parser.parse_args()

    if args.option == 'selection':
        # Make output directory
        if args.output_dir == None:
            output_dir = 'Single_syll_' + args.syll_name
        else:
            output_dir = args.output_dir

        if not os.path.isdir(args.data_dir + '/' + output_dir):
            os.makedirs(args.data_dir + '/' + output_dir)

        # List of recordings
        songfile = glob.glob(args.data_dir + '/' + '*wav')

        select_syll_general(songfile, output_dir, args)

    elif args.option == 'uniform_dur_syllable':
        # List of recordings
        songfile = sorted(glob.glob(args.data_dir + '/' + '*wav'))

        uniform_dur_syllable(songfile, args)

    elif args.option == 'uniform_partition':
        # Make output directory
        if args.output_dir == None:
            output_dir = 'Uniform_partition'
        else:
            output_dir = args.output_dir

        if not os.path.isdir(args.data_dir + '/' + output_dir):
            os.makedirs(args.data_dir + '/' + output_dir)

        # List of recordings
        songfile = glob.glob(args.data_dir + '/' + '*wav')

        uniform_partition_1sec(songfile, output_dir, args)

    elif args.option == 'control':
        # List of recordings
        songfile = sorted(glob.glob(args.data_dir + '/' + '*wav'))

        control_1sec(songfile, args)

    elif args.option == 'control_dur':
        if not os.path.isdir(args.data_dir + '/' + args.error_dir):
            os.makedirs(args.data_dir + '/' + args.error_dir)

        # List of recordings
        #songfile = sorted(glob.glob(args.data_dir + '/' + args.syll_name + '/' + 'Single_syll_' +  args.syll_name + '/' + '*wav'))
        songfile = sorted(glob.glob(args.data_dir + '/' + '*wav'))
        #songfile = sorted(glob.glob(args.data_dir + '/' + 'Single_syll_' +  args.syll_name + '/' + '*wav'))

        control_dur(songfile, args)

    elif args.option == 'filter':
        # List of recordings
        songfile = glob.glob(args.data_dir + '/' + '*wav')

        high_pass_filter(songfile, args)

    elif args.option == 'pad':
        # List of recordings
        songfile = sorted(glob.glob(args.data_dir + '/' + 'NEW_' + args.syll_name + '*wav'))
        if np.size(songfile) >= args.n_template:
            syll_ex = random.sample(songfile, args.n_template)
        else:
            syll_ex = random.sample(songfile, np.size(songfile))

        template = glob.glob(args.data_dir + '/' + args.output_dir + '/' + 'Template_' + args.syll_name + '*.wav')

        pad_syllables(template, syll_ex, args)

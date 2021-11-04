# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:26:54 2019

@author: Mnemosyne

Functions to compute the features of the song
"""

import os
import shutil
import glob
import sys
import random
import re
import numpy as np
import scipy as sp
import scipy.io.wavfile as wav
from scipy.fftpack import fft, rfft
from scipy.optimize import curve_fit
import scipy.signal as signal
from scipy.stats.mstats import gmean
from sklearn.cluster import KMeans
from pydub import AudioSegment
from pydub import silence
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d
import matplotlib.colors as colors
from threading import Thread 
import librosa.feature
import librosa.effects

from songbird_data_analysis import Song_functions

# Onset and offset of the syllables: to compute duration of syllables and gaps
def cut(rawsong, sr, threshold, min_syl_dur, min_silent_dur, f_cut_min, f_cut_max):
    """
    This function is meant to be used on a single recording, create an external loop 
    to apply on several recordings (see function distribution).

    VARIABLES: 
        - rawsong: the wav file a song
        - sr: sampling rate
                
    OUTPUT:
        - onset and offset of each syllable of the song
          So for syllable 1 of a song, its onset is onsets[0] and its offset is offsets[0].
          To get that segment of the spectrogram, you'd take spect[:,onsets[0]:offsets[0]]
    """
    
    # parameters that might be adjusted dependending on the bird
    rawsong = rawsong.astype(float)
    rawsong = rawsong.flatten()
    
    amp = Song_functions.smooth_data(rawsong,sr,freq_cutoffs=(f_cut_min, f_cut_max))
    
    (onsets, offsets) = Song_functions.segment_song(amp,segment_params={'threshold': threshold, 'min_syl_dur': min_syl_dur, 'min_silent_dur': min_silent_dur},samp_freq=sr)    # Detects syllables according to the threshold you set                      

    return amp, onsets, offsets

def test_features(songfile, args):
    """
    A function to tune the parameter depending on the dataset and test the feature extraction

    INPUT:
    One recording.

    OUTPUT
        - plot of the spectrogram, onset & offset and amplitude of the selected syllables
        - plot the pitches
        - plot the coupling of the features two by two
    """
    # read the data
    sr, samples = wav.read(songfile[0])
    y, sr = librosa.load(songfile[0], sr=16000)

    # determine onset and offset of the syllables for this song
    amp, onsets, offsets = cut(samples, sr, args.threshold, args.min_syl_dur, args.min_silent_dur, args.f_cut_min, args.f_cut_max)

    # Make output directory
    aux_output_dir = os.path.join(args.data_dir,args.output_dir)
    if not os.path.isdir(aux_output_dir):
        os.makedirs(aux_output_dir)

    os.chdir(aux_output_dir)

    # Spectrogram with librosa
    X = librosa.stft(y, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
    Y = np.log(1 + 100 * np.abs(X) ** 2)
    T_coef = np.arange(X.shape[1]) * args.H / sr
    K = args.N // 2
    F_coef = np.arange(K + 1) * sr / args.N

    # Plot
    noverlap = args.nperseg - args.overlap
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    # Plots spectrogram
    #(f,t,spect)=sp.signal.spectrogram(samples, sr, args.window, args.nperseg, noverlap, mode='complex')
    #ax1.imshow(10*np.log10(np.square(abs(spect))), origin="lower", aspect="auto", interpolation="none", extent=[0, max(t)*1000, min(f), max(f)], cmap = 'inferno')
    extent = [T_coef[0], T_coef[-1], F_coef[0], 8000]
    ax1.imshow(Y, cmap=args.color, aspect='auto', origin='lower', extent=extent, norm=colors.PowerNorm(gamma=0.2))
    ax1.set_ylabel('Frequency (Hz)')

    # Plots song signal amplitude
    x_amp=np.arange(len(amp))
    ax2.plot(x_amp/sr*1000,samples,color='grey')
    for i in range(0,len(onsets)):
        ax2.axvline(x=onsets[i]/sr*1000,color='olivedrab',linestyle='dashed')
        ax2.axvline(x=offsets[i]/sr*1000,color='darkslategrey',linestyle='dashed')
    ax2.set_ylabel('Amplitude (V)')

    # Plot smoothed amplitude of the song as per spectrogram index
    ax3.plot(x_amp/sr*1000, amp,color='grey')
    for i in range(0,len(onsets)):
        ax3.axvline(x=onsets[i]/sr*1000,color='olivedrab',linestyle='dashed')
        ax3.axvline(x=offsets[i]/sr*1000,color='darkslategrey',linestyle='dashed')
    ax3.axhline(y=args.threshold,color='black',label='Threshold')
    ax3.legend()
    ax3.set_ylabel('Amplitude (V)')
    ax3.set_xlabel('Time (ms)')

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)

    ax1.tick_params(axis='x', labelbottom=False, bottom=False)
    ax2.tick_params(axis='x', labelbottom=False, bottom=False)
    ax3.tick_params(axis='x', labelbottom=True, bottom=True)

    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Test_selection_0' + '.' + args.format)

    # Duration, spectral flatness, mean pitch
    dur_syll = np.zeros((np.size(onsets),))
    dur_gap = np.zeros((np.size(onsets),))
    wiener = np.zeros((np.size(onsets),))
    mean_pitch = np.zeros((np.size(onsets),))
    max_pitch = np.zeros((np.size(onsets),))
    min_pitch = np.zeros((np.size(onsets),))
    direction_pitch = np.zeros((np.size(onsets),))

    for j in range(0,np.size(onsets)):
        # Syllable duration
        dur_syll[j] = offsets[j] - onsets[j]

        if j<(np.size(onsets)-1):
            dur_gap[j] = onsets[j+1] - offsets[j]

        # Spectral flatness/wiener entropy
        wiener[j] = np.mean(librosa.feature.spectral_flatness(samples[onsets[j]:offsets[j]].astype(np.float)))

        # Pitch detection, max and min frequency
        pitches, magnitudes = librosa.core.piptrack(samples[onsets[j]:offsets[j]].astype(np.float), sr=sr, n_fft=256, fmin=1000, fmax=8000) #, win_length=100)

        pitches_all = 0
        for interval in range(0,magnitudes.shape[1]):
            index = magnitudes[:,interval].argmax()
            pitches_all = np.append(pitches_all,pitches[index,interval])

        pitches_all = pitches_all[np.nonzero(pitches_all)]

        mean_pitch[j] = np.mean(pitches_all)
        max_pitch[j] = np.max(pitches_all)
        min_pitch[j] = np.min(pitches_all)

        if pitches_all[0]<pitches_all[-1]:
            direction_pitch[j] = 1
        else:
            direction_pitch[j] = -1

        np.save('pitches_syll_' + str(j) + '.npy', pitches_all)

    # Plot all the pitches
    colors_list = list(colors._colors_full_map.values())[0::5]
    fig, ax = plt.subplots()
    #(f, t, spect) = sp.signal.spectrogram(samples, sr, args.window, args.nperseg, noverlap, mode='complex')
    #ax.imshow(10 * np.log10(np.square(abs(spect))), origin="lower", aspect="auto", interpolation="none", extent=[0, max(t) * 1000, min(f), max(f)], cmap='inferno')
    extent = [T_coef[0], T_coef[-1], F_coef[0], 8000]
    ax.imshow(Y, cmap=args.color, aspect='auto', origin='lower', extent=extent, norm=colors.PowerNorm(gamma=0.2))
    for j in range(0, np.size(onsets)):
        pitches_all= np.load('pitches_syll_' + str(j) + '.npy')
        x = np.linspace(onsets[j]/sr*1000, offsets[j]/sr*1000 - 25, np.size(pitches_all))
        ax.plot(x, pitches_all, 'o', c=colors_list[j])
    plt.ylabel('Frequency(Hz)')
    plt.xlabel('Time(ms)')
    plt.title('Pitch')
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'syll_pitch.' + args.format)

    # Mean pitch over the interval of computation
    x_axis_pitches = np.zeros((np.size(samples),))
    for j in range(0, np.size(onsets)):
        x_axis_pitches[int(np.mean([onsets[j], offsets[j]]) / sr * 1000)] = mean_pitch[j]

    x_axis_pitches_max = np.zeros((np.size(samples),))
    for j in range(0, np.size(onsets)):
        x_axis_pitches_max[int(np.mean([onsets[j], offsets[j]]) / sr * 1000)] = max_pitch[j]

    x_axis_pitches_min = np.zeros((np.size(samples),))
    for j in range(0, np.size(onsets)):
        x_axis_pitches_min[int(np.mean([onsets[j], offsets[j]]) / sr * 1000)] = min_pitch[j]

    # Plot the mean, min and max pitches
    fig, ax = plt.subplots()
    #ax.imshow(10 * np.log10(np.square(abs(spect))), origin="lower", aspect="auto", interpolation="none", extent=[0, max(t) * np.size(samples), min(f), max(f)], cmap='inferno')
    extent = [T_coef[0], T_coef[-1], F_coef[0], 8000]
    ax.imshow(Y, cmap=args.color, aspect='auto', origin='lower', extent=extent, norm=colors.PowerNorm(gamma=0.2))
    ax.plot(x_axis_pitches, color='black', linewidth=0, markersize=4, marker='X', label='mean')
    ax.plot(x_axis_pitches_max, color='red', linewidth=0, markersize=4, marker='*', label='max')
    ax.plot(x_axis_pitches_min, color='red', linewidth=0, markersize=4, marker='*', label='min')
    ax.legend()
    ax.set_xlim([0, T_coef[-1]])
    plt.ylabel('Frequency(Hz)')
    plt.xlabel('Time(ms)')
    plt.title('Pitch')
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Pitch.' + args.format)

    # Cumulative plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,9))
    # Plot the spectrogram
    #ax3.imshow(10*np.log10(np.square(abs(spect))), origin="lower", aspect="auto", interpolation="none", extent=[0, max(t)*1000, min(f), max(f)], cmap = 'inferno')
    ax1.imshow(Y, cmap=args.color, aspect='auto', origin='lower', extent=extent, norm=colors.PowerNorm(gamma=0.2))
    ax1.set_ylabel('Frequency(Hz)', fontsize=15)
    ax1.set_xticks([])
    ax1.title.set_text('Spectrogram')

    # Plots song signal amplitude
    x_amp = np.arange(len(amp))
    ax2.plot(x_amp / sr * 1000, amp, color='grey', label='_Hidden label')
    for i in range(0, len(onsets)-2):
        ax2.axvline(x=onsets[i] / sr * 1000, color='olivedrab', linestyle='dashed')
        ax2.axvline(x=offsets[i] / sr * 1000, color='darkslategrey', linestyle='dashed')
    ax2.axvline(x=onsets[len(onsets)-1] / sr * 1000, color='olivedrab', linestyle='dashed', label='Onset')
    ax2.axvline(x=offsets[len(onsets)-1] / sr * 1000, color='darkslategrey', linestyle='dashed', label='Offset')
    ax2.legend()
    ax2.set_ylabel('Amplitude (V)', fontsize=15)
    ax2.title.set_text('Selection')

    plt.xlabel('Time(ms)', fontsize=15)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Summary.' + args.format)

    # Cumulative plot 2
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
    # Plots song signal amplitude
    x_amp = np.arange(len(amp))
    ax1.plot(x_amp / sr * 1000, amp, color='grey')
    for i in range(0, len(onsets)):
        ax1.axvline(x=onsets[i] / sr * 1000, color='olivedrab', linestyle='dashed')
        ax1.axvline(x=offsets[i] / sr * 1000, color='darkslategrey', linestyle='dashed')
    ax1.set_ylabel('Amplitude (V)')
    ax1.title.set_text('Syllable selection')

    # Plot all the pitches
    #(f, t, spect) = sp.signal.spectrogram(samples, sr, args.window, args.nperseg, noverlap, mode='complex')
    #ax2.imshow(10 * np.log10(np.square(abs(spect))), origin="lower", aspect="auto", interpolation="none", extent=[0, max(t) * 1000, min(f), max(f)], cmap='inferno')
    extent = [T_coef[0], T_coef[-1], F_coef[0], 8000]
    ax2.imshow(Y, cmap=args.color, aspect='auto', origin='lower', extent=extent, norm=colors.PowerNorm(gamma=0.2))
    for j in range(0, np.size(onsets)):
        pitches_all = np.load('pitches_syll_' + str(j) + '.npy')
        x = np.linspace(onsets[j] / sr * 1000, offsets[j] / sr * 1000 - 25, np.size(pitches_all))
        ax2.plot(x, pitches_all, 'o', c=colors_list[j])
    ax2.set_ylabel('Frequency(Hz)')
    ax2.title.set_text('Pitch trajectory')

    plt.xlabel('Time(ms)')
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Summary_solo.' + args.format)

    # Clustering the syllables depending on the features
    # Duration VS pitch
    plt.subplots()
    plt.plot(np.round(dur_syll)/16,mean_pitch, '*')
    plt.xlabel('Duration(ms)')
    plt.ylabel('Pitch(Hz)')
    plt.title('Duration VS pitch')
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'durationVSpitch' + args.format)

    # Duration VS wiener
    plt.subplots()
    plt.plot(np.round(dur_syll)/16,wiener, '*')
    plt.xlabel('Duration(ms)')
    plt.ylabel('Wiener entropy(dB)')
    plt.title('Duration VS Wiener entropy')
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'durationVSwiener.' + args.format)

    # Wiener VS pitch
    plt.subplots()
    plt.plot(wiener,mean_pitch, '*')
    plt.xlabel('Wiener entropy(dB)')
    plt.ylabel('Pitch(Hz)')
    plt.title('Wiener entropy VS pitch')
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'wienerVSpitch.' + args.format)

    print('Done')

def repertoire(songfile, classes, args):
    """
    :param songfile: list of wav files (one per element of the repertoire)
    :param classes: list of the names of each element of the repertoire

    :return: a figure with one example per element of the repertoire
    """
    samples_repertoire = []
    T_coef_all = []
    for s in range(0, np.size(songfile)):
        y, sr = librosa.load(songfile[s], sr=16000)

        # cut the silence
        X = librosa.stft(y, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
        Y = np.log(1 + 100 * np.abs(X) ** 2)
        T_coef = np.arange(X.shape[1]) * args.H / sr * 1000
        K = args.N // 2
        F_coef = np.arange(K + 1) * sr / args.N
        samples_repertoire.append(Y)
        T_coef_all.append(T_coef[-1])

    # Plots spectrogram
    #plt.style.use('dark_background')
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 14), sharey=True)
    for i in range(0, 4):
        for j in range(0,4):
            extent = [0, T_coef_all[4*i + j], 0, 8000]
            axs[i, j].imshow(samples_repertoire[4*i + j], cmap=args.color, extent = extent, aspect='auto', origin='lower', norm=colors.PowerNorm(gamma=0.2))
            axs[i, j].set_title(classes[4*i + j], fontsize=12)
            axs[i, j].set_xlim(0, 350)
            axs[i, j].spines['top'].set_color('none')
            axs[i, j].spines['right'].set_color('none')
            axs[0, j].set_xlabel('Time (ms)', fontsize=15)
        axs[i, 3].set_ylabel('Frequency (Hz)', fontsize=15)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + 'Repertoire.' + args.format)

    print('Done')

def single_syllable_features_from_song(songfile, args):
    """
    VARIABLES: 
        - songfile: list of recordings
        
    OUTPUT:
        - .npy file containing the features stored, one single file for the whole directory
    """
        
    # inizialization of variables
    dur_syll = [0]
    dur_gap = [0]
    wiener = [0]
    pitch = [0]
        
    for i in range(0,np.size(songfile)):
        sr, samples = wav.read(songfile[i]) 

        # determine onset and offset of the syllables for this song
        onsets = cut(samples, sr, args.threshold, args.min_syl_dur, args.min_silent_dur, args.f_cut_min, args.f_cut_max)[0]
        offsets = cut(samples, sr, args.threshold, args.min_syl_dur, args.min_silent_dur, args.f_cut_min, args.f_cut_max)[1]
        
        dur_syll_aux = np.zeros((np.size(onsets),))
        dur_gap_aux = np.zeros((np.size(onsets),))
        wiener_aux = np.zeros((np.size(onsets),))
        pitch_aux = np.zeros((np.size(onsets),))
    
        for j in range(0,np.size(onsets)):  
            # syllable duration
            dur_syll_aux[j] = offsets[j] - onsets[j]
                        
            if j<(np.size(onsets)-1):
                dur_gap_aux[j] = onsets[j+1] - offsets[j]
        
            # spectral flatness/wiener entropy
            wiener_aux[j] = np.mean(librosa.feature.spectral_flatness(samples[onsets[j]:offsets[j]].astype(np.float)))
            
            # pitch detection
            pitches, magnitudes = librosa.core.piptrack(samples[onsets[j]:offsets[j]].astype(np.float), sr=sr, nfft=100, fmin=500, fmax=8000)
            
            pitches_all = 0
            for interval in range(0,magnitudes.shape[1]):
                index = magnitudes[:,interval].argmax()
                pitches_all = np.append(pitches_all,pitches[index,interval])
             
            pitch_aux[j] = np.mean(pitches_all[1::])
        
        dur_syll = np.append(dur_syll, dur_syll_aux) #collect syllable duration for all the i-th recordings
        dur_gap = np.append(dur_gap, dur_gap_aux) #collect gap duration for all the i-th recordings
        wiener = np.append(wiener, wiener_aux) #collect wiener entropy value for all the i-th recordings
        pitch = np.append(pitch, pitch_aux) #collect pitch for all the i-th recordings

    # save the data
    data = {'File_name': songfile[0::], 'How_many': np.size(songfile), 'Duration_syll': dur_syll[1::], 'Duration_Gap': dur_gap[1::], 'Wiener_entropy': wiener[1::], 'Pitch':pitch[1::]}
    np.save(args.data_dir + '/' 'Summary_data_features.npy', data)

def single_syllable_features(songfile, syll_name, template, args):
    """
    VARIABLES:
        - songfile: list of recordings with syllables
        - template: a comparison template (if given) to compute cross-correlation

    OUTPUT:
        - .npy file containing the features stored
    """

    # Count the number of elements in the group
    if args.dataset_dim == None:
        how_many = np.size(songfile)
    elif args.dataset_dim == 0:
        if args.dataset_dim > np.size(songfile):
            how_many = np.size(songfile)
        else:
            how_many = args.dataset_dim
    else:
        songfile = random.sample(songfile, args.dataset_dim)
        how_many = np.size(songfile)

    # Feature variables initialization
    dur_syll = np.zeros((how_many,))
    data_list = []
    pitches_list = []
    mean_pitch = np.zeros((how_many,))
    max_pitch = np.zeros((how_many,))
    min_pitch = np.zeros((how_many,))
    direction_pitch = np.zeros((how_many,))
    wiener = np.zeros((how_many,))

    for j in range(0,how_many):
        # Build the dataset: copy selected file in the training dataset directory
        #shutil.copy(songfile[j], args.data_dir + '/' + args.train_dir)

        # Read the wave
        sr, samples = wav.read(songfile[j])
        dur_syll[j] = samples.size/16

        # Pitch detection
        pitches, magnitudes = librosa.core.piptrack(samples.astype(np.float), sr=sr, n_fft=100, fmin=500, fmax=8000)

        pitches_all = 0
        for interval in range(0, magnitudes.shape[1]):
            index = magnitudes[:, interval].argmax()
            pitches_all = np.append(pitches_all, pitches[index, interval])

        pitches_all = pitches_all[np.nonzero(pitches_all)]

        mean_pitch[j] = np.mean(pitches_all)
        max_pitch[j] = np.max(pitches_all)
        min_pitch[j] = np.min(pitches_all)

        if pitches_all[0] < pitches_all[-1]:
            direction_pitch[j] = 1
        else:
            direction_pitch[j] = -1

        # Spectral flatness (Wiener entropy)
        wiener[j] = np.mean(librosa.feature.spectral_flatness(samples.astype(np.float)))

    # Argmax of duration to determine the longer syllable
    template_size = int(np.max(dur_syll))

    # Template construction (adding silence before and after for alignment)
    if len(template) > 0:
        template_samples, sr = librosa.load(template[0], sr=16000)

        if template_samples.size / 16 < template_size:
            aux_size = template_size - template_samples.size / 16
            silence = np.zeros((int(round(aux_size / 2) * 16)), )
            aux_template = np.append(silence, template_samples)
            aux_template = np.append(aux_template, silence)
        else:
            aux_template = template_samples

        silence = np.zeros((template_size,))
        new_template = np.append(silence, aux_template)
        new_template = np.append(new_template, silence)

        X = librosa.stft(new_template, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
        T_coef = np.arange(X.shape[1]) * args.H / sr * 1000 #save to plot
        spectrogram_template = np.log(1 + 100 * np.abs(X ** 2))
        shape_template = spectrogram_template.shape

    spectrograms = []
    spectrograms_envelope = []
    cross_correlation = np.zeros((how_many,))
    for j in range(0, how_many):
        data_list.append(songfile[j])

        # compute the spectrogram
        samples_aux, sr = librosa.load(songfile[j], sr=16000)

        if samples_aux.size / 16 < template_size:
            aux_size = template_size - samples_aux.size / 16
            silence = np.zeros((int(round(aux_size / 2) * 16)),)
            samples_aux = np.append(silence, samples_aux)
            samples_aux = np.append(samples_aux, silence)

        silence = np.zeros((template_size,))
        new_template_aux = np.append(silence, samples_aux)
        new_template_aux = np.append(new_template_aux, silence)

        if new_template.size > new_template_aux.size:
            silence = np.zeros((int(round(new_template.size-new_template_aux.size))),)
            new_template_aux = np.append(new_template_aux, silence)

        X = librosa.stft(new_template_aux, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
        spectrogram_aux = np.log(1 + 100 * np.abs(X ** 2))

        if spectrogram_aux.shape[1] == shape_template[1] + 1:
            spectrogram_aux = spectrogram_aux[:,0:shape_template[1]]

        fig, ax = plt.subplots()
        (lag_aux, corr_aux, line, b) = ax.xcorr(spectrogram_template.flatten(), spectrogram_aux.flatten(), normed=True, maxlags=args.n_lags, lw=2)
        #plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'xcorr_' + syll_name + '_' + str(j) + '.' + args.format)
        cross_correlation[j] = np.max(corr_aux[1])
        index_aux = lag_aux[np.argmax(corr_aux)]

        if index_aux < 0:
            spectrogram_new = np.append(np.zeros((-index_aux,)), spectrogram_aux.flatten()[-index_aux::])
            spectrograms.append(np.reshape(spectrogram_new, shape_template))
        else:
            spectrogram_new = np.append(spectrogram_aux.flatten()[index_aux::], np.zeros((index_aux,)))
            spectrograms.append(np.reshape(spectrogram_new, shape_template))

        # Envelope based
        rawsong = samples_aux.astype(float)
        rawsong = rawsong.flatten()

        amp = Song_functions.smooth_data(rawsong, sr, freq_cutoffs=(500, 7999))

        # Debug to see amplitude range
        #plt.subplots()
        #plt.plot(amp)
        #plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'ampD.' + args.format)

        #print(songfile[j])

        if syll_name == 'N':
            new_song = rawsong[0:np.where(amp > 0.00001)[0][-1]]
            silence = np.zeros((new_template.size - np.size(new_song),))
            new_song = np.append(silence, new_song)

        else:
            new_song = rawsong[np.where(amp > 0.00001)[0][0]::]
            silence = np.zeros((new_template.size - np.size(new_song),))
            new_song = np.append(new_song, silence)

        X = librosa.stft(new_song, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                         pad_mode='constant', center=True)
        spectrograms_envelope.append(np.log(1 + 100 * np.abs(X ** 2)))

        plt.close('all')

    mean_spectrogram = np.mean(spectrograms, axis=0)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 14), sharey=True, sharex=True)
    #extent = [0, np.max(T_coef), 0, 8000]
    ax.imshow(mean_spectrogram, cmap=args.color, aspect='auto', origin='lower', norm=colors.PowerNorm(gamma=0.2))
    ax.set_title(syll_name, fontsize=15)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_xlabel('Time (ms)', fontsize=15)
    ax.set_ylabel('Frequency (Hz)', fontsize=15)
    plt.tight_layout()
    #plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Mean_spectrogram_' +syll_name + '.' + args.format)

    # Save the data
    data = {'File_name': syll_name, 'Data_list':data_list, 'List_song': data_list, 'How_many': how_many, 'Duration_syll': dur_syll,
                'Mean_pitch': mean_pitch, 'Min_pitch': min_pitch, 'Max_pitch': max_pitch, 'All_pitches': pitches_list,
                'Direction': direction_pitch,
                'Wiener_entropy': wiener, 'Cross_correlation': cross_correlation, 'Spectrograms': spectrograms, 'T_spectro':T_coef, 'Spectrograms_envelope':spectrograms_envelope}

    return data

def syllable_silence_features(songfile, syll_name, template, args):
    """
    VARIABLES:
        - songfile: list of recordings with syllables followed by silence up to 1s
                    produced previously using the trained WaveGAN
        - template: a comparison template (if given) to compute cross-correlation

    OUTPUT:
        - .npy file containing the features stored
    """

    # Count the number of elements in the group
    if args.dataset_dim == None:
        how_many = np.size(songfile)
    elif args.dataset_dim == 0:
        if args.dataset_dim > np.size(songfile):
            how_many = np.size(songfile)
        else:
            how_many = args.dataset_dim
    else:
        songfile = random.sample(songfile, args.dataset_dim)
        how_many = np.size(songfile)

    # Feature variables initialization
    dur_syll = np.zeros((how_many,))
    data_list = []
    pitches_list = []
    mean_pitch = np.zeros((how_many,))
    max_pitch = np.zeros((how_many,))
    min_pitch = np.zeros((how_many,))
    direction_pitch = np.zeros((how_many,))
    wiener = np.zeros((how_many,))

    for j in range(0,how_many):
        # Read the wave
        sr, samples = wav.read(songfile[j])
        trim = librosa.effects.trim(samples.astype(np.float), top_db=20)
        samples = trim[0]

        dur_syll[j] = samples.size/16

        # Pitch detection
        pitches, magnitudes = librosa.core.piptrack(samples, sr=sr, n_fft=100, fmin=500, fmax=8000)

        pitches_all = 0
        for interval in range(0, magnitudes.shape[1]):
            index = magnitudes[:, interval].argmax()
            pitches_all = np.append(pitches_all, pitches[index, interval])

        pitches_all = pitches_all[np.nonzero(pitches_all)]

        mean_pitch[j] = np.mean(pitches_all)
        max_pitch[j] = np.max(pitches_all)
        min_pitch[j] = np.min(pitches_all)

        if pitches_all[0] < pitches_all[-1]:
            direction_pitch[j] = 1
        else:
            direction_pitch[j] = -1

        # Spectral flatness (Wiener entropy)
        wiener[j] = np.mean(librosa.feature.spectral_flatness(samples.astype(np.float)))

    # Argmax of duration to determine the longer syllable
    if how_many > 0:
        template_size = int(np.max(dur_syll))

        # Template construction (adding silence before and after for alignment)
        if len(template) > 0:
            template_samples, sr = librosa.load(template[0], sr=16000)
            trim = librosa.effects.trim(template_samples.astype(np.float), top_db=20)
            template_samples = trim[0]

            if template_samples.size / 16 < template_size:
                aux_size = template_size - template_samples.size / 16
                silence = np.zeros((int(round(aux_size / 2) * 16)), )
                aux_template = np.append(silence, template_samples)
                aux_template = np.append(aux_template, silence)
            else:
                aux_template = template_samples

            silence = np.zeros((template_size,))
            new_template = np.append(silence, aux_template)
            new_template = np.append(new_template, silence)

            X = librosa.stft(new_template, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
            T_coef = np.arange(X.shape[1]) * args.H / sr * 1000 #save to plot
            spectrogram_template = np.log(1 + 100 * np.abs(X ** 2))
            shape_template = spectrogram_template.shape

        spectrograms = []
        spectrograms_envelope = []
        cross_correlation = np.zeros((how_many,))
        for j in range(0, how_many):
            data_list.append(songfile[j])

            # compute the spectrogram
            samples_aux, sr = librosa.load(songfile[j], sr=16000)
            trim = librosa.effects.trim(samples_aux.astype(np.float), top_db=20)
            samples_aux = trim[0]

            if samples_aux.size / 16 < template_size:
                aux_size = template_size - samples_aux.size / 16
                silence = np.zeros((int(round(aux_size / 2) * 16)),)
                samples_aux = np.append(silence, samples_aux)
                samples_aux = np.append(samples_aux, silence)

            silence = np.zeros((template_size,))
            new_template_aux = np.append(silence, samples_aux)
            new_template_aux = np.append(new_template_aux, silence)

            if new_template.size > new_template_aux.size:
                silence = np.zeros((int(round(new_template.size-new_template_aux.size))),)
                new_template_aux = np.append(new_template_aux, silence)

            X = librosa.stft(new_template_aux, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
            spectrogram_aux = np.log(1 + 100 * np.abs(X ** 2))

            if spectrogram_aux.shape[1] == shape_template[1] + 1:
                spectrogram_aux = spectrogram_aux[:,0:shape_template[1]]

            fig, ax = plt.subplots()
            (lag_aux, corr_aux, line, b) = ax.xcorr(spectrogram_template.flatten(), spectrogram_aux.flatten(), normed=True, maxlags=args.n_lags, lw=2)
            #plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'xcorr_' + syll_name + '_' + str(j) + '.' + args.format)
            cross_correlation[j] = np.max(corr_aux[1])
            index_aux = lag_aux[np.argmax(corr_aux)]

            if index_aux < 0:
                spectrogram_new = np.append(np.zeros((-index_aux,)), spectrogram_aux.flatten()[-index_aux::])
                spectrograms.append(np.reshape(spectrogram_new, shape_template))
            else:
                spectrogram_new = np.append(spectrogram_aux.flatten()[index_aux::], np.zeros((index_aux,)))
                spectrograms.append(np.reshape(spectrogram_new, shape_template))

            # Envelope based
            rawsong = samples_aux.astype(float)
            rawsong = rawsong.flatten()

            amp = Song_functions.smooth_data(rawsong, sr, freq_cutoffs=(500, 7999))

            # Debug to see amplitude range
            #plt.subplots()
            #plt.plot(amp)
            #plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'ampD.' + args.format)

            #print(songfile[j])

            if syll_name == 'N':
                # new_song = rawsong[0:np.where(amp > 0.00001)[0][-1]] # new training
                new_song = rawsong[np.where(amp > 0.00001)[0][0]::] #PRE dataset
                silence = np.zeros((new_template.size - np.size(new_song),))
                new_song = np.append(silence, new_song)

            else:
                new_song = rawsong[np.where(amp > 0.00001)[0][0]::]
                silence = np.zeros((new_template.size - np.size(new_song),))
                new_song = np.append(new_song, silence)

            X = librosa.stft(new_song, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                             pad_mode='constant', center=True)
            spectrograms_envelope.append(np.log(1 + 100 * np.abs(X ** 2)))

            plt.close('all')

    else:
        cross_correlation = np.zeros((1,))
        T_coef = 0
        spectrograms = np.zeros((8000,1))
        spectrograms_envelope = np.zeros((8000,1))

    # Save the data
    data = {'File_name': syll_name, 'Data_list':data_list, 'List_song': data_list, 'How_many': how_many, 'Duration_syll': dur_syll,
                'Mean_pitch': mean_pitch, 'Min_pitch': min_pitch, 'Max_pitch': max_pitch, 'All_pitches': pitches_list,
                'Direction': direction_pitch,
                'Wiener_entropy': wiener, 'Cross_correlation': cross_correlation, 'Spectrograms': spectrograms, 'T_spectro':T_coef, 'Spectrograms_envelope':spectrograms_envelope}

    return data

def lag_corr_training_dataset(songfile, classes, args):
    import statistics as stat

    data_syll_list = []
    for c in range(0, np.size(classes)):
        syll_list_aux = []
        for i in range(0, np.size(songfile)):
            if (songfile[i].find('NEW_' + classes[c]) != -1):
                syll_list_aux.append(songfile[i])
        data_syll_list.append(syll_list_aux)

    cross_corr = []
    cross_corr_2 = []
    for c in range(0, np.size(classes)):
        pairs = stat.pairs(range(0, np.size(data_syll_list[c])), range(0, np.size(data_syll_list[c])))
        syll_pairs = random.sample(pairs, args.n_template)

        cross_corr_aux = np.zeros((args.n_template,))
        cross_corr_aux_2 = np.zeros((args.n_template,))
        cross_corr_aux_3 = np.zeros((args.n_template,))

        for j in range(0,args.n_template):
            cross_corr_aux[j] = stat.lag_cross_corr(args.n_lags, data_syll_list[c][syll_pairs[j][0]], data_syll_list[c][syll_pairs[j][1]], args.nperseg, args.overlap)
            plt.close('all')

            sr, samples_1 = wav.read(data_syll_list[c][syll_pairs[j][0]])
            sr, samples_2 = wav.read(data_syll_list[c][syll_pairs[j][1]])

            freq, times, spectrogram_1 = sp.signal.spectrogram(samples_1, sr, window='hann', nperseg=args.nperseg,
                                                               noverlap=args.nperseg - args.overlap)

            freq, times, spectrogram_2 = sp.signal.spectrogram(samples_2, sr, window='hann', nperseg=args.nperseg,
                                                               noverlap=args.nperseg - args.overlap)

            fig, ax = plt.subplots()
            cross_aux = ax.xcorr(spectrogram_1.flatten(), spectrogram_2.flatten(), maxlags=args.n_lags, lw=2)
            cross_correlation_aux_2[j] = np.max(cross_aux[1])

            y, sr = librosa.load(data_syll_list[c][syll_pairs[j][0]], sr=16000)
            y_2, sr = librosa.load(data_syll_list[c][syll_pairs[j][1]], sr=16000)

            X = librosa.stft(y, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                             pad_mode='constant', center=True)
            spectrogram_aux = np.log(1 + 100 * np.abs(X ** 2))

            X_2 = librosa.stft(y_2, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                             pad_mode='constant', center=True)
            spectrogram_aux_2 = np.log(1 + 100 * np.abs(X_2 ** 2))

            freq_downsample_1 = sp.signal.resample(spectrogram_aux, 60, t=None, axis=0)
            time_downsample_1 = sp.signal.resample(freq_downsample_1, 120, t=None, axis=1)

            freq_downsample_2 = sp.signal.resample(spectrogram_aux_2, 60, t=None, axis=0)
            time_downsample_2 = sp.signal.resample(freq_downsample_2, 120, t=None, axis=1)

            fig, ax = plt.subplots()
            (lag_aux, corr_aux, line, b) = ax.xcorr(time_downsample_1.flatten(), time_downsample_2.flatten(), maxlags=args.n_lags, lw=2)
            #(lag_aux, corr_aux, line, b) = ax.xcorr(spectrogram_aux.flatten(), spectrogram_aux_2.flatten(), maxlags=args.n_lags, lw=2)
            cross_corr_aux_3[j] = np.max(corr_aux[1])

            plt.close('all')

        print(cross_corr_aux)
        print(cross_corr_aux_2)
        print(cross_corr_aux_3)

        input()

        cross_corr.append(cross_corr_aux)
        cross_corr_2.append(cross_corr_aux_2)

    np.save(args.data_dir + '/' + args.output_dir + '/' + 'Dataset_cross_corr_distribution.npy', cross_corr)
    np.save(args.data_dir + '/' + args.output_dir + '/' + 'Dataset_cross_corr_distribution_2.npy', cross_corr_2)

    # Plot
    for c in range(0, np.size(classes)):
        plt.subplots()
        n, x, _ = plt.hist(cross_corr[c,:], 20, color='b', alpha=0)
        bin_centers = 0.5 * (x[1:] + x[:-1])
        plt.plot(bin_centers, n, 'k', alpha=0.5)
        plt.ylabel('Cross-correlation')
        plt.title(classes[c], fontsize=15)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' +  'cross_correlation_class_' + classes[c] + '.' + args.format)

        plt.subplots()
        n, x, _ = plt.hist(cross_corr_2[c,:], 20, color='b', alpha=0)
        bin_centers = 0.5 * (x[1:] + x[:-1])
        plt.plot(bin_centers, n, 'k', alpha=0.5)
        plt.ylabel('Cross-correlation')
        plt.title(classes[c], fontsize=15)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'cross_correlation_2_class_' + classes[c] + '.' + args.format)

    plt.close('all')

    print('Done')

def data_features_analysis_plots(plot_data, args):
    #import umap
    #import umap.plot
    # FEATURES
    # Initialization of the features
    how_many = [0]
    name = ['label']
    duration_syll = [0]
    duration_syll_aux = list()
    mean_pitch = [0]
    mean_pitch_aux = list()
    max_pitch =[0]
    max_pitch_aux = list()
    min_pitch = [0]
    min_pitch_aux = list()
    wiener = [0]
    wiener_aux = list()
    all_spectrograms = []
    all_labels = []
    mean_spectrogram = []
    mean_spectrogram_env = []
    mean_cross_dataset = []
    T_coef = []

    for i in range(0, np.size(plot_data)):
        load_data = np.load(plot_data[i], allow_pickle = True) #added allow_pickle because of a new error I never found before opening datasets I created and already opened months ago
        load_data = load_data.item()
        how_many = np.append(how_many, load_data['How_many'])
        name = np.append(name, load_data['File_name'])
        duration_syll_aux.append(np.round(load_data['Duration_syll']))
        duration_syll = np.append(duration_syll, duration_syll_aux[i])
        mean_pitch_aux.append(load_data['Mean_pitch'])
        mean_pitch = np.append(mean_pitch, mean_pitch_aux[i])
        min_pitch_aux.append(load_data['Min_pitch'])
        min_pitch = np.append(min_pitch, min_pitch_aux[i])
        max_pitch_aux.append(load_data['Max_pitch'])
        max_pitch = np.append(max_pitch, max_pitch_aux[i])
        wiener_aux.append(load_data['Wiener_entropy'])
        wiener = np.append(wiener, wiener_aux[i])
        spectrograms = load_data['Spectrograms']
        spectrograms_envelope = load_data['Spectrograms_envelope']

        for s in range(0, len(spectrograms)):
            all_spectrograms.append(spectrograms[s].flatten())
            all_labels.append(name[i+1])
        cross_corr_dataset = load_data['Cross_correlation']
        T_coef.append(np.max(load_data['T_spectro']))

        mean_spectrogram.append(np.mean(spectrograms,axis=0))
        mean_spectrogram_env.append(np.mean(spectrograms_envelope,axis=0))
        mean_cross_dataset.append(np.mean(cross_corr_dataset))

        if np.int(np.round(np.size(duration_syll_aux[i])/10))>0:
            # Duration per type of syllables
            h, bins = np.histogram(duration_syll_aux[i], bins = np.int(np.round(np.size(duration_syll_aux[i])/10)))
            fig, ax = plt.subplots()
            plt.bar(bins[:-1], h, width=0.8, color='b', alpha=0.6, label='Syllable Duration')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
            ax.set_xlim([0,np.max(duration_syll_aux[i])])
            plt.xlabel('Duration(ms)')
            plt.ylabel('Number of occurences')
            plt.title(name[i+1])
            plt.tight_layout()  # to avoid the cut of labels
            plt.savefig(args.data_dir + '/' + 'Duration_syllable_' + name[i+1] + '.' + args.format)

            # Pitch per type of syllable
            # Mean pitch
            h, bins = np.histogram(mean_pitch_aux[i], bins = np.int(np.round(np.size(mean_pitch_aux[i])/10)))
            fig, ax = plt.subplots()
            plt.bar(bins[:-1], h, width=0.8, color='b', alpha=0.6, label='Mean pitch')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
            ax.set_xlim([0,np.max(mean_pitch_aux[i])])
            plt.xlabel('Mean pitch (Hz)')
            plt.ylabel('Number of occurences')
            plt.title(name[i+1])
            plt.tight_layout()  # to avoid the cut of labels
            plt.savefig(args.data_dir + '/' + 'Mean_pitch_syllable_' + name[i+1]  + '.' + args.format)

            # Min pitch
            h, bins = np.histogram(min_pitch_aux[i], bins = np.int(np.round(np.size(min_pitch_aux[i])/10)))
            fig, ax = plt.subplots()
            plt.bar(bins[:-1], h, width=0.8, color='b', alpha=0.6, label='Min pitch')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
            ax.set_xlim([0,np.max(min_pitch_aux[i])])
            plt.xlabel('Mean pitch (Hz)')
            plt.ylabel('Number of occurences')
            plt.title(name[i+1])
            plt.tight_layout()  # to avoid the cut of labels
            plt.savefig(args.data_dir + '/' + 'Min_pitch_syllable_' + name[i+1] + '.' + args.format)

            # Max pitch
            h, bins = np.histogram(max_pitch_aux[i], bins = np.int(np.round(np.size(max_pitch_aux[i])/10)))
            fig, ax = plt.subplots()
            plt.bar(bins[:-1], h, width=0.8, color='b', alpha=0.6, label='Max pitch')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
            ax.set_xlim([0,np.max(max_pitch_aux[i])])
            plt.xlabel('Mean pitch (Hz)')
            plt.ylabel('Number of occurences')
            plt.title(name[i+1])
            plt.tight_layout()  # to avoid the cut of labels
            plt.savefig(args.data_dir + '/' + 'Max_pitch_syllable_' + name[i+1] + '.' + args.format)

            # Wiener entropy
            h, bins = np.histogram(wiener_aux[i], bins = np.int(np.round(np.size(wiener_aux[i])/10)))
            fig, ax = plt.subplots()
            plt.bar(bins[:-1], h, width=0.001, color='b', alpha=0.6, label='Wiener entropy')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
            ax.set_xlim([0,np.max(wiener_aux[i])])
            plt.xlabel('Wiener entropy(dB)')
            plt.ylabel('Number of occurences')
            plt.title(name[i+1])
            plt.tight_layout()  # to avoid the cut of labels
            plt.savefig(args.data_dir + '/' + 'Wiener_entropy_syllable_' + name[i+1] + '.' + args.format)

        # Cross-correlation
        plt.subplots()
        n, x, _ = plt.hist(cross_corr_dataset, 20, color='b', alpha=0)
        bin_centers = 0.5 * (x[1:] + x[:-1])
        plt.plot(bin_centers, n, 'k', alpha=0.5)
        plt.ylabel('Cross-correlation')
        plt.title(name[i+1], fontsize=15)
        plt.savefig(args.data_dir + '/' + 'cross_correlation_class_' + name[i+1] + '.' + args.format)

        plt.close('all')

    how_many = how_many[1::]
    name = name[1::]
    duration_syll = duration_syll[1::]
    mean_pitch = mean_pitch[1::]
    min_pitch = min_pitch[1::]
    max_pitch = max_pitch[1::]
    wiener = wiener[1::]

    # How many syllables per type
    fig, ax = plt.subplots()
    ax.bar(name, how_many)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylabel('Number of samples', fontsize=15)
    plt.xlabel('Classes', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    #plt.savefig(args.data_dir + '/' + 'How_many_syllables.' + args.format)

    # Cumulative duration
    h, bins = np.histogram(duration_syll, bins=np.int(np.round(np.size(duration_syll)/10)))
    fig, ax = plt.subplots()
    plt.bar(bins[:-1], h/np.max(h), width = 0.8, color = 'g', alpha = 0.6)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    #plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
    #ax.set_xlim([0,np.max(duration_syll)])
    plt.xlabel('Duration(ms)', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    #plt.title('Syllable duration')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout() #to avoid the cut of labels
    plt.savefig(args.data_dir + '/' + 'Duration_syll_prob.' + args.format)

    h, bins = np.histogram(duration_syll, bins=np.int(np.round(np.size(duration_syll)/10)))
    fig, ax = plt.subplots()
    plt.bar(bins[:-1], h, width = 0.8, color = 'b', alpha = 1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    #plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
    #ax.set_xlim([0,np.max(duration_syll)])
    plt.xlabel('Duration(ms)', fontsize=15)
    plt.ylabel('Number of occurences',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title('Syllable duration')
    plt.tight_layout() #to avoid the cut of labels
    plt.savefig(args.data_dir + '/' +'Duration_syll.' + args.format)

    # Cumulative pitch
    # Mean pitch
    h, bins = np.histogram(mean_pitch, bins=np.int(np.round(np.size(mean_pitch)/10)))
    fig, ax = plt.subplots()
    plt.bar(bins[:-1], h, width = 0.8, color = 'g', alpha = 0.6, label = 'Pitch')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_xlim([0,np.max(mean_pitch)])
    plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
    plt.xlabel('Mean pitch(Hz)')
    plt.ylabel('Number of occurences')
    plt.title('Mean pitch distribution')
    plt.tight_layout() #to avoid the cut of labels
    plt.savefig(args.data_dir + '/' + 'Mean_pitch_distribution.' + args.format)

    # Mean pitch
    h, bins = np.histogram(min_pitch, bins=np.int(np.round(np.size(min_pitch)/10)))
    fig, ax = plt.subplots()
    plt.bar(bins[:-1], h, width = 0.8, color = 'g', alpha = 0.6, label = 'Pitch')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_xlim([0,np.max(min_pitch)])
    plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
    plt.xlabel('Min pitch(Hz)')
    plt.ylabel('Number of occurences')
    plt.title('Mean pitch distribution')
    plt.tight_layout() #to avoid the cut of labels
    plt.savefig(args.data_dir + '/' + 'Min_pitch_distribution.' + args.format)

    # Mean pitch
    h, bins = np.histogram(max_pitch, bins=np.int(np.round(np.size(max_pitch)/10)))
    fig, ax = plt.subplots()
    plt.bar(bins[:-1], h, width = 0.8, color = 'g', alpha = 0.6, label = 'Pitch')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_xlim([0,np.max(max_pitch)])
    plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
    plt.xlabel('Max pitch(Hz)')
    plt.ylabel('Number of occurences')
    plt.title('Mean pitch distribution')
    plt.tight_layout() #to avoid the cut of labels
    plt.savefig(args.data_dir + '/' + 'Max_pitch_distribution.' + args.format)

    # Cumulative spectral flatness
    h, bins = np.histogram(wiener, bins=np.int(np.round(np.size(wiener)/10)))
    fig, ax = plt.subplots()
    plt.bar(bins[:-1], h, width = 0.001, color = 'g', alpha = 0.6, label = 'Wiener entropy')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    leg = plt.legend(loc='upper right', fontsize=15, ncol=1, shadow=True, fancybox=True)
    ax.set_xlim([0,np.max(wiener)])
    plt.xlabel('Wiener entropy(dB)')
    plt.ylabel('Number of occurences')
    plt.title('Wiener entropy')
    plt.tight_layout() #to avoid the cut of labels
    plt.savefig(args.data_dir + '/' + 'Wiener_entropy.' + args.format)

    # Clustering
    # Duration VS entropy
    fig = plt.subplots()
    plt.plot(duration_syll, wiener, '.')
    plt.title('Duration VS entropy')
    plt.xlabel('Duration(ms)')
    plt.ylabel('Wiener entropy(dB)')
    plt.savefig(args.data_dir + '/' + 'durationVSwiener.' + args.format)

    # Mean pitch VS entropy
    fig = plt.subplots()
    plt.plot(mean_pitch, wiener, '.')
    plt.title('Mean pitch VS entropy')
    plt.xlabel('Mean pitch(Hz)')
    plt.ylabel('Wiener entropy(dB)')
    plt.savefig(args.data_dir + '/' + 'meanpitchVSwiener.' + args.format)

    # Min pitch VS entropy
    fig = plt.subplots()
    plt.plot(min_pitch, wiener, '.')
    plt.title('Min pitch VS entropy')
    plt.xlabel('Min pitch(Hz)')
    plt.ylabel('Wiener entropy(dB)')
    plt.savefig(args.data_dir + '/' + 'minpitchVSwiener.' + args.format)

    # Max pitch vs entropy
    fig = plt.subplots()
    plt.plot(max_pitch, wiener, '.')
    plt.title('Max pitch VS entropy')
    plt.xlabel('Max pitch(Hz)')
    plt.ylabel('Wiener entropy(dB)')
    plt.savefig(args.data_dir + '/' + 'maxpitchVSwiener.' + args.format)

    # KMeans: analysis of the classes of syllables
    # Max pitch VS wiener
    X = [max_pitch, wiener]
    kmeans = KMeans(16, random_state=0).fit(np.transpose(X))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    predict = kmeans.predict(np.transpose(X))

    fig, ax = plt.subplots()
    plt.scatter(np.transpose(X)[:, 0], np.transpose(X)[:, 1], c=predict, s=10, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=30, alpha=0.3)
    plt.xlabel('Pitch(Hz)')
    plt.ylabel('Wiener entropy(dB)')
    plt.title('Max pitch VS Wiener entropy')
    plt.savefig(args.data_dir + '/' + 'max-pitchVSwiener.' + args.format)

    # Min pitch VS wiener
    X = [min_pitch, wiener]
    kmeans = KMeans(16, random_state=0).fit(np.transpose(X))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    predict = kmeans.predict(np.transpose(X))

    fig, ax = plt.subplots()
    plt.scatter(np.transpose(X)[:, 0], np.transpose(X)[:, 1], c=predict, s=10, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=30, alpha=0.3)
    plt.xlabel('Pitch(Hz)')
    plt.ylabel('Wiener entropy(dB)')
    plt.title('Min pitch VS Wiener entropy')
    plt.savefig(args.data_dir + '/' + 'min_pitchVSwiener.' + args.format)

    # Mean pitch VS wiener
    X = [mean_pitch, wiener]
    kmeans = KMeans(16, random_state=0).fit(np.transpose(X))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    predict = kmeans.predict(np.transpose(X))

    fig, ax = plt.subplots()
    plt.scatter(np.transpose(X)[:, 0], np.transpose(X)[:, 1], c=predict, s=10, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=30, alpha=0.3)
    plt.xlabel('Pitch(Hz)')
    plt.ylabel('Wiener entropy(dB)')
    plt.title('Mean pitch VS Wiener entropy')
    plt.savefig(args.data_dir + '/' + 'mean-pitchVSwiener.' + args.format)

    # Duration VS Wiener
    X = [duration_syll, wiener]
    kmeans = KMeans(16, random_state=0).fit(np.transpose(X))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    predict = kmeans.predict(np.transpose(X))

    fig, ax = plt.subplots()
    plt.scatter(np.transpose(X)[:, 0], np.transpose(X)[:, 1], c=predict, s=10, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=30, alpha=0.3)
    plt.xlabel('Duration(ms)')
    plt.ylabel('Wiener entropy(dB)')
    plt.title('Duration VS Wiener entropy')
    plt.savefig(args.data_dir + '/' + 'dur-pitchVSwiener.' + args.format)

    # Mean spectrogram
    #plt.style.use('dark_background')
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 14), sharey=True, sharex=True)
    for i in range(0, 4):
        for j in range(0, 4):
            extent = [0, np.max(T_coef[4 * i + j]), 0, 8000]
            if mean_spectrogram[4 * i + j].size>1:
                axs[i, j].imshow(mean_spectrogram[4 * i + j], extent = extent, cmap=args.color, aspect='auto', origin='lower', norm=colors.PowerNorm(gamma=0.5)) #gamma 0.2 in original data
            axs[i, j].set_title(name[4 * i + j], fontsize=15)
            #axs[i, j].set_xlim(0,350)
            axs[i, j].spines['top'].set_color('none')
            axs[i, j].spines['right'].set_color('none')
            axs[0, j].set_xlabel('Time (ms)', fontsize=15)
        axs[i, 3].set_ylabel('Frequency (Hz)', fontsize=15)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + 'Mean_spectrogram.' + args.format)

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 14), sharey=True, sharex=True)
    for i in range(0, 4):
        for j in range(0, 4):
            extent = [0, np.max(T_coef[4 * i + j]), 0, 8000]
            if mean_spectrogram[4 * i + j].size > 1:
                axs[i, j].imshow(mean_spectrogram_env[4 * i + j], extent=extent, cmap=args.color, aspect='auto', origin='lower', norm=colors.PowerNorm(gamma=0.5))  # gamma 0.2 in original data
            axs[i, j].set_title(name[4 * i + j], fontsize=15)
            axs[i, j].set_xlim(0,300)
            axs[i, j].spines['top'].set_color('none')
            axs[i, j].spines['right'].set_color('none')
            axs[0, j].set_xlabel('Time (ms)', fontsize=15)
        axs[i, 3].set_ylabel('Frequency (Hz)', fontsize=15)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + 'Mean_spectrogram_envelope.' + args.format)

    plt.close('all')
    print('Done')

def data_silence_plot(plot_data, classes, args):

    load_data = np.load(plot_data[0], allow_pickle=True)  # added allow_pickle because of a new error I never found before opening datasets I created and already opened months ago
    load_data = load_data.item()

    # FEATURES
    # Initialization of the features
    T_coef = load_data['T_spectro']
    spectrograms = load_data['Spectrograms']
    spectrograms_envelope = load_data['Spectrograms_envelope']
    cross_corr_dataset = load_data['Cross_correlation']

    all_spectrograms = []
    all_labels = []
    mean_spectrogram = []
    mean_spectrogram_env = []
    mean_cross_dataset = []
    for c in range(0, np.size(classes)):
        for s in range(0, len(spectrograms[c])):
            all_spectrograms.append(spectrograms[c][s].flatten())
            all_labels.append(classes[c])

        mean_spectrogram.append(np.mean(spectrograms[c],axis=0))
        mean_spectrogram_env.append(np.mean(spectrograms_envelope[c], axis=0))
        mean_cross_dataset.append(np.mean(cross_corr_dataset))

    # Mean spectrogram
    #plt.style.use('dark_background')
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 14), sharey=True, sharex=True)
    for i in range(0, 4):
        for j in range(0, 4):
            extent = [0, np.max(T_coef[4 * i + j]), 0, 8000]
            if mean_spectrogram[4 * i + j].size > 1:
                axs[i, j].imshow(mean_spectrogram_env[4 * i + j], extent=extent, cmap=args.color, aspect='auto',
                                 origin='lower', norm=colors.PowerNorm(gamma=0.2))
            axs[i, j].set_title(classes[4 * i + j], fontsize=15)
            axs[i, j].set_xlim(0,350)
            axs[i, j].spines['top'].set_color('none')
            axs[i, j].spines['right'].set_color('none')
            axs[0, j].set_xlabel('Time (ms)', fontsize=15)
        axs[i, 3].set_ylabel('Frequency (Hz)', fontsize=15)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + 'Mean_spectrogram.' + args.format)

    print('Done')
    
if __name__ == '__main__':
    import argparse
    import glob
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str,
                        choices=['repertoire', 'test', 'song_features', 'syll_features', 'syll_silence_features', 'syll_silence_features_GAN', 'lag', 'plot', 'plot_silence'],
                        help="which simulation type should be chosen")
    parser.add_argument('--data_dir', type=str,
                        help='Data directory', default=None)
    parser.add_argument('--output_dir', type=str,
                        help='Output directory', default=None)
    parser.add_argument('--train_dir', type=str, help='Name of the training directory', default=None)

    data_args = parser.add_argument_group('Data')
    data_args.add_argument('--threshold', type=float,
                           help='Threshold to select syllables', default=100000)
    data_args.add_argument('--min_syl_dur', type=float,
                           help='Minimal duration for a syllable to be selected', default=0.003)
    data_args.add_argument('--min_silent_dur', type=float,
                           help='Minimal silence in between two syllables, to cut the selection there', default=0.01)
    data_args.add_argument('--f_cut_min', type = int,
                           help='Minimal cut off in frequency when cutting the syllables, greater than zero',
                           default=500)
    data_args.add_argument('--f_cut_max', type = int,
                           help='Maximal cut off in frequency when cutting the syllables, less than sampling rate - 1',
                           default=7999)

    syll_args = parser.add_argument_group('Single_syllable')
    syll_args.add_argument('--dataset_dim', type = int,
                           help='How many syllables I used for the dataset',
                           default=None)
    syll_args.add_argument('--template_dir', type=str, help='Directory containing the templates', default='Repertoire')

    plot_args = parser.add_argument_group('Plot')
    plot_args.add_argument('--window', type=str,
                           help='Type of window for the visualization of the spectrogram', default='hanning')
    plot_args.add_argument('--overlap', type=int,
                           help='Overlap for the visualization of the spectrogram', default=66)
    plot_args.add_argument('--nperseg', type=int,
                           help='Nperseg for the visualization of the spectrogram', default=1024)
    plot_args.add_argument('--format', type=str, help='Saving format', default='png')
    plot_args.add_argument('--n_lags', type=int, help='Number of lags used to compute the cross-correlation', default=100)
    plot_args.add_argument('--n_template', type=int, help='Number of pairs to compute the cross-correlation', default=100)
    plot_args.add_argument('--syll_name', type = str, help='Name of the syllable')
    plot_args.add_argument('--N', type = int, help='Nftt spectrogram librosa', default=256)
    plot_args.add_argument('--H', type = int, help='Hop length spectrogram librosa', default=64)
    plot_args.add_argument('--color', type = str, help='Colormap', default='inferno')

    args = parser.parse_args()

    # Creation of directories (if needed)
    if args.output_dir != None:
        if not os.path.isdir(args.data_dir + '/' + args.output_dir):
            os.makedirs(args.data_dir + '/' + args.output_dir)

    if args.train_dir != None:
        if not os.path.isdir(args.data_dir + '/' + args.train_dir):
            os.makedirs(args.data_dir + '/' + args.train_dir)

    # Save args
    with open(os.path.join(args.data_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    if args.option == 'repertoire':
        # List of recordings
        songfile = sorted(glob.glob(args.data_dir + '/' + '*wav'))
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']
        repertoire(songfile, classes, args)

    if args.option == 'test':
        # List of recordings
        songfile = glob.glob(args.data_dir + '/' + '*wav')
        test_features(songfile, args)
        
    elif args.option == 'song_features':
        # List of recordings containing several syllables each
        songfile = glob.glob(args.data_dir + '/' + '*wav')

        single_syllable_features_from_song(songfile,args)

    elif args.option == 'syll_features':
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

        # List of recordings containing one syllable followed by silence (produced via the GAN)
        songfile_dir = []
        template_list = []
        for c in range(0, np.size(classes)):
            songfile_dir.append(glob.glob(args.data_dir + '/' + 'Train_nosilence' + '/' + 'NEW_'+ classes[c] + '*.wav'))
            template_list.append(glob.glob(args.data_dir + '/' + args.template_dir + '/' + 'NEW_' + classes[c] + '*.wav'))

        for c in range(0, np.size(classes)):
            print(classes[c])
            data = single_syllable_features(songfile_dir[c], classes[c], template_list[c], args)
            np.save(args.data_dir + '/' + args.output_dir + '/'  + classes[c] +'.npy', data)

        print('Done')

    elif args.option == 'syll_silence_features':
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

        # List of recordings containing one syllable followed by silence (produced via the GAN)
        songfile_dir = []
        template_list = []
        for c in range(0, np.size(classes)):
            songfile_dir.append(glob.glob(args.data_dir + '/' + 'Train_nosilence' + '/' + 'NEW_'+ classes[c] + '*.wav'))
            template_list.append(glob.glob(args.data_dir + '/' + args.template_dir + '/' + 'NEW_' + classes[c] + '*.wav'))

        for c in range(0, np.size(classes)):
            print(classes[c])
            data = syllable_silence_features(songfile_dir[c], classes[c], template_list[c], args)
            np.save(args.data_dir + '/' + args.output_dir + '/'  + classes[c] +'.npy', data)

        print('Done')

    elif args.option == 'syll_silence_features_GAN':
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

        # List of recordings containing one syllable followed by silence (produced via the GAN)
        aux = np.load(args.data_dir + '/' + 'Generation_summary_ep0997_ld3.npy', allow_pickle=True)
        aux = aux.item()
        aux_names = aux['File_name']
        aux_decoder_names = aux['Decoder_name']

        songfile_dir = []
        template_list = []

        for c in range(0, np.size(classes)):
            syll_list_aux = []
            for i in range(0, np.size(aux_names)):
                if (aux_decoder_names[i].find(classes[c]) != -1):
                    syll_list_aux.append(aux_names[i])
            songfile_dir.append(syll_list_aux)

        for c in range(0, np.size(classes)):
            template_list.append(glob.glob(args.data_dir + '/' + args.template_dir + '/' + 'NEW_' + classes[c] + '*.wav'))

        for c in range(0, np.size(classes)):
            print(classes[c])
            data = syllable_silence_features(songfile_dir[c], classes[c], template_list[c], args)
            np.save(args.data_dir + '/' + args.output_dir + '/'  + classes[c] +'.npy', data)

        print('Done')

    elif args.option == 'lag':
        # List of recordings
        songfile = glob.glob(args.data_dir + '/' + '*wav')

        # Classes
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

        lag_corr_training_dataset(songfile, classes, args)

    elif args.option == 'plot':
        # Templates
        songfile = sorted(glob.glob(args.template_dir + '/' + '*wav'))

        # Data
        plot_data = sorted(glob.glob(args.data_dir + '/' + '*.npy'))

        data_features_analysis_plots(plot_data, args)

    elif args.option == 'plot_silence':
        # Classes
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']
        # Data
        plot_data = sorted(glob.glob(args.data_dir + '/' + '*.npy'))

        data_silence_plot(plot_data, classes, args)
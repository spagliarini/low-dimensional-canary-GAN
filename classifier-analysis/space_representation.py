# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:22:29 2019

@author: Mnemosyne
"""

import pickle 
import os
import random
import glob
import numpy as np
import librosa
import librosa.display
import librosa.feature
import librosa.effects
import scipy.io.wavfile as wav
import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import rcParams, cm, colors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import audiosegment
from pydub import AudioSegment

from sklearn.decomposition import PCA
import umap
import umap.plot

import plots
import statistics

def open_pkl(name):
    data = open(name, 'rb')
    z0 = pickle.load(data)
    return z0

def syllable_space_UMAP(summary_dataset, args):
    classes_colors = ['grey', 'gainsboro', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen', 'tan','darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink']
    classes_colors_garbage = ['grey', 'gainsboro', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen', 'tan','darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink', 'white']
    classes_colors_intensity = ['blue', 'deepskyblue', 'aqua', 'lime', 'greenyellow', 'yellow', 'darkorange',
                                'orangered', 'red', 'maroon']
    classes_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("16classes", classes_colors)
    classes_cmap_X = matplotlib.colors.LinearSegmentedColormap.from_list("16classes_garbage", classes_colors_garbage)
    classes_cmap_intensity = matplotlib.colors.LinearSegmentedColormap.from_list("intensity", classes_colors_intensity)
    plt.register_cmap("16classes", classes_cmap)
    plt.register_cmap("16classes_garbage", classes_cmap_X)
    plt.register_cmap("intensity", classes_cmap_intensity)

    # Representation of the data
    load_data = np.load(summary_dataset[0], allow_pickle=True)
    load_data = load_data.item()
    file_name = load_data['File_name']
    real_name = load_data['Real_name']
    decoder_name = load_data['Decoder_name']
    raw_sum_distr = load_data['Annotations']

    all_samples = []
    all_spectrograms = []
    error_labels = []
    intensity_labels = []
    err = 0
    for s in range(0, len(file_name)):
        read_aux, sr = librosa.load(file_name[s], 16000)
        all_samples.append(read_aux[0:15999])
        X = librosa.stft(all_samples[s], n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
        T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
        spect_aux = np.log(1 + 100 * np.abs(X ** 2))
        all_spectrograms.append(spect_aux.flatten())

        if args.all_garbage == True:
            if decoder_name[s] == 'EARLY15':
                decoder_name[s] = 'X'
            if decoder_name[s] == 'EARLY30':
                decoder_name[s] = 'X'
            if decoder_name[s] == 'EARLY45':
                decoder_name[s] = 'X'
            if decoder_name[s] == 'OT':
                decoder_name[s] = 'X'
            if decoder_name[s] == 'WN':
                decoder_name[s] = 'X'

            if real_name[s] == decoder_name[s]:
                error_labels.append('Correct')
            else:
                error_labels.append('Incorrect')
                err = err + 1

        aux_intensity = np.max(raw_sum_distr[s])
        if ((aux_intensity> 0) and (aux_intensity<= 0.1)):
            intensity_labels.append('0.1')
        elif ((aux_intensity> 0.2) and (aux_intensity<= 0.3)):
            intensity_labels.append('0.2')
        elif ((aux_intensity> 0.2) and (aux_intensity<= 0.3)):
            intensity_labels.append('0.3')
        elif ((aux_intensity > 0.3) and (aux_intensity <= 0.4)):
            intensity_labels.append('0.4')
        elif ((aux_intensity > 0.4) and (aux_intensity <= 0.5)):
            intensity_labels.append('0.5')
        elif ((aux_intensity > 0.5) and (aux_intensity <= 0.6)):
            intensity_labels.append('0.6')
        elif ((aux_intensity > 0.6) and (aux_intensity <= 0.7)):
            intensity_labels.append('0.7')
        elif ((aux_intensity > 0.7) and (aux_intensity <= 0.8)):
            intensity_labels.append('0.8')
        elif ((aux_intensity > 0.8) and (aux_intensity <= 0.9)):
            intensity_labels.append('0.9')
        elif ((aux_intensity > 0.9) and (aux_intensity <= 1)):
            intensity_labels.append('1')

    print(err)

    mapper = umap.UMAP(random_state=args.seed, spread=args.spread, n_neighbors=args.n_neigh, min_dist=args.min_d).fit(np.array(all_spectrograms))
    plt.subplots()
    umap.plot.points(mapper, np.asarray(intensity_labels), color_key_cmap='intensity', background='black')
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'umap_intensity' + '.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'umap_intensity' + '.' + 'png')

    plt.subplots()
    umap.plot.points(mapper, np.asarray(real_name), color_key_cmap='16classes', background='black')
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'umap_spec_real_black' + '.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'umap_spec_real_black' + '.' + 'png')

    plt.subplots()
    umap.plot.points(mapper, np.asarray(decoder_name), color_key_cmap='16classes_garbage', background='black')
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'umap_spec_decoder_black' + '.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'umap_spec_decoder_black' + '.' + 'png')

    plt.subplots()
    umap.plot.points(mapper, np.asarray(error_labels), color_key_cmap='Paired', background='black')
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'error_umap_' + '.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'error_umap_' + '.' + 'png')

    print('Done')

def latent_space_UMAP(gen_summary, args):
    # Color map
    if args.all_garbage == True:
        classes_colors = ['grey', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen','darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink', 'white']
        classes_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("14classes_garbage", classes_colors)
        plt.register_cmap("14classes_garbage", classes_cmap)
    else:
        classes_colors = ['grey', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen', 'darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink', 'lightcoral', 'brown', 'red', 'yellow', 'white']
        classes_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("14classes_garbage", classes_colors)
        plt.register_cmap("14classes_garbage", classes_cmap)

    classes_colors_intensity = ['blue', 'deepskyblue', 'aqua', 'lime', 'greenyellow', 'yellow', 'darkorange', 'orangered', 'red', 'maroon', 'white']
    classes_cmap_intensity = matplotlib.colors.LinearSegmentedColormap.from_list("intensity", classes_colors_intensity)
    plt.register_cmap("intensity", classes_cmap_intensity)

    for i in range(0, np.size(gen_summary)):
        load_data = np.load(gen_summary[i], allow_pickle=True)
        load_data = load_data.item()
        file_name = load_data['File_name']
        decoder_name = load_data['Decoder_name']
        epoch = load_data['Epoch']
        raw_sum_distr = load_data['Annotations']

        all_samples = []
        all_spectrograms = []
        intensity_labels = []
        for s in range(0, len(file_name)):
            read_aux, sr = librosa.load(file_name[s], 16000)
            all_samples.append(read_aux[0:15999])
            X = librosa.stft(all_samples[s], n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                             pad_mode='constant', center=True)
            T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
            spect_aux = np.log(1 + 100 * np.abs(X ** 2))
            all_spectrograms.append(spect_aux.flatten())

            if decoder_name[s] == 'B1':
                decoder_name[s] = 'B'
            elif decoder_name[s] == 'B2':
                decoder_name[s] = 'B'
            elif decoder_name[s] == 'J1':
                decoder_name[s] = 'J'
            elif decoder_name[s] == 'J2':
                decoder_name[s] = 'J'

            if args.all_garbage == True:
                if decoder_name[s] == 'EARLY15':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'EARLY30':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'EARLY45':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'OT':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'WN':
                    decoder_name[s] = 'X'

            aux_intensity = np.max(raw_sum_distr[s])
            if decoder_name[s] == 'X':
                aux_intensity = 0
            if (aux_intensity == 0):
                intensity_labels.append('X')
            if ((aux_intensity > 0) and (aux_intensity <= 0.1)):
                intensity_labels.append('0.1')
            elif ((aux_intensity > 0.2) and (aux_intensity <= 0.3)):
                intensity_labels.append('0.2')
            elif ((aux_intensity > 0.2) and (aux_intensity <= 0.3)):
                intensity_labels.append('0.3')
            elif ((aux_intensity > 0.3) and (aux_intensity <= 0.4)):
                intensity_labels.append('0.4')
            elif ((aux_intensity > 0.4) and (aux_intensity <= 0.5)):
                intensity_labels.append('0.5')
            elif ((aux_intensity > 0.5) and (aux_intensity <= 0.6)):
                intensity_labels.append('0.6')
            elif ((aux_intensity > 0.6) and (aux_intensity <= 0.7)):
                intensity_labels.append('0.7')
            elif ((aux_intensity > 0.7) and (aux_intensity <= 0.8)):
                intensity_labels.append('0.8')
            elif ((aux_intensity > 0.8) and (aux_intensity <= 0.9)):
                intensity_labels.append('0.9')
            elif ((aux_intensity > 0.9) and (aux_intensity <= 1)):
                intensity_labels.append('1')

        mapper = umap.UMAP(random_state=args.seed, spread=args.spread, n_neighbors=args.n_neigh, min_dist=args.min_d).fit(np.array(all_spectrograms))

        plt.subplots()
        umap.plot.points(mapper, np.asarray(decoder_name),  color_key_cmap='14classes_garbage', background='black')
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + '_ep_' + str(int(epoch)) + '_black.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + '_ep_' + str(int(epoch)) + '_black.' + 'png')
        plt.close('all)')

        umap.plot.points(mapper, np.asarray(intensity_labels), color_key_cmap='intensity', background='black')
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'intensity_umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + '_ep_' + str(int(epoch)) + '.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'intensity_umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + '_ep_' + str(int(epoch)) + '.' + 'png')
        plt.close('all)')

    print('Done')

def latent_vs_real_UMAP(summary_dataset, generation_summary, classes, args):
    # Color map
    if args.all_garbage == True:
        classes_colors = ['grey', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen','darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink', 'white']
    else:
        classes_colors = ['grey', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen', 'darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink', 'lightcoral', 'brown', 'red', 'yellow', 'white']

    classes_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("14classes_garbage", classes_colors)
    plt.register_cmap("14classes_garbage", classes_cmap)

    # Real data
    load_data = np.load(summary_dataset[0], allow_pickle=True)
    load_data = load_data.item()
    file_name_real = load_data['File_name']
    real_name_real = load_data['Real_name']
    decoder_name = load_data['Decoder_name']

    if args.n_template >0:
        subset = random.sample(range(0,np.size(file_name_real)), args.n_template)

        labels_real = []
        all_samples_real = []
        all_spectrograms_real = []
        all_spectrograms = []
        all_real = []
        all_decoder = []
        for s in range(0, len(subset)):
            read_aux, sr = librosa.load(file_name_real[subset[s]], 16000)
            all_samples_real.append(read_aux[0:15999])
            X = librosa.stft(all_samples_real[s], n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                             pad_mode='constant', center=True)
            T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
            spect_aux = np.log(1 + 100 * np.abs(X ** 2))
            all_spectrograms_real.append(spect_aux.flatten())
            all_spectrograms.append(spect_aux.flatten())
            labels_real.append('Real')
            if real_name_real[subset[s]] == 'B1':
                real_name_real[subset[s]] = 'B'
            elif real_name_real[subset[s]] == 'B2':
                real_name_real[subset[s]] = 'B'
            elif real_name_real[subset[s]] == 'J1':
                real_name_real[subset[s]] = 'J'
            elif real_name_real[subset[s]] == 'J2':
                real_name_real[subset[s]] = 'J'

            if decoder_name[subset[s]] == 'B1':
                decoder_name[subset[s]] = 'B'
            elif decoder_name[subset[s]] == 'B2':
                decoder_name[subset[s]] = 'B'
            elif decoder_name[subset[s]] == 'J1':
                decoder_name[subset[s]] = 'J'
            elif decoder_name[subset[s]] == 'J2':
                decoder_name[subset[s]] = 'J'

            if args.all_garbage == True:
                if decoder_name[subset[s]] == 'EARLY15':
                    decoder_name[subset[s]] = 'X'
                if decoder_name[subset[s]] == 'EARLY30':
                    decoder_name[subset[s]] = 'X'
                if decoder_name[subset[s]] == 'EARLY45':
                    decoder_name[subset[s]] = 'X'
                if decoder_name[subset[s]] == 'OT':
                    decoder_name[subset[s]] = 'X'
                if decoder_name[subset[s]] == 'WN':
                    decoder_name[subset[s]] = 'X'

            all_real.append(real_name_real[subset[s]])
            all_decoder.append(decoder_name[subset[s]])

    else:
        labels_real = []
        all_samples_real = []
        all_spectrograms_real = []
        all_spectrograms = []
        all_real = []
        all_decoder = []
        for s in range(0, len(file_name_real)):
            read_aux, sr = librosa.load(file_name_real[s], 16000)
            all_samples_real.append(read_aux[0:15999])
            X = librosa.stft(all_samples_real[s], n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                             pad_mode='constant', center=True)
            T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
            spect_aux = np.log(1 + 100 * np.abs(X ** 2))
            all_spectrograms_real.append(spect_aux.flatten())
            all_spectrograms.append(spect_aux.flatten())
            labels_real.append('Real')
            if real_name_real[s] == 'B1':
                real_name_real[s] = 'B'
            elif real_name_real[s] == 'B2':
                real_name_real[s] = 'B'
            elif real_name_real[s] == 'J1':
                real_name_real[s] = 'J'
            elif real_name_real[s] == 'J2':
                real_name_real[s] = 'J'

            if decoder_name[s] == 'B1':
                decoder_name[s] = 'B'
            elif decoder_name[s] == 'B2':
                decoder_name[s] = 'B'
            elif decoder_name[s]== 'J1':
                decoder_name[s] = 'J'
            elif decoder_name[s] == 'J2':
                decoder_name[s] = 'J'

            if args.all_garbage == True:
                if decoder_name[s] == 'EARLY15':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'EARLY30':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'EARLY45':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'OT':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'WN':
                    decoder_name[s] = 'X'

            all_real.append(real_name_real[s])
            all_decoder.append(decoder_name[s])

    if args.balanced > 0:
        labels_all_gen = []
        all_samples_gen = []
        all_spectrograms_gen = []

        for i in range(0, np.size(generation_summary)):
            # Generated data
            load_data = np.load(generation_summary[i], allow_pickle=True)
            load_data = load_data.item()
            file_name_gen = load_data['File_name']
            decoder_name_gen = load_data['Decoder_name']
            epoch = load_data['Epoch']

            subsets = []
            labels_balanced_aux= []
            for c in range(0, np.size(classes)):
                subset_aux = []
                labels_aux_classes = []
                for s in range(0,np.size(decoder_name_gen)):
                    if decoder_name_gen[s] == (classes[c]):
                        subset_aux.append(file_name_gen[s])
                        labels_aux_classes.append(decoder_name_gen[s])

                if len(subset_aux) > 0:
                    subsets.append(subset_aux[0:args.balanced])
                    labels_balanced_aux.append(labels_aux_classes[0:args.balanced])

            all_samples_aux = []
            all_spectrograms_aux = []
            labels_aux = []
            labels_balanced = []
            all_subsets = []
            for l in range(0, len(subsets)):
                for j in range(0, np.size(subsets[l])):
                    all_subsets.append(subsets[l][j])
                    labels_balanced.append(labels_balanced_aux[l][j])

            for l in range(0, len(all_subsets)):
                read_aux, sr = librosa.load(all_subsets[l], 16000)
                all_samples_aux.append(read_aux[0:15999])
                all_samples_gen.append(read_aux[0:15999])

                X = librosa.stft(all_samples_aux[l], n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
                T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
                spect_aux = np.log(1 + 100 * np.abs(X ** 2))

                all_spectrograms_aux.append(spect_aux.flatten())
                all_spectrograms_gen.append(spect_aux.flatten())
                all_spectrograms.append(spect_aux.flatten())
                labels_aux.append('Ep_' + str(int(epoch)))

                if labels_balanced[l] == 'B1':
                    labels_balanced[l] = 'B'
                elif labels_balanced[l] == 'B2':
                    labels_balanced[l] = 'B'
                elif labels_balanced[l] == 'J1':
                    labels_balanced[l]  = 'J'
                elif labels_balanced[l] == 'J2':
                    labels_balanced[l] = 'J'

                if args.all_garbage == True:
                    if labels_balanced[l] == 'EARLY15':
                        labels_balanced[l] = 'X'
                    if labels_balanced[l] == 'EARLY30':
                        labels_balanced[l] = 'X'
                    if labels_balanced[l] == 'EARLY45':
                        labels_balanced[l] = 'X'
                    if labels_balanced[l] == 'OT':
                        labels_balanced[l] = 'X'
                    if labels_balanced[l] == 'WN':
                        labels_balanced[l] = 'X'

            labels = np.append(labels_real, labels_aux)
            syll_labels = np.append(all_real, labels_balanced)
            syll_labels_decoder = np.append(all_decoder, labels_balanced)

            mapper = umap.UMAP(random_state=args.seed, spread=args.spread, n_neighbors=args.n_neigh, min_dist=args.min_d).fit(np.array(np.concatenate((all_spectrograms_real, all_spectrograms_aux), axis=0)))
            plt.subplots()
            umap.plot.points(mapper, np.asarray(labels), color_key_cmap='Paired', background='black')
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + 'ep' + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + 'ep' + str(int(epoch)) + '.' + 'png')

            plt.subplots()
            umap.plot.points(mapper, np.asarray(syll_labels), color_key_cmap='14classes_garbage', background='black')
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_across_time_single_syll_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_across_time_single_syll_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + 'png')
            plt.close('all')

            plt.subplots()
            umap.plot.points(mapper, np.asarray(syll_labels_decoder), color_key_cmap='14classes_garbage', background='black')
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_across_time_single_syll_decoder_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_across_time_single_syll_decoder_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + 'png')
            plt.close('all')

            labels_all_gen.append(labels_aux)

    else:
        labels_all_gen = []
        all_samples_gen = []
        all_spectrograms_gen = []
        for i in range(0, np.size(generation_summary)):
            # Generated data
            load_data = np.load(generation_summary[i], allow_pickle=True)
            load_data = load_data.item()
            file_name_gen = load_data['File_name']
            decoder_name_gen = load_data['Decoder_name']
            epoch = load_data['Epoch']

            all_samples_aux = []
            all_spectrograms_aux = []
            labels_aux = []
            for s in range(0,len(file_name_gen)):
                read_aux, sr = librosa.load(file_name_gen[s], 16000)
                all_samples_aux.append(read_aux[0:15999])
                all_samples_gen.append(read_aux[0:15999])
                X = librosa.stft(all_samples_aux[s], n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                                 pad_mode='constant', center=True)
                T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
                spect_aux = np.log(1 + 100 * np.abs(X ** 2))
                all_spectrograms_aux.append(spect_aux.flatten())
                all_spectrograms_gen.append(spect_aux.flatten())
                all_spectrograms.append(spect_aux.flatten())
                labels_aux.append('Ep_' + str(int(epoch)))
                if decoder_name_gen[s] == 'B1':
                    decoder_name_gen[s] = 'B'
                elif decoder_name_gen[s] == 'B2':
                    decoder_name_gen[s] = 'B'
                elif decoder_name_gen[s] == 'J1':
                    decoder_name_gen[s] = 'J'
                elif decoder_name_gen[s] == 'J2':
                    decoder_name_gen[s] = 'J'

                if args.all_garbage == True:
                    if decoder_name_gen[s] == 'EARLY15':
                        decoder_name_gen[s] = 'X'
                    if decoder_name_gen[s] == 'EARLY30':
                        decoder_name_gen[s] = 'X'
                    if decoder_name_gen[s] == 'EARLY45':
                        decoder_name_gen[s] = 'X'
                    if decoder_name_gen[s] == 'OT':
                        decoder_name_gen[s] = 'X'
                    if decoder_name_gen[s] == 'WN':
                        decoder_name_gen[s] = 'X'

            labels = np.append(labels_real,labels_aux)
            syll_labels = np.append(all_real, decoder_name_gen)
            syll_labels_decoder = np.append(all_decoder, decoder_name_gen)

            mapper = umap.UMAP(random_state=args.seed, spread=args.spread, n_neighbors=args.n_neigh, min_dist=args.min_d).fit(np.array(np.concatenate((all_spectrograms_real, all_spectrograms_aux), axis=0)))

            plt.subplots()
            umap.plot.points(mapper, np.asarray(labels), color_key_cmap='Paired', background='black')
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + 'ep' + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + 'ep' + str(int(epoch)) + '.' + 'png')

            plt.subplots()
            umap.plot.points(mapper, np.asarray(syll_labels), color_key_cmap='14classes_garbage', background='black')
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_across_time_single_syll_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_across_time_single_syll_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + 'png')
            plt.close('all')

            plt.subplots()
            umap.plot.points(mapper, np.asarray(syll_labels_decoder), color_key_cmap='14classes_garbage', background='black')
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_across_time_single_syll_decoder_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_across_time_single_syll_decoder_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + 'png')
            plt.close('all')

            labels_all_gen.append(labels_aux)


    print('Done')
    input()

    all_labels = np.append(labels_real, labels_all_gen)
    mapper = umap.UMAP(random_state=args.seed, spread=args.spread, n_neighbors=args.n_neigh, min_dist=args.min_d).fit(np.array(all_spectrograms))

    plt.subplots()
    umap.plot.points(mapper, np.asarray(all_labels), color_key_cmap='Paired', background='black')
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_across_time' + '.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_across_time' + '.' + 'png')

    plt.close('all')
    print('Done')

def latent_vs_gen_UMAP(summary_dataset, generation_summary, classes, args):
    # Color map
    if args.all_garbage == True:
        classes_colors = ['grey', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen','darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink', 'white']
    else:
        classes_colors = ['grey', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen', 'darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink', 'lightcoral', 'brown', 'red', 'yellow', 'white']

    classes_colors_intensity = ['blue', 'deepskyblue', 'aqua', 'lime', 'greenyellow', 'yellow', 'darkorange',
                                'orangered', 'red', 'maroon']
    classes_cmap_intensity = matplotlib.colors.LinearSegmentedColormap.from_list("intensity", classes_colors_intensity)
    plt.register_cmap("intensity", classes_cmap_intensity)

    classes_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("14classes_garbage", classes_colors)
    plt.register_cmap("14classes_garbage", classes_cmap)

    # Real data
    load_data = np.load(summary_dataset[0], allow_pickle=True)
    load_data = load_data.item()
    file_name_real = load_data['File_name']
    decoder_name = load_data['Decoder_name']
    raw_sum_distr = load_data['Annotations']

    if args.n_template >0:
        subset = random.sample(range(0,np.size(file_name_real)), args.n_template)

        labels_real = []
        all_samples_real = []
        all_spectrograms_real = []
        all_spectrograms = []
        all_decoder = []
        intensity_labels = []
        for s in range(0, len(subset)):
            read_aux, sr = librosa.load(file_name_real[subset[s]], 16000)
            all_samples_real.append(read_aux[0:15999])
            X = librosa.stft(all_samples_real[s], n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                             pad_mode='constant', center=True)
            T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
            spect_aux = np.log(1 + 100 * np.abs(X ** 2))
            all_spectrograms_real.append(spect_aux.flatten())
            all_spectrograms.append(spect_aux.flatten())
            labels_real.append('Gen')

            if decoder_name[subset[s]] == 'B1':
                decoder_name[subset[s]] = 'B'
            elif decoder_name[subset[s]] == 'B2':
                decoder_name[subset[s]] = 'B'
            elif decoder_name[subset[s]] == 'J1':
                decoder_name[subset[s]] = 'J'
            elif decoder_name[subset[s]] == 'J2':
                decoder_name[subset[s]] = 'J'

            if args.all_garbage == True:
                if decoder_name[subset[s]] == 'EARLY15':
                    decoder_name[subset[s]] = 'X'
                if decoder_name[subset[s]] == 'EARLY30':
                    decoder_name[subset[s]] = 'X'
                if decoder_name[subset[s]] == 'EARLY45':
                    decoder_name[subset[s]] = 'X'
                if decoder_name[subset[s]] == 'OT':
                    decoder_name[subset[s]] = 'X'
                if decoder_name[subset[s]] == 'WN':
                    decoder_name[subset[s]] = 'X'

            aux_intensity = np.max(raw_sum_distr[s])
            if ((aux_intensity > 0) and (aux_intensity <= 0.1)):
                intensity_labels.append('0.1')
            elif ((aux_intensity > 0.2) and (aux_intensity <= 0.3)):
                intensity_labels.append('0.2')
            elif ((aux_intensity > 0.2) and (aux_intensity <= 0.3)):
                intensity_labels.append('0.3')
            elif ((aux_intensity > 0.3) and (aux_intensity <= 0.4)):
                intensity_labels.append('0.4')
            elif ((aux_intensity > 0.4) and (aux_intensity <= 0.5)):
                intensity_labels.append('0.5')
            elif ((aux_intensity > 0.5) and (aux_intensity <= 0.6)):
                intensity_labels.append('0.6')
            elif ((aux_intensity > 0.6) and (aux_intensity <= 0.7)):
                intensity_labels.append('0.7')
            elif ((aux_intensity > 0.7) and (aux_intensity <= 0.8)):
                intensity_labels.append('0.8')
            elif ((aux_intensity > 0.8) and (aux_intensity <= 0.9)):
                intensity_labels.append('0.9')
            elif ((aux_intensity > 0.9) and (aux_intensity <= 1)):
                intensity_labels.append('1')

            all_decoder.append(decoder_name[subset[s]])

    else:
        labels_real = []
        all_samples_real = []
        all_spectrograms_real = []
        all_spectrograms = []
        all_decoder = []
        intensity_labels = []
        for s in range(0, len(file_name_real)):
            read_aux, sr = librosa.load(file_name_real[s], 16000)
            all_samples_real.append(read_aux[0:15999])
            X = librosa.stft(all_samples_real[s], n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                             pad_mode='constant', center=True)
            T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
            spect_aux = np.log(1 + 100 * np.abs(X ** 2))
            all_spectrograms_real.append(spect_aux.flatten())
            all_spectrograms.append(spect_aux.flatten())
            labels_real.append('Gen')

            if decoder_name[s] == 'B1':
                decoder_name[s] = 'B'
            elif decoder_name[s] == 'B2':
                decoder_name[s] = 'B'
            elif decoder_name[s]== 'J1':
                decoder_name[s] = 'J'
            elif decoder_name[s] == 'J2':
                decoder_name[s] = 'J'

            if args.all_garbage == True:
                if decoder_name[s] == 'EARLY15':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'EARLY30':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'EARLY45':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'OT':
                    decoder_name[s] = 'X'
                if decoder_name[s] == 'WN':
                    decoder_name[s] = 'X'

            aux_intensity = np.max(raw_sum_distr[s])
            if ((aux_intensity > 0) and (aux_intensity <= 0.1)):
                intensity_labels.append('0.1')
            elif ((aux_intensity > 0.2) and (aux_intensity <= 0.3)):
                intensity_labels.append('0.2')
            elif ((aux_intensity > 0.2) and (aux_intensity <= 0.3)):
                intensity_labels.append('0.3')
            elif ((aux_intensity > 0.3) and (aux_intensity <= 0.4)):
                intensity_labels.append('0.4')
            elif ((aux_intensity > 0.4) and (aux_intensity <= 0.5)):
                intensity_labels.append('0.5')
            elif ((aux_intensity > 0.5) and (aux_intensity <= 0.6)):
                intensity_labels.append('0.6')
            elif ((aux_intensity > 0.6) and (aux_intensity <= 0.7)):
                intensity_labels.append('0.7')
            elif ((aux_intensity > 0.7) and (aux_intensity <= 0.8)):
                intensity_labels.append('0.8')
            elif ((aux_intensity > 0.8) and (aux_intensity <= 0.9)):
                intensity_labels.append('0.9')
            elif ((aux_intensity > 0.9) and (aux_intensity <= 1)):
                intensity_labels.append('1')

            all_decoder.append(decoder_name[s])

    if args.balanced > 0:
        labels_all_gen = []
        all_samples_gen = []
        all_spectrograms_gen = []
        intensity_labels = []
        for i in range(0, np.size(generation_summary)):
            # Generated data
            load_data = np.load(generation_summary[i], allow_pickle=True)
            load_data = load_data.item()
            file_name_gen = load_data['File_name']
            decoder_name_gen = load_data['Decoder_name']
            epoch = load_data['Epoch']
            raw_sum_distr = load_data['Annotations']

            subsets = []
            labels_balanced_aux= []
            for c in range(0, np.size(classes)):
                subset_aux = []
                labels_aux_classes = []
                for s in range(0,np.size(decoder_name_gen)):
                    if decoder_name_gen[s] == (classes[c]):
                        subset_aux.append(file_name_gen[s])
                        labels_aux_classes.append(decoder_name_gen[s])

                if len(subset_aux) > 0:
                    subsets.append(subset_aux[0:args.balanced])
                    labels_balanced_aux.append(labels_aux_classes[0:args.balanced])

            all_samples_aux = []
            all_spectrograms_aux = []
            labels_aux = []
            labels_balanced = []
            all_subsets = []
            intensity_labels = []
            for l in range(0, len(subsets)):
                for j in range(0, np.size(subsets[l])):
                    all_subsets.append(subsets[l][j])
                    labels_balanced.append(labels_balanced_aux[l][j])

            for l in range(0, len(all_subsets)):
                read_aux, sr = librosa.load(all_subsets[l], 16000)
                all_samples_aux.append(read_aux[0:15999])
                all_samples_gen.append(read_aux[0:15999])

                X = librosa.stft(all_samples_aux[l], n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
                T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
                spect_aux = np.log(1 + 100 * np.abs(X ** 2))

                all_spectrograms_aux.append(spect_aux.flatten())
                all_spectrograms_gen.append(spect_aux.flatten())
                all_spectrograms.append(spect_aux.flatten())
                labels_aux.append('Expl')

                if labels_balanced[l] == 'B1':
                    labels_balanced[l] = 'B'
                elif labels_balanced[l] == 'B2':
                    labels_balanced[l] = 'B'
                elif labels_balanced[l] == 'J1':
                    labels_balanced[l]  = 'J'
                elif labels_balanced[l] == 'J2':
                    labels_balanced[l] = 'J'

                if args.all_garbage == True:
                    if labels_balanced[l] == 'EARLY15':
                        labels_balanced[l] = 'X'
                    if labels_balanced[l] == 'EARLY30':
                        labels_balanced[l] = 'X'
                    if labels_balanced[l] == 'EARLY45':
                        labels_balanced[l] = 'X'
                    if labels_balanced[l] == 'OT':
                        labels_balanced[l] = 'X'
                    if labels_balanced[l] == 'WN':
                        labels_balanced[l] = 'X'

                aux_intensity = np.max(raw_sum_distr[s])
                if ((aux_intensity > 0) and (aux_intensity <= 0.1)):
                    intensity_labels.append('0.1')
                elif ((aux_intensity > 0.2) and (aux_intensity <= 0.3)):
                    intensity_labels.append('0.2')
                elif ((aux_intensity > 0.2) and (aux_intensity <= 0.3)):
                    intensity_labels.append('0.3')
                elif ((aux_intensity > 0.3) and (aux_intensity <= 0.4)):
                    intensity_labels.append('0.4')
                elif ((aux_intensity > 0.4) and (aux_intensity <= 0.5)):
                    intensity_labels.append('0.5')
                elif ((aux_intensity > 0.5) and (aux_intensity <= 0.6)):
                    intensity_labels.append('0.6')
                elif ((aux_intensity > 0.6) and (aux_intensity <= 0.7)):
                    intensity_labels.append('0.7')
                elif ((aux_intensity > 0.7) and (aux_intensity <= 0.8)):
                    intensity_labels.append('0.8')
                elif ((aux_intensity > 0.8) and (aux_intensity <= 0.9)):
                    intensity_labels.append('0.9')
                elif ((aux_intensity > 0.9) and (aux_intensity <= 1)):
                    intensity_labels.append('1')

            labels = np.append(labels_real, labels_aux)
            syll_labels_decoder = np.append(all_decoder, labels_balanced)

            mapper = umap.UMAP(random_state=args.seed, spread=args.spread, n_neighbors=args.n_neigh, min_dist=args.min_d).fit(np.array(np.concatenate((all_spectrograms_real, all_spectrograms_aux), axis=0)))
            plt.subplots()
            umap.plot.points(mapper, np.asarray(labels), color_key_cmap='Paired', background='black')
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + 'ep' + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'CFR_umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + 'ep' + str(int(epoch)) + '.' + 'png')

            plt.subplots()
            umap.plot.points(mapper, np.asarray(syll_labels_decoder), color_key_cmap='14classes_garbage', background='black')
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'EXPL_CFR_umap_across_time_single_syll_decoder_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'EXPL_CFR_umap_across_time_single_syll_decoder_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + 'png')
            plt.close('all')

            umap.plot.points(mapper, np.asarray(intensity_labels), color_key_cmap='intensity', background='black')
            if args.format != 'png':
                plt.savefig(
                    args.data_dir + '/' + args.output_dir + '/' + 'intensity_umap_' + str(args.n_neigh) + '_' + str(
                        args.min_d) + '_' + str(args.seed) + '_ep_' + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'intensity_umap_' + str(args.n_neigh) + '_' + str(
                args.min_d) + '_' + str(args.seed) + '_ep_' + str(int(epoch)) + '.' + 'png')

            labels_all_gen.append(labels_aux)

    else:
        labels_all_gen = []
        all_samples_gen = []
        all_spectrograms_gen = []
        for i in range(0, np.size(generation_summary)):
            # Generated data
            load_data = np.load(generation_summary[i], allow_pickle=True)
            load_data = load_data.item()
            file_name_gen = load_data['File_name']
            decoder_name_gen = load_data['Decoder_name']
            epoch = load_data['Epoch']

            all_samples_aux = []
            all_spectrograms_aux = []
            labels_aux = []
            for s in range(0,len(file_name_gen)):
                read_aux, sr = librosa.load(file_name_gen[s], 16000)
                all_samples_aux.append(read_aux[0:15999])
                all_samples_gen.append(read_aux[0:15999])
                X = librosa.stft(all_samples_aux[s], n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                                 pad_mode='constant', center=True)
                T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
                spect_aux = np.log(1 + 100 * np.abs(X ** 2))
                all_spectrograms_aux.append(spect_aux.flatten())
                all_spectrograms_gen.append(spect_aux.flatten())
                all_spectrograms.append(spect_aux.flatten())
                labels_aux.append('Expl')
                if decoder_name_gen[s] == 'B1':
                    decoder_name_gen[s] = 'B'
                elif decoder_name_gen[s] == 'B2':
                    decoder_name_gen[s] = 'B'
                elif decoder_name_gen[s] == 'J1':
                    decoder_name_gen[s] = 'J'
                elif decoder_name_gen[s] == 'J2':
                    decoder_name_gen[s] = 'J'

                if args.all_garbage == True:
                    if decoder_name_gen[s] == 'EARLY15':
                        decoder_name_gen[s] = 'X'
                    if decoder_name_gen[s] == 'EARLY30':
                        decoder_name_gen[s] = 'X'
                    if decoder_name_gen[s] == 'EARLY45':
                        decoder_name_gen[s] = 'X'
                    if decoder_name_gen[s] == 'OT':
                        decoder_name_gen[s] = 'X'
                    if decoder_name_gen[s] == 'WN':
                        decoder_name_gen[s] = 'X'

                aux_intensity = np.max(raw_sum_distr[s])
                if ((aux_intensity > 0) and (aux_intensity <= 0.1)):
                    intensity_labels.append('0.1')
                elif ((aux_intensity > 0.2) and (aux_intensity <= 0.3)):
                    intensity_labels.append('0.2')
                elif ((aux_intensity > 0.2) and (aux_intensity <= 0.3)):
                    intensity_labels.append('0.3')
                elif ((aux_intensity > 0.3) and (aux_intensity <= 0.4)):
                    intensity_labels.append('0.4')
                elif ((aux_intensity > 0.4) and (aux_intensity <= 0.5)):
                    intensity_labels.append('0.5')
                elif ((aux_intensity > 0.5) and (aux_intensity <= 0.6)):
                    intensity_labels.append('0.6')
                elif ((aux_intensity > 0.6) and (aux_intensity <= 0.7)):
                    intensity_labels.append('0.7')
                elif ((aux_intensity > 0.7) and (aux_intensity <= 0.8)):
                    intensity_labels.append('0.8')
                elif ((aux_intensity > 0.8) and (aux_intensity <= 0.9)):
                    intensity_labels.append('0.9')
                elif ((aux_intensity > 0.9) and (aux_intensity <= 1)):
                    intensity_labels.append('1')

            labels = np.append(labels_real,labels_aux)
            syll_labels_decoder = np.append(all_decoder, decoder_name_gen)

            mapper = umap.UMAP(random_state=args.seed, spread=args.spread, n_neighbors=args.n_neigh, min_dist=args.min_d).fit(np.array(np.concatenate((all_spectrograms_real, all_spectrograms_aux), axis=0)))

            plt.subplots()
            umap.plot.points(mapper, np.asarray(labels), color_key_cmap='Paired', background='black')
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'EXPL_CFR_umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + 'ep' + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'EXPL_CFR_umap_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + 'ep' + str(int(epoch)) + '.' + 'png')

            plt.subplots()
            umap.plot.points(mapper, np.asarray(syll_labels_decoder), color_key_cmap='14classes_garbage', background='black')
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'EXPL_CFR_umap_across_time_single_syll_decoder_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'EXPL_CFR_umap_across_time_single_syll_decoder_' + str(args.n_neigh) + '_' + str(args.min_d) + '_' + str(args.seed) + str(int(epoch)) + '.' + 'png')
            plt.close('all')

            umap.plot.points(mapper, np.asarray(intensity_labels), color_key_cmap='intensity', background='black')
            if args.format != 'png':
                plt.savefig(
                    args.data_dir + '/' + args.output_dir + '/' + 'intensity_umap_' + str(args.n_neigh) + '_' + str(
                        args.min_d) + '_' + str(args.seed) + '_ep_' + str(int(epoch)) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'intensity_umap_' + str(args.n_neigh) + '_' + str(
                args.min_d) + '_' + str(args.seed) + '_ep_' + str(int(epoch)) + '.' + 'png')

            labels_all_gen.append(labels_aux)

    print('Done')

def phrase(generation_summary_list, classes, template_list, args):
    """
    :param generation_summary_list: list of generation per epoch
    :param classes: list of classes in the training dataset

    :return: spectrogram phrase generated by the generator at a given epoch
             TODO: add gap estimation to make it even more similar to the original ones
    """
    # Silence
    silence = np.zeros((300,))

    # Recognition of which syllables is each recording
    if np.size(generation_summary_list)>0:
        gen_syll_list_ep = []
        all_ep = []
        for sl in range(0, np.size(generation_summary_list)):
            aux = np.load(generation_summary_list[sl], allow_pickle = True)
            aux = aux.item()

            aux_names = aux['File_name'] #was real name in case I had some old file
            aux_decoder_names = aux['Decoder_name']
            aux_epoch = aux['Epoch']

            gen_syll_list = []
            for c in range(0,np.size(classes)):
                syll_list_aux = []
                for i in range(0,np.size(aux_names)):
                    if (aux_decoder_names[i].find(classes[c]) != -1):
                        syll_list_aux.append(aux_names[i])
                gen_syll_list.append(syll_list_aux)

            gen_syll_list_ep.append(gen_syll_list)
            all_ep.append(aux_epoch)

        # Sequence generation
        for sl in range(0, np.size(generation_summary_list)):
            for c in range(0,np.size(classes)):
                if np.size(gen_syll_list_ep[sl][c]) >= args.n_template:
                    syll_ex = random.sample(gen_syll_list_ep[sl][c], args.n_template)
                else:
                    syll_ex = random.sample(gen_syll_list_ep[sl][c], np.size(gen_syll_list_ep[sl][c]))

                # build a sequence
                sequence = [0]
                for s in range(0, int(np.size(syll_ex) / 2)):
                    y, sr = librosa.load(syll_ex[s], sr=16000)

                    # cut the silence
                    trim = librosa.effects.trim(y, top_db=20)
                    y = trim[0]
                    sequence = np.append(sequence, y)
                    sequence = np.append(sequence, silence)

                librosa.output.write_wav(args.data_dir + '/' + args.output_dir + '/' + 'Sequence1_' + str(int(all_ep[sl])) + '_syll_' + str(classes[c]) + '.wav', sequence, sr)

                sequence = [0]
                for s in range(int(np.size(syll_ex) / 2), int(np.size(syll_ex))):
                    y, sr = librosa.load(syll_ex[s], sr=16000)

                    # cut the silence
                    trim = librosa.effects.trim(y, top_db=20)
                    y = trim[0]
                    sequence = np.append(sequence, y)
                    sequence = np.append(sequence, silence)

                librosa.output.write_wav(args.data_dir + '/' + args.output_dir + '/' + 'Sequence2_' + str(int(all_ep[sl])) + '_syll_' + str(classes[c]) + '.wav', sequence, sr)

            # PLOT using librosa
            for c in range(0,np.size(classes)):
                # template
                y_template, sr_template = librosa.load(template_list[c], sr=16000)

                X_template = librosa.stft(y_template, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                                          pad_mode='constant',
                                          center=True)
                Y_template = np.log(1 + 100 * np.abs(X_template) ** 2)
                T_coef_template = np.arange(X_template.shape[1]) * args.H / sr_template
                K_template = args.N // 2
                F_coef_template = np.arange(K_template + 1) * sr_template / args.N

                # sequence
                y, sr = librosa.load(args.data_dir + '/' + args.output_dir + '/' + 'Sequence1_' + str(int(all_ep[sl])) + '_syll_' + str(classes[c]) + '.wav', sr=16000)

                X = librosa.stft(y, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
                Y_1 = np.log(1 + 100 * np.abs(X) ** 2)
                T_coef = np.arange(X.shape[1]) * args.H / sr
                K = args.N // 2
                F_coef = np.arange(K + 1) * sr / args.N

                y, sr = librosa.load(args.data_dir + '/' + args.output_dir + '/' + 'Sequence2_' + str(int(all_ep[sl])) + '_syll_' + str(classes[c]) + '.wav', sr=16000)

                X = librosa.stft(y, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
                Y_2 = np.log(1 + 100 * np.abs(X) ** 2)
                T_coef = np.arange(X.shape[1]) * args.H / sr
                K = args.N // 2
                F_coef = np.arange(K + 1) * sr / args.N

                fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(22, 5))
                extent_template = [T_coef_template[0], T_coef_template[-1], F_coef_template[0], 8000]
                ax1.imshow(Y_template, cmap=args.color, aspect='auto', origin='lower', extent=extent_template, norm=colors.PowerNorm(gamma=args.gamma))
                ax1.set_ylabel('Frequency (Hz)')
                ax1.title.set_text('Template ' + classes[c])

                extent = [T_coef[0], T_coef[-1], F_coef[0], 8000]
                ax2.imshow(Y_1, cmap=args.color, aspect='auto', origin='lower', extent=extent,
                           norm=colors.PowerNorm(gamma=0.5))
                ax2.set_ylabel('Frequency (Hz)')
                ax2.title.set_text('Selected ' + classes[c])

                ax3.imshow(Y_2, cmap=args.color, aspect='auto', origin='lower', extent=extent,
                           norm=colors.PowerNorm(gamma=0.5))
                ax3.set_ylabel('Frequency (Hz)')

                plt.xlabel('Time (seconds)')
                plt.tight_layout()
                if args.format != 'png':
                    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'spectogram_' + str(int(all_ep[sl])) + '_Syll_' + str(classes[c]) + '.' + args.format)
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'spectogram_' + str(int(all_ep[sl])) + '_Syll_' + str(classes[c]) + '.' + 'png')

    print('Done')

def latent_space_xyz(generation_summary, classes, args):
    """
    3D representation of the latent space
    """
    # Color map
    classes_colors = ['grey', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen', 'darkgreen', 'turquoise',
                      'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink', 'white']
    classes_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("14classes_garbage", classes_colors)
    plt.register_cmap("14classes_garbage", classes_cmap)

    # Load data
    for i in range(0, len(generation_summary)):
        load_data = np.load(generation_summary[i], allow_pickle=True)
        load_data = load_data.item()
        file_name_gen = load_data['File_name']
        decoder_name_gen = load_data['Decoder_name']
        epoch = load_data['Epoch']

        xy = np.zeros((len(file_name_gen), 2))
        xz = np.zeros((len(file_name_gen), 2))
        yz = np.zeros((len(file_name_gen), 2))
        z = np.zeros((len(file_name_gen), 3))
        c = []
        for j in range(0, len(file_name_gen)):
            latent_aux_path = args.data_dir + '/' + 'generation_49202' + '/' +'z' + os.path.basename(file_name_gen[j])[10:-4] + '.pkl'
            z_aux = open_pkl(latent_aux_path)[0]
            z[j,:] =z_aux
            xy[j,:] = [z_aux[0], z_aux[1]]
            xz[j,:] = [z_aux[0], z_aux[2]]
            yz[j,:] = [z_aux[1], z_aux[2]]

            # Decoder name additional X class and grouping J and B
            if decoder_name_gen[j] == 'B1':
                decoder_name_gen[j] = 'B'
            elif decoder_name_gen[j] == 'B2':
                decoder_name_gen[j] = 'B'
            elif decoder_name_gen[j] == 'J1':
                decoder_name_gen[j] = 'J'
            elif decoder_name_gen[j] == 'J2':
                decoder_name_gen[j] = 'J'

            if args.all_garbage == True:
                if decoder_name_gen[j] == 'EARLY15':
                    decoder_name_gen[j] = 'X'
                if decoder_name_gen[j] == 'EARLY30':
                    decoder_name_gen[j] = 'X'
                if decoder_name_gen[j] == 'EARLY45':
                    decoder_name_gen[j] = 'X'
                if decoder_name_gen[j] == 'OT':
                    decoder_name_gen[j] = 'X'
                if decoder_name_gen[j] == 'WN':
                    decoder_name_gen[j] = 'X'

            c.append(classes_colors[np.where(np.asarray(classes)==decoder_name_gen[j])[0][0]])

    # Creation of the legend
    legend_elements = []
    for i in range(0, len(classes)):
        legend_elements.append(Line2D([0], [0], marker='o', color=classes_colors[i], label=classes[i]))

    plt.style.use('dark_background')
    
    # Plot
    fig, ax = plt.subplots()
    plt.scatter(xy[:, 0], xy[:, 1], c=c, s=(10,), marker='.', alpha = 0.6, label=np.unique(np.asarray(decoder_name_gen)))
    plt.xlabel('z1')
    plt.ylabel('z2')
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    if args.format != 'png':
        plt.savefig(args.data_dir + '/canary16/' + args.output_dir + '/' + 'latent_xy.' + args.format)
    #plt.savefig(args.data_dir + '/canary16/' + args.output_dir + '/' + 'latent_xy.' + 'png')
    plt.savefig(args.data_dir + '/canary3last/' + args.output_dir + '/' + 'latent_xy.' + 'png')

    fig, ax = plt.subplots()
    plt.scatter(xz[:, 0], xz[:, 1], c=c, s=(10,), marker='.', alpha = 0.6, label=np.unique(np.asarray(decoder_name_gen)))
    plt.xlabel('z1')
    plt.ylabel('z3')
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    if args.format != 'png':
        plt.savefig(args.data_dir + '/canary16/' + args.output_dir + '/' + 'latent_xz.' + args.format)
    #plt.savefig(args.data_dir + '/canary16/' + args.output_dir + '/' + 'latent_xz.' + 'png')
    plt.savefig(args.data_dir + '/canary3last/' + args.output_dir + '/' + 'latent_xz.' + 'png')

    fig, ax = plt.subplots()
    plt.scatter(yz[:, 0], yz[:, 1], c=c, s=(10,), marker='.', alpha = 0.6, label=np.unique(np.asarray(decoder_name_gen)))
    plt.xlabel('z2')
    plt.ylabel('z3')
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    if args.format != 'png':
        plt.savefig(args.data_dir + '/canary16/' + args.output_dir + '/' + 'latent_yz.' + args.format)
    #plt.savefig(args.data_dir + '/canary16/' + args.output_dir + '/' + 'latent_yz.' + 'png')
    plt.savefig(args.data_dir + '/canary3last/' + args.output_dir + '/' + 'latent_yz.' + 'png')

    # 3D plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(z[:, 0], z[:, 1], z[:,2], c=c, marker='.')
    ax.set_xlabel('z0')
    ax.set_ylabel('z1')
    ax.set_zlabel('z2')
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    if args.format != 'png':
        plt.savefig(args.data_dir + '/canary16/' + args.output_dir + '/' + 'latent_z.' + args.format)
    #plt.savefig(args.data_dir + '/canary16/' + args.output_dir + '/' + 'latent_z.' + 'png')
    plt.savefig(args.data_dir + '/canary3last/' + args.output_dir + '/' + 'latent_z.' + 'png')

    plt.close('all)')

    # 3D slices
    bound = -1
    step = 0.25
    while bound < 0.85:
        aux = np.where(np.logical_and(z[:, 1] >= bound, z[:, 1] <= bound + step))[0]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(z[aux, 0], z[aux, 1], z[aux, 2], c=np.asarray(c)[aux], marker='.')
        ax.set_xlabel('z0')
        ax.set_ylabel('z1')
        ax.set_zlabel('z2')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        if args.format != 'png':
            plt.savefig(args.data_dir + '/canary16/' + args.output_dir + '/' + 'slice_' + str(
                bound) + '_latent_z.' + args.format)
        # plt.savefig(args.data_dir + '/canary16/' + args.output_dir + '/' + 'latent_z.' + 'png')
        plt.savefig(
            args.data_dir + '/canary3last/' + args.output_dir + '/' + 'slice_' + str(bound) + '_latent_z.' + 'png')

        bound = bound + step

    plt.close('all)')

    print('Done')

def single_spectro_plot(songs, args):
    """
    :param songs : list of songs to plot

    :return : Spectrogram plot of the exploration of the latent space.
    """
    for i in range(0, np.size(songs)):
        fig = plots.plot_spectro_librosa(songs[i], args.N, args.H, args.color, args.gamma, 'Exploration_' + str(i))
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + 'Exploration_' + str(i) + '.' + args.format)
        plt.savefig(args.data_dir + '/' + 'Exploration_' + str(i) + '.' + 'png')
    print('Done')

def preview_plot(songs,args):
    """
    :param songs : list of songs to plot

    :return : Spectrogram plot of all the songs.
    """
    fig = plots.multi_spectro(songs, args.N, args.H, args.color, args.gamma)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + 'Preview_time.' + args.format)
    plt.savefig(args.data_dir + '/' + 'Preview_time.' + 'png')

    print('Done')

def convex_set(generation_summary, classes, args):
    # Load data
    for i in range(0, len(generation_summary)):
        load_data = np.load(generation_summary[i], allow_pickle=True)
        load_data = load_data.item()
        file_name_gen = load_data['File_name']
        decoder_name_gen = load_data['Decoder_name'][0:5000]

        for c in range(0, len(classes)):
            aux = np.where(np.asarray(decoder_name_gen) == classes[c])[0]

            all_pairs = statistics.pairs(range(0, np.size(aux)), range(0, np.size(aux)))

            if len(all_pairs) < 100:
                pairs = all_pairs
            else:
                pairs = random.sample(all_pairs, 100)

            for p in range(0, len(pairs)):
                latent_aux_path_1 = args.data_dir + '/' + 'generation_49202' + '/' + 'z' + os.path.basename(file_name_gen[aux[pairs[p][0]]])[10:-4] + '.pkl'
                latent_aux_path_2 = args.data_dir + '/' + 'generation_49202' + '/' + 'z' + os.path.basename(file_name_gen[aux[pairs[p][1]]])[10:-4] + '.pkl'
                z_aux_1 = open_pkl(latent_aux_path_1)[0]
                z_aux_2 = open_pkl(latent_aux_path_2)[0]
                mean_z = z_aux_1 + z_aux_2

                if not os.path.isdir(args.data_dir + '/' + args.output_dir + '/' + classes[c]):
                    os.makedirs(args.data_dir + '/' + args.output_dir + '/' + classes[c])

                with open(args.data_dir + '/' + args.output_dir + '/' + classes[c] + '/pair_' + str(p), 'wb') as f:
                    pickle.dump(mean_z, f)

    print('Done')

if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()
  parser.add_argument('--option', type=str, choices=['syll_UMAP', 'latent_UMAP', 'cfr_UMAP', 'gen_expl', 'phrase', 'xyz', 'convex', 'spect_plot', 'preview_spectro'])
  parser.add_argument('--data_dir', type=str, help='Directory containing the data',
                      default=None)
  parser.add_argument('--output_dir', type=str, help='Directory where to save the output',
                      default=None)

  analysis_args = parser.add_argument_group('features_analysis')
  analysis_args.add_argument('--window', type=str, help='Type of window for the visualization of the spectrogram',
                             default='hanning')
  analysis_args.add_argument('--overlap', type=int, help='Overlap for the visualization of the spectrogram', default=64)
  analysis_args.add_argument('--nperseg', type=int, help='Nperseg for the visualization of the spectrogram', default=512)
  analysis_args.add_argument('--seed', type=int, help='Seed for UMAP', default=42)
  analysis_args.add_argument('--n_neigh', type=int, help='How much local is the topology. Smaller means more local', default=100)
  analysis_args.add_argument('--min_d', type=float, help='how tightly UMAP is allowed to pack points together. Higher values provide more details. Range 0.0 to 0.99', default=0.9)
  analysis_args.add_argument('--spread', type=float, help='Additional parameter to change when min_d is changed, it has to be >= min_d', default=0.9)
  analysis_args.add_argument('--balanced', type=int, help='Do we want to consider a balanced genetated set? How many per class?', default=0)

  plot_args = parser.add_argument_group('Plot')
  plot_args.add_argument('--format', type=str, help='Saving format', default='png')
  plot_args.add_argument('--n_template', type=int, help='How many syllable to consider', default=0)
  plot_args.add_argument('--n_seq', type=int, help='How many sequences of syllables', default=4)
  plot_args.add_argument('--color', type=str, help='Colormap', default='inferno')
  plot_args.add_argument('--N', type=int, help='Nftt spectrogram librosa', default=256)
  plot_args.add_argument('--H', type=int, help='Hop length spectrogram librosa', default=64)
  plot_args.add_argument('--gamma', type=float, help='powernorm color, 0.2 training, 0.5 generated', default=0.5)
  plot_args.add_argument('--all_garbage', type=str, help='Do we want to group all the garbage classes together or not', default=True)

  args = parser.parse_args()

  # Output direction creation
  if args.output_dir != None:
      if not os.path.isdir(args.data_dir + '/' + args.output_dir):
        os.makedirs(args.data_dir + '/' + args.output_dir)

  # Save args
  if args.output_dir != None:
    with open(os.path.join(args.data_dir + '/' + args.output_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

  if args.option == 'syll_UMAP':
      # Load data
      summary_dataset = sorted(glob.glob(args.data_dir + '/' + 'Dataset_summary*.npy'))
      syllable_space_UMAP(summary_dataset, args)

  if args.option == 'latent_UMAP':
      # Load data
      gen_summary = sorted(glob.glob(args.data_dir + '/' + 'Generation_summary_ep0984*.npy'))

      latent_space_UMAP(gen_summary, args)

  if args.option == 'cfr_UMAP':
      # Classes
      classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V', 'EARLY15', 'EARLY30', 'EARLY45',  'OT','WN']

      # Load data
      summary_dataset = sorted(glob.glob(args.data_dir + '/' + 'Dataset_summary*.npy'))
      generation_summary = sorted(glob.glob(args.data_dir + '/' + 'Generation_summary_ep*' + '.npy'))
      # generation_summary = sorted(glob.glob(args.data_dir + '/' + 'Generation_summary_ep0984*' + '.npy'))

      latent_vs_real_UMAP(summary_dataset, generation_summary, classes, args)

  if args.option == 'gen_expl':
      # Classes
      classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V', 'EARLY15', 'EARLY30', 'EARLY45',  'OT','WN']

      # Load data
      summary_dataset = sorted(glob.glob(args.data_dir + '/' + 'ALL*.npy'))
      generation_summary = sorted(glob.glob(args.data_dir + '/' + 'Generation_summary_ep*' + '.npy'))

      latent_vs_gen_UMAP(summary_dataset, generation_summary, classes, args)

  if args.option == 'phrase':
      # Classes
      classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']
      # Load data
      generation_summary_list = sorted(glob.glob(args.data_dir + '/' + 'Generation_summary_ep0984*' + '.npy'))
      template_list = sorted(glob.glob(args.data_dir + '/' + 'Template_phrases' + '/' + 'Template_*' + '.wav'))

      phrase(generation_summary_list, classes, template_list, args)

  if args.option == 'xyz':
      # Classes
      classes = ['A', 'B', 'C', 'D', 'E', 'H', 'J', 'L', 'M', 'N', 'O', 'Q', 'R', 'V', 'X']

      # Load data
      #generation_summary = sorted(glob.glob(args.data_dir + '/canary16/' + 'Generation_summary_ep0984*' + '.npy'))
      generation_summary = sorted(glob.glob(args.data_dir + '/canary3last/' + 'Generation_summary_ep0984*' + '.npy'))

      latent_space_xyz(generation_summary, classes, args)

  if args.option == 'spect_plot':
      # Load data
      list = sorted(glob.glob(args.data_dir + '/' + 'Exploration*.wav'))

      single_spectro_plot(list, args)

  if args.option == 'preview_spectro':
      # Load data
      songs = glob.glob(args.data_dir + '/' + '*.wav')
      preview_plot(songs, args)

  if args.option == 'convex':
      # Classes
      classes = ['A', 'B', 'C', 'D', 'E', 'H', 'J', 'L', 'M', 'N', 'O', 'Q', 'R', 'V', 'X']

      # Load data
      # generation_summary = sorted(glob.glob(args.data_dir + '/canary16/' + 'Generation_summary_ep0984*' + '.npy'))
      generation_summary = sorted(glob.glob(args.data_dir + '/canary3last/' + 'Generation_summary_ep0984*' + '.npy'))

      convex_set(generation_summary, classes, args)
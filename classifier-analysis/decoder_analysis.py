import glob
import os
import pickle
import scipy as sp
from scipy import special
import scipy.io.wavfile as wav
import numpy as np
import librosa
import shutil
from math import log2
import sklearn.metrics
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plots
import statistics as stat

def open_pkl(name):
    data = open(name, 'rb')
    z = pickle.load(data)
    return z

def create_annotations(args):
    """
    The input directory contains one or more audio files (.wav) of duration 1s.

    This function creates a dictionary containing for each .wav file the following elements:
    - name (path of the recording)
    - vocab: the list of the whole vocabulary
    - raw: this entry stores the raw outputs produced by the ESN. The ESN produce one annotation vector per timestep;
           the raw output does the same thing but for each timestep of input audio

    The output is saved in the same directory of the data (outside to be able to use the function somewhere else).
    """

    from canarydecoder import load

    # Load the model
    # decoder = load('canary16-deltas')
    # decoder = load('canarygan-3-d')
    # decoder = load('canary16-clean-d')
    # decoder = load('canary16-clean-d-notrim')
    # decoder = load('canary16-filtered')
    # decoder = load('canary16-filtered-notrim') #THIS ONE TO COMPUTE IS CLASSIFIER - REAL
    # decoder = load('canarygan-e-ot-noise')
    # decoder = load('canarygan-clean-e-ot-noise-notrim')
    # decoder = load('canarygan-clean-e-ot-noise')
    # decoder = load('canarygan-f-3e-ot-noise')
    decoder = load('canarygan-f-3e-ot-noise-notrim') #CLASSIFIER - EXT
    # decoder = load('canarygan-2e-ot-noise')
    # decoder = load('canarygan-8e-ot-noise')
    # decoder = load('canarygan-8e-ot-noise-v2')

    # decoder = load('021220-1e')
    # decoder = load('021220-1e-balanced')
    # decoder = load('021220-8e')
    # decoder = load('021220-8e-balanced')

    # Create dictionary
    #annotations = decoder(args.data_dir + '/' + 'generation_' + str(args.ckpt_n)) # if loop on ckpt_n
    annotations = decoder(args.data_dir) # if only one dir as input dir

    return annotations

def analysis_dataset(annotations_dataset, legend_list, args):
    """
    The input directory contains one or more annotations.

    This function is meant to analyse the datasets used to train the GAN and see how the decoder works on a known set of data.
    """

    # Classes
    annotations_aux = open_pkl(annotations_dataset[0])
    classes = annotations_aux[0].vocab
    print(classes)

    GAN_class = np.zeros((np.size(args.GAN_classes),))
    classes_noGAN = classes
    if np.size(args.GAN_classes)>0:
        for gc in range(0,np.size(args.GAN_classes)):
            GAN_class[gc] = np.where(classes == args.GAN_classes[gc])[0][0]
            classes_noGAN = np.delete(classes_noGAN, np.argwhere(classes_noGAN == args.GAN_classes[gc]))

    classes_noGAN_index = np.zeros((np.size(classes) - np.size(args.GAN_classes),))
    for c in range(0,np.size(classes) - np.size(args.GAN_classes)):
        classes_noGAN_index[c] = np.where(classes == classes_noGAN[c])[0][0]
    n_classes = len(annotations_aux[0].vocab)

    # Real distributions
    data_real = []

    # Whole dataset here because the number of samples per class is not exactly a multiple of the number of classes
    data_real_distr = np.ones((n_classes, )) * 1000
    if np.size(args.GAN_classes)>0:
        for gc in range(0,np.size(args.GAN_classes)):
            data_real_distr[np.where(classes == args.GAN_classes[gc])[0][0]] = 0

    # data_real_distr[4] = 1175
    # data_real_distr[16] = 1281
    # data_real_distr = [1488, 1488, 1497, 1487, 1034, 1478, 0, 0, 0, 1465, 1430, 1497, 1496, 1301, 1127, 1462, 1232, 1466, 1432]
    # data_real_distr = [1488, 931, 1497, 1487, 1034, 1478, 0, 0, 0, 1465, 1430, 1497, 1496, 1301, 1127, 1462, 1232, 1466, 1432]
    # data_real_distr = [1488, 931, 1497, 1500, 1034, 1478, 0, 0, 0, 1465, 1430, 1497, 1496, 1301, 1127, 1462, 1232, 1466, 1432]

    fig, ax = plt.subplots()
    ax.bar(classes, data_real_distr)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(rotation='45')
    plt.ylabel('Number of samples')
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'How_many_syllables_1.' + args.format)

    data_real.append(data_real_distr/np.sum(data_real_distr))

    dataset_dim = np.zeros((np.size(legend_list)))
    whole_list_of_recordings = []
    whole_decoder_list_of_recordings = []
    IS = np.zeros((np.size(legend_list),))
    for ds in range(0, np.size(legend_list)):
        annotations_aux = open_pkl(annotations_dataset[ds])
        dataset_dim[ds] = len(annotations_aux)

        if ds > 0:
            data_real_distr = np.ones((n_classes,)) * ((len(annotations_aux))/np.size(classes_noGAN))
            data_real_distr[6:9] = 0

            fig, ax = plt.subplots()
            ax.bar(classes, data_real_distr)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.ylabel('Number of samples')
            plt.tight_layout()
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'How_many_syllables_' + str(legend_list[ds]) + '.' + args.format)

            data_real.append(data_real_distr / np.sum(data_real_distr))

        list_of_recordings = []
        annotations_raw = []
        decoder_list_of_recordings = []
        decoder_list_of_recordings_index= []
        raw_sum_dataset= np.zeros((int(dataset_dim[ds]), n_classes))
        raw_sum_distr= np.zeros((int(dataset_dim[ds]), n_classes))
        raw_max_dataset = np.zeros((int(dataset_dim[ds]),))
        raw_max_indices_dataset = np.zeros((int(dataset_dim[ds]),))
        raw_max_dataset_noGAN = np.zeros((int(dataset_dim[ds]),))
        raw_max_dataset_onlyGAN = np.zeros((int(dataset_dim[ds]),))
        raw_max_indices_dataset_noGAN = np.zeros((int(dataset_dim[ds]),))
        raw_max_indices_dataset_onlyGAN = np.zeros((int(dataset_dim[ds]),))

        for i in range(0, int(dataset_dim[ds])):
            list_of_recordings.append(annotations_aux[i].id)
            annotations_raw.append(annotations_aux[i].vect)

            # This operation should give me an idea of which class is represented the most in my generations
            raw_sum_dataset[i, :] = np.sum(annotations_raw[i], axis=0)
            raw_sum_distr[i,]= sp.special.softmax(raw_sum_dataset[i, :])
            raw_max_dataset[i] = np.max(raw_sum_dataset[i, :])
            raw_max_indices_dataset[i] = np.where(raw_sum_dataset[i, :] == np.max(raw_sum_dataset[i, :]))[0][0]

            # without the GAN classes
            if np.size(args.GAN_classes)>0:
                raw_max_dataset_noGAN[i] = np.max(np.delete(raw_sum_dataset[i, :], GAN_class))
                raw_max_indices_dataset_noGAN[i] = np.where(np.delete(raw_sum_dataset[i,:], GAN_class) == np.max(np.delete(raw_sum_dataset[i,:], GAN_class)))[0][0]

                # only GAN
                raw_max_dataset_onlyGAN[i] = np.max(np.delete(raw_sum_dataset[i, :], classes_noGAN_index))
                raw_max_indices_dataset_onlyGAN[i] = np.where(np.delete(raw_sum_dataset[i, :], classes_noGAN_index) == np.max(np.delete(raw_sum_dataset[i, :], classes_noGAN_index)))[0][0]

            # Assigned class to each recording
            decoder_list_of_recordings.append(classes[int(raw_max_indices_dataset[i])])
            decoder_list_of_recordings_index.append(int(raw_max_indices_dataset[i]))

        original_list_of_recordings = []
        original_list_of_recordings_index = []
        for i in range(0, int(dataset_dim[ds])):
            for c in range(0, n_classes - np.size(args.GAN_classes)):
                if (list_of_recordings[i].find('NEW_' + classes_noGAN[c]) != -1):
                    original_list_of_recordings.append(classes_noGAN[c])
                    original_list_of_recordings_index.append(c)
                    pass

        # Inception score
        IS[ds] = stat.inception_score(raw_sum_distr[:, :])

        # PLOT
        # Plot the distribution of syllables in the dataset, with and without the garbage classes
        title = 'Dataset'
        fig, dataset_h, dataset_mean, dataset_std, dataset_var = plots.plot_distribution_classes(raw_max_indices_dataset, classes, title, n_classes)
        fig.savefig(
            args.data_dir + '/' + args.output_dir + '/' + 'Classes_syllables_dataset_' + legend_list[ds] + '.' + args.format)

        # The next plot does not really work (understand WHY)
        classes_order = np.append(classes_noGAN, args.GAN_classes)
        title = 'Dataset'
        fig, dataset_h, dataset_mean, dataset_std, dataset_var = plots.plot_distribution_classes(
            np.append(raw_max_indices_dataset_noGAN, raw_max_indices_dataset_onlyGAN), classes_order, title, n_classes)
        fig.savefig(
            args.data_dir + '/' + args.output_dir + '/' + 'Classes_syllables_dataset_order_' + legend_list[
                ds] + '.' + args.format)

        # Manual operation to get the plot
        # h, bins = np.histogram(raw_max_indices_dataset, bins=range(n_classes + 1))
        # h = np.append(h[0:6], np.append(h[9:16], np.append(h[17:20], np.append(h[6:9], np.append(h[16], h[20])))))
        # fig, ax = plt.subplots()
        # ax.bar(bins[:-1], h, width=0.8, color='b', alpha=0.6, label='Syllable classes distribution', align='center')
        # ax.spines['top'].set_color('none')
        # ax.spines['right'].set_color('none')
        # # plt.legend(loc='upper right', fontsize=8, ncol=1, shadow=True, fancybox=True)
        # plt.xticks(bins[:-1], np.append(classes_noGAN, args.GAN_classes), rotation='vertical', fontsize=6)
        # plt.xlabel('Classes of syllables', fontsize=15)
        # plt.ylabel('Number of occurences', fontsize=15)
        # plt.title(title, fontsize=15)
        # plt.tight_layout()  # to avoid the cut of labels
        # plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Classes_syllables_dataset_order_' + legend_list[ds] + '.' + args.format)

        if np.size(args.GAN_classes) > 0:
            title = 'Dataset without GAN class'
            fig, dataset_h_noGAN, dataset_mean, dataset_std, dataset_var = plots.plot_distribution_classes(
                raw_max_indices_dataset_noGAN, np.delete(classes, GAN_class), title, n_classes - np.size(args.GAN_classes))
            fig.savefig(
                args.data_dir + '/' + args.output_dir + '/' + 'Classes_syllables_dataset_noGAN' + legend_list[ds] + '.' + args.format)

        # In case we need to save all the recordings
        whole_list_of_recordings.append(list_of_recordings)
        whole_decoder_list_of_recordings.append(decoder_list_of_recordings)

        # Dataset distribution
        dataset_distr = dataset_h / np.sum(dataset_h)

        dataset_summary = {'File_name': list_of_recordings, 'Real_name': original_list_of_recordings, 'Decoder_name': decoder_list_of_recordings, 'Annotations':raw_sum_distr,
                           'Dataset_distr': dataset_distr, 'Dataset_real_distr': data_real, 'IS': IS[ds]}
        print(legend_list[ds])
        np.save(args.data_dir + '/' + 'Dataset_summaryNEW' + legend_list[ds] + '.npy', dataset_summary)

    print('Done')


def analysis_error(annotations_dataset, legend_list, args):
    """
    The input directory contains one or more annotations.

    This function is meant to analyse the errors of the dataset used to train the GAN and see how the decoder works on this set of data.
    """

    # Classes
    annotations_aux = open_pkl(annotations_dataset[0][0])
    classes = annotations_aux[0].vocab

    GAN_class = np.zeros((np.size(args.GAN_classes),))
    classes_noGAN = classes
    if np.size(args.GAN_classes)>0:
        for gc in range(0,np.size(args.GAN_classes)):
            GAN_class[gc] = np.where(classes == args.GAN_classes[gc])[0][0]
            classes_noGAN = np.delete(classes_noGAN, np.argwhere(classes_noGAN == args.GAN_classes[gc]))
    n_classes = len(annotations_aux[0].vocab)

    # Distribution
    dataset_dim = np.zeros((np.size(legend_list)))
    whole_list_of_recordings = []
    whole_decoder_list_of_recordings = []
    IS = np.zeros((np.size(legend_list),))
    for ds in range(0, np.size(legend_list)):
        annotations_aux = open_pkl(annotations_dataset[ds][0])
        dataset_dim[ds] = len(annotations_aux)

        list_of_recordings = []
        annotations_raw = []
        decoder_list_of_recordings = []
        decoder_list_of_recordings_index= []
        raw_sum_dataset= np.zeros((int(dataset_dim[ds]), n_classes))
        raw_sum_distr= np.zeros((int(dataset_dim[ds]), n_classes))
        raw_max_dataset = np.zeros((int(dataset_dim[ds]),))
        raw_max_indices_dataset = np.zeros((int(dataset_dim[ds]),))
        raw_max_dataset_noGAN = np.zeros((int(dataset_dim[ds]),))
        raw_max_indices_dataset_noGAN = np.zeros((int(dataset_dim[ds]),))

        for i in range(0, int(dataset_dim[ds])):
            list_of_recordings.append(annotations_aux[i].id)
            annotations_raw.append(annotations_aux[i].vect)

            # This operation should give me an idea of which class is represented the most in my generations
            raw_sum_dataset[i, :] = np.sum(annotations_raw[i], axis=0)
            raw_sum_distr[i,]= sp.special.softmax(raw_sum_dataset[i, :])
            raw_max_dataset[i] = np.max(raw_sum_dataset[i, :])
            raw_max_indices_dataset[i] = np.where(raw_sum_dataset[i, :] == np.max(raw_sum_dataset[i, :]))[0][0]

            # without the GAN classes
            raw_max_dataset_noGAN[i] = np.max(np.delete(raw_sum_dataset[i, :], GAN_class))
            raw_max_indices_dataset_noGAN[i] = np.where(np.delete(raw_sum_dataset[i,:], GAN_class) == np.max(np.delete(raw_sum_dataset[i,:], GAN_class)))[0][0]

            # Assigned class to each recording
            decoder_list_of_recordings.append(classes[int(raw_max_indices_dataset[i])])
            decoder_list_of_recordings_index.append(int(raw_max_indices_dataset[i]))

        original_list_of_recordings = []
        original_list_of_recordings_index = []
        for i in range(0, int(dataset_dim[ds])):
            for c in range(0, n_classes - np.size(args.GAN_classes)):
                if (list_of_recordings[i].find('NEW_' + classes_noGAN[c]) != -1):
                    original_list_of_recordings.append(classes_noGAN[c])
                    original_list_of_recordings_index.append(c)
                    pass

        # Inception score
        IS[ds] = stat.inception_score(raw_sum_distr[:, :])

        # Control the error of the decoder
        decoder_syll_error = np.zeros((n_classes - np.size(args.GAN_classes),))  # not including the GAN class
        pos_error = []
        for i in range(0,int(dataset_dim[ds])):
            if original_list_of_recordings[i] != decoder_list_of_recordings[i]:
                decoder_syll_error[int(original_list_of_recordings_index[i])] = decoder_syll_error[int(original_list_of_recordings_index[i])] + 1
                pos_error.append(i)

        decoder_total_error = np.sum(decoder_syll_error)

        # PLOT
        # Plot the distribution of syllables in the dataset, with and without the garbage classes
        title = 'Dataset'
        fig, dataset_h, dataset_mean, dataset_std, dataset_var = plots.plot_distribution_classes(raw_max_indices_dataset, classes, title, n_classes)
        fig.savefig(
            args.data_dir + '/' + args.output_dir + '/' + 'Classes_syllables_dataset_' + legend_list[ds] + '.' + args.format)

        if np.size(args.GAN_classes) > 0:
            title = 'Dataset without GAN class'
            fig, dataset_h_noGAN, dataset_mean, dataset_std, dataset_var = plots.plot_distribution_classes(
                raw_max_indices_dataset_noGAN, np.delete(classes, GAN_class), title, n_classes - np.size(args.GAN_classes))
            #fig.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Classes_syllables_dataset_noGAN_' + legend_list[ds] + '.' + args.format)

        # Plot of the error in the dataset
        fig, ax = plt.subplots()
        ax.bar(classes_noGAN, decoder_syll_error)
        plt.ylabel('Number of samples')
        plt.xticks(range(n_classes - np.size(args.GAN_classes)), classes_noGAN, rotation='vertical', fontsize=15)
        plt.ylabel('Number of errors', fontsize=15)
        plt.xlabel('Classes of syllables')
        plt.legend(['decoder_total_error=' + str(decoder_total_error)])
        plt.tight_layout()
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Error in the dataset_' + legend_list[ds]  + '.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Error in the dataset_' + legend_list[ds]  + '.' + 'png')

        # In case we need to save all the recordings
        whole_list_of_recordings.append(list_of_recordings)
        whole_decoder_list_of_recordings.append(decoder_list_of_recordings)

        # Dataset distribution
        dataset_distr = dataset_h / np.sum(dataset_h)

        dataset_summary = {'File_name': list_of_recordings, 'Real_name': original_list_of_recordings, 'Decoder_name': decoder_list_of_recordings, 'Annotations':raw_sum_distr,
                           'Dataset_distr': dataset_distr}
        np.save(args.data_dir + '/' + args.output_dir + '/' + 'Error_summary_' + legend_list[ds] + '.npy', dataset_summary)

        plt.close('all')

    print('Done')

def analysis_generation(annotations_generation, summary_dataset, args):
    """
    :param annotations_generation: list of the annotations (one per epoch)
    :param summary_dataset: dictionary containing the info about the training dataset
    :param args: all the parameters, see below

    Also, the npy dictionary containing the distribution of the dataset needs to be in the same directory
    (to compute the cross entropy).

    :return: save npy dictionary with the summary of the analysis
    """
    annotations_aux = open_pkl(annotations_generation[0])
    classes = annotations_aux[0].vocab

    GAN_class = np.zeros((np.size(args.GAN_classes),))
    classes_noGAN = classes
    if np.size(args.GAN_classes)>0:
        for gc in range(0,np.size(args.GAN_classes)):
            GAN_class[gc] = np.where(classes == args.GAN_classes[gc])[0][0]
            classes_noGAN = np.delete(classes_noGAN, np.argwhere(classes_noGAN == args.GAN_classes[gc]))
    n_classes = len(annotations_aux[0].vocab)

    annotations_aux = open_pkl(annotations_generation[0])
    n_syll = len(annotations_aux)

    ld_generation = np.zeros((np.size(annotations_generation),))

    ckpt_generation = np.zeros((np.size(annotations_generation),))

    epoch  = np.zeros((np.size(annotations_generation),))

    raw_sum = np.zeros((np.size(annotations_generation), n_syll, n_classes))
    raw_sum_distr = np.zeros((np.size(annotations_generation), n_syll, n_classes))
    raw_max = np.zeros((np.size(annotations_generation), n_syll))
    raw_max_indices = np.zeros((np.size(annotations_generation), n_syll))
    raw_max_noGAN = np.zeros((np.size(annotations_generation), n_syll))
    raw_max_indices_noGAN = np.zeros((np.size(annotations_generation), n_syll))

    IS = np.zeros((np.size(annotations_generation),))

    for a in range(0,np.size(annotations_generation)):
        ld_generation[a] = annotations_generation[a][annotations_generation[a].find('_ld') + 3]
        ckpt_generation[a] = annotations_generation[a][annotations_generation[a].find('ckpt') + 4 :annotations_generation[a].find('ckpt')+ 9]
        epoch[a] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)
        annotations_aux = open_pkl(annotations_generation[a])

        GAN_list_of_recordings = []
        GAN_decoder_list_of_recordings = []
        annotations_raw = []
        for i in range(0,n_syll):
            GAN_list_of_recordings.append(annotations_aux[i].id)
            annotations_raw.append(annotations_aux[i].vect)
            # This operation should give me an idea of which class is represented the most in my generations
            raw_sum[a,i,:] = np.sum(annotations_raw[i], axis=0)
            raw_max[a,i] = np.max(raw_sum[a,i,:])
            raw_max_indices[a,i] = np.where(raw_sum[a,i,:] == np.max(raw_sum[a,i,:]))[0][0]
            # Probability distribution to compute the inception score
            raw_sum_distr[a, i, :] = special.softmax(raw_sum[a, i, :])
            # without the GAN class
            raw_max_noGAN[a, i] = np.max(np.delete(raw_sum[a, i, :], GAN_class))
            raw_max_indices_noGAN[a, i] = np.where(raw_sum[a, i, :] == np.max(raw_sum[a, i, :]))[0][0]

            GAN_decoder_list_of_recordings.append(classes[int(raw_max_indices[a,i])])

        # Inception score
        IS[a] = stat.inception_score(raw_sum_distr[a,:,:])

        GAN_summary = {'File_name': GAN_list_of_recordings, 'Decoder_name': GAN_decoder_list_of_recordings, 'Decoder_index': raw_max_indices[a,:], 'Latent_dim': ld_generation[a], 'Epoch': epoch[a], 'IS': IS[a], 'Annotations': raw_sum_distr[a, :, :]}
        if int(epoch[a])<100:
            np.save(args.data_dir + '/' + 'Generation_summary' + '_ep00' + str(round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim))  + '_ld' + str(annotations_generation[a][annotations_generation[a].find('_ld') + 3]) + '.npy', GAN_summary)
        elif int(epoch[a])<1000:
            np.save(args.data_dir + '/' + 'Generation_summary' + '_ep0' + str(round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim))  + '_ld' + str(annotations_generation[a][annotations_generation[a].find('_ld') + 3]) + '.npy', GAN_summary)
        else:
            np.save(args.data_dir + '/' + 'Generation_summary' + '_ep' + str(round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)) + '_ld' + str(annotations_generation[a][annotations_generation[a].find('_ld') + 3]) + '.npy', GAN_summary)

    # PLOT
    # Generation from GAN
    h = np.zeros((np.size(ld_generation), n_classes))
    h_mean = np.zeros((np.size(ld_generation),3)) # column 1 = mean value, column 2 = which dim, column 3 = at which epoch
    h_std = np.zeros((np.size(ld_generation), 3))
    h_var = np.zeros((np.size(ld_generation), 3))
    h_mean[:,1] = ld_generation
    h_std[:, 1] = ld_generation
    h_var[:,1] = ld_generation

    h_noGAN = np.zeros((np.size(ld_generation), n_classes - np.size(args.GAN_classes)))
    h_meannoGAN = np.zeros((np.size(ld_generation),3))
    h_stdnoGAN = np.zeros((np.size(ld_generation), 3))
    h_varnoGAN = np.zeros((np.size(ld_generation), 3))
    h_percentile5noGAN = np.zeros((np.size(ld_generation), 3))
    h_percentile95noGAN = np.zeros((np.size(ld_generation), 3))
    h_maxnoGAN = np.zeros((np.size(ld_generation), 3))
    h_minnoGAN = np.zeros((np.size(ld_generation), 3))
    h_mediannoGAN = np.zeros((np.size(ld_generation), 3))
    h_meannoGAN[:, 1] = ld_generation
    h_stdnoGAN[:, 1] = ld_generation
    h_varnoGAN[:, 1] = ld_generation
    h_mediannoGAN[:, 1] = ld_generation
    h_percentile5noGAN[:, 1] = ld_generation
    h_percentile95noGAN[:, 1] = ld_generation
    h_maxnoGAN[:, 1] = ld_generation
    h_minnoGAN[:, 1] = ld_generation

    rep_classes = np.zeros((np.size(ld_generation),3)) # column 1 = how many classes are represented, column 2 = which dim , column 3 = which epoch
    rep_classes[:,1] = ld_generation
    for a in range(0,np.size(ld_generation)):
        #title = 'Latent space dimension = ' + str(int(ld_generation[a])) + 'at epoch_' + str(round(int(ckpt_generation[a])*5*64/args.dataset_dim))
        title = 'Epoch ' + str(round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim))
        # Plot using mean
        fig, gan_h, gan_mean, gan_std, gan_var = plots.plot_distribution_classes(raw_max_indices[a, :], np.append(classes_noGAN, args.GAN_classes), title, n_classes)
        if a%1 == 0:
            if args.format != 'png':
                fig.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Classes of syllables_ld' + str(int(ld_generation[a])) + '_epoch_' + str(round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)) + '.' + args.format)
            fig.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Classes of syllables_ld' + str(int(ld_generation[a])) + '_epoch_' + str(round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)) + '.' + 'png')

        h_mean[a, 0] = gan_mean
        h_mean[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)
        h_std[a, 0] = gan_std
        h_std[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)
        h_var[a, 0] = gan_var
        h_var[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)

        h[a, :] = gan_h

        # Distribution of the classes (how many classes are represented at each epoch)
        rep_classes[a, 0] = np.count_nonzero(np.delete(gan_h, GAN_class))
        rep_classes[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)

        # NoGAN data
        h_noGAN[a, :] = np.delete(gan_h, GAN_class)
        h_meannoGAN[a, 0] = np.mean(np.delete(gan_h, GAN_class))
        h_meannoGAN[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)
        h_stdnoGAN[a, 0] = np.std(np.delete(gan_h, GAN_class))
        h_stdnoGAN[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)
        h_varnoGAN[a, 0] = np.var(np.delete(gan_h, GAN_class))
        h_varnoGAN[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)
        h_mediannoGAN[a, 0] = np.percentile(np.delete(gan_h, GAN_class),50)
        h_mediannoGAN[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)
        h_percentile5noGAN[a, 0] = np.percentile(np.delete(gan_h, GAN_class), 5)
        h_percentile5noGAN[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)
        h_percentile95noGAN[a, 0] = np.percentile(np.delete(gan_h, GAN_class), 95)
        h_percentile95noGAN[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)
        h_maxnoGAN[a, 0] = np.max(np.delete(gan_h, GAN_class))
        h_maxnoGAN[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)
        h_minnoGAN[a, 0] = np.min(np.delete(gan_h, GAN_class))
        h_minnoGAN[a, 2] = round(int(ckpt_generation[a]) * 5 * 64 / args.dataset_dim)

        plt.close('all')

    # Distribution of the classes
    # Collect all the data in a useful way
    mean_across_time = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation)/np.size(args.n_ld_dim)),3))
    std_across_time = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation)/np.size(args.n_ld_dim)),3))
    var_across_time = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation)/np.size(args.n_ld_dim)),3))
    mean_across_time_noGAN = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation) / np.size(args.n_ld_dim)), 3))
    std_across_time_noGAN = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation) / np.size(args.n_ld_dim)), 3))
    var_across_time_noGAN = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation) / np.size(args.n_ld_dim)), 3))
    median_across_time_noGAN = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation) / np.size(args.n_ld_dim)), 3))
    percentile5_across_time_noGAN = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation) / np.size(args.n_ld_dim)), 3))
    percentile95_across_time_noGAN = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation) / np.size(args.n_ld_dim)), 3))
    max_across_time_noGAN = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation) / np.size(args.n_ld_dim)), 3))
    min_across_time_noGAN = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation) / np.size(args.n_ld_dim)), 3))
    rep_classes_time = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation)/np.size(args.n_ld_dim)),3)) # how many different classes are represented in the generated data
    classes_time = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation) / np.size(args.n_ld_dim)), n_classes))  # how many generations per class across dimensions and time
    for ld in range(0,np.size(args.n_ld_dim)):
        mean_across_time[ld, :, :] = h_mean[np.where(h_mean[:,1] == args.n_ld_dim[ld]),:]
        std_across_time[ld,:,:] = h_std[np.where(h_std[:,1] == args.n_ld_dim[ld]),:]
        var_across_time[ld,:,:] = h_var[np.where(h_var[:,1] == args.n_ld_dim[ld]),:]
        mean_across_time_noGAN[ld, :, :] = h_meannoGAN[np.where(h_meannoGAN[:, 1] == args.n_ld_dim[ld]), :]
        std_across_time_noGAN[ld, :, :] = h_stdnoGAN[np.where(h_stdnoGAN[:, 1] == args.n_ld_dim[ld]), :]
        var_across_time_noGAN[ld, :, :] = h_varnoGAN[np.where(h_varnoGAN[:, 1] == args.n_ld_dim[ld]), :]
        median_across_time_noGAN[ld, :, :] = h_mediannoGAN[np.where(h_mediannoGAN[:, 1] == args.n_ld_dim[ld]), :]
        percentile5_across_time_noGAN[ld, :, :] = h_percentile5noGAN[np.where(h_percentile5noGAN[:, 1] == args.n_ld_dim[ld]), :]
        percentile95_across_time_noGAN[ld, :, :] = h_percentile95noGAN[np.where(h_percentile95noGAN[:, 1] == args.n_ld_dim[ld]), :]
        max_across_time_noGAN[ld, :, :] = h_maxnoGAN[np.where(h_maxnoGAN[:, 1] == args.n_ld_dim[ld]), :]
        min_across_time_noGAN[ld, :, :] = h_minnoGAN[np.where(h_minnoGAN[:, 1] == args.n_ld_dim[ld]), :]
        rep_classes_time[ld,:,:] = rep_classes[np.where(rep_classes[:,1] == args.n_ld_dim[ld]),:]
        classes_time[ld, :, :] = h[np.where(ld_generation == args.n_ld_dim[ld]),:]

    # Distribution of the data
    load_data = np.load(summary_dataset[0], allow_pickle=True)
    load_data = load_data.item()
    dataset_distr = load_data['Dataset_distr']
    dataset_real_distr = load_data['Dataset_real_distr']

    GAN_distr = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation)/np.size(args.n_ld_dim)), n_classes))
    for ld in range(0, np.size(args.n_ld_dim)):
        GAN_distr_aux = h[np.where(ld_generation == args.n_ld_dim[ld]), :]
        for ep in range(0,int(np.size(ld_generation)/np.size(args.n_ld_dim))):
            if np.sum(GAN_distr_aux[0,ep,:]) != 0:
                GAN_distr[ld,ep,:] = GAN_distr_aux[0,ep,:]/np.sum(GAN_distr_aux[0,ep,:]) # probability distribution

    GAN_distr_noGAN = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation)/np.size(args.n_ld_dim)), n_classes-np.size(args.GAN_classes)))
    for ld in range(0, np.size(args.n_ld_dim)):
        GAN_distr_aux = h_noGAN[np.where(ld_generation == args.n_ld_dim[ld]), :]
        for ep in range(0,int(np.size(ld_generation)/np.size(args.n_ld_dim))):
            if np.sum(GAN_distr_aux[0,ep,:]) != 0:
                GAN_distr_noGAN[ld,ep,:] = GAN_distr_aux[0,ep,:]/np.sum(GAN_distr_aux[0,ep,:]) # probability distribution

    # Cross-entropy between the dataset and the generated data
    # Cross entropy using the classic definition
    classic_cross_entropy = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation)/np.size(args.n_ld_dim))))
    classic_cross_entropy_noGAN = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation)/np.size(args.n_ld_dim))))
    for ld in range(0, np.size(args.n_ld_dim)):
        for ep in range(0, int(np.size(ld_generation)/np.size(args.n_ld_dim))):
            classic_cross_entropy[ld,ep] = stat.cross_entropy(dataset_distr, GAN_distr[ld,ep,:])
            classic_cross_entropy_noGAN[ld,ep] = stat.cross_entropy(np.delete(dataset_distr,GAN_class), GAN_distr_noGAN[ld,ep,:])

    # Cross-entropy between the real dataset distribution and the generated data
    # Cross entropy using the classic definition
    classic_cross_entropy_real = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation)/np.size(args.n_ld_dim))))
    classic_cross_entropy_noGAN_real = np.zeros((np.size(args.n_ld_dim), int(np.size(ld_generation)/np.size(args.n_ld_dim))))
    for ld in range(0, np.size(args.n_ld_dim)):
        for ep in range(0, int(np.size(ld_generation)/np.size(args.n_ld_dim))):
            classic_cross_entropy_real[ld,ep] = stat.cross_entropy(dataset_real_distr[0], GAN_distr[ld,ep,:])
            classic_cross_entropy_noGAN_real[ld,ep] = stat.cross_entropy(np.delete(dataset_real_distr[0],GAN_class), GAN_distr_noGAN[ld,ep,:])

    generation_summary = {'mean_across_time': mean_across_time, 'std_across_time': std_across_time, 'var_across_time': var_across_time, 'mean_across_time_noGAN': mean_across_time_noGAN, 'std_across_time_noGAN': std_across_time_noGAN,
                          'median_across_time_noGAN': median_across_time_noGAN, 'var_across_time_noGAN': var_across_time_noGAN, 'percentile5_across_time_noGAN': percentile5_across_time_noGAN,
                          'percentile95_across_time_noGAN': percentile95_across_time_noGAN, 'max_across_time_noGAN': max_across_time_noGAN, 'min_across_time_noGAN': min_across_time_noGAN, 'rep_classes_time': rep_classes_time,
                          'classes_time': classes_time, 'GAN_distr': GAN_distr, 'GAN_distr_noGAN': GAN_distr_noGAN, 'cross_entropy': classic_cross_entropy, 'cross_entropy_noGAN':classic_cross_entropy_noGAN,
                          'cross_entropy_real': classic_cross_entropy_real, 'cross_entropy_noGAN_real':classic_cross_entropy_noGAN_real, 'classes': classes, 'classes_no_GAN': classes_noGAN}
    np.save(args.data_dir + '/' + 'Generation_summary_all' + str(args.n_ld_dim[0]) + '.npy', generation_summary)

    print('Done')

def analysis_latent(generation_data, summary_dataset, legend_list, colors_list, args):
    """
    The input directory contains one or more annotations.

    This function is meant to analyse the output of the GAN and see how it is able to reproduce and or differentiate from the dataset
    depending on a correlation with the input classes.
    """

    # DATASET
    load_data = np.load(summary_dataset[0],allow_pickle=True)
    load_data = load_data.item()
    dataset_distr = load_data['Dataset_distr']
    dataset_distr_real = load_data['Dataset_real_distr']
    data_entropy = stat.cross_entropy(dataset_distr_real[0],dataset_distr)

    colors_list = list(colors._colors_full_map.values())[0::5]

    # GENERATED DATA
    load_data = np.load(generation_data[0][0], allow_pickle=True)
    load_data = load_data.item()

    classes = load_data['classes']

    GAN_class = np.zeros((np.size(args.GAN_classes),))
    for gc in range(0, np.size(args.GAN_classes)):
        GAN_class[gc] = np.where(classes == args.GAN_classes[gc])[0][0]

    # Load the data
    mean_across_time = []
    std_across_time = []
    var_across_time = []
    mean_across_time_noGAN = []
    std_across_time_noGAN = []
    var_across_time_noGAN = []
    median_across_time_noGAN = []
    percentile5_across_time_noGAN = []
    percentile95_across_time_noGAN = []
    max_across_time_noGAN = []
    min_across_time_noGAN = []
    rep_classes_time = []
    classes_time = []
    GAN_distr = []
    GAN_distr_noGAN = []
    classic_cross_entropy = []
    classic_cross_entropy_noGAN = []
    classic_cross_entropy_noGAN_real = []
    classic_cross_entropy_real = []

    for ld in range(0,np.size(args.n_ld_dim)):
        mean_across_time_aux = []
        std_across_time_aux = []
        var_across_time_aux = []
        mean_across_time_noGAN_aux = []
        std_across_time_noGAN_aux = []
        var_across_time_noGAN_aux = []
        median_across_time_noGAN_aux = []
        percentile5_across_time_noGAN_aux = []
        percentile95_across_time_noGAN_aux = []
        max_across_time_noGAN_aux = []
        min_across_time_noGAN_aux = []
        rep_classes_time_aux = []
        classes_time_aux = []
        GAN_distr_aux = []
        GAN_distr_noGAN_aux = []
        classic_cross_entropy_aux = []
        classic_cross_entropy_real_aux = []
        classic_cross_entropy_noGAN_aux = []
        classic_cross_entropy_noGAN_real_aux = []

        for gd in range(0,np.size(generation_data[ld])):
            load_data = np.load(generation_data[ld][gd], allow_pickle=True)
            load_data = load_data.item()
            mean_across_time_aux.append(load_data['mean_across_time'])
            std_across_time_aux.append(load_data['std_across_time'])
            var_across_time_aux.append(load_data['var_across_time'])
            mean_across_time_noGAN_aux.append(load_data['mean_across_time_noGAN'])
            std_across_time_noGAN_aux.append(load_data['std_across_time_noGAN'])
            rep_classes_time_aux.append(load_data['rep_classes_time'])
            var_across_time_noGAN_aux.append(load_data['var_across_time_noGAN'])
            median_across_time_noGAN_aux.append(load_data['median_across_time_noGAN'])
            percentile5_across_time_noGAN_aux.append(load_data['percentile5_across_time_noGAN'])
            percentile95_across_time_noGAN_aux.append(load_data['percentile95_across_time_noGAN'])
            max_across_time_noGAN_aux.append(load_data['max_across_time_noGAN'])
            min_across_time_noGAN_aux.append(load_data['min_across_time_noGAN'])
            GAN_distr_noGAN_aux.append(load_data['GAN_distr_noGAN'])
            classes_time_aux.append(load_data['classes_time'])
            GAN_distr_aux.append(load_data['GAN_distr'])
            classic_cross_entropy_aux.append(load_data['cross_entropy'])
            classic_cross_entropy_real_aux.append(load_data['cross_entropy_real'])
            classic_cross_entropy_noGAN_aux.append(load_data['cross_entropy_noGAN'])
            classic_cross_entropy_noGAN_real_aux.append(load_data['cross_entropy_noGAN_real'])

        mean_across_time.append(mean_across_time_aux)
        std_across_time.append(std_across_time_aux)
        var_across_time.append(var_across_time_aux)
        mean_across_time_noGAN.append(mean_across_time_noGAN_aux)
        std_across_time_noGAN.append(std_across_time_noGAN_aux)
        var_across_time_noGAN.append(var_across_time_noGAN_aux)
        median_across_time_noGAN.append(median_across_time_noGAN_aux)
        percentile5_across_time_noGAN.append(percentile5_across_time_noGAN_aux)
        percentile95_across_time_noGAN.append(percentile95_across_time_noGAN_aux)
        max_across_time_noGAN.append(max_across_time_noGAN_aux)
        min_across_time_noGAN.append(min_across_time_noGAN_aux)
        rep_classes_time.append(rep_classes_time_aux)
        classes_time.append(classes_time_aux)
        GAN_distr.append(GAN_distr_aux)
        GAN_distr_noGAN.append(GAN_distr_noGAN_aux)
        classic_cross_entropy.append(classic_cross_entropy_aux)
        classic_cross_entropy_real.append(classic_cross_entropy_real_aux)
        classic_cross_entropy_noGAN.append(classic_cross_entropy_noGAN_aux)
        classic_cross_entropy_noGAN_real.append(classic_cross_entropy_noGAN_real_aux)

    # Plots across time for a fixed dataset dimension, comparison between different instances
    for ld in range(0,np.size(args.n_ld_dim)):
        # Mean and variance distribution
        # with GAN class
        fig,ax = plt.subplots()
        for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
            plt.plot(mean_across_time[ld][gd][0,:,2], mean_across_time[ld][gd][0,:,0], colors_list[ld])
            ax.fill_between(mean_across_time[ld][gd][0,:,2], mean_across_time[ld][gd][0,:,0] - std_across_time[ld][gd][0,:,0], mean_across_time[ld][gd][0,:,0] + std_across_time[ld][gd][0,:,0], colors_list[gd], alpha = 0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean and std', fontsize=15)
        plt.title('Evolution of mean and std across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['mean' , 'Std'])
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_std_for_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)

        fig,ax = plt.subplots()
        for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
            plt.plot(var_across_time[ld][gd][0,:,2], var_across_time[ld][gd][0,:,0], colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of variance', fontsize=15)
        plt.title('Evolution of variance across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['ld dim' + str(args.n_ld_dim[ld])])
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_var_for_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)

        # without GAN class
        # Mean + std
        fig, ax = plt.subplots()
        for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0] - std_across_time_noGAN[ld][gd][0, :, 0],
                            mean_across_time_noGAN[ld][gd][0, :, 0] + std_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd], alpha=0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean and std', fontsize=15)
        plt.title('Evolution of mean and std across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['Mean', 'Std'])
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_std_for_dim_' + str(
            args.n_ld_dim[ld]) + '_noGAN.' + args.format)

        # Mean + median
        fig, ax = plt.subplots()
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], median_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd], alpha=0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean and median', fontsize=15)
        plt.title('Evolution of mean and median across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['Mean', 'Median'])
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_median_for_dim_' + str(
            args.n_ld_dim[ld]) + '_noGAN.' + args.format)

        # Mean + min/max
        fig, ax = plt.subplots()
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2],
                            min_across_time_noGAN[ld][gd][0, :, 0],
                            max_across_time_noGAN[ld][gd][0, :, 0],
                            colors_list[gd], alpha=0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean', fontsize=15)
        plt.title('Evolution of mean across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['Mean', 'p50'])
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_min_max_for_dim_' + str(
            args.n_ld_dim[ld]) + '_noGAN.' + args.format)

        # Mean + p5 - p95
        fig, ax = plt.subplots()
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2],
                            percentile5_across_time_noGAN[ld][gd][0, :, 0],
                            percentile95_across_time_noGAN[ld][gd][0, :, 0],
                            colors_list[gd], alpha=0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean', fontsize=15)
        plt.title('Evolution of mean across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['Mean', 'p5-p95'])
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_p5p95_for_dim_' + str(
            args.n_ld_dim[ld]) + '_noGAN.' + args.format)

        # Var
        fig, ax = plt.subplots()
        for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
            plt.plot(var_across_time_noGAN[ld][gd][0, :, 2], var_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)')
        plt.ylabel('Evolution of variance')
        plt.title('Evolution of variance across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_var_for_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + args.format)

    plt.close('all')

    # Distribution of classes represented by the dataset
    for ld in range(0, np.size(args.n_ld_dim)):
        fig, ax = plt.subplots()
        for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
            plt.plot(rep_classes_time[ld][gd][0,:,2], rep_classes_time[ld][gd][0,:,0], colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Number of classes', fontsize=15)
        plt.title('Number of represented classes across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Distr_classes_across_latent_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    # GAN class
    for gc in range(0,np.size(args.GAN_classes)):
        for ld in range(0, np.size(args.n_ld_dim)):
            fig, ax = plt.subplots()
            for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
                plt.plot(rep_classes_time[ld][gd][0, :, 2], classes_time[ld][gd][0, :, int(GAN_class[gc])].T, colors_list[gd])
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            plt.xlabel('Time (in number of epochs)', fontsize=15)
            plt.ylabel('Number of syllable (in percentage)', fontsize=15)
            plt.legend(legend_list)
            plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class for latent dim = ' + str(args.n_ld_dim[ld]))
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + classes[int(GAN_class[gc])] + '_class_across_time_and_latent_' + str(args.n_ld_dim[ld]) + '.' + args.format)

        for ld in range(0, np.size(args.n_ld_dim)):
            fig, ax = plt.subplots()
            for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
                plt.plot(rep_classes_time[ld][gd][0, :, 2], classes_time[ld][gd][0, :, int(GAN_class[gc])].T/1000, colors_list[gd])
                plt.axhline(dataset_distr[int(GAN_class[gc])], color='k')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            plt.xlabel('Time (in number of epochs)', fontsize=15)
            plt.ylabel('Number of syllable (in percentage)', fontsize=15)
            plt.legend(np.append(legend_list, 'Training'))
            plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class for latent dim = ' + str(args.n_ld_dim[ld]))
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_' + classes[int(GAN_class[gc])] + '_class_across_time_and_latent_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    plt.close('all')

    # Cross-Entropy
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy[ld][gd][0,:], colors_list[gd])
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        plt.title('Cross-entropy across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_noGAN[ld][gd][0,:], colors_list[gd])
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        plt.title('Cross-entropy across time (without GAN class) for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_noGAN_real[ld][gd][0,:], colors_list[gd])
            plt.axhline(data_entropy, color='k')
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        plt.title('Cross-entropy across time (without GAN class) for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(np.append(legend_list, 'Training'))
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_real_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    plt.close('all')

    # All together
    fig,ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
            plt.plot(mean_across_time[ld][gd][0,:,2], mean_across_time[ld][gd][0,:,0], colors_list[gd])
            ax.fill_between(mean_across_time[ld][gd][0,:,2], mean_across_time[ld][gd][0,:,0] - std_across_time[ld][gd][0,:,0], mean_across_time[ld][gd][0,:,0] + std_across_time[ld][gd][0,:,0], colors_list[gd], alpha = 0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean and std', fontsize=15)
    plt.title('Evolution of mean and std across time')
    plt.legend(['mean' , 'Std'])
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_std_for_dim_all.' + args.format)

    fig,ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
            plt.plot(var_across_time[ld][gd][0,:,2], var_across_time[ld][gd][0,:,0], colors_list[gd])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of variance', fontsize=15)
    plt.title('Evolution of variance across time for latent dim = ' + str(args.n_ld_dim[ld]))
    plt.legend(['ld dim' + str(args.n_ld_dim[ld])])
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_var_for_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    # without GAN class
    # Mean + std
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0] - std_across_time_noGAN[ld][gd][0, :, 0],
                            mean_across_time_noGAN[ld][gd][0, :, 0] + std_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean and std', fontsize=15)
    plt.title('Evolution of mean and std across time')
    plt.legend(['Mean', 'Std'])
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_std_for_dim_all_noGAN.' + args.format)

    # Mean + median
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], median_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean and median', fontsize=15)
    plt.title('Evolution of mean and median across time')
    plt.legend(['Mean', 'Median'])
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_median_for_dim_all_noGAN.' + args.format)

    # Mean + min/max
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2],
                            min_across_time_noGAN[ld][gd][0, :, 0],
                            max_across_time_noGAN[ld][gd][0, :, 0],
                            colors_list[gd], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean', fontsize=15)
    plt.title('Evolution of mean across time')
    plt.legend(['Mean', 'p50'])
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_min_max_for_dim_all_noGAN.' + args.format)

    # Mean + p5 - p95
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2],
                            percentile5_across_time_noGAN[ld][gd][0, :, 0],
                            percentile95_across_time_noGAN[ld][gd][0, :, 0],
                            colors_list[gd], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean', fontsize=15)
    plt.title('Evolution of mean across time')
    plt.legend(['Mean', 'p5-p95'])
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_p5p95_for_dim_all_noGAN.' + args.format)

    # Var
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
            plt.plot(var_across_time_noGAN[ld][gd][0, :, 2], var_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)')
    plt.ylabel('Evolution of variance')
    plt.title('Evolution of variance across time')
    plt.legend(legend_list)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_var_for_dim_all_noGAN.' + args.format)

    plt.close('all')


    # Distribution of classes represented by the dataset
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
            plt.plot(rep_classes_time[ld][gd][0,:,2], rep_classes_time[ld][gd][0,:,0], colors_list[gd])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Number of classes', fontsize=15)
    plt.title('Number of represented classes across time')
    plt.legend(legend_list)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Distr_classes_across_latent_all.' + args.format)

    # GAN class
    for gc in range(0,np.size(args.GAN_classes)):
        fig, ax = plt.subplots()
        for ld in range(0, np.size(args.n_ld_dim)):
            for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
                plt.plot(rep_classes_time[ld][gd][0, :, 2], classes_time[ld][gd][0, :, int(GAN_class[gc])].T, colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Number of syllable (in percentage)', fontsize=15)
        plt.legend(legend_list)
        plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class')
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + classes[int(GAN_class[gc])] + '_class_across_time_all.' + args.format)

        fig, ax = plt.subplots()
        for ld in range(0, np.size(args.n_ld_dim)):
            for gd in range(0, round(np.size(generation_data)/np.size(args.n_ld_dim))):
                plt.plot(rep_classes_time[ld][gd][0, :, 2], classes_time[ld][gd][0, :, int(GAN_class[gc])].T/1000, colors_list[gd])
                plt.axhline(dataset_distr[int(GAN_class[gc])], color='k')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Number of syllable (in percentage)', fontsize=15)
        plt.legend(np.append(legend_list, 'Training'))
        plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class')
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_' + classes[int(GAN_class[gc])] + '_class_across_time_all.' + args.format)

    plt.close('all')

    # Cross-Entropy
    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy[ld][gd][0,:], colors_list[gd])
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    plt.title('Cross-entropy across time')
    plt.legend(legend_list)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_all.' + args.format)

    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_noGAN[ld][gd][0,:], colors_list[gd])
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    plt.title('Cross-entropy across time (without GAN class)')
    plt.legend(legend_list)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_all.' + args.format)

    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        for gd in range(0, round(np.size(generation_data) / np.size(args.n_ld_dim))):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_noGAN_real[ld][gd][0,:], colors_list[gd])
            plt.axhline(data_entropy, color='k')
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    plt.title('Cross-entropy across time (without GAN class)')
    plt.legend(np.append(legend_list, 'Training'))
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_real_all.' + args.format)


    print('Done')

def analysis_dim(generation_data, summary_dataset, legend_list_instances, legend_list, colors_list, args):
    load_data = np.load(generation_data[0][0], allow_pickle=True)
    load_data = load_data.item()

    classes = load_data['classes']

    GAN_class = np.zeros((np.size(args.GAN_classes),))
    for gc in range(0, np.size(args.GAN_classes)):
        GAN_class[gc] = np.where(classes == args.GAN_classes[gc])[0][0]

    colors_list_long = list(colors._colors_full_map.values())[0::5]
    colors_list = ['b', 'r', 'g']

    # DATASET
    dataset_distr = []
    dataset_real_distr = []
    data_entropy = []
    data_entropy_noGAN = []
    for sd in range(0,np.size(args.n_ld_dim)):
        load_data = np.load(summary_dataset[sd][0],allow_pickle=True)
        load_data = load_data.item()
        dataset_distr.append(load_data['Dataset_distr'])
        dataset_real_distr.append(load_data['Dataset_real_distr'])
        data_entropy.append(stat.cross_entropy(dataset_real_distr[sd][0],dataset_distr[sd]))
        data_entropy_noGAN.append(stat.cross_entropy(np.delete(dataset_real_distr[sd][0], GAN_class), np.delete(dataset_distr[sd], GAN_class)))

    # GENERATED DATA
    mean_across_time = []
    std_across_time = []
    var_across_time = []
    mean_across_time_noGAN = []
    std_across_time_noGAN = []
    var_across_time_noGAN = []
    median_across_time_noGAN = []
    percentile5_across_time_noGAN = []
    percentile95_across_time_noGAN = []
    max_across_time_noGAN = []
    min_across_time_noGAN = []
    rep_classes_time = []
    classes_time = []
    GAN_distr = []
    GAN_distr_noGAN = []
    classic_cross_entropy = []
    classic_cross_entropy_real = []
    classic_cross_entropy_noGAN = []
    classic_cross_entropy_noGAN_real = []

    for ld in range(0,np.size(args.n_ld_dim)):
        mean_across_time_aux = []
        std_across_time_aux = []
        var_across_time_aux = []
        mean_across_time_noGAN_aux = []
        std_across_time_noGAN_aux = []
        var_across_time_noGAN_aux = []
        median_across_time_noGAN_aux = []
        percentile5_across_time_noGAN_aux = []
        percentile95_across_time_noGAN_aux = []
        max_across_time_noGAN_aux = []
        min_across_time_noGAN_aux = []
        rep_classes_time_aux = []
        classes_time_aux = []
        GAN_distr_aux = []
        GAN_distr_noGAN_aux = []
        classic_cross_entropy_aux = []
        classic_cross_entropy_real_aux = []
        classic_cross_entropy_noGAN_aux = []
        classic_cross_entropy_noGAN_real_aux = []

        for gd in range(0,np.size(generation_data[ld])):
            load_data = np.load(generation_data[ld][gd], allow_pickle=True)
            load_data = load_data.item()
            mean_across_time_aux.append(load_data['mean_across_time'])
            std_across_time_aux.append(load_data['std_across_time'])
            var_across_time_aux.append(load_data['var_across_time'])
            mean_across_time_noGAN_aux.append(load_data['mean_across_time_noGAN'])
            std_across_time_noGAN_aux.append(load_data['std_across_time_noGAN'])
            rep_classes_time_aux.append(load_data['rep_classes_time'])
            var_across_time_noGAN_aux.append(load_data['var_across_time_noGAN'])
            median_across_time_noGAN_aux.append(load_data['median_across_time_noGAN'])
            min_across_time_noGAN_aux.append(load_data['min_across_time_noGAN'])
            max_across_time_noGAN_aux.append(load_data['max_across_time_noGAN'])
            percentile5_across_time_noGAN_aux.append(load_data['percentile5_across_time_noGAN'])
            percentile95_across_time_noGAN_aux.append(load_data['percentile95_across_time_noGAN'])
            GAN_distr_noGAN_aux.append(load_data['GAN_distr_noGAN'])
            classes_time_aux.append(load_data['classes_time'])
            GAN_distr_aux.append(load_data['GAN_distr'])
            classic_cross_entropy_aux.append(load_data['cross_entropy'])
            classic_cross_entropy_real_aux.append(load_data['cross_entropy_real'])
            classic_cross_entropy_noGAN_aux.append(load_data['cross_entropy_noGAN'])
            classic_cross_entropy_noGAN_real_aux.append(load_data['cross_entropy_noGAN_real'])

        mean_across_time.append(mean_across_time_aux)
        std_across_time.append(std_across_time_aux)
        var_across_time.append(var_across_time_aux)
        mean_across_time_noGAN.append(mean_across_time_noGAN_aux)
        std_across_time_noGAN.append(std_across_time_noGAN_aux)
        var_across_time_noGAN.append(var_across_time_noGAN_aux)
        median_across_time_noGAN.append(median_across_time_noGAN_aux)
        min_across_time_noGAN.append(min_across_time_noGAN_aux)
        max_across_time_noGAN.append(max_across_time_noGAN_aux)
        percentile5_across_time_noGAN.append(percentile5_across_time_noGAN_aux)
        percentile95_across_time_noGAN.append(percentile95_across_time_noGAN_aux)
        rep_classes_time.append(rep_classes_time_aux)
        classes_time.append(classes_time_aux)
        GAN_distr.append(GAN_distr_aux)
        GAN_distr_noGAN.append(GAN_distr_noGAN_aux)
        classic_cross_entropy.append(classic_cross_entropy_aux)
        classic_cross_entropy_real.append(classic_cross_entropy_real_aux)
        classic_cross_entropy_noGAN.append(classic_cross_entropy_noGAN_aux)
        classic_cross_entropy_noGAN_real.append(classic_cross_entropy_noGAN_real_aux)

    # Plots across time for a fixed dataset dimension, comparison between different instances
    for ld in range(0,np.size(args.n_ld_dim)):
        # Mean and variance distribution
        fig,ax = plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(var_across_time[ld][gd][0,:,2], var_across_time[ld][gd][0,:,0], colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Variance', fontsize=15)
        plt.title('Evolution of variance across time for dataset dim = ' + str(args.n_ld_dim[ld]))
        if args.title == 'on':
            plt.legend(['Dataset', 'ld dim' + str(args.n_ld_dim[ld])])
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_var_for_dd_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_var_for_dd_dim_' + str(args.n_ld_dim[ld]) + '.' + 'png')

        # without GAN class
        # Mean + std
        fig, ax = plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            #ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0] - std_across_time_noGAN[ld][gd][0, :, 0], mean_across_time_noGAN[ld][gd][0, :, 0] + std_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd], alpha=0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean and std', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of mean and std across time for dataset dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['Mean', 'Std'])
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_std_for_dd_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_std_for_dd_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + 'png')

        # Mean + median
        fig, ax = plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], median_across_time_noGAN[ld][gd][0, :, 0],
                            colors_list[gd], alpha=0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Mean and median', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of mean and median across time for dataset dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['Mean', 'Median'])
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_median_for_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_median_for_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + 'png')

        # Mean + min/max
        fig, ax = plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            #ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2], min_across_time_noGAN[ld][gd][0, :, 0], max_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd], alpha=0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of mean across time for dataset dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['Mean', 'p50'])
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_min_max_for_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_min_max_for_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + 'png')

        # Mean + p5 - p95
        fig, ax = plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            #ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2], percentile5_across_time_noGAN[ld][gd][0, :, 0], percentile95_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd], alpha=0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of mean across time for dataset dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['Mean', 'p5-p95'])
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_p5p95_for_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_p5p95_for_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + 'png')

        # Var
        fig, ax = plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(var_across_time_noGAN[ld][gd][0, :, 2], var_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)')
        plt.ylabel('Evolution of variance')
        if args.title == 'on':
            plt.title('Evolution of variance across time for dataset dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list_instances)
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_var_for_dd_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_var_for_dd_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + 'png')

    plt.close('all')

    # Distribution of classes represented by the dataset
    for ld in range(0, np.size(args.n_ld_dim)):
        fig, ax = plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(rep_classes_time[ld][gd][0,:,2], rep_classes_time[ld][gd][0,:,0], colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Number of classes', fontsize=15)
        if args.title == 'on':
            plt.title('Number of represented classes across time for dataset dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list_instances)
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Distr_classes_across_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Distr_classes_across_dim_' + str(args.n_ld_dim[ld]) + '.' + 'png')

    # GAN class
    for gc in range(0,np.size(args.GAN_classes)):
        for ld in range(0, np.size(args.n_ld_dim)):
            fig, ax = plt.subplots()
            for gd in range(0, np.size(generation_data[ld])):
                plt.plot(rep_classes_time[ld][gd][0, :, 2], classes_time[ld][gd][0, :, int(GAN_class[gc])].T, colors_list[gd])
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            plt.xlabel('Time (in number of epochs)', fontsize=15)
            plt.ylabel('Number of syllable', fontsize=15)
            plt.legend(legend_list_instances)
            if args.title == 'on':
                plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class for dataset dim = ' + str(args.n_ld_dim[ld]))
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + classes[int(GAN_class[gc])] + '_class_across_time_and_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + classes[int(GAN_class[gc])] + '_class_across_time_and_dim_' + str(args.n_ld_dim[ld]) + '.' + 'png')

        for ld in range(0, np.size(args.n_ld_dim)):
            fig, ax = plt.subplots()
            for gd in range(0, np.size(generation_data[ld])):
                plt.plot(rep_classes_time[ld][gd][0, :, 2], (classes_time[ld][gd][0, :, int(GAN_class[gc])].T/1000)*100, colors_list[gd])
            plt.axhline(dataset_distr[ld][int(GAN_class[gc])], color='k')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_ylim([0,100])
            plt.xlabel('Time (in number of epochs)', fontsize=15)
            plt.ylabel('Percentage of syllables', fontsize=15)
            plt.legend(np.append(legend_list_instances, 'Training'))
            if args.title == 'on':
                plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class for dataset dim = ' + str(args.n_ld_dim[ld]))
            if args.format != 'png':
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_' + classes[int(GAN_class[gc])] + '_class_across_time_and_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_' + classes[int(GAN_class[gc])] + '_class_across_time_and_dim_' + str(args.n_ld_dim[ld]) + '.' + 'png')

    plt.close('all')

    # Cross-Entropy
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy[ld][gd][0,:], colors_list[gd])
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        if args.title == 'on':
            plt.title('Cross-entropy across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list_instances)
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_' + str(args.n_ld_dim[ld]) + '.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_' + str(args.n_ld_dim[ld]) + '.' + 'png')

    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_real[ld][gd][0,:], colors_list[gd])
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        if args.title == 'on':
            plt.title('Cross-entropy real across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list_instances)
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_real_' + str(args.n_ld_dim[ld]) + '.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_real_' + str(args.n_ld_dim[ld]) + '.' + 'png')

    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_noGAN[ld][gd][0,:], colors_list[gd])
        plt.axhline(data_entropy[ld], color='k')
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        if args.title == 'on':
            plt.title('Cross-entropy across time (without GAN class) for dataset dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(np.append(legend_list_instances, 'Training'))
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_' + str(args.n_ld_dim[ld]) + '.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_' + str(args.n_ld_dim[ld]) + '.' + 'png')

    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, np.size(generation_data[ld])):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_noGAN_real[ld][gd][0,:], colors_list[gd])
        plt.axhline(data_entropy_noGAN[ld], color='k')
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        if args.title == 'on':
            plt.title('Cross-entropy across time (without GAN class) for dataset dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(np.append(legend_list_instances, 'Training'))
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_real_' + str(args.n_ld_dim[ld]) + '.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_real_' + str(args.n_ld_dim[ld]) + '.' + 'png')

    plt.close('all')

    # Cumulative results with average of the instances
    avg_mean_across_time = []
    avg_std_across_time = []
    avg_var_across_time = []
    avg_mean_across_time_noGAN = []
    avg_std_across_time_noGAN = []
    avg_var_across_time_noGAN = []
    avg_median_across_time_noGAN = []
    avg_percentile5_across_time_noGAN = []
    avg_percentile95_across_time_noGAN = []
    avg_max_across_time_noGAN = []
    avg_min_across_time_noGAN = []
    avg_rep_classes_time = []
    avg_classes_time = []
    avg_GAN_distr_noGAN = []
    avg_classic_cross_entropy = []
    avg_classic_cross_entropy_real = []
    avg_classic_cross_entropy_noGAN = []
    avg_classic_cross_entropy_noGAN_real = []
    avg_classes_time_onlyGAN = []

    for ld in range(0, np.size(args.n_ld_dim)):
        avg_mean_aux = 0
        avg_std_aux = 0
        avg_var_aux = 0
        avg_mean_noGAN_aux = 0
        avg_std_noGAN_aux = 0
        avg_var_noGAN_aux = 0
        avg_median_across_time_noGAN_aux = 0
        avg_percentile5_across_time_noGAN_aux = 0
        avg_percentile95_across_time_noGAN_aux = 0
        avg_max_across_time_noGAN_aux = 0
        avg_min_across_time_noGAN_aux = 0
        avg_rep_classes_aux = 0
        avg_GAN_distr_noGAN_aux = 0
        avg_classic_cross_entropy_aux = 0
        avg_classic_cross_entropy_real_aux = 0
        avg_classic_cross_entropy_noGAN_aux = 0
        avg_classic_cross_entropy_noGAN_real_aux = 0

        for gd in range(0, np.size(generation_data[ld])):
            avg_mean_aux = avg_mean_aux + mean_across_time[ld][gd][0, :, 0]
            avg_mean_noGAN_aux = avg_mean_noGAN_aux + mean_across_time_noGAN[ld][gd][0, :, 0]
            avg_std_aux = avg_std_aux + std_across_time_noGAN[ld][gd][0, :, 0]
            avg_std_noGAN_aux = avg_std_noGAN_aux + std_across_time[ld][gd][0, :, 0]
            avg_var_aux = avg_var_aux + var_across_time[ld][gd][0, :, 0]
            avg_var_noGAN_aux = avg_var_noGAN_aux + var_across_time_noGAN[ld][gd][0, :, 0]
            avg_median_across_time_noGAN_aux = avg_median_across_time_noGAN_aux + median_across_time_noGAN[ld][gd][0, :, 0]
            avg_percentile5_across_time_noGAN_aux = avg_percentile5_across_time_noGAN_aux + percentile5_across_time_noGAN[ld][gd][0, :, 0]
            avg_percentile95_across_time_noGAN_aux = avg_percentile95_across_time_noGAN_aux + percentile95_across_time_noGAN[ld][gd][0, :, 0]
            avg_max_across_time_noGAN_aux = avg_max_across_time_noGAN_aux + max_across_time_noGAN[ld][gd][0, :, 0]
            avg_min_across_time_noGAN_aux = avg_min_across_time_noGAN_aux + min_across_time_noGAN[ld][gd][0, :, 0]
            avg_rep_classes_aux = avg_rep_classes_aux + rep_classes_time[ld][gd][0, :, 0]
            avg_GAN_distr_noGAN_aux = avg_GAN_distr_noGAN_aux + mean_across_time[ld][gd][0, :, 0]
            avg_classic_cross_entropy_aux = avg_classic_cross_entropy_aux + classic_cross_entropy[ld][gd][0, :]
            avg_classic_cross_entropy_real_aux = avg_classic_cross_entropy_real_aux + classic_cross_entropy_real[ld][gd][0, :]
            avg_classic_cross_entropy_noGAN_aux = avg_classic_cross_entropy_noGAN_aux + classic_cross_entropy_noGAN[ld][gd][0, :]
            avg_classic_cross_entropy_noGAN_real_aux = avg_classic_cross_entropy_noGAN_real_aux + classic_cross_entropy_noGAN_real[ld][gd][0, :]

        avg_mean_across_time.append(avg_mean_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_std_across_time.append(avg_std_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_var_across_time.append(avg_var_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_mean_across_time_noGAN.append(avg_mean_noGAN_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_std_across_time_noGAN.append(avg_std_noGAN_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_var_across_time_noGAN.append(avg_var_noGAN_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_median_across_time_noGAN.append(avg_median_across_time_noGAN_aux/(np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_percentile5_across_time_noGAN.append(avg_percentile5_across_time_noGAN_aux/(np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_percentile95_across_time_noGAN.append(avg_percentile95_across_time_noGAN_aux/(np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_max_across_time_noGAN.append(avg_max_across_time_noGAN_aux/(np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_min_across_time_noGAN.append(avg_min_across_time_noGAN_aux/(np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_rep_classes_time.append(avg_rep_classes_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_GAN_distr_noGAN.append(avg_GAN_distr_noGAN_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_classic_cross_entropy.append(
            avg_classic_cross_entropy_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_classic_cross_entropy_real.append(
            avg_classic_cross_entropy_real_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_classic_cross_entropy_noGAN.append(
            avg_classic_cross_entropy_noGAN_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))
        avg_classic_cross_entropy_noGAN_real.append(
            avg_classic_cross_entropy_noGAN_real_aux / (np.size(generation_data) / np.size(args.n_ld_dim)))

        avg_classes_time_onlyGAN_aux = np.zeros((np.size(GAN_class), np.size(rep_classes_time[ld][0][0, :, 2])))
        for gc in range(0, np.size(GAN_class)):
            aux = []
            for gd in range(0, np.size(legend_list_instances)):
                aux.append(classes_time[ld][gd][0, :, int(GAN_class[gc])].T)
            avg_classes_time_onlyGAN_aux[gc, :] = np.mean(aux, axis=0)
        avg_classes_time_onlyGAN.append(avg_classes_time_onlyGAN_aux)

        avg_classes_time_aux = np.zeros((np.size(args.GAN_classes), np.size(rep_classes_time[ld][0][0, :, 2])))
        for gc in range(0, np.size(args.GAN_classes)):
            aux = []
            for gd in range(0, np.size(generation_data[ld])):
                aux.append(classes_time[ld][gd][0, :, int(GAN_class[gc])])
            avg_classes_time_aux[gc, :] = np.mean(aux, axis=0)
        avg_classes_time.append(avg_classes_time_aux)

    # Distribution of classes represented by the dataset
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(rep_classes_time[ld][0][0, :, 2], avg_rep_classes_time[ld], colors_list[ld])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Number of classes', fontsize=15)
    if args.title == 'on':
        plt.title('Number of represented classes across time')
    plt.legend(legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_dstr_classes_across_dim_.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_dstr_classes_across_dim_.' + 'png')

    # Mean and std across time
    # with GAN class
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(mean_across_time[ld][0][0, :, 2], avg_mean_across_time[ld], colors_list[ld])
        #ax.fill_between(mean_across_time[ld][0][0, :, 2], avg_mean_across_time[ld] - avg_std_across_time[ld], avg_mean_across_time[ld] + avg_std_across_time[ld], colors_list[ld], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean and std', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and std across dimension')
    plt.legend(legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_std_across_dim.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_std_across_dim.' + 'png')

    # without GAN class
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_mean_across_time_noGAN[ld], colors_list[ld])
        #ax.fill_between(mean_across_time_noGAN[ld][0][0, :, 2], avg_mean_across_time_noGAN[ld] - avg_std_across_time_noGAN[ld], avg_mean_across_time_noGAN[ld] + avg_std_across_time_noGAN[ld], colors_list[ld], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean and std', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and std across dimension')
    plt.legend(legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_std_across_dim_noGAN.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_std_across_dim_noGAN.' + 'png')

    legend_aux = []
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plot_aux, = plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_mean_across_time_noGAN[ld], colors_list[ld])
        plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_median_across_time_noGAN[ld], colors_list[ld], alpha=0.3)
        legend_aux.append(plot_aux)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Mean and median', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and median across dimension')
    plt.legend(handles = legend_aux, labels = legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_median_across_dim_noGAN.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_median_across_dim_noGAN.' + 'png')

    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plot_aux, = plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_median_across_time_noGAN[ld], colors_list[ld])
        legend_aux.append(plot_aux)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Median', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and median across dimension')
    plt.legend(handles = legend_aux, labels = legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_median_across_dim_noGAN.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_median_across_dim_noGAN.' + 'png')

    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_mean_across_time_noGAN[ld], colors_list[ld])
        #ax.fill_between(mean_across_time_noGAN[ld][0][0, :, 2], avg_min_across_time_noGAN[ld], avg_max_across_time_noGAN[ld], colors_list[ld], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and min/max across dimension')
    plt.legend(legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_min_max_across_dim_noGAN.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_min_max_across_dim_noGAN.' + 'png')

    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_mean_across_time_noGAN[ld], colors_list[ld])
        #ax.fill_between(mean_across_time_noGAN[ld][0][0, :, 2], avg_percentile5_across_time_noGAN[ld], avg_percentile95_across_time_noGAN[ld], colors_list[ld], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and p5/p95 across dimension')
    plt.legend(legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_p5p95_across_dim_noGAN.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_p5p95_across_dim_noGAN.' + 'png')

    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(median_across_time_noGAN[ld][0][0, :, 2], avg_median_across_time_noGAN[ld], colors_list[ld])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Median', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of median across time')
    plt.legend(legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_median_across_dim_noGAN.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_median_across_dim_noGAN.' + 'png')

    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(var_across_time_noGAN[ld][0][0, :, 2], avg_var_across_time_noGAN[ld], colors_list[ld])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)')
    plt.ylabel('Variance')
    if args.title == 'on':
        plt.title('Evolution of variance across time')
    plt.legend(legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_var_across_dim_noGAN.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_var_across_dim_noGAN.' + 'png')

    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(var_across_time[ld][0][0, :, 2], avg_var_across_time[ld], colors_list[ld])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Variance', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of variance across time')
    plt.legend(legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_var_across_dim.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_var_across_dim.' + 'png')

    # GAN class
    for gc in range(0, np.size(args.GAN_classes)):
        fig, ax = plt.subplots()
        for ld in range(0, np.size(args.n_ld_dim)):
            plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classes_time[ld][gc, :].T, colors_list[ld])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Number of syllable', fontsize=15)
        plt.legend(legend_list)
        if args.title == 'on':
            plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class across dim')
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + classes[int(GAN_class[gc])] + '_avg_class_across_time_and_dim.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + classes[int(GAN_class[gc])] + '_avg_class_across_time_and_dim.' + 'png')

        fig, ax = plt.subplots()
        for ld in range(0, np.size(args.n_ld_dim)):
            plt.plot(rep_classes_time[ld][0][0, :, 2], (avg_classes_time[ld][gc, :].T / 1000)*100, colors_list[ld])
        plt.axhline(dataset_distr[ld][int(GAN_class[gc])], color='k')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_ylim([0,100])
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Percentage of syllable', fontsize=15)
        plt.legend(np.append(legend_list, 'Training'))
        if args.title == 'on':
            plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class across dim')
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_' + classes[int(GAN_class[gc])] + '_avg_class_across_time_and_dim.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_' + classes[int(GAN_class[gc])] + '_avg_class_across_time_and_dim.' + 'png')

    # Cumulative GAN classes
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        sum = 0
        for gc in range(0, np.size(args.GAN_classes)):
            sum = sum + avg_classes_time_onlyGAN[ld][gc, :].T / 1000

        plt.plot(rep_classes_time[ld][0][0, :, 2], sum * 100,
                 colors_list[ld])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_ylim([0, 100])
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Percentage of syllable', fontsize=15)
    plt.legend(np.append(legend_list, 'Training'))
    if args.title == 'on':
        plt.title('Number of syllable in ALL class across dim')
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_ALL_avg_class_across_time_and_dim.' + args.format)

    # Cross-Entropy
    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classic_cross_entropy[ld], colors_list[ld])
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    if args.title == 'on':
        plt.title('Cross-entropy across time')
    plt.legend(legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy.' + 'png')

    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classic_cross_entropy_real[ld], colors_list[ld])
    plt.axhline(data_entropy[ld], color='k')
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    if args.title == 'on':
        plt.title('Cross-entropy real across time')
    plt.legend(np.append(legend_list, 'Training'))
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy_real.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy_real.' + 'png')

    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classic_cross_entropy_noGAN[ld], colors_list[ld])
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    if args.title == 'on':
        plt.title('Cross-entropy across time (without GAN class)')
    plt.legend(legend_list)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy_noGAN.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy_noGAN.' + 'png')

    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classic_cross_entropy_noGAN_real[ld], colors_list[ld])
    plt.axhline(data_entropy_noGAN[ld], color='k')
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    if args.title == 'on':
        plt.title('Cross-entropy across time (without GAN class)')
    plt.legend(np.append(legend_list, 'Training'))
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy_noGAN_real.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy_noGAN_real.' + 'png')

    plt.close('all')

    print('Done')

def several_instances(generation_data, summary_dataset, legend_list_instances, legend_list_avg, colors_list, classes_colors, args):
    """
    This function allows to compare several instances between them and to visualize an average plot
    :param generation_data: summary of the generations (one per training)
    :param summary_dataset: to get the distribution of the training data
    :param legend_list_avg: name of latent space dim if multiple
    :param legend_list_instances: name of the instances
    :param: colors_list : to have uniform plots across different analysis (for the instances)
    :param classes_colors: to have uniform plots across different analysis (for the classes)

    :return: comparison figures
    """
    load_data = np.load(generation_data[0][0], allow_pickle=True)
    load_data = load_data.item()

    classes = load_data['classes']

    GAN_class = np.zeros((np.size(args.GAN_classes),))
    for gc in range(0, np.size(args.GAN_classes)):
        GAN_class[gc] = np.where(classes == args.GAN_classes[gc])[0][0]

    # DATASET
    load_data = np.load(summary_dataset[0],allow_pickle=True)
    load_data = load_data.item()
    dataset_distr = load_data['Dataset_distr']
    dataset_distr_real = load_data['Dataset_real_distr']
    data_entropy = stat.cross_entropy(dataset_distr_real[0],dataset_distr)
    data_entropy_noGAN = stat.cross_entropy(np.delete(dataset_distr_real[0],GAN_class),np.delete(dataset_distr,GAN_class))

    # GENERATED DATA
    mean_across_time = []
    std_across_time = []
    var_across_time = []
    mean_across_time_noGAN = []
    std_across_time_noGAN = []
    var_across_time_noGAN = []
    median_across_time_noGAN = []
    percentile5_across_time_noGAN = []
    percentile95_across_time_noGAN = []
    max_across_time_noGAN = []
    min_across_time_noGAN = []
    rep_classes_time = []
    classes_time = []
    GAN_distr = []
    GAN_distr_noGAN = []
    classic_cross_entropy = []
    classic_cross_entropy_real = []
    classic_cross_entropy_noGAN = []
    instances = []
    classic_cross_entropy_noGAN_real = []

    for ld in range(0,np.size(args.n_ld_dim)):
        mean_across_time_aux = []
        std_across_time_aux = []
        var_across_time_aux = []
        mean_across_time_noGAN_aux = []
        std_across_time_noGAN_aux = []
        var_across_time_noGAN_aux = []
        median_across_time_noGAN_aux = []
        percentile5_across_time_noGAN_aux = []
        percentile95_across_time_noGAN_aux = []
        max_across_time_noGAN_aux = []
        min_across_time_noGAN_aux = []
        rep_classes_time_aux = []
        classes_time_aux = []
        GAN_distr_aux = []
        GAN_distr_noGAN_aux = []
        classic_cross_entropy_aux = []
        classic_cross_entropy_real_aux = []
        classic_cross_entropy_noGAN_aux = []
        classic_cross_entropy_noGAN_real_aux = []

        instances_aux = np.zeros((np.size(generation_data),))
        for gd in range(0,np.size(generation_data[ld])):
            load_data = np.load(generation_data[ld][gd], allow_pickle=True)
            load_data = load_data.item()
            mean_across_time_aux.append(load_data['mean_across_time'])
            std_across_time_aux.append(load_data['std_across_time'])
            var_across_time_aux.append(load_data['var_across_time'])
            median_across_time_noGAN_aux.append(load_data['median_across_time_noGAN'])
            percentile5_across_time_noGAN_aux.append(load_data['percentile5_across_time_noGAN'])
            percentile95_across_time_noGAN_aux.append(load_data['percentile95_across_time_noGAN'])
            max_across_time_noGAN_aux.append(load_data['max_across_time_noGAN'])
            min_across_time_noGAN_aux.append(load_data['min_across_time_noGAN'])
            mean_across_time_noGAN_aux.append(load_data['mean_across_time_noGAN'])
            std_across_time_noGAN_aux.append(load_data['std_across_time_noGAN'])
            rep_classes_time_aux.append(load_data['rep_classes_time'])
            var_across_time_noGAN_aux.append(load_data['var_across_time_noGAN'])
            GAN_distr_noGAN_aux.append(load_data['GAN_distr_noGAN'])
            classes_time_aux.append(load_data['classes_time'])
            GAN_distr_aux.append(load_data['GAN_distr'])
            classic_cross_entropy_aux.append(load_data['cross_entropy'])
            classic_cross_entropy_real_aux.append(load_data['cross_entropy_real'])
            classic_cross_entropy_noGAN_aux.append(load_data['cross_entropy_noGAN'])
            classic_cross_entropy_noGAN_real_aux.append(load_data['cross_entropy_noGAN_real'])

            instances_aux[gd] = generation_data[ld][gd][generation_data[ld][gd].find('Generation_summary_i') + len('Generation_summary_i')]

        mean_across_time.append(mean_across_time_aux)
        std_across_time.append(std_across_time_aux)
        var_across_time.append(var_across_time_aux)
        mean_across_time_noGAN.append(mean_across_time_noGAN_aux)
        std_across_time_noGAN.append(std_across_time_noGAN_aux)
        var_across_time_noGAN.append(var_across_time_noGAN_aux)
        median_across_time_noGAN.append(median_across_time_noGAN_aux)
        percentile5_across_time_noGAN.append(percentile5_across_time_noGAN_aux)
        percentile95_across_time_noGAN.append(percentile95_across_time_noGAN_aux)
        max_across_time_noGAN.append(max_across_time_noGAN_aux)
        min_across_time_noGAN.append(min_across_time_noGAN_aux)
        rep_classes_time.append(rep_classes_time_aux)
        classes_time.append(classes_time_aux)
        GAN_distr.append(GAN_distr_aux)
        GAN_distr_noGAN.append(GAN_distr_noGAN_aux)
        classic_cross_entropy.append(classic_cross_entropy_aux)
        classic_cross_entropy_real.append(classic_cross_entropy_real_aux)
        classic_cross_entropy_noGAN.append(classic_cross_entropy_noGAN_aux)
        classic_cross_entropy_noGAN_real.append(classic_cross_entropy_noGAN_real_aux)
        instances.append(instances_aux)

    # All the instances all the dim together
    # legend_aux = []
    # plt.subplots()
    # for ld in range(0, np.size(args.n_ld_dim)):
    #     for gd in range(0, np.size(legend_list_instances)):
    #         plot_aux, = plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_noGAN_real[ld][gd][0,:], classes_colors[ld])
    #     legend_aux.append(plot_aux)
    # plot_aux = plt.axhline(data_entropy, color='k')
    # legend_aux.append(plot_aux)
    # plt.ylabel('Cross entropy', fontsize=15)
    # plt.xlabel('Time (in epochs)', fontsize=15)
    # if args.title == 'on':
    #     plt.title('Cross-entropy across time (without GAN class)')
    # plt.legend(handles = legend_aux, labels = list(np.append(legend_list_avg, 'Training')))
    # plt.tight_layout()
    # if args.format != 'png':
    #     plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_real_ALL.' + args.format)
    # plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_real_ALL.' + 'png')

    # plt.subplots()
    # for ld in range(0, np.size(args.n_ld_dim)):
    #     for gd in range(0, np.size(legend_list_instances)):
    #         plot_aux, = plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_real[ld][gd][0,:], colors_list[ld])
    #     legend_aux.append(plot_aux)
    # plot_aux = plt.axhline(data_entropy, color='k')
    # legend_aux.append(plot_aux)
    # plt.ylabel('Cross entropy', fontsize=15)
    # plt.xlabel('Time (in epochs)', fontsize=15)
    # if args.title == 'on':
    #     plt.title('Cross-entropy across time')
    # plt.legend(handles = legend_aux, labels = list(np.append(legend_list_avg, 'Training')))
    # plt.tight_layout()
    # if args.format != 'png':
    #     plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_real_ALL.' + args.format)
    # plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_real_ALL.' + 'png')
    #
    # plt.close('all')

    # Plots across time for a fixed latent space dimension, comparison between different instances
    for ld in range(0,np.size(args.n_ld_dim)):
        # Mean + median
        legend_aux = []
        fig, ax = plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plot_aux, = plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], median_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd], alpha=0.3)
            legend_aux.append(plot_aux)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Mean and median', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of mean and median across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(handles = legend_aux, labels = legend_list_instances)
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_median_for_dim_' + str(
            args.n_ld_dim[ld]) + '_noGAN.' + args.format)

        # Variance
        fig,ax = plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plt.plot(var_across_time[ld][gd][0,:,2], var_across_time[ld][gd][0,:,0], colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Variance', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of variance across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(handles = legend_aux, labels = legend_list_instances)
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_var_for_ld_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)

        # Mean
        legend_aux = []
        fig, ax = plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plot_aux, = plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            legend_aux.append(plot_aux)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of mean across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(handles = legend_aux, labels = legend_list_instances)
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_for_dim_' + str(
            args.n_ld_dim[ld]) + '_noGAN.' + args.format)

        # Median
        legend_aux = []
        fig, ax = plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plot_aux, = plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], median_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            legend_aux.append(plot_aux)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Median', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of median across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(handles = legend_aux, labels = legend_list_instances)
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_median_for_dim_' + str(
            args.n_ld_dim[ld]) + '_noGAN.' + args.format)

        # Mean + min/max
        fig, ax = plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            #ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2], min_across_time_noGAN[ld][gd][0, :, 0], max_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd], alpha=0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of mean across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['Mean', 'p50'])
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_min_max_for_dim_' + str(
            args.n_ld_dim[ld]) + '_noGAN.' + args.format)

        # Mean + p5 - p95
        fig, ax = plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plt.plot(mean_across_time_noGAN[ld][gd][0, :, 2], mean_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
            #ax.fill_between(mean_across_time_noGAN[ld][gd][0, :, 2], percentile5_across_time_noGAN[ld][gd][0, :, 0], percentile95_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd], alpha=0.3)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Evolution of mean', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of mean across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(['Mean', 'p5-p95'])
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_mean_p5p95_for_dim_' + str(
            args.n_ld_dim[ld]) + '_noGAN.' + args.format)

        # Var
        fig, ax = plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plt.plot(var_across_time_noGAN[ld][gd][0, :, 2], var_across_time_noGAN[ld][gd][0, :, 0], colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Variance', fontsize=15)
        if args.title == 'on':
            plt.title('Evolution of variance across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(handles = legend_aux, labels = legend_list_instances)
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Evolution_var_for_ld_dim_' + str(args.n_ld_dim[ld]) + '_noGAN.' + args.format)

    plt.close('all')

    # Distribution of classes represented by the dataset
    for ld in range(0, np.size(args.n_ld_dim)):
        fig, ax = plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plt.plot(rep_classes_time[ld][gd][0,:,2], rep_classes_time[ld][gd][0,:,0], colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Number of classes', fontsize=15)
        if args.title == 'on':
            plt.title('Number of represented classes across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list_instances)
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Distr_classes_across_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    # GAN class
    for gc in range(0,np.size(args.GAN_classes)):
        for ld in range(0, np.size(args.n_ld_dim)):
            fig, ax = plt.subplots()
            for gd in range(0, np.size(legend_list_instances)):
                plt.plot(rep_classes_time[ld][gd][0, :, 2], classes_time[ld][gd][0, :, int(GAN_class[gc])].T, colors_list[gd])
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            plt.xlabel('Time (in number of epochs)', fontsize=15)
            plt.ylabel('Number of syllable', fontsize=15)
            plt.legend(legend_list_instances)
            if args.title == 'on':
                plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class for latent dim = ' + str(args.n_ld_dim[ld]))
            plt.tight_layout()
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + classes[int(GAN_class[gc])] + '_class_across_time_and_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)

        for ld in range(0, np.size(args.n_ld_dim)):
            fig, ax = plt.subplots()
            for gd in range(0, np.size(legend_list_instances)):
                plt.plot(rep_classes_time[ld][gd][0, :, 2], (classes_time[ld][gd][0, :, int(GAN_class[gc])].T/1000)*100, colors_list[gd])
            plt.axhline(dataset_distr[int(GAN_class[gc])], color='k')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_ylim([0,100])
            plt.xlabel('Time (in number of epochs)', fontsize=15)
            plt.ylabel('Percentage of syllable', fontsize=15)
            plt.legend(np.append(legend_list_instances, 'Training'))
            if args.title == 'on':
                plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class for latent dim = ' + str(args.n_ld_dim[ld]))
            plt.tight_layout()
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_' + classes[int(GAN_class[gc])] + '_class_across_time_and_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    plt.close('all')

    # Cumulative GAN classes
    for ld in range(0, np.size(args.n_ld_dim)):
        fig, ax = plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            sum = 0
            for gc in range(0, np.size(args.GAN_classes)):
                sum = sum + classes_time[ld][gd][0, :, int(GAN_class[gc])].T/1000

            plt.plot(rep_classes_time[ld][0][0, :, 2], sum * 100,colors_list[gd])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_ylim([0, 100])
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Percentage of syllable', fontsize=15)
        plt.legend(np.append(legend_list_instances, 'Training'))
        if args.title == 'on':
            plt.title('Number of syllable in ALL class across dim')
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_ALL_class_across_time_and_dim_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    # Cross-Entropy
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy[ld][gd][0,:], colors_list[gd])
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        if args.title == 'on':
            plt.title('Cross-entropy across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list_instances)
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_real[ld][gd][0,:], colors_list[gd])
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        if args.title == 'on':
            plt.title('Cross-entropy real across time for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list_instances)
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_real_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_noGAN[ld][gd][0,:], colors_list[gd])
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        if args.title == 'on':
            plt.title('Cross-entropy across time (without GAN class) for latent dim = ' + str(args.n_ld_dim[ld]))
        plt.legend(legend_list_instances)
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_' + str(args.n_ld_dim[ld]) + '.' + args.format)

    for ld in range(0, np.size(args.n_ld_dim)):
        plt.subplots()
        for gd in range(0, np.size(legend_list_instances)):
            plt.plot(rep_classes_time[ld][gd][0,:,2], classic_cross_entropy_noGAN_real[ld][gd][0,:], colors_list[gd])
        plt.axhline(data_entropy, color='k')
        plt.ylabel('Cross entropy', fontsize=15)
        plt.xlabel('Time (in epochs)', fontsize=15)
        if args.title == 'on':
            plt.title('Cross-entropy across time (without GAN class) for latent dim = ' + str(args.n_ld_dim[ld]))
        #plt.legend(np.append(legend_list_avg, 'Training'))
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Cross_entropy_noGAN_real' + str(args.n_ld_dim[ld]) + '.' + args.format)

    plt.close('all')

    # Cumulative results with average of the instances
    avg_mean_across_time = []
    avg_std_across_time = []
    avg_var_across_time = []
    avg_mean_across_time_noGAN = []
    avg_std_across_time_noGAN = []
    avg_var_across_time_noGAN = []
    avg_median_across_time_noGAN = []
    avg_percentile5_across_time_noGAN = []
    avg_percentile95_across_time_noGAN = []
    avg_max_across_time_noGAN = []
    avg_min_across_time_noGAN = []
    avg_rep_classes_time = []
    avg_classes_time = []
    avg_classes_time_onlyGAN = []
    avg_GAN_distr_noGAN = []
    avg_classic_cross_entropy = []
    avg_classic_cross_entropy_real = []
    avg_classic_cross_entropy_noGAN = []
    avg_classic_cross_entropy_noGAN_real = []

    for ld in range(0, np.size(args.n_ld_dim)):
        avg_mean_aux = 0
        avg_std_aux = 0
        avg_var_aux = 0
        avg_mean_noGAN_aux = 0
        avg_std_noGAN_aux = 0
        avg_var_noGAN_aux = 0
        avg_median_across_time_noGAN_aux = 0
        avg_percentile5_across_time_noGAN_aux = 0
        avg_percentile95_across_time_noGAN_aux = 0
        avg_max_across_time_noGAN_aux = 0
        avg_min_across_time_noGAN_aux = 0
        avg_rep_classes_aux = 0
        avg_GAN_distr_noGAN_aux = 0
        avg_classic_cross_entropy_aux = 0
        avg_classic_cross_entropy_real_aux = 0
        avg_classic_cross_entropy_noGAN_aux = 0
        avg_classic_cross_entropy_noGAN_real_aux = 0

        for gd in range(0, np.size(legend_list_instances)):
            avg_mean_aux = avg_mean_aux + mean_across_time[ld][gd][0, :, 0]
            avg_mean_noGAN_aux = avg_mean_noGAN_aux + mean_across_time_noGAN[ld][gd][0, :, 0]
            avg_std_aux = avg_std_aux + std_across_time_noGAN[ld][gd][0, :, 0]
            avg_std_noGAN_aux = avg_std_noGAN_aux + std_across_time[ld][gd][0, :, 0]
            avg_var_aux = avg_var_aux + var_across_time[ld][gd][0, :, 0]
            avg_var_noGAN_aux = avg_var_noGAN_aux + var_across_time_noGAN[ld][gd][0, :, 0]
            avg_median_across_time_noGAN_aux = avg_median_across_time_noGAN_aux + median_across_time_noGAN[ld][gd][0, :, 0]
            avg_percentile5_across_time_noGAN_aux = avg_percentile5_across_time_noGAN_aux + percentile5_across_time_noGAN[ld][gd][0, :, 0]
            avg_percentile95_across_time_noGAN_aux = avg_percentile95_across_time_noGAN_aux + percentile95_across_time_noGAN[ld][gd][0, :, 0]
            avg_max_across_time_noGAN_aux = avg_max_across_time_noGAN_aux + max_across_time_noGAN[ld][gd][0, :, 0]
            avg_min_across_time_noGAN_aux = avg_min_across_time_noGAN_aux + min_across_time_noGAN[ld][gd][0, :, 0]
            avg_rep_classes_aux = avg_rep_classes_aux + rep_classes_time[ld][gd][0, :, 0]
            avg_GAN_distr_noGAN_aux = avg_GAN_distr_noGAN_aux + mean_across_time[ld][gd][0, :, 0]
            avg_classic_cross_entropy_aux = avg_classic_cross_entropy_aux + classic_cross_entropy[ld][gd][0, :]
            avg_classic_cross_entropy_real_aux = avg_classic_cross_entropy_real_aux + classic_cross_entropy_real[ld][gd][0, :]
            avg_classic_cross_entropy_noGAN_aux = avg_classic_cross_entropy_noGAN_aux + classic_cross_entropy_noGAN[ld][gd][0, :]
            avg_classic_cross_entropy_noGAN_real_aux = avg_classic_cross_entropy_noGAN_real_aux + classic_cross_entropy_noGAN_real[ld][gd][0, :]

        avg_mean_across_time.append(avg_mean_aux/args.n_instances)
        avg_std_across_time.append(avg_std_aux/args.n_instances)
        avg_var_across_time.append(avg_var_aux/args.n_instances)
        avg_mean_across_time_noGAN.append(avg_mean_noGAN_aux/args.n_instances)
        avg_std_across_time_noGAN.append(avg_std_noGAN_aux/args.n_instances)
        avg_var_across_time_noGAN.append(avg_var_noGAN_aux/args.n_instances)
        avg_median_across_time_noGAN.append(avg_median_across_time_noGAN_aux/args.n_instances)
        avg_percentile5_across_time_noGAN.append(avg_percentile5_across_time_noGAN_aux/args.n_instances)
        avg_percentile95_across_time_noGAN.append(avg_percentile95_across_time_noGAN_aux/args.n_instances)
        avg_max_across_time_noGAN.append(avg_max_across_time_noGAN_aux/args.n_instances)
        avg_min_across_time_noGAN.append(avg_min_across_time_noGAN_aux/args.n_instances)
        avg_rep_classes_time.append(avg_rep_classes_aux/args.n_instances)
        avg_GAN_distr_noGAN.append(avg_GAN_distr_noGAN_aux/args.n_instances)
        avg_classic_cross_entropy.append(avg_classic_cross_entropy_aux/args.n_instances)
        avg_classic_cross_entropy_real.append(avg_classic_cross_entropy_real_aux/args.n_instances)
        avg_classic_cross_entropy_noGAN.append(avg_classic_cross_entropy_noGAN_aux/args.n_instances)
        avg_classic_cross_entropy_noGAN_real.append(avg_classic_cross_entropy_noGAN_real_aux/args.n_instances)


        avg_classes_time_onlyGAN_aux = np.zeros((np.size(GAN_class), np.size(rep_classes_time[ld][0][0, :, 2])))
        for gc in range(0, np.size(GAN_class)):
            aux = []
            for gd in range(0, np.size(legend_list_instances)):
                aux.append(classes_time[ld][gd][0, :, int(GAN_class[gc])].T)
            avg_classes_time_onlyGAN_aux[gc, :] = np.mean(aux, axis = 0)
        avg_classes_time_onlyGAN.append(avg_classes_time_onlyGAN_aux)

        avg_classes_time_aux = np.zeros((np.size(classes), np.size(rep_classes_time[ld][0][0, :, 2])))
        for gc in range(0, np.size(classes)):
            aux = []
            for gd in range(0, np.size(legend_list_instances)):
                aux.append(classes_time[ld][gd][0, :, gc])
            avg_classes_time_aux[gc, :] = np.mean(aux, axis=0)
        avg_classes_time.append(avg_classes_time_aux)

    # Distribution of classes represented by the dataset
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(rep_classes_time[ld][0][0,:,2], avg_rep_classes_time[ld], colors_list[ld])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Number of classes', fontsize=15)
    if args.title == 'on':
        plt.title('Number of represented classes across time')
    plt.legend(legend_list_avg)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_distr_classes_across_dim_.' + args.format)

    # Mean and std across time
    # with GAN class
    fig, ax = plt.subplots()
    for ld in range(0,np.size(args.n_ld_dim)):
        plt.plot(mean_across_time[ld][0][0, :, 2], avg_mean_across_time[ld], colors_list[ld])
        #ax.fill_between(mean_across_time[ld][0][0, :, 2], avg_mean_across_time[ld] - avg_std_across_time[ld], avg_mean_across_time[ld] + avg_std_across_time[ld], colors_list[ld], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean and std', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and std across dimension')
    plt.legend(legend_list_avg)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_std_across_dim.' + args.format)

    # without GAN class
    fig, ax = plt.subplots()
    for ld in range(0,np.size(args.n_ld_dim)):
        plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_mean_across_time_noGAN[ld], colors_list[ld])
        #ax.fill_between(mean_across_time_noGAN[ld][0][0, :, 2], avg_mean_across_time_noGAN[ld] - avg_std_across_time_noGAN[ld], vg_mean_across_time_noGAN[ld] + avg_std_across_time_noGAN[ld], colors_list[ld], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean and std', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and std across dimension')
    plt.legend(legend_list_avg)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_std_across_dim_noGAN.' + args.format)

    legend_aux = []
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plot_aux, = plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_mean_across_time_noGAN[ld], colors_list[ld])
        plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_median_across_time_noGAN[ld], colors_list[ld], alpha=0.3)
        legend_aux.append(plot_aux)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Mean and median', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and median across dimension')
    plt.legend(handles = legend_aux, labels = legend_list_avg)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_median_across_dim_noGAN.' + args.format)

    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plot_aux, = plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_median_across_time_noGAN[ld], colors_list[ld])
        legend_aux.append(plot_aux)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Median and median', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and median across dimension')
    plt.legend(handles = legend_aux, labels = legend_list_avg)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_median_across_dim_noGAN.' + args.format)

    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_mean_across_time_noGAN[ld], colors_list[ld])
        #ax.fill_between(mean_across_time_noGAN[ld][0][0, :, 2], avg_min_across_time_noGAN[ld], avg_max_across_time_noGAN[ld], colors_list[ld], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and min/max across dimension')
    plt.legend(legend_list_avg)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_minmax_across_dim_noGAN.' + args.format)

    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(mean_across_time_noGAN[ld][0][0, :, 2], avg_mean_across_time_noGAN[ld], colors_list[ld])
        #ax.fill_between(mean_across_time_noGAN[ld][0][0, :, 2], avg_percentile5_across_time_noGAN[ld], avg_percentile95_across_time_noGAN[ld], colors_list[ld], alpha=0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Evolution of mean', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of mean and p5/p95 across dimension')
    plt.legend(legend_list_avg)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_mean_p5p95_across_dim_noGAN.' + args.format)

    fig, ax =plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(var_across_time_noGAN[ld][0][0, :, 2], avg_var_across_time_noGAN[ld], colors_list[ld])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Variance', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of variance across time')
    plt.legend(handles = legend_aux, labels = legend_list_instances)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_var_across_dim_noGAN.' + args.format)

    fig, ax =plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(var_across_time[ld][0][0, :, 2], avg_var_across_time[ld], colors_list[ld])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Variance', fontsize=15)
    if args.title == 'on':
        plt.title('Evolution of variance across time')
    plt.legend(handles = legend_aux, labels = legend_list_instances)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_var_across_dim.' + args.format)

    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(median_across_time_noGAN[ld][0][0, :, 2], avg_median_across_time_noGAN[ld], colors_list[ld])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)')
    plt.ylabel('Median')
    if args.title == 'on':
        plt.title('Evolution of median across time')
    plt.legend(handles=legend_aux, labels=legend_list_instances)
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_median_across_dim_noGAN.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_evolution_media_across_dim_noGAN.' + 'png')

    # ALL the classes
    fig, ax = plt.subplots()
    for gc in range(0, np.size(classes)):
        for ld in range(0, np.size(args.n_ld_dim)):
            plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classes_time[ld][gc, :].T, classes_colors[gc])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Number of syllable', fontsize=15)
    plt.legend(legend_list_avg)
    if args.title == 'on':
        plt.title('Number of syllable per class across dim')
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'avg_class_across_time_and_dim.' + args.format)

    fig, ax = plt.subplots()
    for gc in range(0, np.size(classes)):
        for ld in range(0, np.size(args.n_ld_dim)):
            plt.plot(rep_classes_time[ld][0][0, :, 2], (avg_classes_time[ld][gc, :].T / 1000) * 100, classes_colors[gc])
        plt.axhline(dataset_distr[gc], color='k')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Percentage of syllable', fontsize=15)
    plt.legend(np.append(legend_list_avg, 'Training'))
    if args.title == 'on':
        plt.title('Number of syllable per class across dim')
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_class_across_time_and_dim.' + args.format)

    # ALL the classes NOGAN
    fig, ax = plt.subplots()
    for gc in range(0, np.size(classes)):
        for ld in range(0, np.size(args.n_ld_dim)):
            if np.size(GAN_class) == 5:
                if gc == 6 or gc == 7 or gc == 8 or gc == 16 or gc == 20:
                    pass
                else:
                    plt.plot(rep_classes_time[ld][0][0, :, 2], (avg_classes_time[ld][gc, :].T / 1000)*100, classes_colors[gc])
            if np.size(GAN_class) == 10:
                if gc == 6 or gc == 7 or gc == 8 or gc == 9 or gc == 10 or gc == 11 or gc == 12 or gc == 13 or gc == 21 or gc == 25:
                    pass
                else:
                    plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classes_time[ld][gc, :].T / 1000, classes_colors[gc])
            elif np.size(GAN_class) == 0:
                plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classes_time[ld][gc, :].T, classes_colors[gc])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Number of syllable (in percentage)', fontsize=15)
    plt.legend(['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V'])
    if args.title == 'on':
        plt.title('Number of syllable per class across dim')
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'avg_class_NOGAN_across_time_and_dim.' + args.format)

    fig, ax = plt.subplots()
    for gc in range(0, np.size(classes)):
        for ld in range(0, np.size(args.n_ld_dim)):
            if np.size(GAN_class) == 5:
                if gc == 6 or gc ==  7 or gc == 8 or gc == 16 or gc == 20:
                    pass
                else:
                    plt.plot(rep_classes_time[ld][0][0, :, 2], (avg_classes_time[ld][gc, :].T / 1000)*100, classes_colors[gc])
            if np.size(GAN_class) == 10:
                if gc == 6 or gc == 7 or gc == 8 or gc == 9 or gc == 10 or gc == 11 or gc == 12 or gc == 13 or gc == 21 or gc == 25:
                    pass
                else:
                    plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classes_time[ld][gc, :].T / 1000, classes_colors[gc])
            elif np.size(GAN_class) == 0:
                plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classes_time[ld][gc, :].T, classes_colors[gc])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Number of syllable (in percentage)', fontsize=15)
    plt.legend(['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V'])
    if args.title == 'on':
        plt.title('Number of syllable per class across dim')
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_class_NOGAN_across_time_and_dim.' + args.format)

    # GAN classes ONLY
    for gc in range(0,np.size(args.GAN_classes)):
        fig, ax = plt.subplots()
        for ld in range(0, np.size(args.n_ld_dim)):
            plt.plot(rep_classes_time[ld][0][0,:,2], avg_classes_time_onlyGAN[ld][gc,:].T, colors_list[ld])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Number of syllable', fontsize=15)
        plt.legend(legend_list_avg)
        if args.title == 'on':
            plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class across dim')
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + classes[int(GAN_class[gc])] + '_avg_class_across_time_and_dim.' + args.format)

        fig, ax = plt.subplots()
        for ld in range(0, np.size(args.n_ld_dim)):
            plt.plot(rep_classes_time[ld][0][0,:,2], (avg_classes_time_onlyGAN[ld][gc,:].T/1000)*100, colors_list[ld])
        plt.axhline(dataset_distr[int(GAN_class[gc])], color='k')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_ylim([0,100])
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        plt.ylabel('Percentage of syllable', fontsize=15)
        plt.legend(np.append(legend_list_avg, 'Training'))
        if args.title == 'on':
            plt.title('Number of syllable in ' + classes[int(GAN_class[gc])] + ' class across dim')
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_' + classes[int(GAN_class[gc])] + '_avg_class_across_time_and_dim.' + args.format)

    # Cumulative GAN classes
    fig, ax = plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        sum = 0
        for gc in range(0, np.size(args.GAN_classes)):
            sum = sum + avg_classes_time_onlyGAN[ld][gc, :].T / 1000

        plt.plot(rep_classes_time[ld][0][0, :, 2], sum * 100,
                 colors_list[ld])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_ylim([0, 100])
    plt.xlabel('Time (in number of epochs)', fontsize=15)
    plt.ylabel('Percentage of syllable', fontsize=15)
    plt.legend(np.append(legend_list_avg, 'Training'))
    if args.title == 'on':
        plt.title('Number of syllable in ALL class across dim')
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Percentage_ALL_avg_class_across_time_and_dim.' + args.format)

    # Cross-Entropy
    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(rep_classes_time[ld][0][0,:,2], avg_classic_cross_entropy[ld], colors_list[ld])
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    if args.title == 'on':
        plt.title('Cross-entropy across time')
    plt.legend(legend_list_avg)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy.' + args.format)

    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(rep_classes_time[ld][0][0,:,2], avg_classic_cross_entropy_real[ld], colors_list[ld])
    plt.axhline(data_entropy, color='k')
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    if args.title == 'on':
        plt.title('Cross-entropy across time')
    plt.legend(np.append(legend_list_avg, 'Training'))
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy_real.' + args.format)

    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(rep_classes_time[ld][0][0,:,2], avg_classic_cross_entropy_noGAN[ld], colors_list[ld])
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    if args.title == 'on':
        plt.title('Cross-entropy across time (without GAN class)')
    plt.legend(legend_list_avg)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy_noGAN.' + args.format)

    plt.subplots()
    for ld in range(0, np.size(args.n_ld_dim)):
        plt.plot(rep_classes_time[ld][0][0, :, 2], avg_classic_cross_entropy_noGAN_real[ld], colors_list[ld])
    plt.axhline(data_entropy_noGAN, color='k')
    plt.ylabel('Cross entropy', fontsize=15)
    plt.xlabel('Time (in epochs)', fontsize=15)
    if args.title == 'on':
        plt.title('Cross-entropy across time (without GAN class)')
    plt.legend(np.append(legend_list_avg, 'Training'))
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Avg_cross_entropy_noGAN_real.' + args.format)

    plt.close('all')
    print('Done')

def plot_inception(generation_summary_list, dataset_summary, all_IS, colors_list, legend_list, args):
    if len(generation_summary_list) > 0:
        # Training
        #aux = np.load(dataset_summary[0], allow_pickle=True)
        #aux = aux.item()

        #IS_train = aux['IS']
        #print(IS_train)

        # Generated
        IS_aux = np.zeros((np.size(generation_summary_list),))
        lat_dim = np.zeros((np.size(generation_summary_list),))
        epoch = np.zeros((np.size(generation_summary_list),))
        for sl in range(0, np.size(generation_summary_list)):
            aux = np.load(generation_summary_list[sl], allow_pickle=True)
            aux = aux.item()

            IS_aux[sl] = aux['IS']
            lat_dim[sl]= aux['Latent_dim']
            epoch[sl] = aux['Epoch']

        IS = np.zeros((int(np.size(generation_summary_list)/np.size(args.n_ld_dim)),np.size(args.n_ld_dim)))
        for d in range(0, np.size(args.n_ld_dim)):
            IS[:,d] = IS_aux[lat_dim==args.n_ld_dim[d]]
        print(IS)

        IS_summary = {'IS': IS, 'epochs': epoch,} # 'IS_train': IS_train}
        np.save(args.data_dir + '/' + 'IS.npy', IS_summary)


        fig, ax = plt.subplots()
        for d in range(0, np.size(args.n_ld_dim)):
            plt.plot(epoch[lat_dim==args.n_ld_dim[d]], IS[:,d], color=colors_list[d])
        #plt.plot(epoch[lat_dim==args.n_ld_dim[d]], np.ones((np.size(epoch[lat_dim==args.n_ld_dim[d]]),))*IS_train, 'k')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.ylabel('Inception Score', fontsize=15)
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        if args.title == 'on':
            plt.title('Inception score across time')
        #plt.legend([legend_list[d], 'Training'])
        plt.tight_layout()
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Inception_score.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Inception_score.' + 'png')

    if len(all_IS)>0:
        IS = []
        epochs = []
        for i in range(0, len(all_IS)):
            aux = np.load(all_IS[i], allow_pickle=True)
            aux = aux.item()

            IS.append(aux['IS'])
            epochs.append(aux['epochs'])
            print(epochs)
            print(IS)

            if i == 0:
                IS_train = aux['IS_train']

        fig, ax = plt.subplots()
        plt.plot(epochs[0], np.ones((np.size(epochs[0]),)) * IS_train, 'k')
        for i in range(0, len(all_IS)):
            plt.plot(epochs[i], IS[i], color=colors_list[i])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.ylabel('Inception Score', fontsize=15)
        plt.xlabel('Time (in number of epochs)', fontsize=15)
        if args.title == 'on':
            plt.title('Inception score across time')
        plt.legend(np.append('Training', legend_list), loc='upper left')
        plt.tight_layout()
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + 'Inception_score.' + args.format)
        plt.savefig(args.data_dir + '/' + 'Inception_score.' + 'png')

    print('Done')

if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()
  parser.add_argument('--option', type=str, choices=['annotations', 'analysis_dataset', 'analysis_error', 'analysis_gen', 'single', 'analysis_dim', 'instances', 'IS'])
  parser.add_argument('--data_dir', type=str, help='Directory containing the data',
                      default=None)
  parser.add_argument('--output_dir', type=str, help='Directory where to save the output',
                      default=None)

  dataset_args = parser.add_argument_group('Dataset')
  dataset_args.add_argument('--wavegan_latent_dim', type=int,
                help='Number of dimensions of the latent space',
                default=2)
  dataset_args.add_argument('--dd', type=int,
                            help='Scaling parameter to reduce the number of elements in the dataset',
                            default=1)
  dataset_args.add_argument('--dataset_dim', type=int,
                            help='How many elements in the dataset',
                            default=16000) #23456 16000
  dataset_args.add_argument('--ckpt_n', type=int,
                            help='At which chekpoint',
                            default=False)
  dataset_args.add_argument('--n_ld_dim', type=list,
                            help='How many dimension for the analysis or different dataset to be compared (depending on the scaling factor) for the analysis',
                            default=[3]) # if dataset size cond: [1,2,4] # if latent space dim all [1,2,3,4,5,6]
  dataset_args.add_argument('--n_instances', type=int, help='How many instances per type', default=1)
  dataset_args.add_argument('--GAN_classes', type=list,
                            help='GAN classes in the decoder',
                            default=['EARLY15', 'EARLY30', 'EARLY45', 'OT', 'WN'])
                            # 'EARLY', 'OT', 'WN'
                            #'EARLY15', 'EARLY30', 'OT', 'WN'
                            #'EARLY15_1', 'EARLY15_2', 'EARLY15_3', 'EARLY15_4', 'EARLY30_1', 'EARLY30_2', 'EARLY30_3', 'EARLY30_4', 'OT', 'WN'
                            #'EARLY15', 'EARLY30', 'EARLY45', 'OT', 'WN'
                            #'GAN1', 'GAN2', 'GAN3'

  plot_args = parser.add_argument_group('Plot')
  plot_args.add_argument('--format', type=str, help='Saving format', default='png')
  plot_args.add_argument('--title', type=str, help='Add the title or not: default yes', default='off')

  args = parser.parse_args()

  # Output direction creation
  if args.output_dir != None:
      if not os.path.isdir(args.data_dir + '/' + args.output_dir):
        os.makedirs(args.data_dir + '/' + args.output_dir)

  if args.option == 'annotations':
      annotations = create_annotations(args)
      epoch = int((args.ckpt_n * 5 * 64) / args.dataset_dim)
      if len(str(args.ckpt_n)) == 2:
          with open(args.data_dir + '/' + 'annotations_generation_ckpt000' + str(args.ckpt_n) + '_ld' + str(args.wavegan_latent_dim) + '_dd' + str(args.dd) + '_ep' + str(epoch) + '.pkl', 'wb+') as f:
              pickle.dump(annotations, f)
      elif len(str(args.ckpt_n)) == 3:
          with open(args.data_dir + '/' + 'annotations_generation_ckpt00' + str(args.ckpt_n)+ '_ld' + str(args.wavegan_latent_dim) + '_dd' + str(args.dd)  + '_ep' + str(epoch) + '.pkl', 'wb+') as f:
              pickle.dump(annotations, f)
      elif len(str(args.ckpt_n)) == 4:
          with open(args.data_dir + '/' + 'annotations_generation_ckpt0' + str(args.ckpt_n)+ '_ld' + str(args.wavegan_latent_dim) + '_dd' + str(args.dd)  + '_ep' + str(epoch) + '.pkl', 'wb+') as f:
              pickle.dump(annotations, f)
      else:
          with open(args.data_dir + '/' + 'annotations_generation_ckpt' + str(args.ckpt_n) + '_ld' + str(args.wavegan_latent_dim) + '_dd' + str(args.dd) + '_ep' + str(epoch) + '.pkl', 'wb+') as f:
              pickle.dump(annotations, f)

  if args.option == 'analysis_dataset':
      # List of the dataset we want to compare (to have the right legend in the figures)
      legend_list = ['Training dataset']

      # List of the annotations ( dataset)
      #annotations_dataset = sorted(glob.glob(args.data_dir + '/' + 'annotations_training_dataset_' + str(legend_list[0]) + '_*.pkl'))
      # annotations_dataset = sorted(glob.glob(args.data_dir + '/' + 'annotations_training_dataset_*.pkl'))
      annotations_dataset= glob.glob(args.data_dir + '/' + 'annotations_training_dataset_2_16.pkl')
      analysis_dataset(annotations_dataset, legend_list, args)

  if args.option == 'analysis_error':
      # List of the dataset we want to compare (to have the right legend in the figures)
      legend_list = ['A_noisy', 'A_notA', 'B1_noisy', 'B1_not_complete_ex', 'B1_not_cut', 'B1_too_short',
                     'B2_noisy', 'C_noisy', 'C_not_cut', 'C1', 'C2', 'D_not_cut', 'D_notD', 'E_noisy', 'E_not_cut', 'E_notE', 'E_too_long', 'E_too_short',
                     'H_noisy', 'H_not_cut', 'H_notH', 'H_prova', 'H_too_long', 'J1_noisy', 'J1_not_cut', 'J2_noisy', 'L_noisy',
                     'M_noisy', 'M_not_cut', 'M_notM', 'M_too_short', 'N_noisy', 'N_not_cut', 'N_too_short', 'O_noisy',
                     'O_notO', 'Q_noisy', 'Q_too_short', 'R_noisy', 'R_notR', 'R_too_short', 'V_noisy', 'V_notV',
                     'V_too_short']

      legend_list = ['H']

      # List of the annotations (dataset)
      annotations_dataset = []
      for ll in range(0, np.size(legend_list)):
          #annotations_dataset_aux = glob.glob(args.data_dir + '/' + 'annotations_errors_training_' + legend_list[ll] + '.pkl')
          annotations_dataset_aux = glob.glob(args.data_dir + '/' + 'annotations_dataset_' + legend_list[ll] + '*.pkl')
          annotations_dataset.append(annotations_dataset_aux)

      analysis_error(annotations_dataset, legend_list, args)

  if args.option == 'analysis_gen':
      # List of the annotations (generations and dataset)
      summary_dataset = sorted(glob.glob(args.data_dir + '/' + 'Dataset_summary*.npy'))
      annotations_generation = sorted(glob.glob(args.data_dir + '/' + 'annotations_generation*.pkl'))

      analysis_generation(annotations_generation, summary_dataset, args)

  if args.option == 'single':
      # List of the dataset we want to compare (to have the right legend in the figures)
      legend_list = ['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6']

      # List of the generated data
      generation_data = []
      for ld in range(0, np.size(args.n_ld_dim)):
          generation_data_aux = sorted(
              glob.glob(args.data_dir + '/' + 'Generation_summary_all' + str(args.n_ld_dim[ld]) + '*.npy'))
          generation_data.append(generation_data_aux)

      # Training dataset
      summary_dataset = sorted(glob.glob(args.data_dir + '/' + 'Dataset_summary*.npy'))

      analysis_latent(generation_data, summary_dataset, legend_list, args)

  if args.option == 'analysis_dim':
      # List of the dataset we want to compare (to have the right legend in the figures)
      legend_list_instances = ['Ex 0'] #, 'Ex 1', 'Ex 2'] #['NEW', 'OLD'] #
      legend_list = ['1', '1/2', '1/4'] #[3, 4]
      dd_dim = [16000, 8000, 4000] #[23456, 3600, 1600]

      colors_list = ['b', 'r', 'g']

      # List of the generated data
      generation_data = []
      for ld in range(0, np.size(args.n_ld_dim)):
        generation_data_aux = sorted(glob.glob(args.data_dir + '/' + 'Generation_summary_dd' + str(args.n_ld_dim[ld]) +'*.npy'))
        generation_data.append(generation_data_aux)

      # Training dataset
      summary_dataset = []
      for ld in range(0, np.size(args.n_ld_dim)):
        summary_dataset_aux = sorted(glob.glob(args.data_dir + '/' + 'Dataset_summary' + str(args.n_ld_dim[ld]) + '.npy'))
        summary_dataset.append(summary_dataset_aux)

      analysis_dim(generation_data, summary_dataset, legend_list_instances, legend_list, colors_list, args)

  if args.option == 'instances':
      # Color map
      color_list = ['purple', 'gold', 'orange', 'blue', 'darkgreen',  'magenta', 'darkcyan', 'gray', 'brown', 'red']
      #color_list = ['darkcyan']

      classes_colors = ['grey', 'sandybrown', 'orange', 'gold', 'olive', 'lightcoral', 'brown', 'red',
                        'limegreen', 'darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'purple', 'indigo', 'violet',
                        'deeppink', 'white', 'magenta', 'black']

      # List of the generated data
      generation_data = []
      for ld in range(0, np.size(args.n_ld_dim)):
        generation_data_aux = sorted(glob.glob(args.data_dir + '/' + '*_ld' + str(args.n_ld_dim[ld]) +'.npy'))
        generation_data.append(generation_data_aux)

      # Training dataset
      summary_dataset = sorted(glob.glob(args.data_dir + '/' + 'Dataset_summary*.npy'))

      # List of the dataset we want to compare (to have the right legend in the figures)
      legend_list_instances = []
      for l in range(0, np.size(generation_data)):
          legend_list_instances.append('Ex ' + str(l))
      # legend_list_avg = ['Dim 3']
      legend_list_avg = ['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6'] #'Dim 4', 'Dim 5',
      legend_list_instances = ['Ex 0']
      # legend_list_instances = ['Ex 0', 'Ex 1', 'Ex 2', 'Ex 3', 'Ex 4']

      several_instances(generation_data, summary_dataset, legend_list_instances, legend_list_avg, color_list, classes_colors, args)

  if args.option == 'IS':
      # Load data
      generation_summary_list = sorted(glob.glob(args.data_dir + '/' + 'Generation_summary_ep*.npy'))
      dataset_summary = sorted(glob.glob(args.data_dir + '/' + 'Dataset_summary*.npy'))
      all_IS = sorted(glob.glob(args.data_dir + '/' + '*IS_*.npy'))

      # Legend to plot
      legend_list = []
      for l in range(0, np.size(all_IS)):
          legend_list. append('Ex ' + str(l+1))
      #legend_list = ['Ex 0']
      #legend_list = ['1', '1/2', '1/4']
      #legend_list = ['Dim 1', 'Dim 2', 'Dim 3', 'Dim4', 'Dim5', 'Dim 6']

      # Colors list
      colors_list = ['purple', 'gold', 'orange', 'blue', 'darkgreen', 'magenta', 'darkcyan', 'gray', 'brown', 'red'] #10 instances dim 3
      #colors_list = ['darkcyan'] # example dim 3
      #colors_list = ['purple' , 'gold', 'orange', 'blue', 'darkgreen',  'magenta'] # different dim comparison
      # colors_list = ['b', 'r', 'g'] # dimension dataset comparison
      #colors_list = ['r'] # just to set one color

      plot_inception(generation_summary_list, dataset_summary, all_IS, colors_list, legend_list, args)





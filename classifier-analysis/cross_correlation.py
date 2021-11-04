import os
import glob
import random
import numpy as np
import scipy as sp
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import librosa

import statistics as stat
import plots as plot

def spectrogam_pearson_correlation(template_name, template_list, data_list, args):
    """
    :param template_name: name of the templates
    :param template_list: list of the templates
    :param data_list: list of the data

    :return: pearson correlation evaluation of each generated syllables wrt the template elements
    """
    # Load the data
    n_template = np.size(template_list)
    n_data = np.size(data_list)

    template_spectrogram = []
    for t in range(0,n_template):
        sr, samples = wav.read(template_list[t])
        freq, times, spectrogram = sp.signal.spectrogram(samples, sr, window='hann', nperseg=args.nperseg, noverlap=args.nperseg - args.overlap)
        template_spectrogram.append(spectrogram)

    # Correlation between the templates (test) with plot
    template_correlation = np.zeros((n_template, n_template))
    for t_1 in range(0, n_template):
        for t_2 in range(0, n_template):
            template_correlation[t_1, t_2] = stat.pearson_corr_coeff(template_spectrogram[t_1].flatten(), template_spectrogram[t_2].flatten())

    min_val = -1
    max_val = 1
    fig, ax = plt.subplots()
    plt.matshow(template_correlation, cmap=plt.cm.Blues)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks(np.arange(max_val))
    ax.set_yticks(np.arange(max_val))
    plt.xticks(range(n_template), template_name, rotation=90, fontsize=5)
    plt.yticks(range(n_template), template_name, fontsize=5)
    plt.colorbar()
    plt.tight_layout()  # to avoid the cut of labels
    plt.title('Pearson correlation template')
    plt.savefig(args.template_dir + '/' + 'template_pearson_corr.png')

    for d in range(0, n_data):
        wav_list = glob.glob(data_list[d] + '/' + '*.wav')
        data_spectrogram = []
        for s in range(0,np.size(wav_list)):
            sr, samples = wav.read(wav_list[s])
            freq, times, spectrogram = sp.signal.spectrogram(samples, sr, window='hann', nperseg=args.nperseg, noverlap=args.nperseg - args.overlap)
            data_spectrogram.append(spectrogram)

        # Correlation between the template and the generations
        data_template_correlation = np.zeros((n_template, np.size(wav_list)))
        for t in range(0, n_template):
            for s in range(0, np.size(wav_list)):
                data_template_correlation[t, s] = stat.pearson_corr_coeff(template_spectrogram[t][:,0:np.min([np.shape(template_spectrogram[t][1]), np.shape(data_spectrogram[s][1])])].flatten(), data_spectrogram[s][:,0:np.min([np.shape(template_spectrogram[t][1]), np.shape(data_spectrogram[s][1])])].flatten())

        np.save(args.data_dir + '/' + args.output_dir + '/' + 'dataVStemplate_pearson_corr' + str(d) + '.npy', data_template_correlation)

    print('Done')

def test_lag(syll_list, classes, args):
    """
    :param syll_list: list of wav files

    :return: the lag correlation matrix computed using xcorr, plus the plot of the matrix.
    """
    n_lags = 100
    correlation_matrix = np.zeros((np.size(syll_list),np.size(syll_list)))
    for s in range(0,np.size(syll_list)):
        sr, samples_s = wav.read(syll_list[s])
        # Plot spectrogram (control)
        fig, ax = plt.subplots()
        (f, t, spect) = sp.signal.spectrogram(samples_s, sr, args.window, args.nperseg, args.nperseg - 64, mode='complex')
        ax.imshow(10 * np.log10(np.square(abs(spect))), origin="lower", aspect="auto", interpolation="none", extent=[0, max(t) * 1000, min(f), max(f)], cmap='inferno')
        ax.set_ylabel('Frequency (Hz)')
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'test' + str(s) +'.png')
        for t in range(0, np.size(syll_list)):
            sr, samples_t = wav.read(syll_list[t])
            freq, times, spectrogram_s = sp.signal.spectrogram(samples_s, sr, window='hann', nperseg=args.nperseg, noverlap=args.nperseg - 64)
            freq_downsample_s = sp.signal.resample(spectrogram_s, 60, t=None, axis=0)
            time_downsample_s = sp.signal.resample(freq_downsample_s, 120, t=None, axis=1)
            freq, times, spectrogram_t = sp.signal.spectrogram(samples_t, sr, window='hann', nperseg=args.nperseg, noverlap=args.nperseg - 64)
            freq_downsample_t = sp.signal.resample(spectrogram_t, 60, t=None, axis=0)
            time_downsample_t = sp.signal.resample(freq_downsample_t, 120, t=None, axis=1)
            # Plot reduced spectrogram (control)
            fig,ax = plt.subplots()
            ax.imshow(10 * np.log10(np.square(abs(time_downsample_t))), origin="lower", aspect="auto", interpolation="none", cmap='inferno')
            ax.set_ylabel('Frequency (Hz)')
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'test' + str(t) +'.png')
            fig, ax = plt.subplots()
            corr_aux = ax.xcorr(time_downsample_s.flatten(), time_downsample_t.flatten(), maxlags=n_lags, lw=2)
            #print(corr_aux)
            #plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'prova' + str(s) + str(t) + '.png')
            correlation_matrix[s, t] = np.max(corr_aux[1])
            #print(correlation_matrix)
            #input()

        plt.close('all')

    title = 'Cross correlation'
    fig = plot.plot_correlation_general(correlation_matrix, classes, classes, title)
    fig.savefig(args.data_dir + '/' + args.output_dir + '/' + 'cross_correlation_' + str(n_lags) +'lags.png')

    print('Done')

def lag_correlation(classes, GAN_classes, dataset_summary_list, generation_summary_list, args):
    """
    :param classes: dictionary of the classes I have in the training dataset
    :param GAN_classes: garbage classes
    :param dataset_summary_list: list of the wav files (training dataset)
    :param generation_summary_list: list of the wav files (generated dataset)

    :return: A dictionary for the training dataset and one for the generated dataset. Each of them contains the lag
    cross-correlation computed between 100 random couples of syllables per class. In addition, the dictionary
    related to the generated dataset contains the lag cross correlation computed between (maximum) 100 couples of a
    training syllabled and a generated syllable.
    """
    # How many classes and definition of GAN classes
    n_classes = np.size(classes) + np.size(GAN_classes)

    if np.size(dataset_summary_list)>0:
        for sl in range(0, np.size(dataset_summary_list)):
            aux = np.load(dataset_summary_list[sl], allow_pickle = True)
            aux = aux.item()

            aux_names = aux['File_name']

            data_syll_list = []
            for c in range(0,np.size(classes)):
                syll_list_aux = []
                for i in range(0,np.size(aux_names)):
                    if (aux_names[i].find('NEW_' + classes[c]) != -1):
                        syll_list_aux.append(aux_names[i])
                data_syll_list.append(syll_list_aux)

            cross_corr = []
            for c in range(0, np.size(classes)):
                pairs = stat.pairs(range(0,np.size(data_syll_list[c])),range(0,np.size(data_syll_list[c])))
                syll_pairs = random.sample(pairs, args.n_template)
                data_syll_single = random.sample(range(0,np.size(data_syll_list[c])), args.n_template)

                cross_corr_aux = np.zeros((args.n_template,))
                for j in range(0,args.n_template):
                    if j == 0:
                        title = 'Real ex spectrogram class' + classes[c]
                        fig = plot.plot_spectro(data_syll_list[c][syll_pairs[j][0]], args.window, args.nperseg, title)
                        fig.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Real_ex_spectrogram_class_' + classes[c] + '.' + args.format)
                    cross_corr_aux[j] = stat.lag_cross_corr(args.n_lags, data_syll_list[c][syll_pairs[j][0]], data_syll_list[c][syll_pairs[j][1]], args.nperseg, args.overlap)
                    plt.close('all')
                cross_corr.append(cross_corr_aux)

            np.save(args.data_dir + '/' + 'Dataset_cross_corr_distribution.npy', cross_corr)

    if np.size(generation_summary_list)>0:
        for sl in range(0, np.size(generation_summary_list)):
            aux = np.load(generation_summary_list[sl], allow_pickle = True)
            aux = aux.item()

            aux_names = aux['File_name'] #was real name in case I had some old file
            aux_decoder_names = aux['Decoder_name']
            aux_latent_dim = aux['Latent_dim']
            aux_epoch = aux['Epoch']

            gen_syll_list = []
            for c in range(0,np.size(classes)):
                syll_list_aux = []
                for i in range(0,np.size(aux_names)):
                    if (aux_decoder_names[i].find(classes[c]) != -1):
                        syll_list_aux.append(aux_names[i])
                gen_syll_list.append(syll_list_aux)

            for g in range(0,np.size(GAN_classes)):
                syll_list_aux = []
                for i in range(0,np.size(aux_names)):
                    if (aux_decoder_names[i].find(GAN_classes[g]) != -1):
                        syll_list_aux.append(aux_names[i])
                gen_syll_list.append(syll_list_aux)

            cross_corr_within = []
            cross_corr_across = []
            for c in range(0, np.size(classes)):
                if np.size(gen_syll_list[c])>0:
                    pairs = stat.pairs(range(0, np.size(gen_syll_list[c])), range(0, np.size(gen_syll_list[c])))
                    if np.shape(pairs)[0]<args.n_template:
                        syll_pairs = random.sample(pairs, np.shape(pairs)[0])
                    else:
                        syll_pairs = random.sample(pairs, args.n_template)

                    if args.n_template<np.size(gen_syll_list[c]):
                        gen_syll_single = random.sample(range(0, np.size(gen_syll_list[c])),args.n_template)
                        data_syll_single = random.sample(range(0, np.size(data_syll_list[c])), args.n_template)
                    else:
                        gen_syll_single = random.sample(range(0, np.size(gen_syll_list[c])),np.size(gen_syll_list[c]))
                        data_syll_single = random.sample(range(0, np.size(data_syll_list[c])), np.size(gen_syll_list[c]))

                    cross_corr_aux_w = np.zeros((np.size(gen_syll_single),))
                    cross_corr_aux_a = np.zeros((np.size(gen_syll_single),))
                    for j in range(0, np.size(gen_syll_single)):
                        #if j == 0:
                            #title = 'Gen ex spectrogram class' + classes[c]
                            #fig = plot.plot_spectro(gen_syll_list[c][syll_pairs[j][0]], args.window, args.nperseg, title)
                            #fig.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Gen_ex_spectrogram_class_' + classes[c] + '.png')
                        cross_corr_aux_w[j] = stat.lag_cross_corr(args.n_lags, gen_syll_list[c][syll_pairs[j][0]], gen_syll_list[c][syll_pairs[j][1]], args.nperseg, args.overlap)
                        cross_corr_aux_a[j] = stat.lag_cross_corr(args.n_lags, gen_syll_list[c][gen_syll_single[j]], data_syll_list[c][data_syll_single[j]], args.nperseg, args.overlap)
                        plt.close('all')

                    cross_corr_within.append(cross_corr_aux_w)
                    cross_corr_across.append(cross_corr_aux_a)

                else:
                    cross_corr_within.append(np.zeros((args.n_template,)))
                    cross_corr_across.append(np.zeros((args.n_template,)))

            dict = {'Epoch': aux_epoch, 'Latent_dim': aux_latent_dim, 'Cross_corr_within': cross_corr_within, 'Cross_corr_across': cross_corr_across}
            np.save(args.data_dir + '/' + 'Generation_cross_corr_distribution_ep' + str(aux_epoch) + '_ld' + str(aux_latent_dim) + '.npy', dict)

    print('Done')



def plot_correlation(cross_corr_data, cross_corr_gen, classes, args):
    # Load the data
    mean_cross_dataset = np.zeros((np.size(classes),))
    for sd in range(0, np.size(cross_corr_data)):
        cross_corr_dataset = np.load(cross_corr_data[sd])
        for c in range(0, np.size(classes)):
            mean_cross_dataset[c] = np.mean(cross_corr_dataset[c,:])

    cross_corr_generation = []
    cross_corr_across = []
    ld = []
    epoch =[]
    mean_cross_gen = np.zeros((np.size(classes),np.size(cross_corr_gen)))
    mean_cross_gen_across = np.zeros((np.size(classes),np.size(cross_corr_gen)))
    for sg in range(0, np.size(cross_corr_gen)):
        aux = np.load(cross_corr_gen[sg], allow_pickle=True)
        aux = aux.item()

        ld.append(aux['Latent_dim'])
        epoch.append(int(aux['Epoch']))

        cross_corr_generation_aux = aux['Cross_corr_within']
        cross_corr_across_aux = aux['Cross_corr_across']
        cross_corr_generation.append(cross_corr_generation_aux)
        cross_corr_across.append(cross_corr_across_aux)
        for c in range(0, np.size(classes)):
            mean_cross_gen[c,sg] = np.mean(cross_corr_generation_aux[c])
            mean_cross_gen_across[c,sg] = np.mean(cross_corr_across_aux[c])

    colors_list_long = list(colors._colors_full_map.values())[0::5]
    colors_list = ['r', 'b', 'g', 'orange', 'violet']
    # Comparison dataset - generation per each class
    # Cross-correlation within the same type of syllable (training VS generated data)
    for c in range(0,np.size(classes)):
        plt.hist(cross_corr_dataset[c], 20, color='b', alpha=0.6)
        plt.hist(cross_corr_generation[0][c], 20, color='r', alpha=0.6)
        plt.title('Cross-correlation distribution class ' + classes[c] + ' at epoch ' + str(epoch[0]))
        plt.legend(['Dataset', 'Generation'])
        #plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'cross_correlation_class_' + classes[c] + '_ld_' + str(args.ld) + '.' + args.format)
        plt.close('all')

    # Comparison dataset - generation per each class
    # Cross-correlation within the same type of syllable across trianing and generated data
    for c in range(0, np.size(classes)):
        plt.hist(cross_corr_across[0][c], 20, color='r', alpha=0.6)
        plt.title('Cross-correlation distribution class ' + classes[c] + ' at epoch ' + str(epoch[0]))
        plt.legend(['Dataset vs Generation'])
        #plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'cross_correlation_across_' + classes[c] + '_ld_' + str(args.ld) + '.' + args.format)
        plt.close('all')

    # Comparison generations across epochs per each class
    # Within vs training data
    for c in range(0,np.size(classes)):
        n,x,_ = plt.hist(cross_corr_dataset[c], 20, color='k', alpha=0)
        bin_centers = 0.5 * (x[1:] + x[:-1])
        plt.plot(bin_centers, n, 'k', alpha=0.5)
        for sg in range(0, np.size(cross_corr_gen)):
            n,x,_ = plt.hist(cross_corr_generation[sg][c], 20, color=colors_list[sg], alpha=0)
            bin_centers = 0.5 * (x[1:] + x[:-1])
            plt.plot(bin_centers, n, color=colors_list[sg], alpha=0.5)
        plt.legend(np.append('Training', epoch))
        plt.title('Cross-correlation distribution class ' + classes[c] + ' across epochs')
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'hist_cross_correlation_across_epochs_class_' + classes[c] + '_ld_' + str(args.ld) + '.' + args.format)
        plt.close('all')

    for c in range(0,np.size(classes)):
        for sg in range(0, np.size(cross_corr_gen)):
            n,x,_ = plt.hist(cross_corr_across[sg][c], 20, color=colors_list[sg], alpha=0)
            bin_centers = 0.5 * (x[1:] + x[:-1])
            plt.plot(bin_centers, n, color=colors_list[sg], alpha=0.5)
        plt.legend(epoch)
        plt.title('Cross-correlation across distribution class ' + classes[c] + ' across epochs')
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'hist_cross_correlation_across_across_epochs_class_' + classes[c] + '_ld_' + str(args.ld) + '.' + args.format)
        plt.close('all')

    # Mean cross correlation across epochs
    # Across generated data
    fig, ax = plt.subplots()
    for c in range(0, np.size(classes)):
        plt.plot(np.linspace(0,np.size(cross_corr_gen),np.size(cross_corr_gen)), mean_cross_gen_across[c,:]/mean_cross_gen_across[c, 0], color = colors_list_long[c+3])
    plt.legend(classes)
    ax.set_ylabel('Mean cross-correlation')
    ax.set_xlabel('Epochs')
    plt.title('Mean cross-correlation across generated data across epochs')
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'mean_cross_correlation_across_generated_across_epochs.' + args.format)
    plt.close('all')

    # Real and generated data within
    fig, ax = plt.subplots()
    for c in range(0, np.size(classes)):
        plt.plot(mean_cross_gen_across[c, :], np.ones((np.size(cross_corr_gen),))*mean_cross_dataset[c], '*', color=colors_list_long[c + 3])
    plt.legend(classes)
    plt.plot([0, 1], [0, 1], 'k')
    ax.set_xlabel('Generated data')
    ax.set_ylabel('Real data')
    plt.title('Mean cross-correlation within syllables across epochs')
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'mean_cross_correlation_within_syllables_across_epochs.' + args.format)
    plt.close('all')

    fig, ax = plt.subplots()
    for c in range(0, np.size(classes)):
        plt.plot(np.linspace(0,np.size(cross_corr_gen),np.size(cross_corr_gen)), mean_cross_gen[c,:]/mean_cross_gen[c, 0], color = colors_list_long[c+3])
    plt.legend(classes)
    ax.set_ylabel('Mean cross-correlation')
    ax.set_xlabel('Epochs')
    plt.title('Mean cross-correlation generated data across epochs')
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'mean_cross_correlation_generated_across_epochs.' + args.format)
    plt.close('all')

    print('Done')

def CFR_plot(cross_corr_data, cross_corr_gen, classes, legend_data, legend_gen, args):
    colors_list_long = list(colors._colors_full_map.values())[0::5]

    # Load the data
    # Real data
    cross_corr_dataset = []
    mean_cross_dataset = np.zeros((np.size(classes), np.size(cross_corr_data)))
    if np.size(cross_corr_data)>0:
        for j in range(0,np.size(cross_corr_data)):
            cross_corr_dataset.append(np.load(cross_corr_data[j]))
            for c in range(0, np.size(classes)):
                mean_cross_dataset[c,j] = np.mean(cross_corr_dataset[j][c, :])

        for c in range(0, np.size(classes)):
            plt.subplots()
            for j in range(0,np.size(cross_corr_data)):
                n, x, _ = plt.hist(cross_corr_dataset[j][c], 20, color=colors_list_long[j], alpha=0)
                bin_centers = 0.5 * (x[1:] + x[:-1])
                plt.plot(bin_centers, n, color=colors_list_long[j])
            plt.title(classes[c])
            plt.legend(legend_data)
            plt.tight_layout()  # to avoid the cut of labels
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'mean_cross_correlation_within_class' + classes[c] + '_cfr_datasetOLD.' + args.format)

    input()

    cross_corr_generation = []
    cross_corr_across = []
    ld = []
    epoch =[]
    mean_cross_gen = []
    mean_cross_gen_across = []

    if np.size(cross_corr_gen)>0:
        for i in range(0,np.size(cross_corr_gen)):
            loc_aux = sorted(glob.glob(cross_corr_gen[i] + '/' + 'Generation_cross*'))
            mean_cross_gen_aux = np.zeros((np.size(classes), np.size(loc_aux)))
            mean_cross_gen_across_aux = np.zeros((np.size(classes), np.size(loc_aux)))

            ld_aux = []
            epoch_aux = []
            cross_corr_generation_aux = []
            cross_corr_across_aux = []
            for sg in range(0, np.size(loc_aux)):
                aux = np.load(loc_aux[sg], allow_pickle=True)
                aux = aux.item()

                ld_aux.append(aux['Latent_dim'])
                epoch_aux.append(int(aux['Epoch']))

                load_corr_aux = aux['Cross_corr_within']
                cross_corr_generation_aux.append(load_corr_aux)
                load_corr_across_aux = aux['Cross_corr_across']
                cross_corr_across_aux.append(load_corr_across_aux)

                for c in range(0, np.size(classes)):
                    mean_cross_gen_aux[c,sg] = np.mean(load_corr_aux[c])
                    mean_cross_gen_across_aux[c,sg] = np.mean(load_corr_across_aux[c])

            mean_cross_gen.append(mean_cross_gen_aux)
            mean_cross_gen_across.append(mean_cross_gen_across_aux)

            cross_corr_generation.append(cross_corr_generation_aux)
            cross_corr_across.append(cross_corr_across_aux)
            ld.append(ld_aux)
            epoch.append(epoch_aux)

        for c in range(0, np.size(classes)):
            for e in range(0,np.size(epoch[0])):
                plt.subplots()
                for i in range(0, np.size(cross_corr_gen)):
                    n, x, _ = plt.hist(cross_corr_across[i][e][c], 20, color=colors_list_long[i], alpha=0)
                    bin_centers = 0.5 * (x[1:] + x[:-1])
                    plt.plot(bin_centers, n, color=colors_list_long[i])
                plt.legend(legend_gen)
                plt.title(classes[c] + '_epoch' + str(epoch[0][e]))
                plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'hist_cross_correlation_across_class_' + classes[c] + '_epoch_' + str(epoch[0][e]) + '.' + args.format)
                plt.close('all')

    print('Done')

if __name__ == '__main__':
    import argparse
    import glob
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument('--option', type=str, choices=['pearson', 'test_lag', 'lag', 'plot', 'cfr'])
    parser.add_argument('--template_dir', type=str, help='Directory where there are the templates for spectrogram correlation', default='Template')
    parser.add_argument('--data_dir', type=str, help='Directory containing the data to evaluate')
    parser.add_argument('--output_dir', type=str, help='Directory to save the ouptut', default='Plots')

    analysis_args = parser.add_argument_group('features_analysis')
    analysis_args.add_argument('--window', type=str, help='Type of window for the visualization of the spectrogram', default='hanning')
    analysis_args.add_argument('--overlap', type=int, help='Overlap for the visualization of the spectrogram', default=66)
    analysis_args.add_argument('--nperseg', type=int, help='Nperseg for the visualization of the spectrogram', default=512)
    analysis_args.add_argument('--n_template', type=int, help='How many template I use to compute a pairwise comparison with the generated data', default=200)
    analysis_args.add_argument('--dataset_dim', type=int, help='How many syllables I used for the dataset', default=None)
    analysis_args.add_argument('--n_lags', type=int, help='Number of lags used to compute the cross-correlation', default=100)

    plot_args = parser.add_argument_group('plot')
    plot_args.add_argument('--ld', type=int, help='dimension of the latent space used for training', default=None)
    plot_args.add_argument('--format', type=str, help='Saving format', default='png')

    args = parser.parse_args()

    # Output direction creation
    if args.output_dir != None:
        if not os.path.isdir(args.data_dir + '/' + args.output_dir):
            os.makedirs(args.data_dir + '/' + args.output_dir)

    if args.option=='pearson':
        # Define the name of the template syllables
        template_name = ['Syll_A', 'Syll_B1', 'Syll_B2', 'Syll_C', 'Syll_D', 'Syll_E', 'Syll_H', 'Syll_J1', 'Syll_J2', 'Syll_L', 'Syll_M', 'Syll_N', 'Syll_O', 'Syll_Q', 'Syll_R', 'Syll_V']
        template_list = glob.glob(args.template_dir + '/' + '*.wav')
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

        # Dataset to evaluate
        data_list = glob.glob(args.data_dir + '/generation*')
        spectrogam_pearson_correlation(template_name, template_list, data_list, args)

    if args.option == 'test_lag':
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']
        syll_list = glob.glob(args.data_dir + '/' +'*.wav')

        test_lag(syll_list, classes, args)

    if args.option == 'lag':
        # Name of the classes
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']
        GAN_classes = ['GAN1', 'GAN2', 'GAN3']

        # Load data
        dataset_summary_list = glob.glob(args.data_dir + '/' + 'Dataset_summary1NEWB1_C*')
        generation_summary_list = glob.glob(args.data_dir + '/' + 'Generation_summary_ep*')

        lag_correlation(classes, GAN_classes, dataset_summary_list, generation_summary_list, args)

    if args.option == 'plot':
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

        cross_corr_data = sorted(glob.glob(args.data_dir + '/' + 'Dataset_cross*'))
        cross_corr_gen = sorted(glob.glob(args.data_dir + '/' + 'Generation_cross*'))

        plot_correlation(cross_corr_data, cross_corr_gen, classes, args)

    if args.option == 'cfr':
        classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

        cross_corr_data = sorted(glob.glob(args.data_dir + '/' + 'Dataset_cross*'))
        cross_corr_gen = sorted(glob.glob(args.data_dir + '/' + 'Ex*'))

        legend_data = ['NEW_v1', 'NEW_v2'] #, 'NEW_v3', 'OLD']
        legend_gen = ['Ex0', 'Ex1', 'Ex2']

        CFR_plot(cross_corr_data, cross_corr_gen, classes, legend_data, legend_gen, args)





# Useful path
# dataset general directory
# D:\\PhD_Bordeaux\\DATA\\Dataset_GAN\\Datasets_Marron\\Dataset_2

# generated data for a dimension directory
# D:\\PhD_Bordeaux\\Python_Scripts\\Generative_Models\\wavegan-master\\GPUbox_Results\\trainPLAFRIM_16s_Marron1_ld3\\Evaluation

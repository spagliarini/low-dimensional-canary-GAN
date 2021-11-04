import numpy as np
import scipy as sp
import librosa.display
import librosa.feature
import librosa.effects
import scipy.io.wavfile as wav
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_distribution_classes(X, my_ticks, title, n_classes):
    """
    :param X: data
    :param my_ticks: personal ticks for x label (name of the classes)
    :param title: title of the plot
    :param n_classes: number of classes

    :return: the figure, the distribution and its features (mean, standard deviation, variance)
    """
    h, bins = np.histogram(X, bins=range(n_classes+1))
    #if n_classes>16:  #to have the plot in order (check the correct option - add more if the classifier changes)
        #h = np.append(h[0:6], np.append(h[9:16], np.append(h[17:20], np.append(h[6:9], np.append(h[16], h[20]))))) #NEW DECODER 5 classes
        #my_ticks = np.append(my_ticks[0:6], np.append(my_ticks[9:16], np.append(my_ticks[17:20], np.append(my_ticks[6:9], np.append(my_ticks[16], my_ticks[20])))))
        # h =  np.append(h[0:6], np.append(h[9:19], h[6:9])) #OLD DECODER 3 classes
        # my_ticks = np.append(my_ticks[0:6], np.append(my_ticks[9:19], my_ticks[6:9]))
    h_mean = np.mean(h)
    h_std = np.std(h)
    h_var = np.var(h)
    fig, ax = plt.subplots()
    ax.bar(bins[:-1], h, width=0.8, color='b', alpha=0.6, label='Syllable classes distribution', align='center')
    #plt.plot(bins[:-1], np.ones((n_classes))*h_mean, 'k') # to visualize mean
    #ax.fill_between(bins[:-1], h_mean - h_std, h_mean + h_std, 'k', alpha=0.2) # to visualize the variance
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    #plt.legend(loc='upper right', fontsize=8, ncol=1, shadow=True, fancybox=True)
    plt.xticks(bins[:-1], my_ticks, rotation='vertical', fontsize=15)
    plt.xlabel('Classes of syllables', fontsize=15)
    plt.ylabel('Number of occurences', fontsize=15)
    plt.title(title, fontsize=15)
    plt.tight_layout()  # to avoid the cut of labels

    return fig, h, h_mean, h_std, h_var

def plot_correlation_general(correlation_matrix, x_label, y_label, title):
    """
    :param correlation_matrix: pre-computed correlation matrix
    :param template_name: the name of the classes
    :param title: title of the plot

    :return: the cross correlation matrix figure
    """
    fig, ax = plt.subplots()
    plt.imshow(correlation_matrix, cmap=plt.cm.Blues, aspect='auto')
    plt.xticks(range(np.size(x_label)), x_label, fontsize=7)
    plt.yticks(range(np.size(y_label)), y_label, fontsize=7)
    plt.colorbar()
    #plt.clim(0, 1)
    plt.tight_layout()  # to avoid the cut of labels
    plt.title(title)

    return fig


def plot_histogram(X, n_bins, my_color, title):
    """
    :param X: data
    :param n_bins: number of bins
    :param my_color: my_color
    :param title: title of the plot

    :return: histogram of the data
    """
    fig, ax = plt.subplots()
    plt.hist(X, n_bins, color= my_color, alpha=0.6)
    plt.title(title)
    plt.tight_layout()  # to avoid the cut of labels

    return fig

def plot_spectro_sp(filename, window, nperseg, title):
    """
    :param filename: name of the file to plot (.wav file)
    :param window: as in the documentation of scipy.signal.spectrogram
    :param nperseg: scipy.signal.spectrogram
    :param title: title of the figure

    :return: the spectrogram figure
    """
    sr, samples = wav.read(filename)

    fig, ax = plt.subplots()
    (f, t, spect) = sp.signal.spectrogram(samples, sr, window, nperseg, nperseg - 64, mode='complex')
    ax.imshow(10 * np.log10(np.square(abs(spect))), origin="lower", aspect="auto", interpolation="none", extent=[0, max(t) * 200, min(f), max(f)], cmap='inferno')
    ax.set_ylabel('Frequency (Hz)', fontsize=15)
    ax.set_xlabel('Time (ms)', fontsize=15)
    plt.title(title)
    plt.tight_layout()  # to avoid the cut of labels

    return fig

def plot_spectro_librosa(filename, N, H, color, gamma, title):
    """
    :param filename: name of the file to plot (.wav file)
    :param N: nperseg
    :param H: hoplength
    :param gamma : color powernorm

    ref: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C2/C2_STFT-Conventions.html

    :return: the spectrogram figure
    """
    y, sr = librosa.load(filename, sr=16000)

    X = librosa.stft(y, n_fft=N, hop_length=H, win_length=N, window='hann', pad_mode='constant', center=True)
    Y = np.log(1 + 100 * np.abs(X) ** 2)
    T_coef = np.arange(X.shape[1]) * H / sr
    K = N // 2
    F_coef = np.arange(K + 1) * sr / N

    fig = plt.figure(figsize=(15, 3))
    extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
    plt.imshow(Y, cmap=color, aspect='auto', origin='lower', extent=extent, norm=colors.PowerNorm(gamma=gamma))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.tight_layout()

    return fig

def multi_spectro(songs, N, H, color, gamma):
    """
    :param songs: list of songs to plot
    :param N: nperseg
    :param H: hoplength
    :param gamma : color powernorm

    :return: the spectrogram figure
    """
    fig, axs = plt.subplots(ncols=1, nrows=np.size(songs), figsize=(22, 5))
    for i in range(0, np.size(songs)):
        y, sr = librosa.load(songs[i], sr=16000)

        X = librosa.stft(y, n_fft=N, hop_length=H, win_length=N, window='hann', pad_mode='constant', center=True)
        Y = np.log(1 + 100 * np.abs(X) ** 2)
        T_coef = np.arange(X.shape[1]) * H / sr
        K = N // 2
        F_coef = np.arange(K + 1) * sr / N
        extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]

        axs[i].imshow(Y, cmap=color, aspect='auto', origin='lower', extent=extent, norm=colors.PowerNorm(gamma=gamma))
        axs[i].set_ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.tight_layout()

    return fig
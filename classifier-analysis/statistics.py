import numpy as np
import scipy as sp
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import itertools

def pearson_corr_coeff(syllable_1_template, syllable_2_template):
    '''

    :param syllable_1_template: spectrographic template syllable 1
    :param syllable_2_template: spectrographic template syllable 2

    :return: pearson correlation coefficient
    '''
    N = np.size(syllable_1_template)
    covariance = np.sum((syllable_1_template - np.mean(syllable_1_template))*(syllable_2_template - np.mean(syllable_2_template)))*(1/(N-1))
    pearson_corr = covariance/(np.std(syllable_1_template)*np.std(syllable_2_template))

    return pearson_corr

def cross_entropy(p, q):
    """
    Function to compute the discrete cross entropy between two probabilty distributions using Shannon entropy classic formulation.
    Here log base-2 is to ensure the result has units in bitsuse log base-2 to ensure the result has units in bits.
    Ig np.log then the unit of measure is in nats.

    :param p: true distribution
    :param q: distribution to compare
    :return: cross entropy in bits
    """
    H = 0
    for i in range(len(p)):
        if q[i] != 0:
            H = H + p[i]*np.log(q[i])

    return -H

def KL_divergence(p,q):
    """
    Function to compute the KL divergence between two probabilty distributions.

    :param p: true distribution
    :param q: distribution to compare
    :return: KL divergence
    """
    KL_div = 0
    for i in range(len(p)):
        if q[i] != 0:
            KL_div = KL_div + p[i] * np.log(p[i] / q[i])

    return KL_div

def KL_cross_entropy(p, q):
    """
    Function to compute the discrete cross entropy between two probabilty distributions using KL formulation.
    Here log base-2 is to ensure the result has units in bitsuse log base-2 to ensure the result has units in bits.
    Ig np.log then the unit of measure is in nats.

    :param p: true distribution
    :param q: distribution to compare
    :return: KL cross entropy
    """
    KL_entropy = cross_entropy(p,p) + KL_divergence(p, q)

    return KL_entropy

def cross_entropy_for_class_labels(p,q):
    """
    Function to compute the discrete cross entropy between two label distirbutions.

    :param p: true distribution
    :param q: distribution to compare
    :return: cross entropy between labels
    """
    results = list()
    for i in range(len(p)):
        # create the distribution for each event {0, 1}
        expected = [1.0 - p[i], p[i]]
        predicted = [1.0 - q[i], q[i]]
        # calculate cross entropy for the two events
        ce = cross_entropy(expected, predicted)
        results.append(ce)

    return results

def pairs(list1,list2):
    """
    :param list1: list of elements
    :param list2: list of elements

    :return: pairs of elements with no repetition
    """
    temp = list(itertools.product(list1, list2))

    # output list initialization
    out = []

    # iteration
    for elem in temp:
        if elem[0] != elem[1]:
            out.append(elem)

    return out

def lag_cross_corr(n_lags, filename_1, filename_2, nperseg, overlap):
    sr, samples_1 = wav.read(filename_1)
    sr, samples_2 = wav.read(filename_2)

    freq, times, spectrogram_1 = sp.signal.spectrogram(samples_1, sr, window='hann', nperseg=nperseg,
                                                       noverlap=nperseg - overlap)
    freq_downsample_1 = sp.signal.resample(spectrogram_1, 60, t=None, axis=0)
    time_downsample_1 = sp.signal.resample(freq_downsample_1, 120, t=None, axis=1)

    freq, times, spectrogram_2 = sp.signal.spectrogram(samples_2, sr, window='hann', nperseg=nperseg,
                                                       noverlap=nperseg - overlap)
    freq_downsample_2 = sp.signal.resample(spectrogram_2, 60, t=None, axis=0)
    time_downsample_2 = sp.signal.resample(freq_downsample_2, 120, t=None, axis=1)

    fig, ax = plt.subplots()
    cross_aux = ax.xcorr(time_downsample_1.flatten(), time_downsample_2.flatten(), maxlags=n_lags, lw=2)
    cross_correlation = np.max(cross_aux[1])

    return cross_correlation

def inception_score(p, eps=1E-16):
    """
    :param p: probability vector

    :return: inception score

    doc: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
    """
    p_aux = np.expand_dims(np.mean(p, axis=0), 0)
    # kl divergence
    kl_d = p * (np.log(p + eps) - np.log(p_aux + eps))
    # sum over classes
    sum_kl_d = np.sum(kl_d, axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    IS = np.exp(avg_kl_d)

    return IS
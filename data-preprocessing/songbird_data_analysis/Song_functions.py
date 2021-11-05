#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:33:13 2019

@author: rsankar
"""

"""
Compilation of functions used by Roman to segment songs
"""
import numpy as np
import scipy as sp
import scipy.signal
from scipy.io import loadmat
from scipy.io import wavfile

#Parameters to be used
#
#window =('hamming')
#overlap = 64
#nperseg = 1024
#noverlap = nperseg-overlap
#colormap = "jet"
#
##threshold=2e-9 # for files recorded with Neuralynx
##threshold=2e-5
##threshold=2e-10
#threshold=1.25e-9
#min_syl_dur=0.02
#min_silent_dur= 0.003
#smooth_win=10
#
##rec_system = 'Alpha_omega' # or 'Neuralynx' or 'Other'
#rec_system = 'Neuralynx'

def bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs=(500, 10000)):
    """filter song audio with band pass filter, run through filtfilt
        (zero-phase filter)
        
        Parameters
        ----------
        rawsong : ndarray
        audio
        samp_freq : int
        sampling frequency
        freq_cutoffs : list
        2 elements long, cutoff frequencies for bandpass filter.
        If None, no cutoffs; filtering is done with cutoffs set
        to range from 0 to the Nyquist rate.
        Default is [500, 10000].
        
        Returns
        -------
        filtsong : ndarray
        """
    if freq_cutoffs[0] <= 0:
        raise ValueError('Low frequency cutoff {} is invalid, '
                         'must be greater than zero.'
                         .format(freq_cutoffs[0]))
    
    Nyquist_rate = samp_freq / 2
    if freq_cutoffs[1] >= Nyquist_rate:
        raise ValueError('High frequency cutoff {} is invalid, '
                         'must be less than Nyquist rate, {}.'
                         .format(freq_cutoffs[1], Nyquist_rate))
    
    if rawsong.shape[-1] < 387:
        numtaps = 64
    elif rawsong.shape[-1] < 771:
        numtaps = 128
    elif rawsong.shape[-1] < 1539:
        numtaps = 256
    else:
        numtaps = 512

    cutoffs = np.asarray([freq_cutoffs[0] / Nyquist_rate,
                          freq_cutoffs[1] / Nyquist_rate])
                          # code on which this is based, bandpass_filtfilt.m, says it uses Hann(ing)
# window to design filter, but default for matlab's fir1
# is actually Hamming
# note that first parameter for scipy.signal.firwin is filter *length*
# whereas argument to matlab's fir1 is filter *order*
# for linear FIR, filter length is filter order + 1
    b = scipy.signal.firwin(numtaps + 1, cutoffs, pass_zero=False)
    a = np.zeros((numtaps+1,))
    a[0] = 1  # make an "all-zero filter"
    padlen = np.max((b.shape[-1] - 1, a.shape[-1] - 1))
    filtsong = scipy.signal.filtfilt(b, a, rawsong, padlen=padlen)
    #filtsong = filter_song(b, a, rawsong)
    return (filtsong)


def smooth_data(rawsong, samp_freq, freq_cutoffs = None, smooth_win=10):
    
    if freq_cutoffs is None:
        # then don't do bandpass_filtfilt
        filtsong = rawsong
    else:
        filtsong = bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs)

    squared_song = np.power(filtsong, 2)

    len = np.round(samp_freq * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    smooth = np.convolve(squared_song, h)
    offset = round((smooth.shape[-1] - filtsong.shape[-1]) / 2)
    smooth = smooth[offset:filtsong.shape[-1] + offset]
    return smooth



def segment_song(amp,
                 segment_params={'threshold': 5000, 'min_syl_dur': 0.2, 'min_silent_dur': 0.02},
                 time_bins=None,
                 samp_freq=None):
    """Divides songs into segments based on threshold crossings of amplitude.
        Returns onsets and offsets of segments, corresponding (hopefully) to syllables in a song.
        Parameters
        ----------
        amp : 1-d numpy array
        Either amplitude of power spectral density, returned by compute_amp,
        or smoothed amplitude of filtered audio, returned by evfuncs.smooth_data
        segment_params : dict
        with the following keys
        threshold : int
        value above which amplitude is considered part of a segment. default is 5000.
        min_syl_dur : float
        minimum duration of a segment. default is 0.02, i.e. 20 ms.
        min_silent_dur : float
        minimum duration of silent gap between segment. default is 0.002, i.e. 2 ms.
        time_bins : 1-d numpy array
        time in s, must be same length as log amp. Returned by Spectrogram.make.
        samp_freq : int
        sampling frequency
        
        Returns
        -------
        onsets : 1-d numpy array
        offsets : 1-d numpy array
        arrays of onsets and offsets of segments.
        
        So for syllable 1 of a song, its onset is onsets[0] and its offset is offsets[0].
        To get that segment of the spectrogram, you'd take spect[:,onsets[0]:offsets[0]]
        """
    
    if time_bins is None and samp_freq is None:
        raise ValueError('Values needed for either time_bins or samp_freq parameters '
                         'needed to segment song.')
    if time_bins is not None and samp_freq is not None:
        raise ValueError('Can only use one of time_bins or samp_freq to segment song, '
                         'but values were passed for both parameters')
    
    if time_bins is not None:
        if amp.shape[-1] != time_bins.shape[-1]:
            raise ValueError('if using time_bins, '
                             'amp and time_bins must have same length')

    above_th = amp > segment_params['threshold']
    h = [1, -1]
    # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    # and -1 whenever above_th changes from 1 to 0
    above_th_convoluted = np.convolve(h, above_th)
    
    if time_bins is not None:
        # if amp was taken from time_bins using compute_amp
        # note that np.where calls np.nonzero which returns a tuple
        # but numpy "knows" to use this tuple to index into time_bins
        onsets = time_bins[np.where(above_th_convoluted > 0)]
        offsets = time_bins[np.where(above_th_convoluted < 0)]
    elif samp_freq is not None:
        # if amp was taken from smoothed audio using smooth_data
        # here, need to get the array out of the tuple returned by np.where
        # **also note we avoid converting from samples to s
        # until *after* we find segments**
        onsets = np.where(above_th_convoluted > 0)[0]
        offsets = np.where(above_th_convoluted < 0)[0]
        
    if onsets.shape[0] < 1 or offsets.shape[0] < 1:
        return onsets, offsets #I'VE CHANGED
#        return None, None  # because no onsets or offsets in this file

    # get rid of silent intervals that are shorter than min_silent_dur
    silent_gap_durs = onsets[1:] - offsets[:-1]  # duration of silent gaps
    if samp_freq is not None:
        # need to convert to s
        silent_gap_durs = silent_gap_durs / samp_freq
    keep_these = np.nonzero(silent_gap_durs > segment_params['min_silent_dur'])
    onsets = np.concatenate(
                            (onsets[0, np.newaxis], onsets[1:][keep_these]))
    offsets = np.concatenate(
                         (offsets[:-1][keep_these], offsets[-1, np.newaxis]))
    
    # eliminate syllables with duration shorter than min_syl_dur
    syl_durs = offsets - onsets
    if samp_freq is not None:
        syl_durs = syl_durs / samp_freq
    keep_these = np.nonzero(syl_durs > segment_params['min_syl_dur'])
    onsets = onsets[keep_these]
    offsets = offsets[keep_these]

    #    if samp_freq is not None:
    #        onsets = onsets / samp_freq
    #        offsets = offsets / samp_freq

    return onsets, offsets



#Used to circumvent filtfilt if needed. See bandpass filtfilt function
def filter_song(b,a,rawsong):
    filt_len = len(b)
    rawsong_len = len(rawsong)
    filtered_song = []
    extended_rawsong = np.zeros(filt_len+rawsong_len)
    extended_rawsong[filt_len:len(extended_rawsong)] = rawsong
    for n in range(0,rawsong_len):
        result=0
        local_signal = extended_rawsong[n:(filt_len+n)]
        local_signal = local_signal[::-1]
        local_signal = local_signal*b
        result = np.sum(local_signal)
        filtered_song.append(result)
    
    filtered_song = np.asarray(filtered_song)
    return filtered_song

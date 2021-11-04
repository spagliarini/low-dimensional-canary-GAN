# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:36:29 2019

@author: Mnemosyne

"""

import os
import pickle
import scipy as sp
from scipy import signal
import numpy as np
import glob
from matplotlib import rcParams, cm, colors
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io.wavfile as wav
import tensorflow as tf
from scipy.io.wavfile import write as wavwrite
import librosa
import umap
import umap.plot

def open_pkl(name):
    data = open(name, 'rb')
    z0 = pickle.load(data)
    return z0

def pad_wavfile(args):
    """
    After having applied the function to generate variations in the latent space (in train_wavegan.py), this
    function merges the samples generated to obtain a easier visualization.
    """
    # pad together wav files
    sr, generation_baseline = wav.read(args.data_dir + '/' + 'generation_baseline.wav')
    latent_aux = open(args.data_dir + '/' + 'z_baseline.pkl', 'rb')
    latent_baseline = pickle.load(latent_aux)
    latent_dimension = args.wavegan_latent_dim

    silence = np.zeros((1500,))
    for i in range(0, latent_dimension):
        generation_pos_list = glob.glob(args.data_dir + '/' + 'gen_component_' + str(i) + '_pos*.wav')
        generation_neg_list = glob.glob(args.data_dir + '/' +'gen_component_' + str(i) + '_neg*.wav')

        exploration_song = silence
        exploration_latent = np.zeros((latent_dimension, np.size(generation_neg_list) + 1 + np.size(generation_pos_list)))
        if generation_neg_list != []:
            for j in range(0, np.size(generation_neg_list)):
                sr, generation_neg_aux = wav.read(args.data_dir + '/' +'gen_component_' + str(i) + '_neg' + str(j) + '.wav')
                exploration_song = np.append(generation_neg_aux[0:3000], exploration_song)
                exploration_song = np.append(silence, exploration_song)

                latent_aux = pickle.load(open(args.data_dir + '/' +'z_component_' + str(i) + '_neg' + str(j) + '.pkl', 'rb'))
                exploration_latent[:, np.size(generation_neg_list) - j - 1] = latent_aux[0]

        # add the original syllable
        exploration_song = np.append(exploration_song, generation_baseline[0:3000])
        # two silences before the original syllable
        exploration_song = np.append(exploration_song, silence)
        exploration_song = np.append(exploration_song, silence)

        if np.size(generation_neg_list) > 0:
            exploration_latent[:, np.size(generation_neg_list)] = latent_baseline
        else:
            exploration_latent[:, np.size(generation_neg_list)] = latent_baseline

        if generation_pos_list != []:
            for j in range(0, np.size(generation_pos_list)):
                sr, generation_pos_aux = wav.read(args.data_dir + '/' +'gen_component_' + str(i) + '_pos' + str(j) + '.wav')
                exploration_song = np.append(exploration_song, generation_pos_aux[0:3000])
                exploration_song = np.append(exploration_song, silence)

                latent_aux = pickle.load(open(args.data_dir + '/' +'z_component_' + str(i) + '_pos' + str(j) + '.pkl', 'rb'))
                exploration_latent[:, j + np.size(generation_neg_list) + 1] = latent_aux[0]

        wavwrite(args.data_dir + '/' + 'exploration_' + str(i) + '.wav', sr, exploration_song)

    print('Done')

def focus(args):
    """
    :return: take two latent vectors and go from one to the other component by component (meaning only one component is moved, the other two components are fixed)
    """
    # Open and save z
    z_l = pickle.load(open(args.data_dir + '/' + args.variation_dir + '/' + 'z_component_' + str(args.comp) + '_' + args.sign + str(args.id1) + '.pkl', 'rb'))
    generation_fp = os.path.join(variation_dir, 'z_l.pkl')
    with open(generation_fp, 'wb') as f:
        pickle.dump(z_l, f)

    z_r = pickle.load(open(args.data_dir + '/' + args.variation_dir + '/' + 'z_component_' + str(args.comp) + '_' + args.sign + str(args.id2) + '.pkl', 'rb'))
    generation_fp = os.path.join(variation_dir, 'z_r.pkl')
    with open(generation_fp, 'wb') as f:
        pickle.dump(z_r, f)

    # Load the graph
    tf.reset_default_graph()
    infer_metagraph_fp = os.path.join(args.data_dir, 'infer', 'infer.meta')
    saver = tf.train.import_meta_graph(infer_metagraph_fp)
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    if args.ckpt_n == False:
        ckpt_fp = tf.train.latest_checkpoint(args.data_dir)
    else:
        ckpt_fp = os.path.join(args.data_dir, 'model.ckpt-' + str(args.ckpt_n))
    saver.restore(sess, ckpt_fp)

    # Generate left and right
    # left
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]

    _G_z = sess.run(G_z, {z: z_l})

    generation_fp = os.path.join(variation_dir, 'gen_component_' + str(args.comp) + '_' + args.sign + '_l' + '.wav')
    wavwrite(generation_fp, args.data_sample_rate, _G_z.T)

    # right
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]

    _G_z = sess.run(G_z, {z: z_r})

    generation_fp = os.path.join(variation_dir, 'gen_component_' + str(args.comp) + '_' + args.sign + '_r' + '.wav')
    wavwrite(generation_fp, args.data_sample_rate, _G_z.T)

    counter = 0
    while ((z_l[0][args.comp] + args.variation_step) < z_r[0][args.comp]):
        z_l[0][args.comp] = z_l[0][args.comp] + args.variation_step

        _z = np.reshape(z_l, (1, args.wavegan_latent_dim))

        generation_fp = os.path.join(variation_dir, 'z_component_' + str(args.comp) + '_' + args.sign + str(counter) + '.pkl')
        # Save z
        with open(generation_fp, 'wb') as f:
            pickle.dump(_z, f)

        # Generate
        z = graph.get_tensor_by_name('z:0')
        G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]

        _G_z = sess.run(G_z, {z: _z})

        generation_fp = os.path.join(variation_dir, 'gen_component_' + str(args.comp) + '_' + args.sign + str(counter) + '.wav')
        wavwrite(generation_fp, args.data_sample_rate, _G_z.T)

        counter = counter + 1

    # Create sequence
    sr, generation_l = wav.read(variation_dir + '/' + 'gen_component_' + str(args.comp) + '_' + args.sign + '_l' + '.wav')
    sr, generation_r = wav.read(variation_dir + '/' + 'gen_component_' + str(args.comp) + '_' + args.sign + '_r' + '.wav')

    silence = np.zeros((1500,))
    exploration_song = silence
    exploration_song = np.append(generation_l[0:3000],silence)
    for c in range(0, counter):
        sr, gen_aux =  wav.read(variation_dir + '/' + 'gen_component_' + str(args.comp) + '_' + args.sign + str(c) + '.wav')
        exploration_song = np.append(exploration_song, gen_aux[0:3000])
        exploration_song = np.append(exploration_song, silence)
    exploration_song = np.append(exploration_song, generation_r[0:3000])

    wavwrite(variation_dir + '/' + 'exploration_comp_' + str(args.comp) + '_' + args.sign + '.wav', args.data_sample_rate, exploration_song)

    print('Done')

def bridge(args):
    """
    :return: take two latent vectors and go from one to the other component by component
    """
    # Open and save z
    z_l = pickle.load(open(
        args.data_dir + '/' + args.variation_dir + '/' + 'z_' + str(args.syll_0) + '.pkl', 'rb'))
    generation_fp = os.path.join(bridge_dir, 'z_l.pkl')
    with open(generation_fp, 'wb') as f:
        pickle.dump(z_l, f)

    print(z_l)

    z_r = pickle.load(open(
        args.data_dir + '/' + args.variation_dir + '/' + 'z_' + str(args.syll_1) + '.pkl', 'rb'))
    generation_fp = os.path.join(bridge_dir, 'z_r.pkl')
    with open(generation_fp, 'wb') as f:
        pickle.dump(z_r, f)

    print(z_r)
    # Load the graph
    tf.reset_default_graph()
    infer_metagraph_fp = os.path.join(args.data_dir, 'infer', 'infer.meta')
    saver = tf.train.import_meta_graph(infer_metagraph_fp)
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    if args.ckpt_n == False:
        ckpt_fp = tf.train.latest_checkpoint(args.data_dir)
    else:
        ckpt_fp = os.path.join(args.data_dir, 'model.ckpt-' +  str(args.ckpt_n))
    saver.restore(sess, ckpt_fp)

    # Generate left and right
    # left
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]

    _G_z = sess.run(G_z, {z: z_l})

    generation_fp = os.path.join(bridge_dir, 'gen_l' + '.wav')
    wavwrite(generation_fp, args.data_sample_rate, _G_z.T)

    # right
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]

    _G_z = sess.run(G_z, {z: z_r})

    generation_fp = os.path.join(bridge_dir, 'gen_r' + '.wav')
    wavwrite(generation_fp, args.data_sample_rate, _G_z.T)

    # Define the step per each component
    step = np.zeros((args.wavegan_latent_dim,))
    for i in range(0, args.wavegan_latent_dim):
        if z_r[0][i]>z_l[0][i]:
            step[i] = (z_r[0][i] - z_l[0][i])/args.steps
        else:
            step[i] = -abs(z_r[0][i] - z_l[0][i]) / args.steps

    # Generate intermediate syllables
    counter = 0
    while counter < args.steps:
        z_l[0] = z_l[0] + step

        _z = np.reshape(z_l, (1, args.wavegan_latent_dim))

        generation_fp = os.path.join(bridge_dir, 'z_' + str(counter) + '.pkl')
        # Save z
        with open(generation_fp, 'wb') as f:
            pickle.dump(_z, f)

        # Generate
        z = graph.get_tensor_by_name('z:0')
        G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]

        _G_z = sess.run(G_z, {z: _z})

        generation_fp = os.path.join(bridge_dir,'gen_' + str(counter) + '.wav')
        wavwrite(generation_fp, args.data_sample_rate, _G_z.T)

        counter = counter + 1

    print('Done')

def plot_exploration(data, args):
    for d in range(0, np.size(data)):
        y, sr = librosa.load(data[d], sr=16000)

        X = librosa.stft(y, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                                  pad_mode='constant',
                                  center=True)
        Y = np.log(1 + 100 * np.abs(X) ** 2)
        T_coef = np.arange(X.shape[1]) * args.H / sr
        K = args.N // 2
        F_coef = np.arange(K + 1) * sr / args.N

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 5)) #vary the size depending on the number of syllables
        extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
        ax.imshow(Y, cmap=args.color, aspect='auto', origin='lower', extent=extent, norm=colors.PowerNorm(gamma=0.5))
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (seconds)')
        plt.tight_layout()
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + os.path.basename(data[d])[0:-4] + '.' + args.format)
        plt.savefig(args.data_dir + '/' + os.path.basename(data[d])[0:-4] + '.' + 'png')

    print('Done')

def syllable(args):
    """
    :return: the name of the syllables in the sequence. This function is thought for a manual inspection of a limited sequence of syllables.
    """
    annotations = open_pkl(glob.glob(args.data_dir + '/' + 'annotations_generation*.pkl')[0])
    classes = annotations[0].vocab
    n_classes = len(annotations[0].vocab)

    list_of_recordings = []
    annotations_raw = []
    decoder_list_of_recordings = []
    raw_sum_dataset = np.zeros((len(annotations), n_classes))
    raw_sum_distr = np.zeros((len(annotations), n_classes))
    raw_max_dataset = np.zeros((len(annotations),))
    raw_max_indices_dataset = np.zeros((len(annotations),))
    classes_found = []
    for i in range(0, len(annotations)):
        list_of_recordings.append(os.path.basename(annotations[i].id))
        #print(list_of_recordings[i])

        annotations_raw.append(annotations[i].vect)
        # This operation should give me an idea of which class is represented the most in my generations
        raw_sum_dataset[i, :] = np.sum(annotations_raw[i], axis=0)
        raw_sum_distr[i,] = sp.special.softmax(raw_sum_dataset[i, :])
        raw_max_dataset[i] = np.max(raw_sum_dataset[i, :])
        raw_max_indices_dataset[i] = np.where(raw_sum_dataset[i, :] == np.max(raw_sum_dataset[i, :]))[0][0]
        classes_found.append(classes[int(raw_max_indices_dataset[i])])

        # Assigned class to each recording
        decoder_list_of_recordings.append(classes[int(raw_max_indices_dataset[i])])
        #print(decoder_list_of_recordings[i])
        #input()

    plt.subplots()
    plt.plot(raw_max_dataset)
    plt.ylabel('Probability')
    plt.xlabel('Syllable')
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/'  + 'probability_distr.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/'  + 'probability_distr.' + 'png')

    summary = {'File_name': list_of_recordings, 'Decoder_name': decoder_list_of_recordings, 'Annotations': raw_sum_distr}
    np.save(args.data_dir + '/' + 'Summary.npy', summary)

    print('Done')

def evolution_syll_plot(classes, ckpt_list, args):
    """
    :param classes: list of the classes
    In data_dir we need to have the generative model saved, in output_dir the latent vectors (and we use it to save too).

    :return: plot across epochs of the syllables
    """
    # Load the graph
    tf.reset_default_graph()
    infer_metagraph_fp = os.path.join(args.data_dir, 'infer', 'infer.meta')
    saver = tf.train.import_meta_graph(infer_metagraph_fp)
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()

    # Generate syllables across time
    for c in range(0, np.size(classes)):
        z_aux = open_pkl(args.data_dir + '/' + args.output_dir + '/' + 'z_' + classes[c] + '.pkl')

        for e in range(0, np.size(ckpt_list)):
            ckpt_fp = os.path.join(args.data_dir, 'model.ckpt-' + str(ckpt_list[e]))
            saver.restore(sess, ckpt_fp)

            z = graph.get_tensor_by_name('z:0')
            G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]

            _G_z = sess.run(G_z, {z: z_aux})

            generation_fp = os.path.join(args.data_dir + '/' + args.output_dir, 'evolution_' + classes[c] + '_ckpt_' + str(ckpt_list[e]) + '.wav')
            wavwrite(generation_fp, args.data_sample_rate, _G_z.T)

    # Plot sequence of all the syllables
    fig, axs = plt.subplots(ncols=1, nrows=np.size(ckpt_list), figsize=(8, 5))
    for e in range(0, np.size(ckpt_list)):
        sequence = [0]
        for c in range(0, np.size(classes)):
            y, sr = librosa.load(args.data_dir + '/' + args.output_dir + '/' + 'evolution_' + classes[c] + '_ckpt_' + str(ckpt_list[e]) + '.wav', sr=args.data_sample_rate)
            trim = librosa.effects.trim(y, top_db=20)
            y = trim[0]
            sequence = np.append(sequence, y)

        X = librosa.stft(sequence, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                         pad_mode='constant',
                         center=True)
        Y = np.log(1 + 100 * np.abs(X) ** 2)
        T_coef = np.arange(X.shape[1]) * args.H / args.data_sample_rate
        K = args.N // 2
        F_coef = np.arange(K + 1) * args.data_sample_rate / args.N

        extent_template = [T_coef[0], T_coef[-1], F_coef[0], 8000]
        axs[e].imshow(Y, cmap=args.color, aspect='auto', origin='lower', extent=extent_template,
                   norm=colors.PowerNorm(gamma=args.gamma))
        axs[e].set_ylabel('Frequency (Hz)')
        axs[e].set_xticklabels([])
        axs[e].title.set_text('Epoch ' + str(int(ckpt_list[e]*64*5/16000)))

    plt.tight_layout()
    if args.format != 'png':
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'evolution_plot.' + args.format)
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'evolution_plot.' + 'png')

    # Plot one syllable at the time
    for c in range(0, np.size(classes)):
        fig, axs = plt.subplots(ncols=1, nrows=np.size(ckpt_list), figsize=(2, 11))
        for e in range(0, np.size(ckpt_list)):
            y, sr = librosa.load(args.data_dir + '/' + args.output_dir + '/' + 'evolution_' + classes[c] + '_ckpt_' + str(ckpt_list[e]) + '.wav', sr=args.data_sample_rate)
            trim = librosa.effects.trim(y, top_db=20)
            y = trim[0]

            X = librosa.stft(y, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann',
                             pad_mode='constant',
                             center=True)
            Y = np.log(1 + 100 * np.abs(X) ** 2)
            T_coef = np.arange(X.shape[1]) * args.H / args.data_sample_rate
            K = args.N // 2
            F_coef = np.arange(K + 1) * args.data_sample_rate / args.N

            extent_template = [T_coef[0], T_coef[-1], F_coef[0], 8000]
            axs[e].imshow(Y, cmap=args.color, aspect='auto', origin='lower', extent=extent_template,
                          norm=colors.PowerNorm(gamma=args.gamma))
            axs[e].set_ylabel('Frequency (Hz)')
            axs[e].set_xticklabels([])
        plt.tight_layout()
        if args.format != 'png':
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'evolution_plot_class_' + classes[c] + '.' + args.format)
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'evolution_plot_class_' + classes[c] + '.' + 'png')

    print('Done')

if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()

  parser.add_argument('--option', type=str, choices=['padwav', 'focus', 'bridge', 'plot', 'syll', 'evolution'])
  parser.add_argument('--data_dir', type=str, help='Directory where to find the data')
  parser.add_argument('--output_dir', type=str, help='Directory where to save the output, if it is desired to save them somewhere else', default=None)
  parser.add_argument('--variation_dir', type=str, help='Directory where there are already variations saved, to apply focus', default=None)

  wavegan_args = parser.add_argument_group('WaveGAN')
  wavegan_args.add_argument('--wavegan_latent_dim', type=int, help='Number of dimensions of the latent space', default = 3)

  analysis_args = parser.add_argument_group('features_analysis')
  analysis_args.add_argument('--window', type=str, help='Type of window for the visualization of the spectrogram',
                             default='hanning')
  analysis_args.add_argument('--overlap', type=int, help='Overlap for the visualization of the spectrogram', default=64)
  analysis_args.add_argument('--nperseg', type=int, help='Nperseg for the visualization of the spectrogram', default=512)
  analysis_args.add_argument('--data_sample_rate', type=int, help='Sampling rate used to train and generated', default=16000)

  plot_args = parser.add_argument_group('Plot')
  plot_args.add_argument('--format', type=str, help='Saving format', default='png')
  plot_args.add_argument('--color', type=str, help='Colormap', default='inferno')
  plot_args.add_argument('--N', type=int, help='Nftt spectrogram librosa', default=256)
  plot_args.add_argument('--H', type=int, help='Hop length spectrogram librosa', default=64)
  plot_args.add_argument('--gamma', type=float, help='powernorm color, 0.2 training, 0.5 generated', default=0.5)

  variation_args = parser.add_argument_group('Variation_Bridge')
  variation_args.add_argument('--variation_step', type=float,
      help='Variation step applied to each component of the latent vector')
  variation_args.add_argument('--sign', type=str, help='Pos or neg value, options pos/neg abd eventually pos1 2 3 ..')
  variation_args.add_argument('--id1', type=int, help="Number to identify the first element")
  variation_args.add_argument('--id2', type=int, help="Number to identify the second element")
  variation_args.add_argument('--comp', type=int, help='Which component: 0 1 2 ..')
  variation_args.add_argument('--ckpt_n', type=int, help='At which chekckpoint')
  variation_args.add_argument('--syll_0', type=str, help='From wich syllable')
  variation_args.add_argument('--syll_1', type=str, help='To wich syllable')
  variation_args.add_argument('--steps', type=str, help='In how many steps', default=999)

  args = parser.parse_args()

  # Output direction creation
  if args.output_dir != None:
      if not os.path.isdir(args.data_dir + '/' + args.output_dir):
        os.makedirs(args.data_dir + '/' + args.output_dir)

  if args.option == 'padwav':
      pad_wavfile(args)

  if args.option == 'focus':
      # Create the output directory
      variation_dir = os.path.join(args.data_dir, 'variation_' + str(args.variation_step)[-2::])
      if not os.path.isdir(variation_dir):
          os.makedirs(variation_dir)

      # Save args
      with open(os.path.join(variation_dir, 'args.txt'), 'w') as f:
          f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

      focus(args)

  if args.option == 'bridge':
      # Create the output directory
      bridge_dir = os.path.join(args.data_dir, args.variation_dir, 'bridge_' + args.syll_0 + '_' + args.syll_1)
      if not os.path.isdir(bridge_dir):
          os.makedirs(bridge_dir)

      # Save args
      with open(os.path.join(bridge_dir, 'args.txt'), 'w') as f:
          f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

      bridge(args)

  if args.option == 'plot':
      data = sorted(glob.glob(args.data_dir + '/' + 'exploration*wav'))
      plot_exploration(data, args)

  if args.option == 'syll':
      syllable(args)

  if args.option == 'evolution':
      # Classes
      classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

      # Checkpoint list
      ckpt_list = [0, 741, 2252, 5277, 26483, 49202]

      evolution_syll_plot(classes, ckpt_list, args)
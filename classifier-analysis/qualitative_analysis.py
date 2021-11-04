# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:22:29 2019

@author: Mnemosyne
"""

import os
import random
import glob
import numpy as np
import librosa
import librosa.display
import librosa.feature
import librosa.effects
import scipy.io.wavfile as wav
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sklearn
import shutil
import xlsxwriter
import xlrd
import pandas as pd

def qualitative_test(songs, args):
    '''
    Select N syllables in a directory and copy to a new one.
    '''

    # Random select the syllables
    names = random.sample(songs, args.n_template)
    save_names = []
    test_names = []
    for i in range(0, np.size(names)):
        shutil.copyfile(args.data_dir + '/' + os.path.basename(names[i]), args.data_dir + '/' + args.output_dir + '/' + 'test_' + str(100+i) + '.wav')
        save_names.append(os.path.basename(names[i]))
        test_names.append(args.data_dir + '/' + args.output_dir + '/' + 'test_' + str(i) + '.wav')

    np.save(args.data_dir + '/' + args.output_dir + '/' + 'real_names.npy', save_names)

    print('Done')

def qualitative_table(args):
    """
    Build an excel table to run the test
    """

    # Initialize sheet
    workbook = xlsxwriter.Workbook(args.data_dir + '/' + 'Qualitative_table_test.xlsx')
    worksheet = workbook.add_worksheet()

    # Start from the first cell.
    content = ["Test", "Guess", "Real name", "Classifier"]
    # Rows and columns are zero indexed.
    row = 0
    column = 0
        # iterating through content list
    for item in content:
        # write operation perform
        worksheet.write(row, column, item)

        # incrementing the value of row by one
        # with each iteratons.
        column += 1

    row=1
    column=0
    for item in range(100,args.n_template+100):
        # write test names
        worksheet.write(row, column, item)

        # incrementing the value of row by one
        # with each iteratons.
        row += 1

    workbook.close()

    print('Done')

def qualitative_analysis(args):
    """
    Build an excel table to run the analysis.
    Then to this one one needs to add manually the answers of the judges.
    """
    # Read sheet
    workbook = xlsxwriter.Workbook(args.data_dir + '/' + 'Qualitative_table_analysis.xlsx')
    worksheet = workbook.add_worksheet()

    # Start from the first cell.
    content = ["Test", "Guess", "Real name", "Classifier"]
    # Rows and columns are zero indexed.
    row = 0
    column = 0
        # iterating through content list
    for item in content:
        # write operation perform
        worksheet.write(row, column, item)

        # incrementing the value of row by one
        # with each iteratons.
        column += 1

    real_names = np.load(args.data_dir + '/' + 'real_names.npy')
    row=1
    column=2
    for item in range(0, np.size(real_names)):
        # write test names
        worksheet.write(row, column, real_names[item][0:6])

        # incrementing the value of row by one
        # with each iteratons.
        row += 1

    filename = glob.glob(args.data_dir + '/' + '*summary.npy')
    aux = np.load(filename[0], allow_pickle=True)
    aux = aux.item()
    decoder_names = aux['Decoder_name']
    row=1
    column=3
    for item in range(0, np.size(real_names)):
        # write test names
        worksheet.write(row, column, decoder_names[item])

        # incrementing the value of row by one
        # with each iteratons.
        row += 1

    workbook.close()

    print('Done')

def cohen_kappa(classes, args):
    """
    :param classes: vocabulary
    :return: cohen's kappa coefficient
    """
    workbook = xlrd.open_workbook(args.data_dir + '/' + 'Qualitative_table_generation.xlsx')
    sheet = workbook.sheet_by_index(0)

    column_name = []
    judges = []
    for j in range(1,sheet.ncols):
        column_name.append(sheet.cell_value(0, j))
        aux = []
        for i in range(1, sheet.nrows):
            aux.append(sheet.cell_value(i, j))
        judges.append(aux)

    judges_classes = np.zeros((np.size(classes),len(judges)))
    for j in range(0, len(judges)):
        for c in range(0, np.size(classes)):
            judges_classes[c,j] = np.size(np.where(np.array(judges[j]) == classes[c]))/sheet.nrows

    # Operator to apply 1 - matrix
    w_mat = np.ones([np.size(classes), np.size(classes)], dtype=int)
    w_mat.flat[:: np.size(classes) + 1] = 0

    Cohen_k = []
    confusion = []
    k_MAX = []
    for i in range(0,len(judges)):
        print(column_name[i])
        expected = []
        for j in range(0, len(judges)):
            print(column_name[i])
            print(column_name[j])
            # Find max k
            confusion.append(sklearn.metrics.confusion_matrix(judges[i], judges[j], labels=classes, sample_weight=None))
            sum0 = np.sum(confusion[j], axis=0)
            sum1 = np.sum(confusion[j], axis=1)
            expected.append(np.outer(sum0, sum1) / np.sum(sum0))

            judges_classes_aux = []
            judges_classes_aux.append(judges_classes[:, i])
            judges_classes_aux.append(judges_classes[:,j])
            min_classes = np.min(judges_classes_aux, axis = 0)

            k_MAX.append(1 - (1 - np.sum(min_classes))/np.sum(w_mat*expected[j]))

            # k
            Cohen_k.append(sklearn.metrics.cohen_kappa_score(judges[i], judges[j]))

            print(k_MAX[j])
            print(Cohen_k[j])

    print('Done')

if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()
  parser.add_argument('--option', type=str, choices=['test', 'table', 'analysis', 'k'])
  parser.add_argument('--data_dir', type=str, help='Directory containing the data',
                      default=None)
  parser.add_argument('--output_dir', type=str, help='Directory where to save the output',
                      default=None)

  plot_args = parser.add_argument_group('Plot')
  plot_args.add_argument('--format', type=str, help='Saving format', default='png')
  plot_args.add_argument('--n_template', type=int, help='How many syllable to consider', default=200)

  args = parser.parse_args()

  # Output direction creation
  if args.output_dir != None:
      if not os.path.isdir(args.data_dir + '/' + args.output_dir):
        os.makedirs(args.data_dir + '/' + args.output_dir)

  if args.option =='test':
      songs = glob.glob(args.data_dir + '/' + '*.wav')
      qualitative_test(songs, args)

  if args.option == 'table':
      qualitative_table(args)

  if args.option == 'analysis':
      qualitative_analysis(args)

  if args.option == 'k':
      classes = ['A', 'B', 'C', 'D', 'E', 'H', 'J', 'L', 'M', 'N', 'O', 'Q', 'R', 'V', 'X']
      cohen_kappa(classes, args)

import numpy as np
import csv
import os
import glob
import matplotlib.pyplot as plt

def loss_plot(args):
    """
    Function to plot the loss functions. The value should be downloaded from Tensorboard in csv format.
    """
    # Read the loss data (downloaded from Tensorboard)
    D_loss_file = glob.glob(args.data_dir + '/' + '*D_loss.csv')
    G_loss_file = glob.glob(args.data_dir + '/' + '*G_loss.csv')
    csv_reader_D = csv.reader(open(D_loss_file[0]), delimiter=',')
    csv_reader_G = csv.reader(open(G_loss_file[0]), delimiter=',')
    xD = list(csv_reader_D)
    D = np.array(xD)[:, 1:3]
    D[0, 0:args.limit_epoch] = 0
    D = D[1::]
    D = D.astype("float")

    xG = list(csv_reader_G)
    G = np.array(xG)[:, 1:3]
    G[0, 0:args.limit_epoch] = 0
    G = G[1::]
    G = G.astype("float")

    D_steps = np.linspace(0, np.int(D[-1, 0]) + 10, D.shape[0] - 1)
    G_steps = np.linspace(0, np.int(G[-1, 0]) + 10, G.shape[0] - 1)

    D_last_epoch = round(np.int(D[-1, 0] * args.wavegan_disc_nupdates * args.train_batch_size) / args.dataset_dim)
    G_last_epoch = round(np.int(G[-1, 0] * args.wavegan_disc_nupdates * args.train_batch_size) / args.dataset_dim)

    # Plots D loss
    fig = plt.subplots()
    plt.plot(D_steps, D[1::, 1])
    plt.xticks((0, np.int(D[-1, 0])), ('0', D_last_epoch))
    plt.xlabel('Epoch')
    plt.ylabel('D loss')
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(args.data_dir + '/' + 'D_loss_' + str(args.limit_epoch) + '.png')

    # Plot G_loss
    fig = plt.subplots()
    plt.plot(G_steps, G[1::, 1])
    plt.xticks((0, np.int(G[-1, 0])), ('0', G_last_epoch))
    plt.xlabel('Epoch')
    plt.ylabel('G loss')
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(args.data_dir + '/' + 'G_loss_' + str(args.limit_epoch) + '.png')

    print('Done')

def cfr(args, list_directory):
    """
    Function to plot the loss functions. The value should be downloaded from Tensorboard in csv format.
    Comparison between different instances, indicated in the parameter "list_directory"
    """
    D_steps = list()
    G_steps = list()
    D_epochs = list()
    G_epochs = list()
    D_last_epoch = list()
    G_last_epoch = list()
    D_all = list()
    G_all = list()

    # Read the data
    for l in range(0, np.size(list_directory)):
        # Read the loss data (downloaded from Tensorboard)
        D_loss_file = glob.glob(list_directory[l] +'/' + '*D_loss.csv')
        G_loss_file = glob.glob(list_directory[l] +'/' + '*G_loss.csv')
        csv_reader_D = csv.reader(open(D_loss_file[0]), delimiter=',')
        csv_reader_G = csv.reader(open(G_loss_file[0]), delimiter=',')
        xD = list(csv_reader_D)
        D = np.array(xD)[0:args.limit_epoch, 1:3]
        D[0, :] = 0
        D = D[1::]
        D_all.append(D.astype("float"))

        xG = list(csv_reader_G)
        G = np.array(xG)[0:args.limit_epoch, 1:3]
        G[0, :] = 0
        G = G[1::]
        G_all.append(G.astype("float"))

        D_steps.append(np.linspace(0, np.int(D_all[l][-1, 0]) + 10, D_all[l].shape[0] - 1))
        D_epochs.append(np.round((D_steps[l] * args.wavegan_disc_nupdates * args.train_batch_size) / args.dataset_dim))
        G_steps.append(np.linspace(0, np.int(G_all[l][-1, 0]) + 10, G_all[l].shape[0] - 1))
        G_epochs.append(np.round((G_steps[l] * args.wavegan_disc_nupdates * args.train_batch_size) / args.dataset_dim))

        D_last_epoch.append(round(np.int(D_all[l][-1, 0] * args.wavegan_disc_nupdates * args.train_batch_size) / args.dataset_dim))
        G_last_epoch.append(round(np.int(G_all[l][-1, 0] * args.wavegan_disc_nupdates * args.train_batch_size) / args.dataset_dim))

    # Plot D_loss
    fig, ax = plt.subplots()
    for l in range(0,np.size(list_directory)):
        plt.plot(D_epochs[l], D_all[l][1::, 1]/np.sum(D_all[l][1::, 1]), label='latent dim = ' + list_directory[l][-1])
    #ax.set_yscale('log')
    ax.set_xlim(0, D_epochs[1][-1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Time (in number of epochs)')
    plt.ylabel('D loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + 'CFR_D_loss_' + str(args.limit_epoch) + '.png')

    # Plot G_loss
    fig, ax = plt.subplots()
    for l in range(0,np.size(list_directory)):
        plt.plot(G_epochs[l], G_all[l][1::, 1]/np.sum(G_all[l][1::, 1]), label='latent dim = ' + list_directory[l][-1])
    #ax.set_yscale('log')
    ax.set_xlim(0, G_epochs[1][-1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Time (in number of epochs)')
    plt.ylabel('G loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + 'CFR_G_loss_' + str(args.limit_epoch) + '.png')

    print('Done')

if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()
  parser.add_argument('--option', type=str, choices=['loss', 'cfr'])

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_dir', type=str,
                         help='Data directory containing the data to plot',
                         default=None)

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--dataset_dim', type=int,
                         help='Dimension of the training dataser (to determine the number of epochs)',
                         default=23456)
  train_args.add_argument('--train_batch_size', type=int,
                          help='Batch size',
                          default=64)
  train_args.add_argument('--wavegan_disc_nupdates', type=int,
                          help='Number of discriminator updates per generator update',
                          default=5)

  plot_args = parser.add_argument_group('Plot')
  plot_args.add_argument('--limit_epoch', type=int,
                         help='Last epoch we wont to plot. Pay attention to choose one lower than the shorter training. '
                              'Number in term of the number of lines in the excel tables.',
                         default=-1)

  args = parser.parse_args()

  # Save args
  with open(os.path.join(args.data_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

  if args.option == 'loss':
      loss_plot(args)

  elif args.option == 'cfr':
      # List of the directories where to find the excel tables relative to the loss values
      # The name should be *****LATENT_DIM (such as: "Tensorboard1" where 1 is the latent dimension used in the training
      list_directory = sorted(glob.glob(args.data_dir + '/' + 'Tensorboard*'))
      cfr(args, list_directory)
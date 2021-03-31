import argparse
import os

import torch

import learnergy.visual.tensor as t
import utils.loader as l

# Caveat to enable image showing on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Reconstructs a pre-trained FSRBM/RBM model.')

    parser.add_argument('dataset', help='Dataset identifier', type=str, choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('model_file', help='Model file identifier', type=str)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    model_file = args.model_file
    seed = args.seed

    # Loads the data
    _, _, test = l.load_dataset(name=dataset)

    # Defining the torch seed
    torch.manual_seed(seed)
    
    # Loads the pre-trained model
    rbm = torch.load(model_file)

    # Reconstructs the model
    mse, v = rbm.reconstruct(test)

    # Showing a reconstructed sample
    t.show_tensor(v[0].reshape(28, 28))

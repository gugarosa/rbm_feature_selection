import argparse
import glob
import os

import learnergy.visual.tensor as t
import torch
from natsort import natsorted

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Reconstructs a batch of pre-trained FSRBM/RBM models.')

    parser.add_argument('dataset', help='Dataset identifier', type=str, choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('model_file_stem', help='Model file identifier without epoch and extension', type=str)

    parser.add_argument('-mask_file_stem', help='Mask file identifier without epoch and extension', type=str)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    model_file_stem = args.model_file_stem
    mask_file_stem = args.mask_file_stem
    seed = args.seed

    # Checks the folder and creates a list of model files
    model_files = natsorted(glob.glob(f'{model_file_stem}*'))

    # Checks if there are supplied mask files
    if mask_file_stem:
        # If yes, creates a list of mask files
        mask_files = natsorted(glob.glob(f'{mask_file_stem}*'))

    # If not
    else:
        # Creates a list of None
        mask_files = [None for _ in range(len(model_files))]

    # Instantiates a list for holding the MSE
    mse_values = []
    mask_features_values = []

    # Iterates through every possible file
    for (model_file, mask_file) in zip(model_files, mask_files):
        # Loads the data
        _, _, test, mask_features = l.load_dataset(name=dataset, mask_file=mask_file)

        # Defines the torch seed
        torch.manual_seed(seed)

        # Loads the pre-trained model
        rbm = torch.load(model_file)
    
        # Reconstructs the model
        mse, v = rbm.reconstruct(test)

        # Appends the MSE
        mse_values.append(mse.detach().numpy().item())
        mask_features_values.append(mask_features)

        # Creates the output path for the tensor
        output_path = f'mask_features_{mask_features}_{os.path.splitext(os.path.basename(model_file))[0]}.png'

        # Saves a reconstructed sample
        t.save_tensor(v[0].reshape(28, 28), output_path)
        
    print(f'MSE: {mse_values}')
    print(f'Mask Features: {mask_features_values}')

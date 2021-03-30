import argparse
import os

import torch

import learnergy.visual.tensor as t
import utils.loader as l
from core.fsrbm import FSRBM

# Caveat to enable image showing on MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Trains, reconstructs and saves a FSRBM model.')

    parser.add_argument('dataset', help='Dataset identifier', type=str, choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('-n_visible', help='Number of visible units', type=int, default=784)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.1)

    parser.add_argument('-momentum', help='Momentum', type=float, default=0)

    parser.add_argument('-decay', help='Weight decay', type=float, default=0)

    parser.add_argument('-temperature', help='Temperature', type=float, default=1)

    parser.add_argument('-mask_type', help='Type of mask', type=str, choices=['sigmoid', 'diff'], default='sigmoid')

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=10)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    parser.add_argument('--use_binary_sampling', help='Usage of binary sampling', action='store_true')

    parser.add_argument('--use_gpu', help='Usage of GPU', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    n_visible = args.n_visible
    n_hidden = args.n_hidden
    steps = args.steps
    lr = args.lr
    momentum = args.momentum
    decay = args.decay
    T = args.temperature
    mask_type = args.mask_type
    batch_size = args.batch_size
    epochs = args.epochs
    seed = args.seed
    use_binary_sampling = args.use_binary_sampling
    use_gpu = args.use_gpu

    # Loads the data
    train, _, test = l.load_dataset(name=dataset)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Instantiates the model
    rbm = FSRBM(n_visible=n_visible, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                momentum=momentum, decay=decay, temperature=T, mask_type=mask_type,
                use_binary_sampling=use_binary_sampling, use_gpu=use_gpu)

    # Fitting the model
    rbm.fit(train, batch_size=batch_size, epochs=epochs)

    # Reconstructs the model
    mse, v = rbm.reconstruct(test)

    # Showing a reconstructed sample
    t.show_tensor(v[0].reshape(28, 28))

    # Saving the model
    # torch.save(rbm, f'outputs/{n_hidden}hid_{lr}lr_fsrbm_{dataset}_{seed}.pth')

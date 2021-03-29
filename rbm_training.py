import argparse

import torch

import utils.loader as l
from core.fsrbm import FSRBM
from learnergy.models.bernoulli import RBM


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Trains, reconstructs and saves a FSRBM model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('-n_visible', help='Number of visible units', type=int, default=784)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.1)

    parser.add_argument('-momentum', help='Momentum', type=float, default=0)

    parser.add_argument('-decay', help='Weight decay', type=float, default=0)

    parser.add_argument('-temperature', help='Temperature', type=float, default=1)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=5)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

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
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    seed = args.seed

    # Checks for the name of device
    if device == 'cpu':
        # Updates accordingly
        use_gpu = False
    else:
        # Updates accordingly
        use_gpu = True

    # Loads the data
    train, _, test = l.load_dataset(name=dataset)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Instantiates the model
    rbm = FSRBM(n_visible=n_visible, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                momentum=momentum, decay=decay, temperature=T, use_gpu=use_gpu)

    # Fitting the model
    rbm.fit(train, batch_size=batch_size, epochs=epochs)

    # f = (rbm.f - torch.min(rbm.f)) / (torch.max(rbm.f) - torch.min(rbm.f))
    # f = torch.bernoulli(f)

    # print(f)

    # Reconstructs the model
    # mse, _ = rbm.reconstruct(test)

    # Saving the model
    # torch.save(rbm, f'models/{n_hidden}hid_{lr}lr_fsrbm_{dataset}_{seed}.pth')

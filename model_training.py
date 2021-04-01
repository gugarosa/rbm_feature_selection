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
    parser = argparse.ArgumentParser(usage='Trains, reconstructs and saves a FSRBM/RBM model.')

    parser.add_argument('dataset', help='Dataset identifier', type=str, choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('model', help='Model identifier', type=str, choices=['fsrbm', 'rbm'])

    parser.add_argument('-n_visible', help='Number of visible units', type=int, default=784)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.1)

    parser.add_argument('-momentum', help='Momentum', type=float, default=0)

    parser.add_argument('-decay', help='Weight decay', type=float, default=0)

    parser.add_argument('-temperature', help='Temperature', type=float, default=1)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=10)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    parser.add_argument('-input_mask_fn', help='Mask function', type=str, default='sigmoid', choices=['sigmoid', 'soft_step'])

    parser.add_argument('--use_binary_sampling', help='Usage of binary sampling', action='store_true')

    parser.add_argument('--use_gpu', help='Usage of GPU', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    model = args.model
    n_visible = args.n_visible
    n_hidden = args.n_hidden
    steps = args.steps
    lr = args.lr
    momentum = args.momentum
    decay = args.decay
    T = args.temperature
    batch_size = args.batch_size
    epochs = args.epochs
    seed = args.seed
    input_mask_fn = args.input_mask_fn
    use_binary_sampling = args.use_binary_sampling
    use_gpu = args.use_gpu

    # Loads the data
    train, _, _ = l.load_dataset(name=dataset)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Checks if supplied model is a FSRBM
    if model == 'fsrbm':
        # Instantiates the model
        rbm = FSRBM(n_visible=n_visible, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                    momentum=momentum, decay=decay, temperature=T, input_mask_fn=input_mask_fn,
                    use_binary_sampling=use_binary_sampling, use_gpu=use_gpu)
    
    # If not, it is a standard RBM
    else:
        # Instantiates the model
        rbm = RBM(n_visible=n_visible, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                  momentum=momentum, decay=decay, temperature=T, use_gpu=use_gpu)

    # Fitting the model
    rbm.fit(train, batch_size=batch_size, epochs=epochs)

    # Saving the model
    torch.save(rbm, f'outputs/{n_hidden}hid_{lr}lr_{model}_{dataset}_{seed}.pth')

    # Checks if supplied model is a FSRBM
    if model == 'fsrbm':
        # Checks if input mask is sigmoid
        if input_mask_fn == 'sigmoid':
            f = torch.sigmoid(rbm.f)
        
        # Checks if input mask is soft step
        elif input_mask_fn == 'soft_step':
            f = rbm.soft_step(rbm.f)

        # Samples the mask and saves it
        mask = torch.bernoulli(f)
        torch.save(mask, f'outputs/{n_hidden}hid_{lr}lr_mask_{dataset}_{seed}.pth')

        print(f'Number of features in the mask: {torch.count_nonzero(mask)}')

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Classifies a pre-trained FSRBM/RBM model.')

    parser.add_argument('dataset', help='Dataset identifier', type=str, choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('model_file', help='Model file identifier', type=str)

    parser.add_argument('-mask_file', help='Mask file identifier', type=str)

    parser.add_argument('-n_classes', help='Number of classes', type=int, default=10)

    parser.add_argument('-fine_tune_rbm_lr', help='Fine-tuning RBM learning rate', type=float, default=0.00001)

    parser.add_argument('-fine_tune_lr', help='Fine-tuning learning rate', type=float, default=0.001)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-fine_tune_epochs', help='Fine-tuning epochs', type=int, default=10)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    model_file = args.model_file
    mask_file = args.mask_file
    n_classes = args.n_classes
    fine_tune_rbm_lr = args.fine_tune_rbm_lr
    fine_tune_lr = args.fine_tune_lr
    batch_size = args.batch_size
    fine_tune_epochs = args.fine_tune_epochs
    seed = args.seed
    mask_str = ''

    # Checks if there are supplied mask files
    if mask_file:
        # If yes, creates a list of mask files
        mask_str = 'mask_'

    # Loads the data
    train, _, test, _ = l.load_dataset(name=dataset, mask_file=mask_file)

    # Defines the torch seed
    torch.manual_seed(seed)
    
    # Loads the pre-trained model
    model = torch.load(model_file)

    # Creating the Fully Connected layer to append on top of RBM
    fc = nn.Linear(model.n_hidden, n_classes)

    # Check if model uses GPU
    if model.device == 'cuda':
        # If yes, put fully-connected on GPU
        fc = fc.cuda()

    # Cross-Entropy loss is used for the discriminative fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Creating the optimzers
    optimizer = [optim.Adam(model.parameters(), lr=fine_tune_rbm_lr),
                optim.Adam(fc.parameters(), lr=fine_tune_lr)]

    # Creating training and validation batches
    train_batch = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
    val_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=1)

    # Instatiates a list of metric outputs
    train_losses, val_accs = [], []

    # Defines the output path
    output_path = f'{mask_str}{os.path.splitext(os.path.basename(model_file))[0]}.txt'

    # For amount of fine-tuning epochs
    for e in range(fine_tune_epochs):
        print(f'Epoch {e+1}/{fine_tune_epochs}')

        # Resetting metrics
        train_loss, val_acc = 0, 0
        
        # For every possible batch
        for x_batch, y_batch in tqdm(train_batch):
            # For every possible optimizer
            for opt in optimizer:
                # Resets the optimizer
                opt.zero_grad()
            
            # Flatenning the samples batch
            x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if model.device == 'cuda':
                # Applies the GPU usage to the data and labels
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            # Passing the batch down the model
            y = model(x_batch)

            # Calculating the fully-connected outputs
            y = fc(y)
            
            # Calculating loss
            loss = criterion(y, y_batch)
            
            # Propagating the loss to calculate the gradients
            loss.backward()
            
            # For every possible optimizer
            for opt in optimizer:
                # Performs the gradient update
                opt.step()

            # Adding current batch loss
            train_loss += loss.item()
            
        # Calculate the test accuracy for the model:
        for x_batch, y_batch in tqdm(val_batch):
            # Flatenning the testing samples batch
            x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if model.device == 'cuda':
                # Applies the GPU usage to the data and labels
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            # Passing the batch down the model
            y = model(x_batch)

            # Calculating the fully-connected outputs
            y = fc(y)

            # Calculating predictions
            _, preds = torch.max(y, 1)

            # Calculating validation set accuracy
            val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

        # Appends metrics to list
        train_losses.append(train_loss.detach().numpy() / len(train_batch))
        val_accs.append(val_acc.detach().numpy())

        print(f'Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}')

    # Converts lists to a dataframe
    df = pd.DataFrame({'train_loss': train_losses,
                       'val_acc': val_accs})

    # Saves the dataframe to an output .csv file
    df.to_csv(output_path, index=False)

import torch
import torchvision as tv

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'mnist': tv.datasets.MNIST,
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST
}


class BinaryMask:
    """Applies a binary mask in a sample.

    """

    def __init__(self, mask_file):
        """Initialization method.

        Args:
            mask_file (str): File holding the binary mask to be loaded.

        """

        # Loads and defines a mask property
        self.mask = torch.load(mask_file)

    def __call__(self, sample):
        # Gathers the size of the current sample
        size = sample.size()

        # Re-shapes the mask and applies over the tensor
        mask = torch.reshape(self.mask, size)
        sample = torch.mul(sample, mask)

        return sample


def load_dataset(name='mnist', val_split=0.2, mask_file=None):
    """Loads an input dataset.

    Args:
        name (str): Name of dataset to be loaded.
        val_split (float): Percentage of split for the validation set.
        mask_file (str): File holding the mask to be applied over the input.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(0)

    # Checks if there is a supplied mask file
    if mask_file:
        # If yes, creates the composed transform
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            BinaryMask(mask_file)
        ])

    # If there is no supplied mask file
    else:
        # Just uses the standard transform
        transform = tv.transforms.ToTensor()

    # Loads the training data
    train = DATASETS[name](root='./data', train=True, download=True,
                           transform=transform)

    # Splitting the training data into training/validation
    train, val = torch.utils.data.random_split(
        train, [int(len(train) * (1 - val_split)), int(len(train) * val_split)])

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
                          transform=transform)

    return train, val, test

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

    def __init__(self, mask):
        """Initialization method.

        Args:
            mask (torch.Tensor): Tensor holding the binary mask to be applied.

        """

        # Loads and defines a mask property
        self.mask = mask

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
        # Loads the mask file
        mask = torch.load(mask_file)

        # Gathers the number of features
        mask_features = torch.count_nonzero(mask).numpy().item()

        # If yes, creates the composed transform
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            BinaryMask(mask)
        ])

    # If there is no supplied mask file
    else:
        # Number of mask features will be None
        mask_features = None

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

    return train, val, test, mask_features

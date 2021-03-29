"""Bernoulli-Bernoulli Restricted Boltzmann Machines with Feature Selection.
"""

import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.constants as c
import learnergy.utils.exception as ex
import learnergy.utils.logging as l
from learnergy.models.bernoulli import RBM

logger = l.get_logger(__name__)


class FSRBM(RBM):
    """A FSRBM class provides the basic implementation for
    Bernoulli-Bernoulli Restricted Boltzmann Machines along with Feature Selection.

    References:
        To be published.

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, use_gpu=False):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: RBM -> FSRBM.')

        # Override its parent class
        super(FSRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                    momentum, decay, temperature, use_gpu)

        # Feature selection mask
        self.f = nn.Parameter(torch.randn(n_visible))

        # Updating optimizer's parameters with `f`
        self.optimizer.add_param_group({'params': self.f})

        # Re-checks if current device is CUDA-based due to new parameter
        if self.device == 'cuda':
            # If yes, re-uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')

    @property
    def f(self):
        """torch.Tensor: Feature selection mask.

        """

        return self._f

    @f.setter
    def f(self, f):
        if not isinstance(f, nn.Parameter):
            raise ex.TypeError('`f` should be a PyTorch parameter')

        self._f = f

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        f = (self.f - torch.min(self.f)) / (torch.max(self.f) - torch.min(self.f))
        f = torch.bernoulli(f)

        # Masking the input layer
        v = torch.mul(v, f)

        # Calculating neurons' activations
        activations = F.linear(v, self.W.t(), self.b)

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.sigmoid(torch.div(activations, self.T))

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.sigmoid(activations)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states

    def energy(self, samples):
        """Calculates and frees the system's energy.

        Args:
            samples (torch.Tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        f = (self.f - torch.min(self.f)) / (torch.max(self.f) - torch.min(self.f))
        f = torch.bernoulli(f)

        samples = torch.mul(samples, f)

        # Calculate samples' activations
        activations = F.linear(samples, self.W.t(), self.b)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculate the hidden term
        h = torch.sum(s(activations), dim=1)

        # Calculate the visible term
        v = torch.mv(samples, self.a)

        # Finally, gathers the system's energy
        energy = -v - h

        return energy

    def pseudo_likelihood(self, samples):
        """Calculates the logarithm of the pseudo-likelihood.

        Args:
            samples (torch.Tensor): Samples to be calculated.

        Returns:
            The logarithm of the pseudo-likelihood based on input samples.

        """

        f = (self.f - torch.min(self.f)) / (torch.max(self.f) - torch.min(self.f))
        f = torch.bernoulli(f)

        samples = torch.mul(samples, f)

        # Gathering a new array to hold the rounded samples
        samples_binary = torch.round(samples)

        # Calculates the energy of samples before flipping the bits
        energy = self.energy(samples_binary)

        # Samples an array of indexes to flip the bits
        indexes = torch.randint(0, self.n_visible, size=(
            samples.size(0), 1), device=self.device)

        # Creates an empty vector for filling the indexes
        bits = torch.zeros(samples.size(
            0), samples.size(1), device=self.device)

        # Fills the sampled indexes with 1
        bits = bits.scatter_(1, indexes, 1)

        # Actually flips the bits
        samples_binary = torch.where(
            bits == 0, samples_binary, 1 - samples_binary)

        # Calculates the energy after flipping the bits
        energy1 = self.energy(samples_binary)

        # Calculate the logarithm of the pseudo-likelihood
        pl = torch.mean(self.n_visible *
                        torch.log(torch.sigmoid(energy1 - energy) + c.EPSILON))

        return pl

    def fit(self, dataset, batch_size=128, epochs=10):
        """Fits a new RBM model.
        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.
        Returns:
            MSE (mean squared error) and log pseudo-likelihood from the training step.
        """

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0)

        # For every epoch
        for epoch in range(epochs):
            logger.info('Epoch %d/%d', epoch+1, epochs)

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse = 0
            pl = 0

            # For every batch
            for samples, _ in tqdm(batches):
                # Flattening the samples' batch
                samples = samples.reshape(len(samples), self.n_visible)

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    # Applies the GPU usage to the data
                    samples = samples.cuda()

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples)

                # Detaching the visible states from GPU for further computation
                visible_states = visible_states.detach()

                # Calculates the loss for further gradients' computation
                cost = torch.mean(self.energy(samples)) - \
                    torch.mean(self.energy(visible_states))

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Calculating current's batch MSE
                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size).detach()

                # Calculating the current's batch logarithm pseudo-likelihood
                batch_pl = self.pseudo_likelihood(samples).detach()

                # Summing up to epochs' MSE and pseudo-likelihood
                mse += batch_mse
                pl += batch_pl

            # Normalizing the MSE and pseudo-likelihood with the number of batches
            mse /= len(batches)
            pl /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(mse=mse.item(), pl=pl.item(), time=end-start)

            logger.info('MSE: %f | log-PL: %f', mse, pl)

        return mse, pl
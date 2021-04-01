"""Bernoulli-Bernoulli Restricted Boltzmann Machines with Feature Selection.
"""

import time

import learnergy.utils.exception as e
import learnergy.utils.logging as l
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.rbm import RBM

logger = l.get_logger(__name__)


class FSRBM(RBM):
    """A FSRBM class provides the basic implementation for
    Bernoulli-Bernoulli Restricted Boltzmann Machines along with Feature Selection.

    References:
        To be published.

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, input_mask_fn='sigmoid',
                 use_binary_sampling=False, use_gpu=False):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            input_mask_fn (str): Mask function to be applied over the input.
            use_binary_sampling (boolean): Whether a binary sampling should be used in reconstruction or not.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: RBM -> FSRBM.')

        # Override its parent class
        super(FSRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                    momentum, decay, temperature, use_gpu)

        # Mask function to be applied over the input
        self.input_mask_fn = input_mask_fn

        # Whether a binary sampling should be used in reconstruction or not
        self.use_binary_sampling = use_binary_sampling

        # Feature selection mask
        self.f = nn.Parameter(torch.zeros(n_visible))

        # Updating optimizer's parameters with `f`
        self.optimizer.add_param_group({'params': self.f})

        # Re-checks if current device is CUDA-based due to new parameter
        if self.device == 'cuda':
            # If yes, re-uses CUDA in the whole class
            self.cuda()

        logger.debug('Mask Function: %s | Binary Sampling: %s',
                     self.input_mask_fn, self.use_binary_sampling)
        logger.info('Class overrided.')

    @property
    def input_mask_fn(self):
        """str: Mask function to be applied over the input.

        """

        return self._input_mask_fn

    @input_mask_fn.setter
    def input_mask_fn(self, input_mask_fn):
        if input_mask_fn not in ['sigmoid', 'soft_step']:
            raise e.TypeError('`input_mask_fn` should be `sigmoid` or `soft_step`')

        self._input_mask_fn = input_mask_fn

    @property
    def use_binary_sampling(self):
        """boolean: Binary sampling over reconstruction.

        """

        return self._use_binary_sampling

    @use_binary_sampling.setter
    def use_binary_sampling(self, use_binary_sampling):
        if not isinstance(use_binary_sampling, bool):
            raise e.TypeError('`use_binary_sampling` should be a boolean')

        self._use_binary_sampling = use_binary_sampling

    @property
    def f(self):
        """nn.Parameter: Feature selection mask.

        """

        return self._f

    @f.setter
    def f(self, f):
        if not isinstance(f, nn.Parameter):
            raise e.TypeError('`f` should be a PyTorch parameter')

        self._f = f

    def soft_step(self, x):
        """Calculates the Soft Step's function.

        Args:
            x (torch.Tensor): Tensor with incoming data.

        Returns:
            A f(x) based on Soft Step's function.

        """

        # Calculates the step function over the input
        step = torch.heaviside(x, torch.tensor([0.0]))

        return step * (1 - torch.exp(-x)) + (1 - step) * torch.exp(x)

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling with input masking, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Checks if input mask function is sigmoid
        if self.input_mask_fn == 'sigmoid':
            # Applies sigmoid over the mask
            f = torch.sigmoid(self.f)
        
        # Checks if input mask function is soft step
        elif self.input_mask_fn == 'soft_step':
            # Applies soft step over the mask
            f = self.soft_step(self.f)

        # Checks if reconstruction mask should be used
        if self.use_binary_sampling:
            # Samples the mask
            f = torch.bernoulli(f)

            # Logs the amount of features
            # logger.info(f'Number of features: {torch.count_nonzero(f)}')

        # Applies the feature selection mask over the input
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

    def hidden_sampling_without_mask(self, v, scale=False):
        """Performs the hidden layer sampling without input masking, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

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

        # Checks if input mask function is sigmoid
        if self.input_mask_fn == 'sigmoid':
            # Applies sigmoid over the mask
            f = torch.sigmoid(self.f)
        
        # Checks if input mask function is soft step
        elif self.input_mask_fn == 'soft_step':
            # Applies soft step over the mask
            f = self.soft_step(self.f)

        # Applies the feature selection mask over the input
        samples = torch.mul(samples, f)

        # Calculate samples' activations
        activations = F.linear(samples, self.W.t(), self.b)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculates the hidden term
        h = torch.sum(s(activations), dim=1)

        # Calculates the visible term
        v = torch.mv(samples, self.a)

        # Finally, gathers the system's energy
        energy = -v - h

        return energy

    def fit(self, dataset, batch_size=128, epochs=10, epochs_per_snapshot=10):
        """Fits a new RBM model.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.
            epochs_per_snapshot (int): Amount of epochs per snapshot.

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

            # Verifies if it is supposed to take a snapshot
            if (epoch + 1) % epochs_per_snapshot == 0:
                # Performs a model snapshot
                torch.save(self, f'outputs/fsrbm_snapshot_epoch_{epoch+1}.pth')

                # Checks if input mask is sigmoid
                if self.input_mask_fn == 'sigmoid':
                    f = torch.sigmoid(self.f)
        
                # Checks if input mask is soft step
                elif self.input_mask_fn == 'soft_step':
                    f = rbm.soft_step(self.f)
                
                # Samples the mask and saves it
                mask = torch.bernoulli(f)
                torch.save(mask, f'outputs/mask_snapshot_epoch_{epoch+1}.pth')
                logger.debug('Mask features: %d', torch.count_nonzero(mask))

        return mse, pl

    def reconstruct(self, dataset):
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the testing data.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info('Reconstructing new samples ...')

        # Resetting MSE to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

        # For every batch
        for samples, _ in tqdm(batches):
            # Flattening the samples' batch
            samples = samples.reshape(len(samples), self.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating positive phase hidden probabilities and states
            # Note that we do not use input masking when reconstructing
            _, pos_hidden_states = self.hidden_sampling_without_mask(samples)

            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(
                pos_hidden_states)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size)

            # Summing up the reconstruction's MSE
            mse += batch_mse

        # Normalizing the MSE with the number of batches
        mse /= len(batches)

        logger.info('MSE: %f', mse)

        return mse, visible_probs

"""Bernoulli-Bernoulli Restricted Boltzmann Machines with Feature Selection.
"""

import torch
import torch.nn.functional as F
from torch import nn

import learnergy.utils.exception as e
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
                 momentum=0, decay=0, temperature=1, use_binary_sampling=False,
                 use_gpu=False):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            use_binary_sampling (boolean): Whether a binary sampling should be used in reconstruction or not.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: RBM -> FSRBM.')

        # Override its parent class
        super(FSRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                    momentum, decay, temperature, use_gpu)

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

        logger.debug('Binary Sampling: %s', self.use_binary_sampling)
        logger.info('Class overrided.')

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

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Applies sigmoid over the mask
        f = torch.sigmoid(self.f)

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

    def energy(self, samples):
        """Calculates and frees the system's energy.

        Args:
            samples (torch.Tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        # Applies sigmoid over the mask
        f = torch.sigmoid(self.f)

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

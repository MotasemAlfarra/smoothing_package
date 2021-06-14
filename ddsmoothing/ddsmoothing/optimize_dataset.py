import torch
from tqdm import tqdm
import logging

from .certificate import Certificate
from .optimization import optimize_isotropic_dds


class OptimizeSmoothingParameters():
    def __init__(self):
        """Optimize smoothing parameters over a dataset.
        """
        self.logger = logging.getLogger('simple_example')
        self.logger.setLevel(logging.DEBUG)

    def save_theta(self, **kwargs):
        """Save the optimized theta.
        """
        raise NotImplementedError

    def run_optimization(self, **kwargs):
        """Run the optimization over the dataset.
        """
        raise NotImplementedError

    def log(self, message: str):
        """Log any message as a debug message.

        Args:
            message (str): message contents
        """
        self.logger.debug(message)


class OptimizeIsotropicSmoothingParameters(OptimizeSmoothingParameters):
    def __init__(
            self, model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader, device: str = "cuda:0",
    ):
        """Optimize isotropic smoothing parameters over a dataset given by
        the loader in test_loader. 

        Args:
            model (torch.nn.Module): trained base model
            test_loader (torch.utils.data.DataLoader): dataset of inputs
            device (str, optional): device on which to perform the computations
        """
        super().__init__()
        self.model = model
        self.device = device
        self.loader = test_loader
        self.data_samples = 0

        # Getting the number of samples in the testloader
        for _, _, idx in self.loader:
            self.data_samples += len(idx)

        self.log(
            "There are in total {} instances in the testloader".format(
                self.data_samples
            )
        )

    def save_theta(self, thetas: torch.Tensor, filename: str):
        """Save the optimized isotropic thetas

        Args:
            thetas (torch.Tensor): optimized thetas
            filename (str): filename to the folder where the thetas should be
                saved
        """
        torch.save(thetas, filename)
        self.log('Optimized smoothing parameters are saved')

    def run_optimization(
            self, certificate: Certificate, lr: float,
            theta_0: torch.Tensor, iterations: int,
            num_samples: int, filename: str = './'
    ):
        """Run the Isotropic DDS optimization for the dataset

        Args:
            certificate (Certificate): instance of desired certification object
            lr (float, optional): optimization learning rate for Isotropic DDS
            theta_0 (torch.Tensor): initialization value per input of the test
                loader
            iterations (int): Description
            num_samples (int): number of samples per input and iteration
            filename (str, optional): name of the file of the saved thetas
        """
        theta_0 = theta_0.reshape(-1)
        assert torch.numel(theta_0) == self.data_samples, \
            "Dimension of theta_0 should be the number of " +\
            "examples in the testloader"

        for batch, _, idx in tqdm(self.loader):
            batch = batch.to(self.device)
            thetas = optimize_isotropic_dds(
                model=self.model, batch=batch,
                certificate=certificate, learning_rate=lr,
                sig_0=theta_0[idx], iterations=iterations,
                samples=num_samples, device=self.device
            )

            theta_0[idx] = thetas.detach()

        self.save_theta(theta_0, filename)

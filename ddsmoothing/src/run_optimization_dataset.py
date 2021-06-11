import torch
from tqdm import tqdm

from .isotropic_dds import isotropic_dds


class optimize_smoothing_parameters():
    def save_theta(self, **kwargs):
        raise NotImplementedError

    def run_optimization(self, **kwargs):
        raise NotImplementedError


class optimize_isotropic_smoothing_parameters(optimize_smoothing_parameters):
    def __init__(self, 
        model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
        device: str = "cuda:0"
    ):
    
        self.model = model
        self.device = device
        self.loader = test_loader
        self.num_samples = 0
        # Getting the number of samples in the testloader
        for _, _, idx in self.loader:
            self.num_samples += len(idx)
        print("There are in total {} instances in the testloader".format(self.num_samples))
    
    def save_theta(self, theta: torch.Tensor, path: str):
        torch.save(theta, path+'theta_isotropic.pth')
        print('optimized smoothing parameters are saved')
        return

    def run_optimization(self,
        norm: str, learning_rate: float, 
        theta_0: torch.Tensor, iterations: int,
        samples: int, save_path: str ='./'
    ):
        theta_0 = theta_0.reshape(-1)
        assert torch.numel(theta_0) == self.num_samples, \
            "Dimension of theta_0 should be the number of examples in the testloader"

        for batch, _, idx in tqdm(self.loader):
            batch = batch.to(self.device)
            theta = isotropic_dds(
                model=self.model, batch=batch,
                norm=norm, learning_rate=learning_rate, 
                sig_0=theta_0[idx], iterations=iterations,
                samples=samples, device=self.device
            )
            theta_0[idx] = theta

        self.save_theta(theta, save_path)

        return theta

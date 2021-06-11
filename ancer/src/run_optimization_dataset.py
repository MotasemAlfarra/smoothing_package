import torch
from tqdm import tqdm
from .anisotropic_dds import anisotropic_dds
from ...ddsmoothing.src.run_optimization_dataset import optimize_smoothing_parameters


class optimize_anisotropic_smoothing_parameters(optimize_smoothing_parameters):
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
    
    def save_theta(self,
        theta: torch.Tensor, idx: torch.Tensor, path: str ):
        for i, j in enumerate(idx):
            torch.save(
                theta[i],
                path + 'theta' + '_' + str(j.item()) + '.pt'
            )
        return

    def run_optimization(self,
        norm: str, learning_rate: float, 
        isotropic_theta: torch.Tensor, iterations: int,
        samples: int, kappa: float, save_path: str ='./'
    ):
    
        isotropic_theta = isotropic_theta.reshape(-1)
        assert torch.numel(isotropic_theta) == self.num_samples, \
            "Dimension of theta_0 should be the number of examples in the testloader"

        for batch, _, idx in tqdm(self.loader):
            batch = batch.to(self.device)
            theta_0 = torch.ones_like(batch) * isotropic_theta[idx].reshape(-1, 1, 1, 1)

            theta = anisotropic_dds(
                model=self.model, batch=batch,
                norm=norm, learning_rate=learning_rate, 
                isotropic_theta=theta_0, iterations=iterations,
                samples=samples, kappa=kappa, device=self.device
            )

            self.save_theta(theta, idx, save_path)

        return
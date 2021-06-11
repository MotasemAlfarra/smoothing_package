import argparse
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import all_datasets
from datasets import DATASETS, get_num_classes
from models.utils import load_model

from anisotropic_dds import anisotropic_dds
from isotropic_dds import isotropic_dds


class optimize_smoothing_parameters():
    def __init__(self, 
        model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
        device: str = "cuda:0"
    ):
    
        self.model = model
        self.device = device
        self.loader = test_loader

    def run_isotropic_optimization(self,
        norm: str, learning_rate: float, 
        sig_0: torch.Tensor, iterations: int,
        samples: int
    ):


        return

    def run_anisotropic_optimization(self,
        norm: str, learning_rate: float,
        isotropic_theta: torch.Tensor, iterations: int,
        samples: int, kappa: float
    ):


        return

    def save_sigma(self,
            sigma: torch.Tensor, idx: torch.Tensor, path: str, norm: str,
            flag: str = 'test'
            ):
        if norm == "l1":
            prefix = '/lambda_'
        else:
            prefix = '/sigma_'

        for i, j in enumerate(idx):
            torch.save(
                sigma[i],
                path + prefix + flag + '_' + str(j.item()) + '.pt'
            )

        return












def run_ancer_optimization_on_dataset(
        model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
        isotropic_sigmas: torch.Tensor, output_folder: str, iterations: int,
        certificate: Certificate, lr: float = 0.04, num_samples: int = 100,
        regularization_weight: float = 2, device: str = "cuda:0"
        ):

    for batch_idx, (batch, targets, idx) in enumerate(tqdm(test_loader)):
        batch, targets = batch.to(device), targets.to(device)
        sigma = torch.ones_like(batch) * isotropic_sigmas[idx].reshape(-1, 1, 1, 1)

        sigma = ancer_optimization(
            model, batch, certificate, lr, sigma, iterations,
            num_samples, kappa=regularization_weight,
            device=device
        )

        # save the optimized sigmas
        save_sigma(sigma, idx, output_folder, certificate.norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Certify dataset examples')
    parser.add_argument(
        "--dataset", required=True,
        choices=DATASETS, help="which dataset to use"
    )
    parser.add_argument(
        "--model", required=True,
        type=str, help="path to model of the base classifier"
    )
    parser.add_argument(
        "--model-type", required=True,
        choices=["resnet18", "wideresnet40", "resnet50"],
        type=str, help="type of model to load"
    )
    parser.add_argument(
        "--norm", required=True,
        choices=["l1", "l2"], type=str,
        help="norm of the desired certificate"
    )
    parser.add_argument(
        "--output-folder", required=True,
        type=str, help="output folder for the optimized sigmas"
    )
    parser.add_argument(
        "--isotropic-file", required=True,
        type=str, help="isotropic_dd initialization sigmas"
    )

    # dataset options
    parser.add_argument(
        "--folder-path", type=str, default=None,
        help="dataset folder path, required for ImageNet"
    )

    # optimization options
    parser.add_argument(
        "--iterations", type=int,
        default=100, help="optimization iterations per sample"
    )
    parser.add_argument(
        "--batch-sz", type=int,
        default=128, help="optimization batch size"
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float,
        default=0.04, help="optimization learning rate"
    )
    parser.add_argument(
        "-n", "--num-samples", type=float,
        default=100, help="number of samples per example and iteration"
    )
    parser.add_argument(
        "--regularization-weight", type=float,
        default=2, help="regularization weight"
    )

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the base classifier
    num_classes = get_num_classes(args.dataset)
    model = load_model(args.model, args.model_type, num_classes, device=device)
    model = model.eval()

    # get the dataset
    if args.dataset == "cifar10":
        _, test_loader, img_sz, _, testset_len = all_datasets.cifar10(
            args.batch_sz
        )
    else:
        _, test_loader, img_sz, _, testset_len = all_datasets.ImageNet(
            args.batch_sz,
            directory=args.folder_path
        )

    # get the type of certificate
    certificate = L1Certificate(device=device) if args.norm == "l1" else \
        L2Certificate(1, device=device)

    # open the isotropic ones
    isotropic_sigmas = torch.load(args.isotropic_file, map_location=device)

    run_ancer_optimization_on_dataset(
        model, test_loader, isotropic_sigmas, args.output_folder,
        args.iterations, certificate,
        lr=args.learning_rate, num_samples=args.num_samples,
        regularization_weight=args.regularization_weight,
        device=device
    )

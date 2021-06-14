from math import ceil

import numpy as np
import torch
from torch.distributions.normal import Normal
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint


class Certificate():
    def compute_proxy_gap(self, logits: torch.Tensor):
        """Compute the proxy gap

        Args:
            logits (torch.Tensor): network outputs
        """
        raise NotImplementedError(
            "base class does not implement this method"
        )

    def sample_noise(self, batch: torch.Tensor, repeated_theta: torch.Tensor):
        """Sample noise to obtain the desired certificate

        Args:
            batch (torch.Tensor): original inputs
            repeated_theta (torch.Tensor): thetas repeated to the same batch
                size as `batch`
        """
        raise NotImplementedError(
            "base class does not implement this method"
        )

    def compute_gap(self, pABar: float):
        """Compute the gap given by this certificate

        Args:
            pABar (float): lower bound of p_A as per the definition of Cohen
                et al
        """
        raise NotImplementedError(
            "base class does not implement this method"
        )

    def compute_radius_estimate(
            self, logits: torch.Tensor, theta: torch.Tensor
    ):
        """Compute a differentiable radius estimate

        Args:
            logits (torch.Tensor): network outputs
            theta (torch.Tensor): parameters of the distribution
        """
        raise NotImplementedError(
            "base class does not implement this method"
        )


class L2Certificate(Certificate):
    norm = "l2"

    def __init__(self, batch_size: int, device: str = "cuda:0"):
        self.m = Normal(
            torch.zeros(batch_size).to(device),
            torch.ones(batch_size).to(device)
        )
        self.device = device

    def compute_proxy_gap(self, logits: torch.Tensor) -> torch.Tensor:
        return self.m.icdf(logits[:, 0].clamp_(0.001, 0.999)) - \
            self.m.icdf(logits[:, 1].clamp_(0.001, 0.999))

    def sample_noise(
            self, batch: torch.Tensor, repeated_theta: torch.Tensor
    ) -> torch.Tensor:
        return torch.randn_like(batch, device=self.device) * repeated_theta

    def compute_gap(self, pABar: float) -> float:
        return norm.ppf(pABar)

    def compute_radius_estimate(
            self, logits: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        return theta/2 * self.compute_proxy_gap(logits)


class L1Certificate(Certificate):
    norm = "l1"

    def __init__(self, device="cuda:0"):
        self.device = device

    def compute_proxy_gap(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, 0] - logits[:, 1]

    def sample_noise(
            self, batch: torch.Tensor, repeated_theta: torch.Tensor
    ) -> torch.Tensor:
        return 2 * (torch.rand_like(batch, device=self.device) - 0.5) * \
            repeated_theta

    def compute_gap(self, pABar: float) -> float:
        return 2 * (pABar - 0.5)

    def compute_radius_estimate(
            self, logits: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        return theta * self.compute_proxy_gap(logits)


class Smooth():
    """A smoothed classifier g

    Adapted from:
            https://github.com/locuslab/smoothing/blob/master/code/core.py
        to use an arbitrary certificate Certificate
    """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(
            self, base_classifier: torch.nn.Module, num_classes: int,
            sigma: torch.Tensor, certificate: Certificate
    ):
        """
        Args:
            base_classifier (torch.nn.Module): maps from
                [batch x channel x height x width]
            to [batch x num_classes]
            num_classes (int): number of classes
            sigma (torch.Tensor): distribution parameter
            certificate (Certificate): certificate desired
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.certificate = certificate

    def certify(
            self, x: torch.tensor, n0: int, n: int, alpha: float,
            batch_size: int, device: torch.device = torch.device('cuda:0')
    ) -> (int, float):
        """Monte Carlo algorithm for certifying that g's prediction around x
        is constant within some L2/L1 radius.
        With probability at least 1 - alpha, the class returned by this method
        will equal g(x), and g's prediction will robust within a L2/L1 ball of
        radius R around x.

        Args:
            x (torch.tensor): the input [channel x height x width]
            n0 (int): the number of Monte Carlo samples to use for selection
            n (int): the number of Monte Carlo samples to use for estimation
            alpha (float): the failure probability
            batch_size (int): batch size to use when evaluating the base
                classifier
            device (torch.device, optional): Description

        Returns:
            int, float: (predicted class, gap term in the certified radius)
            in the case of abstention, the class will be ABSTAIN and the
            radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size, device=device)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size, device=device)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            return cAHat, self.certificate.compute_gap(pABar)

    def predict(
            self, x: torch.tensor, n: int, alpha: float, batch_size: int,
            device: torch.device = torch.device('cuda:0')
    ) -> int:
        """Monte Carlo algorithm for evaluating the prediction of g at x.
        With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in
        https://arxiv.org/abs/1610.03944 for identifying the top category of
        a multinomial distribution.

        Args:
            x (torch.tensor): the input [channel x height x width]
            n (int): the number of Monte Carlo samples to use
            alpha (float): the failure probability
            batch_size (int): batch size to use when evaluating the base
                classifier
            device (torch.device, optional): device on which to perform the
                computations

        Returns:
            int: output class
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size, device=device)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(
            self, x: torch.tensor, num: int, batch_size,
            device: torch.device = torch.device('cuda:0')
    ) -> np.ndarray:
        """Sample the base classifier's prediction under noisy corruptions of
        the input x.

        Args:
            x (torch.tensor): the input [channel x width x height]
            num (int): number of samples to collect
            batch_size (TYPE): Description
            device (torch.device, optional): device on which to perform the
                computations

        Returns:
            np.ndarray: an ndarray[int] of length num_classes containing the
            per-class counts
        """
        with torch.no_grad():
            # counts = np.zeros(self.num_classes, dtype=int)
            counts = torch.zeros(self.num_classes, dtype=float, device=device)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = self.certificate.sample_noise(batch, self.sigma)
                # noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions,
                                          device, self.num_classes)
            return counts.cpu().numpy()

    def _count_arr(
            self, arr: torch.tensor, device: torch.device, length: int
    ) -> torch.tensor:
        counts = torch.zeros(length, dtype=torch.long, device=device)
        unique, c = arr.unique(sorted=False, return_counts=True)
        counts[unique] = c
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """Returns a (1 - alpha) lower confidence bound on a bernoulli
        proportion.

        This function uses the Clopper-Pearson method.

        Args:
            NA (int): the number of "successes"
            N (int): the number of total draws
            alpha (float): the confidence level

        Returns:
            float: a lower bound on the binomial proportion which holds true
            w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


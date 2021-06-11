import torch
from torch.autograd import Variable
from .certificate import L1Certificate, L2Certificate


def isotropic_dds(
    model: torch.nn.Module, batch: torch.Tensor,
    norm: str, learning_rate: float, 
    sig_0: torch.Tensor, iterations: int,
    samples: int, device: str = 'cuda:0'):
    """Optimize smoothing parameters for a batch.

    Args:
        model: trained network.
        batch: inputs to certify around.
        certificate_type: type of certificate that you want to optimize (either l1 or l2).
        learning_rate: optimization learning rate for ANCER.
        sig_0: initialization value per input in batch.
        iterations: number of iterations to run the optimization.
        samples: number of samples per input and iteration.
    """

    batch_size = batch.shape[0]

    certificate = L1Certificate(batch_size, device=device) if norm == "l1" else \
        L2Certificate(batch_size, device=device)

    sig = Variable(sig_0, requires_grad=True).view(batch_size, 1, 1, 1)

    for param in model.parameters():
        param.requires_grad_(False)

    #Reshaping so for n > 1
    new_shape = [batch_size * samples]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1,samples, 1, 1)).view(new_shape)

    for _ in range(iterations):
        sigma_repeated = sig.repeat((1, samples, 1, 1)).view(-1,1,1,1)
        eps = certificate.sample_noise(new_batch, sigma_repeated) #Reparamitrization trick
        out = model(new_batch + eps).reshape(batch_size, samples, -1).mean(1)#This is \psi in the algorithm
        
        vals, _ = torch.topk(out, 2)
        vals.transpose_(0, 1)
        radius = certificate.compute_radius_estimate(vals, sig.reshape(-1))
        grad = torch.autograd.grad(radius.sum(), sig)

        sig.data += learning_rate*grad[0]  # Gradient Ascent step

    #For training purposes after getting the sigma
    for param in model.parameters():
        param.requires_grad_(True)    

    return sig.reshape(-1)

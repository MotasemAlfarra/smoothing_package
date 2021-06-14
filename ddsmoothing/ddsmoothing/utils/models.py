import torch
from torchvision.models.resnet import resnet50

from .resnet import resnet18
from .wideresnet import WideResNet


def load_model(
        path: str, model_type: str, num_classes: int, device: str = "cuda"
        ):
    # crate model base on model type
    if model_type == "resnet18":
        model = resnet18(num_classes=num_classes).to(device)
    elif model_type == "wideresnet40":
        model = WideResNet(
            depth=40,
            widen_factor=2,
            num_classes=num_classes
        ).to(device)
    elif model_type == "resnet50":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).to(device)

        checkpoint = torch.load(path, map_location=device)

        keys = list(checkpoint['state_dict'].keys())
        count = 0
        for key in model.state_dict().keys():
            model.state_dict()[key].copy_(
                checkpoint['state_dict'][keys[count]].data
            )
            count += 1

        return model
    else:
        raise ValueError("model_type requested not available")

    # load the checkpoint
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

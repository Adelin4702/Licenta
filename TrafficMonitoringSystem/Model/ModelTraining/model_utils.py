import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import subprocess


def get_device():
    """Setup device for training with automatic selection of best GPU"""
    if torch.cuda.is_available():
        # Find GPU with most free memory
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                                    stdout=subprocess.PIPE, text=True)
            free_memory = [int(x) for x in result.stdout.strip().split("\n")]
            device_id = free_memory.index(max(free_memory))
            print(f"Using GPU {device_id} with {max(free_memory)} MB free memory")
            return torch.device(f'cuda:{device_id}')
        except:
            print("Falling back to first GPU")
            return torch.device('cuda:0')
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')


def get_object_detection_model(num_classes):
    """Create an object detection model with custom anchor sizes"""
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Modify anchor sizes to better match small objects
    model.rpn.anchor_generator.sizes = ((32,), (64,), (128,), (256,), (512,))

    return model


def create_optimizer(model):
    """Create optimizer with custom settings for different layers"""
    params = [
        {"params": [p for n, p in model.named_parameters() if "box_predictor" in n], "lr": 0.005},
        {"params": [p for n, p in model.named_parameters() if "box_predictor" not in n], "lr": 0.0005},
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.0001)
    return optimizer


def create_lr_scheduler(optimizer):
    """Create learning rate scheduler"""
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,
        T_mult=1,
        eta_min=1e-6
    )
    return lr_scheduler
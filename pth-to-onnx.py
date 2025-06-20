import torch
import torchvision.models as models  # sau importă propriul model

# 1. Încarcă modelul (exemplu cu ResNet18)
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load("final_model_subset_binary_epoch_5_map_0.7004.pth"))
model.eval()

# 2. Creează un input dummy (cu aceeași formă ca datele de antrenare)
dummy_input = torch.randn(1, 3, 480, 480)  # Exemplu pentru imagini RGB 224x224

# 3. Exportă în format ONNX
torch.onnx.export(
    model, dummy_input, "model.onnx",
    export_params=True,        # Salvează greutățile modelului
    opset_version=11,          # Versiune ONNX (11 sau mai mare e recomandat)
    do_constant_folding=True,  # Optimizează constantele
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

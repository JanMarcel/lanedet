import torch

from lanedet.models.backbones.mobilenet import MobileNetV2

# if __name__ == "__main__":
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_fp32 = MobileNetV2()
model_fp32.load_state_dict(
    torch.load('models/mobilenet_tusimple_200epochs.pth'), strict=False)

# model_fp32 = model_fp32.to(device)  # Ensure the model is on the correct device
model_fp32.eval()
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
# model_fp32 = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)

tensor_state = torch.randn(1, 3, 640, 360)
model_fp32_prepared(tensor_state)

model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
torch.save(model_int8.state_dict(), "models/mobilenet_tusimple_200epochs_quant.pth")
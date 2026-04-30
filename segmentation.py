import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from config import device, model_paths

model = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None,
                 in_channels=3, classes=1).to(device)

def load_selected_model(model_key):
    """Загружает веса выбранной модели"""
    if not model_key or "❌" in model_key:
        return False
    try:
        model.load_state_dict(torch.load(model_paths[model_key], map_location=device))
        model.eval()
        return True
    except:
        return False

def core_inference(img):
    """Выполняет сегментацию одного RGB‑изображения"""
    img_res = cv2.resize(img, (256, 256))
    x = (img_res.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_t)).cpu().numpy()[0, 0]
        mask = (prob > 0.5).astype(np.uint8)
    return img_res, mask, prob
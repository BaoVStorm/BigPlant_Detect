import torch
import torchvision.transforms as T
from PIL import Image

_tfm = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def preprocess_pil(img: Image.Image, device: torch.device) -> torch.Tensor:
    x = _tfm(img.convert("RGB")).unsqueeze(0)
    return x.to(device)

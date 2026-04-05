import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import SegformerForSemanticSegmentation
except Exception as exc:
    raise ImportError("transformers is required for efficientnetv2-segformer inference") from exc


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_efficientnetv2_s(num_classes: int):
    try:
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
                    break
        return model, "torchvision"
    except Exception:
        pass

    try:
        import timm

        for name in ["tf_efficientnetv2_s", "efficientnetv2_s", "tf_efficientnetv2_s_in21k"]:
            try:
                model = timm.create_model(name, pretrained=True, num_classes=num_classes)
                return model, f"timm:{name}"
            except Exception:
                continue
    except Exception:
        pass

    raise RuntimeError("Could not build EfficientNetV2-S. Install torchvision or timm.")


class EffNetV2SegFormerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        segformer_name: str = "nvidia/segformer-b1-finetuned-ade-512-512",
        seg_input_size: int = 512,
        seg_threshold: float = 0.5,
        seg_temperature: float = 12.0,
        min_keep_bg: float = 0.15,
        freeze_segformer: bool = True,
    ):
        super().__init__()
        self.classifier, self.cls_backend = build_efficientnetv2_s(num_classes)
        self.seg_threshold = float(seg_threshold)
        self.seg_temperature = float(seg_temperature)
        self.min_keep_bg = float(min_keep_bg)
        self.seg_input_size = int(seg_input_size)

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(segformer_name)
        self.seg_backend = segformer_name

        if freeze_segformer:
            for p in self.segformer.parameters():
                p.requires_grad = False
            self.segformer.eval()

        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _build_foreground_mask(self, x_raw: torch.Tensor) -> torch.Tensor:
        h, w = x_raw.shape[-2], x_raw.shape[-1]
        x_seg = F.interpolate(
            x_raw,
            size=(self.seg_input_size, self.seg_input_size),
            mode="bilinear",
            align_corners=False,
        )
        x_seg = (x_seg - self.mean) / self.std

        with torch.no_grad():
            seg_out = self.segformer(pixel_values=x_seg)
            logits = seg_out.logits

        probs = torch.softmax(logits, dim=1)
        conf = probs.max(dim=1, keepdim=True).values
        conf = F.interpolate(conf, size=(h, w), mode="bilinear", align_corners=False)

        conf_min = conf.amin(dim=(2, 3), keepdim=True)
        conf_max = conf.amax(dim=(2, 3), keepdim=True)
        conf_norm = (conf - conf_min) / (conf_max - conf_min + 1e-6)
        return torch.sigmoid((conf_norm - self.seg_threshold) * self.seg_temperature)

    def forward(self, x_raw: torch.Tensor):
        fg_mask = self._build_foreground_mask(x_raw)
        x_focus = x_raw * (self.min_keep_bg + (1.0 - self.min_keep_bg) * fg_mask)
        x_cls = (x_focus - self.mean) / self.std
        logits = self.classifier(x_cls)
        return logits, fg_mask


class EfficientSegformerInferenceAdapter(nn.Module):
    def __init__(self, model: EffNetV2SegFormerClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None, training: bool = False):
        logits, _ = self.model(x)
        return logits, None, None, None

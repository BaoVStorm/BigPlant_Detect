import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


def build_mobilenetv3_large(num_classes: int):
    try:
        import timm

        model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=num_classes)
        return model, "timm"
    except Exception:
        weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = tv_models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model, "torchvision"


class MobileNetV3DeepLabV3Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seg_pretrained: bool = True,
        seg_freeze: bool = True,
        mask_mode: str = "attention",
    ):
        super().__init__()
        self.seg_freeze = bool(seg_freeze)
        self.mask_mode = str(mask_mode or "attention").strip().lower()

        try:
            seg_weights = tv_models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if seg_pretrained else None
            self.seg_model = tv_models.segmentation.deeplabv3_resnet50(weights=seg_weights, aux_loss=True)
        except Exception:
            self.seg_model = tv_models.segmentation.deeplabv3_resnet50(pretrained=seg_pretrained, aux_loss=True)

        if self.seg_freeze:
            for p in self.seg_model.parameters():
                p.requires_grad = False
            self.seg_model.eval()

        self.classifier, self.cls_backend = build_mobilenetv3_large(num_classes)

    @staticmethod
    def _build_foreground_mask(x: torch.Tensor, seg_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(seg_logits, dim=1)
        fg = 1.0 - probs[:, :1, :, :]
        fg = F.interpolate(fg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return fg.clamp(0.0, 1.0)

    def _apply_mask(self, x: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
        if self.mask_mode == "hard":
            hard_mask = (fg_mask > 0.5).to(dtype=x.dtype)
            return x * hard_mask
        if self.mask_mode == "residual":
            return x * (0.5 + 0.5 * fg_mask)
        return x * fg_mask

    def build_classifier_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.seg_freeze:
            self.seg_model.eval()
            with torch.no_grad():
                seg_logits = self.seg_model(x)["out"]
        else:
            seg_logits = self.seg_model(x)["out"]
        fg_mask = self._build_foreground_mask(x, seg_logits)
        return self._apply_mask(x, fg_mask)

    def forward(self, x: torch.Tensor):
        x_cls = self.build_classifier_input(x)
        logits = self.classifier(x_cls)
        return logits, None


class MobileNetDeepLabV3InferenceAdapter(nn.Module):
    def __init__(self, model: MobileNetV3DeepLabV3Classifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None, training: bool = False):
        logits, _ = self.model(x)
        return logits, None, None, None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


def build_resnet50(num_classes: int):
    try:
        import timm

        model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
        return model, "timm"
    except Exception:
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        tv_resnet = tv_models.resnet50(weights=weights)
        in_features = tv_resnet.fc.in_features
        tv_resnet.fc = nn.Linear(in_features, num_classes)
        return tv_resnet, "torchvision"


class ResNet50DeepLabV3Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seg_pretrained: bool = False,
        seg_freeze: bool = False,
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

        self.cls_model, self.cls_backend = build_resnet50(num_classes)

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
        logits = self.cls_model(x_cls)
        return logits, None


class ResNet50DeepLabV3InferenceAdapter(nn.Module):
    def __init__(self, model: ResNet50DeepLabV3Classifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None, training: bool = False):
        logits, _ = self.model(x)
        return logits, None, None, None

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import SegformerForSemanticSegmentation
except Exception as exc:
    raise ImportError("transformers is required for mobilenetv3large-segformer inference") from exc


def build_mobilenetv3_large(num_classes: int):
    try:
        import timm

        model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=num_classes)
        return model, "timm"
    except Exception:
        from torchvision import models as tv_models

        weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = tv_models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model, "torchvision"


def infer_plant_class_ids(segformer_model, explicit_keywords=None):
    if explicit_keywords is None:
        explicit_keywords = [
            "plant",
            "tree",
            "flower",
            "grass",
            "leaf",
            "palm",
            "bush",
            "forest",
            "garden",
            "vegetation",
        ]

    id2label = getattr(segformer_model.config, "id2label", {}) or {}
    plant_ids = []
    for k, v in id2label.items():
        try:
            idx = int(k) if isinstance(k, str) else int(k)
        except Exception:
            continue
        name = str(v).lower()
        if any(word in name for word in explicit_keywords):
            plant_ids.append(idx)
    return sorted(set(plant_ids))


class MobileNetV3SegFormerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        segformer_name: str = "nvidia/segformer-b4-finetuned-ade-512-512",
        freeze_segformer: bool = True,
        mask_blend: float = 1.0,
        background_keep: float = 0.15,
    ):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(segformer_name)
        self.classifier, self.cls_backend = build_mobilenetv3_large(num_classes)

        self.mask_blend = float(mask_blend)
        self.background_keep = float(background_keep)
        self.plant_class_ids = infer_plant_class_ids(self.segformer)
        self.freeze_segformer = bool(freeze_segformer)

        if self.freeze_segformer:
            for p in self.segformer.parameters():
                p.requires_grad = False
            self.segformer.eval()

    def _build_foreground_mask(self, seg_logits: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        logits_up = F.interpolate(seg_logits, size=(out_h, out_w), mode="bilinear", align_corners=False)
        probs = torch.softmax(logits_up, dim=1)

        if self.plant_class_ids:
            ids = [i for i in self.plant_class_ids if i < probs.shape[1]]
            if ids:
                fg_prob = probs[:, ids, :, :].sum(dim=1, keepdim=True)
            else:
                fg_prob = probs.max(dim=1, keepdim=True).values
        else:
            fg_prob = probs.max(dim=1, keepdim=True).values

        return fg_prob.clamp(0.0, 1.0)

    def build_classifier_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_segformer:
            self.segformer.eval()
            with torch.no_grad():
                seg_out = self.segformer(pixel_values=x)
        else:
            seg_out = self.segformer(pixel_values=x)

        fg_mask = self._build_foreground_mask(seg_out.logits, x.shape[-2], x.shape[-1])
        effective_mask = self.background_keep + (1.0 - self.background_keep) * fg_mask
        effective_mask = (1.0 - self.mask_blend) + self.mask_blend * effective_mask
        return x * effective_mask

    def forward(self, x: torch.Tensor):
        x_cls = self.build_classifier_input(x)
        logits = self.classifier(x_cls)
        return logits, None


class MobileNetSegFormerInferenceAdapter(nn.Module):
    def __init__(self, model: MobileNetV3SegFormerClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None, training: bool = False):
        logits, _ = self.model(x)
        return logits, None, None, None

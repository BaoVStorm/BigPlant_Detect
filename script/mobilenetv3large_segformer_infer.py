import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import SegformerForSemanticSegmentation
except Exception as exc:
    raise ImportError("transformers is required for mobilenetv3large-segformer inference") from exc


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_mobilenetv3_large(num_classes: int):
    try:
        import timm

        model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=num_classes)
        return model, "timm"
    except Exception:
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
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


class MobileNetV3LargeSegFormerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        segformer_name: str = "nvidia/segformer-b4-finetuned-ade-512-512",
        mask_blend: float = 1.0,
        background_keep: float = 0.15,
        freeze_segformer: bool = True,
    ):
        super().__init__()

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(segformer_name)
        self.classifier, self.cls_backend = build_mobilenetv3_large(num_classes)

        self.mask_blend = float(mask_blend)
        self.background_keep = float(background_keep)
        self.plant_class_ids = infer_plant_class_ids(self.segformer)

        if freeze_segformer:
            for p in self.segformer.parameters():
                p.requires_grad = False
            self.segformer.eval()

        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

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

    def build_classifier_input(self, x_raw: torch.Tensor) -> torch.Tensor:
        x_norm = (x_raw - self.mean) / self.std
        with torch.no_grad():
            seg_out = self.segformer(pixel_values=x_norm)

        fg_mask = self._build_foreground_mask(seg_out.logits, x_norm.shape[-2], x_norm.shape[-1])
        effective_mask = self.background_keep + (1.0 - self.background_keep) * fg_mask
        effective_mask = (1.0 - self.mask_blend) + self.mask_blend * effective_mask
        return x_norm * effective_mask

    def forward(self, x_raw: torch.Tensor):
        x_cls = self.build_classifier_input(x_raw)
        logits = self.classifier(x_cls)
        return logits, None


class MobileNetSegFormerInferenceAdapter(nn.Module):
    def __init__(self, model: MobileNetV3LargeSegFormerClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None, training: bool = False):
        logits, _ = self.model(x)
        return logits, None, None, None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

try:
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
except Exception as exc:
    raise ImportError("transformers is required for resnet50-mask2former inference") from exc


def build_resnet50(num_classes: int):
    try:
        import timm

        model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=num_classes)
        return model, "timm"
    except Exception:
        from torchvision.models import ResNet50_Weights, resnet50

        tv_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = tv_model.fc.in_features
        tv_model.fc = nn.Linear(in_features, num_classes)
        return tv_model, "torchvision"


class ResNet50Mask2FormerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mask2former_name: str = "facebook/mask2former-swin-large-ade-semantic",
        seg_input_size: int = 384,
        mask_floor: float = 0.15,
        freeze_segmentation: bool = True,
    ):
        super().__init__()

        self.seg_processor = AutoImageProcessor.from_pretrained(mask2former_name)
        self.seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(mask2former_name)
        self.seg_input_size = int(seg_input_size)
        self.mask_floor = float(mask_floor)

        if freeze_segmentation:
            for p in self.seg_model.parameters():
                p.requires_grad = False
            self.seg_model.eval()

        self.classifier, self.cls_backend = build_resnet50(num_classes)

        cls_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        cls_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("cls_mean", cls_mean)
        self.register_buffer("cls_std", cls_std)

        seg_mean = torch.tensor(self.seg_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        seg_std = torch.tensor(self.seg_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("seg_mean", seg_mean)
        self.register_buffer("seg_std", seg_std)

    @torch.no_grad()
    def _predict_foreground_mask(self, images_raw: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images_raw.shape

        seg_in = TF.resize(images_raw, [self.seg_input_size, self.seg_input_size], antialias=True)
        seg_in = (seg_in - self.seg_mean) / self.seg_std

        outputs = self.seg_model(pixel_values=seg_in)
        semantic_maps = self.seg_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(h, w)] * b,
        )

        masks = []
        for sem in semantic_maps:
            fg = (sem != 0).to(dtype=images_raw.dtype).unsqueeze(0)
            masks.append(fg)

        mask = torch.stack(masks, dim=0).to(images_raw.device)
        mask = F.avg_pool2d(mask, kernel_size=7, stride=1, padding=3)
        mask = self.mask_floor + (1.0 - self.mask_floor) * mask
        return mask.clamp(0.0, 1.0)

    def build_classifier_input(self, images_raw: torch.Tensor) -> torch.Tensor:
        self.seg_model.eval()
        foreground_mask = self._predict_foreground_mask(images_raw)
        focused = images_raw * foreground_mask
        return (focused - self.cls_mean) / self.cls_std

    def forward(self, images_raw: torch.Tensor):
        cls_in = self.build_classifier_input(images_raw)
        logits = self.classifier(cls_in)
        return logits, None


class ResNet50Mask2FormerInferenceAdapter(nn.Module):
    def __init__(self, model: ResNet50Mask2FormerClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None, training: bool = False):
        logits, _ = self.model(x)
        return logits, None, None, None

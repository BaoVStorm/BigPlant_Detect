import numpy as np
from collections import OrderedDict
import hashlib
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

try:
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
except Exception as exc:
    raise ImportError("transformers is required for efficientnetv2-mask2former inference") from exc


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


class EffNetV2Mask2FormerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mask2former_model_id: str = "facebook/mask2former-swin-small-coco-panoptic",
        mask_score_threshold: float = 0.35,
        freeze_mask2former: bool = True,
        mask_cache_size: int = 256,
    ):
        super().__init__()
        self.classifier, self.cls_backend = build_efficientnetv2_s(num_classes)
        self.mask_score_threshold = float(mask_score_threshold)

        self.mask2former_processor = AutoImageProcessor.from_pretrained(mask2former_model_id)
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(mask2former_model_id)
        self.mask_backend = mask2former_model_id
        self.mask_cache_size = int(mask_cache_size)
        self._mask_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        if freeze_mask2former:
            for p in self.mask2former.parameters():
                p.requires_grad = False
            self.mask2former.eval()

        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _foreground_from_panoptic(self, panoptic: dict, h: int, w: int) -> np.ndarray:
        seg = panoptic["segmentation"].detach().cpu().numpy()
        info = panoptic.get("segments_info", [])
        if not info:
            return np.ones((h, w), dtype=np.float32)

        center_y, center_x = h / 2.0, w / 2.0
        scored = []
        for s in info:
            sid = s["id"]
            score = float(s.get("score", 1.0))
            if score < self.mask_score_threshold:
                continue
            m = seg == sid
            area = int(m.sum())
            if area <= 0:
                continue
            ys, xs = np.where(m)
            cy, cx = ys.mean(), xs.mean()
            dist = float(((cy - center_y) ** 2 + (cx - center_x) ** 2) ** 0.5)
            scored.append((area, -dist, sid))

        if not scored:
            return np.ones((h, w), dtype=np.float32)

        scored.sort(reverse=True)
        keep_ids = [x[2] for x in scored[:3]]
        fg = np.isin(seg, keep_ids).astype(np.float32)

        fg_ratio = float(fg.mean())
        if fg_ratio < 0.01 or fg_ratio > 0.98:
            return np.ones((h, w), dtype=np.float32)
        return fg

    @torch.no_grad()
    def _build_foreground_mask(self, x_raw: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x_raw.shape
        masks = []
        for i in range(b):
            img = x_raw[i].detach().cpu().clamp(0.0, 1.0)
            img_u8 = (img * 255.0).to(torch.uint8)
            cache_key = hashlib.sha1(img_u8.numpy().tobytes()).hexdigest()

            cached = self._mask_cache.get(cache_key)
            if cached is not None:
                self._mask_cache.move_to_end(cache_key)
                masks.append(torch.from_numpy(cached.copy()))
                continue

            pil = TF.to_pil_image(img)

            inputs = self.mask2former_processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(x_raw.device) for k, v in inputs.items()}
            if x_raw.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = self.mask2former(**inputs)
            else:
                outputs = self.mask2former(**inputs)

            panoptic = self.mask2former_processor.post_process_panoptic_segmentation(
                outputs,
                target_sizes=[(h, w)],
                threshold=self.mask_score_threshold,
            )[0]

            fg_np = self._foreground_from_panoptic(panoptic, h, w)
            self._mask_cache[cache_key] = fg_np
            if len(self._mask_cache) > self.mask_cache_size:
                self._mask_cache.popitem(last=False)
            masks.append(torch.from_numpy(fg_np))

        fg = torch.stack(masks, dim=0).unsqueeze(1).to(device=x_raw.device, dtype=x_raw.dtype)
        return fg

    @staticmethod
    def _apply_blur_background(x_raw: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
        mixed = []
        for i in range(x_raw.shape[0]):
            img = x_raw[i]
            m = fg_mask[i]
            blurred = TF.gaussian_blur(img, kernel_size=[11, 11], sigma=[5.0, 5.0])
            mixed.append(img * m + blurred * (1.0 - m))
        return torch.stack(mixed, dim=0)

    def build_classifier_input(self, x_raw: torch.Tensor) -> torch.Tensor:
        fg_mask = self._build_foreground_mask(x_raw)
        x_focus = self._apply_blur_background(x_raw, fg_mask)
        return (x_focus - self.mean) / self.std

    def forward(self, x_raw: torch.Tensor):
        x_cls = self.build_classifier_input(x_raw)
        logits = self.classifier(x_cls)
        return logits, None


class EfficientMask2FormerInferenceAdapter(nn.Module):
    def __init__(self, model: EffNetV2Mask2FormerClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None, training: bool = False):
        logits, _ = self.model(x)
        return logits, None, None, None

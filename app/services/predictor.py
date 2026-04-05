import time
from typing import List, Dict, Any
import torch
import torch.nn.functional as F


def _forward_once(model_or_runtime, x: torch.Tensor, prior: torch.Tensor):
    if hasattr(model_or_runtime, "forward") and not isinstance(model_or_runtime, torch.nn.Module):
        return model_or_runtime.forward(x, prior)
    return model_or_runtime(x, prior, training=False)


@torch.inference_mode()
def predict_one(
    model,
    x: torch.Tensor,
    class_names: List[str],
    organ_dim: int,
    topk: int = 5,
    two_pass: bool = True,
) -> Dict[str, Any]:
    prior = torch.full((1, organ_dim), 1.0 / organ_dim, device=x.device)

    # pass1
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    logits, aux_org, probs, entropy = _forward_once(model, x, prior)

    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    infer_ms = (t1 - t0) * 1000.0
    used_prior = prior
    used_two_pass = False

    # pass2
    if two_pass and aux_org is not None:
        prior2 = F.softmax(aux_org, dim=-1)

        if x.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        logits, aux_org, probs, entropy = _forward_once(model, x, prior2)

        if x.device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        infer_ms += (t3 - t2) * 1000.0
        used_prior = prior2
        used_two_pass = True

    pred_probs = F.softmax(logits, dim=-1)[0]
    pred_idx = int(torch.argmax(pred_probs).item())
    pred_label = class_names[pred_idx]
    confidence = float(pred_probs[pred_idx].item())

    k = min(topk, pred_probs.numel())
    top_vals, top_idxs = torch.topk(pred_probs, k=k)
    top_list = [
        {
            "rank": i + 1,
            "class_idx": int(ci.item()),
            "label": class_names[int(ci.item())],
            "prob": float(p.item()),
        }
        for i, (ci, p) in enumerate(zip(top_idxs, top_vals))
    ]

    routing: Dict[str, Any] = {}
    if probs is not None:
        p_mean = probs.mean(dim=(0, 1))
        routing["expert_prob_mean"] = [float(v.item()) for v in p_mean]
        routing["top_expert"] = int(torch.argmax(p_mean).item())
    if entropy is not None:
        routing["entropy_mean"] = float(entropy.mean().item())
        routing["entropy_max"] = float(entropy.max().item())

    return {
        "pred": {
            "class_idx": pred_idx,
            "label": pred_label,
            "confidence": confidence,
            "topk": top_list,
        },
        "organ_prior_used": [float(v) for v in used_prior[0].detach().cpu().tolist()],
        "two_pass": used_two_pass,
        "timing_ms": {"inference_ms": infer_ms},
        "routing": routing,
    }

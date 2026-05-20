from __future__ import annotations

import copy
import io
import math
import sys
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.quantization import ResNet18_QuantizedWeights, resnet18 as quantized_resnet18

APP_TITLE = "Демонстрационный проект: выбор стратегии квантования"
APP_SUBTITLE = (
    "Наглядный GUI: стресс-тест входа, сравнение стратегий квантования, "
    "анализ чувствительности слоёв и выбор рекомендации под конкретную цель."
)
DEFAULT_TOPK = 5
DEFAULT_REPEATS = 12
CPU = torch.device("cpu")

STRATEGY_LABELS = {
    "per_tensor_int8": "Per-tensor INT8",
    "per_channel_int8": "Per-channel INT8",
    "mixed_precision": "Mixed precision",
    "native_int8": "Native INT8",
}

GOAL_MODES = [
    "Простота внедрения",
    "Баланс",
    "Максимальная надёжность",
]

COMPLEXITY_PENALTY = {
    "per_tensor_int8": 0.05,
    "per_channel_int8": 0.18,
    "mixed_precision": 0.35,
    "native_int8": 0.12,
}

GOAL_PRIOR = {
    "Простота внедрения": {
        "per_tensor_int8": 0.08,
        "per_channel_int8": 0.02,
        "mixed_precision": -0.06,
        "native_int8": 0.04,
    },
    "Баланс": {
        "per_tensor_int8": 0.00,
        "per_channel_int8": 0.05,
        "mixed_precision": 0.00,
        "native_int8": 0.03,
    },
    "Максимальная надёжность": {
        "per_tensor_int8": -0.04,
        "per_channel_int8": 0.02,
        "mixed_precision": 0.08,
        "native_int8": 0.02,
    },
}

DISTORTION_MODES = [
    "none",
    "gaussian_noise",
    "blur",
    "low_contrast",
    "darken",
    "jpeg_compression",
]


def get_supported_quant_backend() -> Tuple[Optional[str], List[str]]:
    engines = list(torch.backends.quantized.supported_engines)
    chosen = None
    for candidate in ["x86", "fbgemm", "qnnpack"]:
        if candidate in engines:
            chosen = candidate
            break
    if chosen is not None:
        torch.backends.quantized.engine = chosen
    return chosen, engines


@lru_cache(maxsize=1)
def get_runtime_config() -> Dict[str, str]:
    backend, engines = get_supported_quant_backend()
    cuda_available = torch.cuda.is_available()
    gpu_name = "нет"
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA обнаружена, но имя GPU не получено"
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "backend": backend or "нет",
        "supported_backends": ", ".join(engines) if engines else "нет",
        "native_int8_available": str(backend is not None),
        "cuda_available": str(cuda_available),
        "gpu_name": gpu_name,
    }


@lru_cache(maxsize=1)
def load_assets() -> Dict[str, object]:
    weights = ResNet18_Weights.DEFAULT
    fp32_model = resnet18(weights=weights).eval().cpu()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    native_model = None
    if get_runtime_config()["native_int8_available"] == "True":
        try:
            native_model = quantized_resnet18(
                weights=ResNet18_QuantizedWeights.DEFAULT,
                quantize=True,
            ).eval().cpu()
        except Exception:
            native_model = None
    return {
        "fp32_model": fp32_model,
        "native_int8_model": native_model,
        "preprocess": preprocess,
        "categories": categories,
        "model_title": "ResNet18",
    }


def affine_fake_quantize_tensor_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    w = weight.detach().float().cpu()
    qmin, qmax = -128, 127
    w_min = float(w.min().item())
    w_max = float(w.max().item())
    if abs(w_max - w_min) < 1e-12:
        return w.clone(), {"scale": 1.0, "zero_point": 0}
    scale = max((w_max - w_min) / float(qmax - qmin), 1e-12)
    zero_point = qmin - round(w_min / scale)
    zero_point = int(np.clip(zero_point, qmin, qmax))
    q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax)
    dq = (q - zero_point) * scale
    return dq, {"scale": scale, "zero_point": zero_point}


def affine_fake_quantize_per_channel_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    w = weight.detach().float().cpu()
    original_shape = w.shape
    flat = w.reshape(w.shape[0], -1)
    rows = []
    scales = []
    for i in range(flat.shape[0]):
        dq_i, qp = affine_fake_quantize_tensor_int8(flat[i])
        rows.append(dq_i)
        scales.append(qp["scale"])
    dq = torch.stack(rows, dim=0).reshape(original_shape)
    return dq, {
        "scale_mean": float(np.mean(scales)) if scales else 0.0,
        "scale_std": float(np.std(scales)) if scales else 0.0,
    }


def build_strategy_model(strategy: str) -> nn.Module:
    assets = load_assets()
    base = assets["fp32_model"]
    native = assets["native_int8_model"]
    if strategy == "native_int8":
        if native is None:
            raise ValueError("Native INT8 недоступен в текущей среде")
        return native
    model = copy.deepcopy(base).cpu().eval()
    sensitive_layers, _ = get_reference_sensitive_layers()
    sensitive_set = set(sensitive_layers)
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if getattr(module, "weight", None) is None:
            continue
        if strategy == "mixed_precision" and name in sensitive_set:
            continue
        if strategy == "per_channel_int8" and module.weight.ndim >= 2:
            dq, _ = affine_fake_quantize_per_channel_int8(module.weight.data)
        else:
            dq, _ = affine_fake_quantize_tensor_int8(module.weight.data)
        module.weight.data.copy_(dq.to(module.weight.data.dtype))
    return model


@lru_cache(maxsize=8)
def get_strategy_model(strategy: str) -> nn.Module:
    return build_strategy_model(strategy)


def apply_stress_test(image: Image.Image, mode: str, severity: float) -> Image.Image:
    image = image.convert("RGB")
    severity = float(np.clip(severity, 0.0, 1.0))
    if mode == "none" or severity <= 1e-6:
        return image
    if mode == "gaussian_noise":
        arr = np.asarray(image).astype(np.float32) / 255.0
        sigma = 0.04 + 0.28 * severity
        noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((out * 255).astype(np.uint8))
    if mode == "blur":
        radius = 0.3 + 4.5 * severity
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    if mode == "low_contrast":
        factor = max(0.12, 1.0 - 0.82 * severity)
        return ImageEnhance.Contrast(image).enhance(factor)
    if mode == "darken":
        factor = max(0.12, 1.0 - 0.75 * severity)
        return ImageEnhance.Brightness(image).enhance(factor)
    if mode == "jpeg_compression":
        quality = int(max(8, round(95 - 82 * severity)))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    return image


def preprocess_pil_image(image: Image.Image) -> torch.Tensor:
    preprocess = load_assets()["preprocess"]
    return preprocess(image).unsqueeze(0)


def run_model_once(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(x).detach().cpu()


def benchmark_single_image(model: nn.Module, x: torch.Tensor, repeats: int = 10, warmup: int = 3) -> Dict[str, float]:
    with torch.inference_mode():
        for _ in range(max(1, warmup)):
            _ = model(x)
    values = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(x)
        t1 = time.perf_counter()
        values.append((t1 - t0) * 1000.0)
    arr = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def topk_dataframe(logits: torch.Tensor, categories: List[str], topk: int = 5) -> pd.DataFrame:
    probs = torch.softmax(logits[0].float(), dim=0)
    values, indices = probs.topk(topk)
    rows = []
    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        rows.append({
            "rank": rank,
            "class_id": int(idx),
            "class_name": categories[int(idx)],
            "probability": round(float(score), 6),
        })
    return pd.DataFrame(rows)


def collect_output_metrics(logits_fp32: torch.Tensor, logits_test: torch.Tensor, topk: int = 5) -> Dict[str, float]:
    probs_ref = torch.softmax(logits_fp32.float(), dim=1)
    probs_test = torch.softmax(logits_test.float(), dim=1)
    top1_ref = probs_ref.argmax(dim=1)
    top1_test = probs_test.argmax(dim=1)
    topk_ref = probs_ref.topk(topk, dim=1).indices
    topk_test = probs_test.topk(topk, dim=1).indices
    top1_agreement = float((top1_ref == top1_test).float().mean().item())
    overlaps = []
    for i in range(probs_ref.shape[0]):
        a = set(topk_ref[i].tolist())
        b = set(topk_test[i].tolist())
        overlaps.append(len(a & b) / topk)
    topk_overlap = float(np.mean(overlaps))
    eps = 1e-12
    kl_div = float((probs_ref * (probs_ref.clamp_min(eps).log() - probs_test.clamp_min(eps).log())).sum(dim=1).mean().item())
    cosine = float(F.cosine_similarity(logits_fp32.float(), logits_test.float(), dim=1).mean().item())
    mean_abs_logit_diff = float(torch.mean(torch.abs(logits_fp32.float() - logits_test.float())).item())
    conf_ref = float(probs_ref.max(dim=1).values.mean().item())
    conf_test = float(probs_test.max(dim=1).values.mean().item())
    confidence_change = abs(conf_ref - conf_test)
    return {
        "top1_agreement": top1_agreement,
        "topk_overlap": topk_overlap,
        "mean_kl_divergence": kl_div,
        "mean_cosine_similarity": cosine,
        "mean_abs_logit_diff": mean_abs_logit_diff,
        "mean_top1_confidence_fp32": conf_ref,
        "mean_top1_confidence_test": conf_test,
        "confidence_change": confidence_change,
    }


def layer_stats_from_pair(base_weight: torch.Tensor, test_weight: torch.Tensor) -> Dict[str, float]:
    w0 = base_weight.detach().float().cpu().reshape(-1)
    w1 = test_weight.detach().float().cpu().reshape(-1)
    diff = w0 - w1
    abs_w0 = w0.abs()
    std = float(w0.std(unbiased=False).item())
    max_abs = float(abs_w0.max().item())
    outlier_ratio_3sigma = float((abs_w0 > (3.0 * max(std, 1e-12))).float().mean().item())
    risk_score = max_abs / max(std, 1e-12)
    rel_l2 = float(torch.linalg.norm(diff).item() / max(torch.linalg.norm(w0).item(), 1e-12))
    cosine = float(F.cosine_similarity(w0.unsqueeze(0), w1.unsqueeze(0), dim=1).item())
    mse = float(torch.mean(diff ** 2).item())
    return {
        "risk_score": risk_score,
        "outlier_ratio_3sigma": outlier_ratio_3sigma,
        "relative_l2_error": rel_l2,
        "relative_l2_error_pct": 100.0 * rel_l2,
        "cosine_after_quant": cosine,
        "mse": mse,
    }


@lru_cache(maxsize=1)
def get_reference_sensitive_layers() -> Tuple[List[str], pd.DataFrame]:
    base = load_assets()["fp32_model"]
    rows = []
    for name, module in base.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and getattr(module, "weight", None) is not None:
            dq, _ = affine_fake_quantize_tensor_int8(module.weight.data)
            stats = layer_stats_from_pair(module.weight.data, dq)
            rows.append({"layer": name, "layer_type": module.__class__.__name__, **stats})
    df = pd.DataFrame(rows).sort_values("relative_l2_error", ascending=False).reset_index(drop=True)
    q75 = float(df["relative_l2_error"].quantile(0.75))
    risk75 = float(df["risk_score"].quantile(0.75))
    mask = (df["relative_l2_error"] >= q75) | (df["risk_score"] >= risk75)
    return df.loc[mask, "layer"].tolist(), df


def analysis_strategy_key(strategy: str) -> str:
    return "per_tensor_int8" if strategy == "native_int8" else strategy


def analyze_layers_for_strategy(strategy: str) -> pd.DataFrame:
    base = load_assets()["fp32_model"]
    test = get_strategy_model(analysis_strategy_key(strategy))
    base_modules = dict(base.named_modules())
    test_modules = dict(test.named_modules())
    rows = []
    for name, module in base_modules.items():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if getattr(module, "weight", None) is None:
            continue
        if name not in test_modules or getattr(test_modules[name], "weight", None) is None:
            continue
        stats = layer_stats_from_pair(module.weight.data, test_modules[name].weight.data)
        rows.append({
            "layer": name,
            "layer_type": module.__class__.__name__,
            "shape": str(tuple(module.weight.shape)),
            **stats,
        })
    return pd.DataFrame(rows).sort_values("relative_l2_error", ascending=False).reset_index(drop=True)


def summarize_layer_df(layer_df: pd.DataFrame) -> Dict[str, float]:
    q75 = float(layer_df["relative_l2_error_pct"].quantile(0.75))
    risk75 = float(layer_df["risk_score"].quantile(0.75))
    sensitive_mask = (layer_df["relative_l2_error_pct"] >= q75) | (layer_df["risk_score"] >= risk75)
    sensitive_share = float(sensitive_mask.mean())
    corr = float(layer_df["risk_score"].corr(layer_df["relative_l2_error"]))
    if math.isnan(corr):
        corr = 0.0
    return {
        "layer_q75_error_pct": q75,
        "sensitive_layers_share": sensitive_share,
        "risk_error_corr": max(0.0, corr),
    }


def normalize_output_stability(m: Dict[str, float]) -> float:
    score = 0.0
    score += 0.30 * m["top1_agreement"]
    score += 0.20 * m["topk_overlap"]
    score += 0.25 * np.clip((m["mean_cosine_similarity"] - 0.96) / 0.04, 0.0, 1.0)
    score += 0.15 * np.clip(1.0 - m["mean_kl_divergence"] / 0.03, 0.0, 1.0)
    score += 0.10 * np.clip(1.0 - m["confidence_change"] / 0.12, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def normalize_layer_stability(layer_s: Dict[str, float]) -> float:
    a = np.clip(1.0 - layer_s["layer_q75_error_pct"] / 6.0, 0.0, 1.0)
    b = np.clip(1.0 - layer_s["sensitive_layers_share"] / 0.70, 0.0, 1.0)
    return float(0.60 * a + 0.40 * b)


def compute_goal_based_score(row: Dict[str, float], goal_mode: str) -> float:
    strategy_key = row["strategy_key"]
    simplicity_term = 1.0 - COMPLEXITY_PENALTY.get(strategy_key, 0.20)
    confidence_term = max(0.0, 1.0 - row["confidence_change"] / 0.12)
    prior = GOAL_PRIOR.get(goal_mode, {}).get(strategy_key, 0.0)
    if goal_mode == "Простота внедрения":
        score = (
            0.35 * row["output_stability_score"] +
            0.15 * row["layer_stability_score"] +
            0.35 * simplicity_term +
            0.15 * confidence_term +
            prior
        )
    elif goal_mode == "Баланс":
        score = (
            0.42 * row["output_stability_score"] +
            0.28 * row["layer_stability_score"] +
            0.18 * simplicity_term +
            0.12 * confidence_term +
            prior
        )
    else:
        score = (
            0.45 * row["output_stability_score"] +
            0.35 * row["layer_stability_score"] +
            0.10 * confidence_term +
            0.10 * simplicity_term +
            prior
        )
    return float(np.clip(score, 0.0, 1.0))


def choose_recommendation_by_goal(strategy_rows: List[Dict[str, float]], goal_mode: str) -> Tuple[str, str, List[Dict[str, float]]]:
    scored_rows: List[Dict[str, float]] = []
    for row in strategy_rows:
        row_copy = dict(row)
        row_copy["goal_based_score"] = compute_goal_based_score(row_copy, goal_mode)
        scored_rows.append(row_copy)
    if goal_mode == "Простота внедрения":
        eligible = [r for r in scored_rows if r["top1_agreement"] >= 1.0 and r["mean_cosine_similarity"] >= 0.995]
        best = max(eligible, key=lambda r: r["goal_based_score"]) if eligible else max(scored_rows, key=lambda r: r["goal_based_score"])
        reason = "Выбрана стратегия с минимальной инженерной сложностью при сохранении приемлемой устойчивости выходов."
    elif goal_mode == "Баланс":
        best = max(scored_rows, key=lambda r: r["goal_based_score"])
        reason = "Выбрана стратегия с лучшим компромиссом между устойчивостью модели и сложностью внедрения."
    else:
        best = max(scored_rows, key=lambda r: r["goal_based_score"])
        reason = "Выбрана стратегия с приоритетом на сохранение поведения модели и снижение риска деградации чувствительных слоёв."
    return best["strategy_key"], reason, scored_rows


def make_topk_plot(pred_tables: Dict[str, pd.DataFrame]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = list(pred_tables.keys())
    if not keys:
        return fig
    topk = len(next(iter(pred_tables.values())))
    x = np.arange(topk)
    width = 0.18
    for i, key in enumerate(keys):
        df = pred_tables[key]
        probs = df["probability"].values
        ax.bar(x + i * width, probs, width=width, label=STRATEGY_LABELS.get(key, key))
    ax.set_xticks(x + width * (len(keys) - 1) / 2)
    ax.set_xticklabels([f"Top-{i + 1}" for i in range(topk)])
    ax.set_ylabel("Probability")
    ax.set_title("Сравнение top-k вероятностей по стратегиям")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def make_latency_plot(rows: List[Dict[str, float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [STRATEGY_LABELS[r["strategy_key"]] for r in rows]
    means = [r["mean_ms"] for r in rows]
    errs = [r["std_ms"] for r in rows]
    ax.bar(labels, means, yerr=errs, capsize=4)
    ax.set_ylabel("ms / image")
    ax.set_title("Среднее время инференса")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15)
    fig.tight_layout()
    return fig


def make_output_stability_plot(rows: List[Dict[str, float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [STRATEGY_LABELS[r["strategy_key"]] for r in rows]
    vals = [r["output_stability_score"] for r in rows]
    ax.bar(labels, vals)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Стабильность выходов после квантования")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15)
    fig.tight_layout()
    return fig


def make_scatter_plot(layer_df: pd.DataFrame) -> plt.Figure:
    corr = float(layer_df["risk_score"].corr(layer_df["relative_l2_error"]))
    if math.isnan(corr):
        corr = 0.0
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(layer_df["risk_score"], layer_df["relative_l2_error"], alpha=0.8)
    for _, row in layer_df.head(min(6, len(layer_df))).iterrows():
        ax.annotate(row["layer"], (row["risk_score"], row["relative_l2_error"]), fontsize=8, alpha=0.8)
    ax.set_title(f"Risk-score vs relative L2 error (corr = {corr:.3f})")
    ax.set_xlabel("risk_score = max(|w|) / std(w)")
    ax.set_ylabel("relative L2 error")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def make_sensitive_layers_bar(layer_df: pd.DataFrame) -> plt.Figure:
    top = layer_df.head(min(8, len(layer_df))).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["layer"], top["relative_l2_error_pct"])
    ax.set_xlabel("relative L2 error, %")
    ax.set_title("Наиболее чувствительные слои")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def make_distribution_plots(layer_name: str, strategy: str) -> plt.Figure:
    base_model = load_assets()["fp32_model"]
    test_model = get_strategy_model(analysis_strategy_key(strategy))
    m0 = dict(base_model.named_modules())[layer_name]
    m1 = dict(test_model.named_modules())[layer_name]
    w0 = m0.weight.detach().float().cpu().numpy().ravel()
    w1 = m1.weight.detach().float().cpu().numpy().ravel()
    err = w0 - w1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(w0, bins=100, alpha=0.65, label="FP32")
    axes[0].hist(w1, bins=100, alpha=0.65, label=STRATEGY_LABELS.get(strategy, strategy))
    axes[0].set_title("Распределение весов")
    axes[0].set_xlabel("weight")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].hist(err, bins=100)
    axes[1].set_title("Распределение ошибки квантования")
    axes[1].set_xlabel("FP32 - quantized")
    axes[1].set_ylabel("count")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    return fig


def make_matrix_plot(layer_name: str, strategy: str) -> plt.Figure:
    base_model = load_assets()["fp32_model"]
    test_model = get_strategy_model(analysis_strategy_key(strategy))
    w0 = dict(base_model.named_modules())[layer_name].weight.detach().float().cpu()
    w1 = dict(test_model.named_modules())[layer_name].weight.detach().float().cpu()
    if w0.ndim == 4:
        a = w0[0, 0].numpy()
        b = w1[0, 0].numpy()
    elif w0.ndim == 2:
        a = w0[: min(64, w0.shape[0]), : min(64, w0.shape[1])].numpy()
        b = w1[: min(64, w1.shape[0]), : min(64, w1.shape[1])].numpy()
    else:
        flat0 = w0.flatten()[:256].numpy().reshape(16, 16)
        flat1 = w1.flatten()[:256].numpy().reshape(16, 16)
        a, b = flat0, flat1
    e = a - b
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(a, cmap="coolwarm")
    axes[0].set_title("FP32")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(b, cmap="coolwarm")
    axes[1].set_title(STRATEGY_LABELS.get(strategy, strategy))
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(e, cmap="coolwarm")
    axes[2].set_title("Ошибка")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    return fig


def make_recommendation_plot(rows: List[Dict[str, float]]) -> plt.Figure:
    labels = [STRATEGY_LABELS[r["strategy_key"]] for r in rows]
    output = [r["output_stability_score"] for r in rows]
    layer = [r["layer_stability_score"] for r in rows]
    simplicity = [1.0 - COMPLEXITY_PENALTY.get(r["strategy_key"], 0.20) for r in rows]
    goal_score = [r["goal_based_score"] for r in rows]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 1.5 * width, output, width, label="Стабильность выходов")
    ax.bar(x - 0.5 * width, layer, width, label="Послойная устойчивость")
    ax.bar(x + 0.5 * width, simplicity, width, label="Простота внедрения")
    ax.bar(x + 1.5 * width, goal_score, width, label="Итог по выбранной цели")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Сводные оценки проекта")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def available_strategies() -> List[str]:
    items = ["per_tensor_int8", "per_channel_int8", "mixed_precision"]
    if load_assets()["native_int8_model"] is not None:
        items.append("native_int8")
    return items


def run_full_analysis(image: Image.Image, distortion_mode: str, severity: float, repeats: int, topk: int, goal_mode: str):
    if image is None:
        raise gr.Error("Загрузите изображение")
    assets = load_assets()
    categories = assets["categories"]
    fp32_model = assets["fp32_model"]
    stressed = apply_stress_test(image, distortion_mode, severity)
    x = preprocess_pil_image(stressed)
    logits_fp32 = run_model_once(fp32_model, x)
    preds_fp32 = topk_dataframe(logits_fp32, categories, topk=topk)
    rows = []
    pred_tables = {}
    for strategy in available_strategies():
        model = get_strategy_model(strategy)
        logits = run_model_once(model, x)
        preds = topk_dataframe(logits, categories, topk=topk)
        metrics = collect_output_metrics(logits_fp32, logits, topk=topk)
        bench = benchmark_single_image(model, x, repeats=repeats, warmup=3)
        layer_df = analyze_layers_for_strategy(strategy)
        layer_stats = summarize_layer_df(layer_df)
        output_stability_score = normalize_output_stability(metrics)
        layer_stability_score = normalize_layer_stability(layer_stats)
        criterion_interpretability = float(np.clip(layer_stats["risk_error_corr"], 0.0, 1.0))
        base_readiness_score = float(0.50 * output_stability_score + 0.35 * layer_stability_score + 0.15 * criterion_interpretability)
        rows.append({
            "strategy_key": strategy,
            "strategy": STRATEGY_LABELS[strategy],
            **metrics,
            **layer_stats,
            **bench,
            "output_stability_score": output_stability_score,
            "layer_stability_score": layer_stability_score,
            "criterion_interpretability": criterion_interpretability,
            "base_readiness_score": base_readiness_score,
        })
        pred_tables[strategy] = preds
    chosen_key, explanation, rows = choose_recommendation_by_goal(rows, goal_mode)
    chosen_row = next(r for r in rows if r["strategy_key"] == chosen_key)
    strategy_metric_rows = []
    for r in rows:
        strategy_metric_rows.append({
            "Стратегия": STRATEGY_LABELS[r["strategy_key"]],
            "Top-1 agreement": round(r["top1_agreement"], 4),
            f"Top-{topk} overlap": round(r["topk_overlap"], 4),
            "Cosine": round(r["mean_cosine_similarity"], 4),
            "KL": round(r["mean_kl_divergence"], 6),
            "|Δ logits|": round(r["mean_abs_logit_diff"], 4),
            "Confidence change": round(r["confidence_change"], 4),
            "Q75 error, %": round(r["layer_q75_error_pct"], 4),
            "Sensitive share": round(r["sensitive_layers_share"], 4),
            "Базовая готовность": round(r["base_readiness_score"], 4),
            "Итог по цели": round(r["goal_based_score"], 4),
        })
    summary_md = (
        "### Краткий вывод\n"
        f"- **Искажение:** {distortion_mode}, severity = {severity:.2f}.\n"
        f"- **Цель оптимизации:** {goal_mode}.\n"
        f"- **Рекомендованная стратегия:** **{STRATEGY_LABELS[chosen_key]}**.\n"
        f"- **Почему:** {explanation}\n"
        f"- **Стабильность выходов выбранной стратегии:** {chosen_row['output_stability_score']:.3f}.\n"
        f"- **Послойная устойчивость:** {chosen_row['layer_stability_score']:.3f}.\n"
        f"- **Интерпретируемость критерия:** {chosen_row['criterion_interpretability']:.3f}.\n"
        f"- **Итог по выбранной цели:** {chosen_row['goal_based_score']:.3f}."
    )
    metrics_df = pd.DataFrame(strategy_metric_rows)
    topk_plot = make_topk_plot(pred_tables)
    latency_plot = make_latency_plot(rows)
    stability_plot = make_output_stability_plot(rows)
    chosen_layer_df = analyze_layers_for_strategy(chosen_key)
    layer_scatter = make_scatter_plot(chosen_layer_df)
    layer_bar = make_sensitive_layers_bar(chosen_layer_df)
    layer_table = chosen_layer_df[["layer", "layer_type", "shape", "risk_score", "outlier_ratio_3sigma", "relative_l2_error_pct", "cosine_after_quant"]].copy().round({
        "risk_score": 4,
        "outlier_ratio_3sigma": 6,
        "relative_l2_error_pct": 6,
        "cosine_after_quant": 6,
    })
    layer_choices = chosen_layer_df["layer"].tolist()
    default_layer = layer_choices[0] if layer_choices else None
    dist_plot = make_distribution_plots(default_layer, chosen_key) if default_layer else None
    matrix_plot = make_matrix_plot(default_layer, chosen_key) if default_layer else None
    rec_table = pd.DataFrame([
        {
            "Стратегия": r["strategy"],
            "Стабильность выходов": round(r["output_stability_score"], 4),
            "Послойная устойчивость": round(r["layer_stability_score"], 4),
            "Простота внедрения": round(1.0 - COMPLEXITY_PENALTY.get(r["strategy_key"], 0.20), 4),
            "Базовая готовность": round(r["base_readiness_score"], 4),
            "Итог по цели": round(r["goal_based_score"], 4),
        }
        for r in rows
    ])
    rec_plot = make_recommendation_plot(rows)
    chosen_pred_df = pred_tables[chosen_key]
    return (
        stressed,
        summary_md,
        STRATEGY_LABELS[chosen_key],
        preds_fp32,
        chosen_pred_df,
        metrics_df,
        topk_plot,
        latency_plot,
        stability_plot,
        layer_table,
        layer_scatter,
        layer_bar,
        gr.Dropdown(choices=layer_choices, value=default_layer),
        dist_plot,
        matrix_plot,
        rec_table,
        rec_plot,
    )


def update_layer_visuals(layer_name: str, chosen_strategy_label: str):
    strategy_key = next((k for k, v in STRATEGY_LABELS.items() if v == chosen_strategy_label), "per_tensor_int8")
    return make_distribution_plots(layer_name, strategy_key), make_matrix_plot(layer_name, strategy_key)


def get_environment_table() -> pd.DataFrame:
    cfg = get_runtime_config()
    return pd.DataFrame({
        "Параметр": [
            "Python", "torch", "CUDA доступна", "GPU", "Поддерживаемые quantized backend",
            "Активный quantized backend", "Нативный INT8 backend доступен", "Архитектура"
        ],
        "Значение": [
            cfg["python"], cfg["torch"], cfg["cuda_available"], cfg["gpu_name"],
            cfg["supported_backends"], cfg["backend"], cfg["native_int8_available"], "ResNet18"
        ],
    })


def build_app() -> gr.Blocks:
    env_df = get_environment_table()
    strategy_options = [STRATEGY_LABELS[s] for s in available_strategies()]
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}\n\n{APP_SUBTITLE}")
        gr.Markdown(
            "**Идея демо:** на одной и той же картинке можно управляемо усиливать искажения и менять цель оптимизации. "
            "За счёт этого рекомендация наглядно переключается между `Per-tensor INT8`, `Per-channel INT8` и `Mixed precision`."
        )
        gr.Dataframe(value=env_df, label="Среда выполнения", interactive=False, wrap=True)
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Исходное изображение")
                distortion_mode = gr.Dropdown(choices=DISTORTION_MODES, value="none", label="Тип искажения")
                severity = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.0, label="Сила искажения")
                goal_mode = gr.Dropdown(choices=GOAL_MODES, value="Баланс", label="Цель оптимизации")
                repeats = gr.Slider(minimum=5, maximum=40, step=1, value=DEFAULT_REPEATS, label="Повторы для времени")
                topk = gr.Slider(minimum=3, maximum=8, step=1, value=DEFAULT_TOPK, label="Top-k")
                run_btn = gr.Button("Запустить стресс-тест", variant="primary")
            with gr.Column(scale=1):
                stressed_img = gr.Image(type="pil", label="Изображение после искажения")
                summary_md = gr.Markdown()
                chosen_strategy = gr.Dropdown(choices=strategy_options, value=strategy_options[0], label="Рекомендованная стратегия", interactive=False)
        with gr.Tab("Вкладка 1 — Демонстрация на примере"):
            with gr.Row():
                pred_fp32 = gr.Dataframe(label="FP32: top-k предсказания", interactive=False, wrap=True)
                pred_best = gr.Dataframe(label="Рекомендованная стратегия: top-k предсказания", interactive=False, wrap=True)
            strategy_metrics = gr.Dataframe(label="Сводка метрик по стратегиям", interactive=False, wrap=True)
            with gr.Row():
                topk_plot = gr.Plot(label="Сравнение top-k вероятностей")
                latency_plot = gr.Plot(label="Среднее время инференса")
            stability_plot = gr.Plot(label="Стабильность выходов после квантования")
        with gr.Tab("Вкладка 2 — Анализ слоёв"):
            layer_table = gr.Dataframe(label="Послойные метрики рекомендованной стратегии", interactive=False, wrap=True)
            with gr.Row():
                layer_scatter = gr.Plot(label="Risk-score vs error")
                layer_bar = gr.Plot(label="Наиболее чувствительные слои")
        with gr.Tab("Вкладка 3 — Распределения и матрицы"):
            layer_selector = gr.Dropdown(label="Слой для детального просмотра", choices=[])
            with gr.Row():
                dist_plot = gr.Plot(label="Распределения")
                matrix_plot = gr.Plot(label="Матрицы FP32 / quantized / error")
        with gr.Tab("Вкладка 4 — Рекомендация"):
            rec_table = gr.Dataframe(label="Сводные оценки стратегий", interactive=False, wrap=True)
            rec_plot = gr.Plot(label="Сводные оценки проекта")
            
        run_btn.click(
            fn=run_full_analysis,
            inputs=[image_input, distortion_mode, severity, repeats, topk, goal_mode],
            outputs=[
                stressed_img, summary_md, chosen_strategy, pred_fp32, pred_best, strategy_metrics,
                topk_plot, latency_plot, stability_plot,
                layer_table, layer_scatter, layer_bar, layer_selector, dist_plot, matrix_plot,
                rec_table, rec_plot,
            ],
        )
        layer_selector.change(
            fn=update_layer_visuals,
            inputs=[layer_selector, chosen_strategy],
            outputs=[dist_plot, matrix_plot],
        )
    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()

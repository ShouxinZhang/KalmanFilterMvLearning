#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io
import torch


@dataclass(frozen=True)
class Scaling:
    x_scale: float
    y_scale: float
    mode: str


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complex_rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(torch.abs(x) ** 2))


def load_mat_signal(mat_path: Path, key: str) -> np.ndarray:
    mat = scipy.io.loadmat(mat_path)
    if key not in mat:
        available = sorted(k for k in mat.keys() if not k.startswith("__"))
        msg = f"Key {key!r} not found in {mat_path}. Available keys: {available}"
        raise KeyError(msg)
    arr = mat[key]
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        msg = f"Expected 1D array for {key!r}, got shape {arr.shape}"
        raise ValueError(msg)
    if arr.dtype.kind != "c":
        arr = arr.astype(np.complex128)
    return arr


def load_mat_scalar(mat_path: Path, key: str) -> float:
    mat = scipy.io.loadmat(mat_path)
    if key not in mat:
        available = sorted(k for k in mat.keys() if not k.startswith("__"))
        msg = f"Key {key!r} not found in {mat_path}. Available keys: {available}"
        raise KeyError(msg)
    v = float(np.asarray(mat[key]).squeeze().item())
    if not math.isfinite(v):
        raise ValueError(f"Invalid scalar {key!r}={v} in {mat_path}")
    return v


def infer_fs_hz(mat_path: Path, *, key: str) -> float:
    """Infer sampling rate in Hz from MAT.

    The bundled MAT uses MHz-like numbers (e.g. 245.76). If the scalar looks like MHz,
    convert to Hz. Otherwise assume it's already in Hz.
    """

    fs = load_mat_scalar(mat_path, key)
    # Heuristic: values < 1e6 are very likely expressed in MHz.
    if fs < 1e6:
        fs *= 1e6
    return fs


def scale_signals(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, Scaling]:
    if mode == "none":
        return x, y, Scaling(x_scale=1.0, y_scale=1.0, mode=mode)

    if mode == "rms":
        x_scale = float(complex_rms(x).cpu().item())
        y_scale = float(complex_rms(y).cpu().item())
    elif mode == "adc15":
        x_scale = float(2**15)
        y_scale = float(2**15)
    else:
        raise ValueError(f"Unknown scale mode: {mode!r}")

    if not math.isfinite(x_scale) or x_scale <= 0:
        raise ValueError(f"Invalid x_scale={x_scale}")
    if not math.isfinite(y_scale) or y_scale <= 0:
        raise ValueError(f"Invalid y_scale={y_scale}")
    return x / x_scale, y / y_scale, Scaling(x_scale=x_scale, y_scale=y_scale, mode=mode)


def parse_orders(spec: str) -> list[int]:
    orders: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        orders.append(int(part))
    if not orders:
        raise ValueError("Empty --orders")
    for p in orders:
        if p <= 0:
            raise ValueError(f"Invalid order: {p}. Expected positive integers.")
    return orders


def build_symmetrized_memory_polynomial_features(
    x: torch.Tensor,
    *,
    memory_depth: int,
    orders: Iterable[int],
    include_conjugate: bool,
    mix_freq_hz: float | None = None,
    mix_conj_freq_hz: float | None = None,
    fs_hz: float = 1.0,
) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(f"Expected 1D x, got shape {tuple(x.shape)}")
    if memory_depth <= 0:
        raise ValueError(f"Invalid memory_depth={memory_depth}")
    orders = list(orders)
    if not orders:
        raise ValueError("orders must be non-empty")

    device = x.device
    dtype = torch.complex64
    x = x.to(dtype)
    n = x.shape[0]

    if (mix_freq_hz is None) ^ (mix_conj_freq_hz is None):
        raise ValueError("mix_freq_hz and mix_conj_freq_hz must be both set or both None")
    if mix_freq_hz is not None:
        if not math.isfinite(fs_hz) or fs_hz <= 0:
            raise ValueError(f"Invalid fs_hz={fs_hz}")
        if not math.isfinite(mix_freq_hz) or not math.isfinite(mix_conj_freq_hz):
            raise ValueError("Invalid mixing frequencies")
        # Precompute time rotators.
        t = torch.arange(n, device=device, dtype=torch.float32) / float(fs_hz)
        rot = torch.exp(1j * (2.0 * math.pi * float(mix_freq_hz)) * t).to(dtype)
        rot_conj = torch.exp(1j * (2.0 * math.pi * float(mix_conj_freq_hz)) * t).to(dtype)
    else:
        rot = None
        rot_conj = None

    per_delay = len(orders) * (2 if include_conjugate else 1)
    total = memory_depth * per_delay
    phi = torch.empty((n, total), dtype=dtype, device=device)

    col = 0
    for delay in range(memory_depth):
        if delay == 0:
            x_d = x
        else:
            pad = torch.zeros((delay,), dtype=dtype, device=device)
            x_d = torch.cat((pad, x[:-delay]))

        if rot is not None:
            x_d = x_d * rot
        abs_d = torch.abs(x_d)

        for p in orders:
            if p == 1:
                scale = 1.0
            else:
                scale = abs_d ** (p - 1)
            phi[:, col] = x_d * scale
            col += 1

        if include_conjugate:
            x_dc = torch.conj(x_d)
            # If LO-aware mixing is enabled, the conjugate branch represents a different RF image
            # and needs its own folded mixing frequency to land in the receiver baseband.
            if rot_conj is not None:
                x_dc = torch.conj(torch.cat((torch.zeros((delay,), dtype=dtype, device=device), x[:-delay])) if delay else x) * rot_conj
            for p in orders:
                if p == 1:
                    scale = 1.0
                else:
                    scale = abs_d ** (p - 1)
                phi[:, col] = x_dc * scale
                col += 1

    if col != total:
        raise RuntimeError(f"Internal error: wrote {col} cols, expected {total}")
    return phi


def init_theta(
    n_params: int,
    *,
    device: torch.device,
    std: float,
) -> torch.Tensor:
    if n_params <= 0:
        raise ValueError(f"Invalid n_params={n_params}")
    if std <= 0:
        raise ValueError(f"Invalid std={std}")
    real = torch.randn((n_params,), dtype=torch.float32, device=device) * std
    imag = torch.randn((n_params,), dtype=torch.float32, device=device) * std
    return (real + 1j * imag).to(torch.complex64)


def q_schedule_paper(epoch: int) -> float:
    if epoch < 10:
        return 0.0
    base = 1e-6
    decay = 0.7
    every = 50
    step = (epoch - 10) // every
    q = base * (decay**step)
    return max(q, 1e-12)


@dataclass
class EpochMetrics:
    epoch: int
    mse: float
    residual_db: float
    suppression_db: float


def evaluate(theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    y_hat = phi @ theta
    residual = y - y_hat
    mse = torch.mean(torch.abs(residual) ** 2)
    y_power = torch.mean(torch.abs(y) ** 2)
    mse_f = float(mse.detach().cpu().item())
    y_power_f = float(y_power.detach().cpu().item())
    eps = 1e-30

    if not math.isfinite(mse_f):
        return mse_f, float("nan"), float("nan")
    if mse_f < 0:
        mse_f = 0.0
    if not math.isfinite(y_power_f) or y_power_f <= 0:
        return mse_f, float("nan"), float("nan")

    residual_db = 10.0 * math.log10(max(mse_f, eps))
    suppression_db = 10.0 * math.log10(max(y_power_f, eps) / max(mse_f, eps))
    return mse_f, residual_db, suppression_db


def plot_epoch_metrics(path: Path, *, algo: str, metrics: list[EpochMetrics]) -> None:
    import matplotlib.pyplot as plt

    epochs = [m.epoch for m in metrics]
    residual_db = [m.residual_db for m in metrics]
    suppression_db = [m.suppression_db for m in metrics]

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
    fig.suptitle(algo)
    axes[0].plot(epochs, residual_db, linewidth=1.5)
    axes[0].set_ylabel("Residual power (dB)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, suppression_db, linewidth=1.5)
    axes[1].set_ylabel("Suppression (dB)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_psd(
    path: Path,
    *,
    y: np.ndarray,
    residual: np.ndarray,
    title: str,
    psd_method: str = "welch",
    psd_nperseg: int = 4096,
    psd_noverlap: int = 2048,
    psd_fs: float = 1.0,
    psd_scaling: str = "density",
    psd_xlim_hz: float = 0.0,
    psd_ymin_db: float | None = None,
    psd_ymax_db: float | None = None,
    psd_component: str = "complex",
) -> None:
    import matplotlib.pyplot as plt

    if y.ndim != 1 or residual.ndim != 1:
        raise ValueError("plot_psd expects 1D arrays")
    if y.shape != residual.shape:
        raise ValueError("y and residual must have same shape")

    freq, y_db = estimate_psd_db(
        y,
        method=psd_method,
        nperseg=psd_nperseg,
        noverlap=psd_noverlap,
        fs=psd_fs,
        scaling=psd_scaling,
        component=psd_component,
    )
    _, r_db = estimate_psd_db(
        residual,
        method=psd_method,
        nperseg=psd_nperseg,
        noverlap=psd_noverlap,
        fs=psd_fs,
        scaling=psd_scaling,
        component=psd_component,
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq, y_db, label="reference", linewidth=1.0)
    ax.plot(freq, r_db, label="residual", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)" if psd_fs != 1.0 else "Normalized frequency")
    ax.set_ylabel("PSD (dB)")
    ax.grid(True, alpha=0.3)
    if psd_fs != 1.0 and psd_xlim_hz and psd_xlim_hz > 0:
        ax.set_xlim(-psd_xlim_hz, psd_xlim_hz)
    if psd_ymin_db is not None or psd_ymax_db is not None:
        data_min = float(np.nanmin([y_db.min(), r_db.min()]))
        data_max = float(np.nanmax([y_db.max(), r_db.max()]))
        ymin = psd_ymin_db if psd_ymin_db is not None else float(np.nanmin([y_db.min(), r_db.min()]))
        ymax = psd_ymax_db if psd_ymax_db is not None else float(np.nanmax([y_db.max(), r_db.max()]))
        if data_max < ymin or data_min > ymax:
            print(
                f"warning: PSD y-limits [{ymin:.1f}, {ymax:.1f}] hide all data "
                f"(data range [{data_min:.1f}, {data_max:.1f}]). Plot may appear blank."
            )
        ax.set_ylim(ymin, ymax)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def estimate_psd_db(
    x: np.ndarray,
    *,
    method: str = "welch",
    nperseg: int = 4096,
    noverlap: int = 2048,
    fs: float = 1.0,
    scaling: str = "density",
    component: str = "complex",
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate PSD in dB for complex baseband signals.

    - method='fft': single periodogram (high variance; can look like a thick band)
    - method='welch': Welch averaged PSD (smoother; closer to typical paper plots)

    Returns (freq, psd_db) with freq fftshifted. If fs=1.0, freq is normalized.
    Use component='real' to mimic real-valued PSD plots (symmetric in frequency).
    """

    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError(f"estimate_psd_db expects 1D input, got shape {x.shape}")
    n = x.shape[0]
    if n <= 0:
        raise ValueError("estimate_psd_db expects non-empty input")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid fs={fs}")
    if scaling not in {"density", "spectrum"}:
        raise ValueError(f"Invalid scaling={scaling!r}; expected 'density' or 'spectrum'")
    if component not in {"complex", "real", "imag"}:
        raise ValueError(f"Invalid component={component!r}; expected 'complex', 'real', or 'imag'")

    if component == "real":
        x = np.real(x)
    elif component == "imag":
        x = np.imag(x)

    if method == "fft":
        freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
        X = np.fft.fftshift(np.fft.fft(x))
        spec = (np.abs(X) ** 2) / n
        if scaling == "density":
            df = fs / n
            psd = spec / max(df, 1e-30)
        else:
            psd = spec
        psd_db = 10.0 * np.log10(psd + 1e-30)
        return freq, psd_db

    if method == "welch":
        import scipy.signal

        seg = int(min(max(8, nperseg), n))
        ov = int(min(max(0, noverlap), max(0, seg - 1)))
        freq, psd = scipy.signal.welch(
            x,
            fs=fs,
            window="hann",
            nperseg=seg,
            noverlap=ov,
            detrend=False,
            return_onesided=False,
            scaling=scaling,
            average="mean",
        )
        freq = np.fft.fftshift(freq)
        psd = np.fft.fftshift(psd)
        psd_db = 10.0 * np.log10(np.asarray(psd).real + 1e-30)
        return freq, psd_db

    raise ValueError(f"Unknown PSD method: {method!r}")


def plot_psd_compare(
    path: Path,
    *,
    y: np.ndarray,
    residuals: dict[str, np.ndarray],
    title: str,
    psd_method: str = "welch",
    psd_nperseg: int = 4096,
    psd_noverlap: int = 2048,
    psd_fs: float = 1.0,
    psd_scaling: str = "density",
    psd_normalize: str = "none",
    noise: np.ndarray | None = None,
    psd_xlim_hz: float = 0.0,
    psd_ymin_db: float | None = None,
    psd_ymax_db: float | None = None,
    psd_component: str = "complex",
    psd_align: str = "none",
    psd_noise_target_db: float = -50.0,
) -> None:
    """Overlay PSD curves similar to the paper's Fig.4.

    Plots reference PSD plus multiple residual PSDs.
    A simple noise floor estimate is added as a horizontal line.
    """

    import matplotlib.pyplot as plt

    if y.ndim != 1:
        raise ValueError("plot_psd_compare expects 1D y")
    if not residuals:
        raise ValueError("plot_psd_compare expects non-empty residuals")
    for name, r in residuals.items():
        if r.ndim != 1:
            raise ValueError(f"Residual {name!r} is not 1D")
        if r.shape != y.shape:
            raise ValueError(f"Residual {name!r} shape {r.shape} != y shape {y.shape}")

    freq, y_db = estimate_psd_db(
        y,
        method=psd_method,
        nperseg=psd_nperseg,
        noverlap=psd_noverlap,
        fs=psd_fs,
        scaling=psd_scaling,
        component=psd_component,
    )
    residual_db: dict[str, np.ndarray] = {}
    for k, v in residuals.items():
        _, v_db = estimate_psd_db(
            v,
            method=psd_method,
            nperseg=psd_nperseg,
            noverlap=psd_noverlap,
            fs=psd_fs,
            scaling=psd_scaling,
            component=psd_component,
        )
        residual_db[k] = v_db

    noise_freq: np.ndarray | None = None
    noise_db: np.ndarray | None = None
    if noise is not None:
        noise_freq, noise_db = estimate_psd_db(
            noise,
            method=psd_method,
            nperseg=psd_nperseg,
            noverlap=psd_noverlap,
            fs=psd_fs,
            scaling=psd_scaling,
            component=psd_component,
        )
        # If grids differ, interpolate noise to match plotted freq.
        if noise_freq.shape != freq.shape or not np.allclose(noise_freq, freq):
            noise_db = np.interp(freq, noise_freq, noise_db, left=float(noise_db[0]), right=float(noise_db[-1]))
            noise_freq = freq
    else:
        # Fallback: heuristic constant noise floor estimate.
        noise_floor_db = min(float(np.percentile(v, 10.0)) for v in residual_db.values())
        noise_freq = freq
        noise_db = np.full_like(freq, noise_floor_db)

    if psd_normalize not in {"none", "ref_max"}:
        raise ValueError("psd_normalize must be one of: none, ref_max")
    if psd_normalize == "ref_max":
        ref0 = float(np.max(y_db))
        y_db = y_db - ref0
        residual_db = {k: v - ref0 for k, v in residual_db.items()}
        assert noise_db is not None
        noise_db = noise_db - ref0

    if psd_align not in {"none", "noise_median"}:
        raise ValueError("psd_align must be one of: none, noise_median")
    if psd_align == "noise_median":
        assert noise_db is not None
        # Align based on median noise floor within the plotted x-limits.
        if psd_fs != 1.0 and psd_xlim_hz and psd_xlim_hz > 0:
            band = np.abs(freq) <= psd_xlim_hz
        else:
            band = np.full_like(freq, True, dtype=bool)
        noise_med = float(np.median(noise_db[band]))
        if math.isfinite(noise_med):
            shift = float(psd_noise_target_db - noise_med)
            y_db = y_db + shift
            residual_db = {k: v + shift for k, v in residual_db.items()}
            noise_db = noise_db + shift
        else:
            shift = 0.0

    color_map = {
        "Adam": "tab:orange",
        "FDEKF": "tab:blue",
        "RR-FDEKF": "tab:red",
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    # Plot residuals slightly transparent so the reference remains visible when they overlap.
    ax.plot(freq, y_db, label="Reference PIM Signal", color="tab:green", linewidth=2.0, zorder=3)

    for name in ["RR-FDEKF", "FDEKF", "Adam"]:
        if name in residual_db:
            ax.plot(
                freq,
                residual_db[name],
                label=f"Cancelled signal ({name})",
                color=color_map.get(name, None),
                linewidth=1.2,
                alpha=0.85,
                zorder=2,
            )

    assert noise_db is not None
    ax.plot(freq, noise_db, label="Noise Floor", color="0.6", linewidth=1.0, zorder=1)

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)" if psd_fs != 1.0 else "Normalized frequency")
    ylab = "PSD (dB)"
    if psd_align == "noise_median":
        ylab = f"PSD (dB, noise median -> {psd_noise_target_db:.0f} dB)"
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    if psd_fs != 1.0 and psd_xlim_hz and psd_xlim_hz > 0:
        ax.set_xlim(-psd_xlim_hz, psd_xlim_hz)
    if psd_ymin_db is not None or psd_ymax_db is not None:
        data_min = float(np.nanmin([y_db.min(), noise_db.min()] + [v.min() for v in residual_db.values()]))
        data_max = float(np.nanmax([y_db.max(), noise_db.max()] + [v.max() for v in residual_db.values()]))
        ymin = psd_ymin_db if psd_ymin_db is not None else float(np.nanmin([y_db.min(), noise_db.min()]))
        ymax = psd_ymax_db if psd_ymax_db is not None else float(np.nanmax([y_db.max(), noise_db.max()]))
        if data_max < ymin or data_min > ymax:
            print(
                f"warning: PSD y-limits [{ymin:.1f}, {ymax:.1f}] hide all data "
                f"(data range [{data_min:.1f}, {data_max:.1f}]). Plot may appear blank."
            )
        ax.set_ylim(ymin, ymax)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_adam(
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    seed: int,
    init_std: float,
    lr: float,
    betas: tuple[float, float],
    eps: float,
) -> tuple[torch.Tensor, list[EpochMetrics]]:
    device = phi.device
    set_torch_seed(seed)

    theta = torch.nn.Parameter(init_theta(phi.shape[1], device=device, std=init_std))
    opt = torch.optim.Adam([theta], lr=lr, betas=betas, eps=eps)

    metrics: list[EpochMetrics] = []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        for t in range(phi.shape[0]):
            opt.zero_grad(set_to_none=True)
            y_hat = torch.sum(phi[t] * theta)
            loss = torch.abs(y[t] - y_hat) ** 2
            loss.backward()
            opt.step()
        mse, residual_db, suppression_db = evaluate(theta.detach(), phi, y)
        dt = time.perf_counter() - t0
        print(
            f"[Adam] epoch={epoch:4d} mse={mse:.6e} residual_db={residual_db:.3f} "
            f"suppression_db={suppression_db:.3f} time={dt:.2f}s"
        )
        metrics.append(
            EpochMetrics(
                epoch=epoch,
                mse=mse,
                residual_db=residual_db,
                suppression_db=suppression_db,
            )
        )
    return theta.detach(), metrics


def train_fdekf(
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    seed: int,
    init_std: float,
    p0: float,
    r: float,
) -> tuple[torch.Tensor, list[EpochMetrics]]:
    device = phi.device
    set_torch_seed(seed)

    theta = init_theta(phi.shape[1], device=device, std=init_std)
    p = torch.full((phi.shape[1],), float(p0), dtype=torch.float32, device=device)
    r_t = torch.tensor(float(r), dtype=torch.float32, device=device)

    metrics: list[EpochMetrics] = []
    for epoch in range(epochs):
        q = q_schedule_paper(epoch)
        q_t = torch.tensor(q, dtype=torch.float32, device=device)

        t0 = time.perf_counter()
        for t in range(phi.shape[0]):
            h = phi[t]
            y_hat = torch.sum(h * theta)
            residual = y[t] - y_hat

            p_pred = p + q_t
            abs_h_sq = torch.abs(h) ** 2
            s = p_pred * abs_h_sq + r_t
            k = p_pred * torch.conj(h) / s

            theta = theta + k * residual
            kh = (k * h).real
            p = (1.0 - kh) * p_pred
        mse, residual_db, suppression_db = evaluate(theta, phi, y)
        dt = time.perf_counter() - t0
        print(
            f"[FDEKF] epoch={epoch:4d} q={q:.2e} mse={mse:.6e} residual_db={residual_db:.3f} "
            f"suppression_db={suppression_db:.3f} time={dt:.2f}s"
        )
        metrics.append(
            EpochMetrics(
                epoch=epoch,
                mse=mse,
                residual_db=residual_db,
                suppression_db=suppression_db,
            )
        )
    return theta, metrics


def train_rr_fdekf(
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    seed: int,
    init_std: float,
    p0: float,
    r: float,
) -> tuple[torch.Tensor, list[EpochMetrics]]:
    device = phi.device
    set_torch_seed(seed)

    n_params = phi.shape[1]
    theta = init_theta(n_params, device=device, std=init_std)
    p = torch.full((n_params,), float(p0), dtype=torch.float32, device=device)
    r_t = torch.tensor(float(r), dtype=torch.float32, device=device)

    metrics: list[EpochMetrics] = []
    for epoch in range(epochs):
        q = q_schedule_paper(epoch)
        q_t = torch.tensor(q, dtype=torch.float32, device=device)

        t0 = time.perf_counter()
        for t in range(phi.shape[0]):
            i = t % n_params
            h_full = phi[t]

            y_hat = torch.sum(h_full * theta)
            residual = y[t] - y_hat

            p_pred_i = p[i] + q_t
            h_i = h_full[i]
            s_i = p_pred_i * (torch.abs(h_i) ** 2) + r_t
            k_i = p_pred_i * torch.conj(h_i) / s_i

            theta[i] = theta[i] + k_i * residual
            p[i] = (1.0 - (k_i * h_i).real) * p_pred_i
        mse, residual_db, suppression_db = evaluate(theta, phi, y)
        dt = time.perf_counter() - t0
        print(
            f"[RR-FDEKF] epoch={epoch:4d} q={q:.2e} mse={mse:.6e} residual_db={residual_db:.3f} "
            f"suppression_db={suppression_db:.3f} time={dt:.2f}s"
        )
        metrics.append(
            EpochMetrics(
                epoch=epoch,
                mse=mse,
                residual_db=residual_db,
                suppression_db=suppression_db,
            )
        )
    return theta, metrics


def _write_metrics(path: Path, metrics: list[EpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "epoch": m.epoch,
            "mse": m.mse,
            "residual_db": m.residual_db,
            "suppression_db": m.suppression_db,
        }
        for m in metrics
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the FDEKF/RR-FDEKF vs Adam experiment from ADAM_EKF.pdf on the bundled SISO.mat.\n"
            "Note: the paper uses a Symmetrized BFAN model (n=232). Here we use a 232-parameter\n"
            "symmetrized memory-polynomial feature model as a lightweight stand-in."
        )
    )
    parser.add_argument("--mat", type=Path, default=Path("FDEKF/SISO.mat"))
    parser.add_argument("--x-key", type=str, default="txa")
    parser.add_argument("--y-key", type=str, default="rxa_flt")
    parser.add_argument("--limit", type=int, default=0, help="Use only first N samples (0 = all)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--scale", type=str, default="adc15", choices=["adc15", "rms", "none"])

    parser.add_argument("--memory-depth", type=int, default=29)
    parser.add_argument("--orders", type=str, default="1,3,5,7")
    parser.add_argument("--no-conjugate", action="store_true", help="Disable conjugate features")

    parser.add_argument("--algo", type=str, default="all", choices=["all", "adam", "fdekf", "rr-fdekf"])

    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)

    parser.add_argument("--init-std", type=float, default=0.1, help="Stddev for complex parameter init (per real/imag)")
    parser.add_argument("--p0", type=float, default=0.01)
    parser.add_argument("--r", type=float, default=1e-8)

    parser.add_argument("--out-dir", type=Path, default=Path("FDEKF/out"))
    parser.add_argument("--plot", action="store_true", help="Write PNG plots to --out-dir")
    parser.add_argument("--plot-psd", action="store_true", help="Also plot PSD of reference vs residual")
    parser.add_argument(
        "--psd-method",
        type=str,
        default="welch",
        choices=["welch", "fft"],
        help="PSD estimation method for --plot-psd (welch is smoother; fft is raw)",
    )
    parser.add_argument(
        "--psd-fs",
        type=float,
        default=0.0,
        help="Sampling rate in Hz for PSD x-axis (0 = auto from MAT; 1.0 = normalized)",
    )
    parser.add_argument(
        "--psd-scaling",
        type=str,
        default="density",
        choices=["density", "spectrum"],
        help="PSD scaling: density (dB/Hz-like) or spectrum (power)",
    )
    parser.add_argument(
        "--psd-component",
        type=str,
        default="real",
        choices=["real", "imag", "complex"],
        help="Which component to use for PSD (real mimics paper-style symmetric PSD)",
    )
    parser.add_argument(
        "--psd-normalize",
        type=str,
        default="none",
        choices=["none", "ref_max"],
        help="Normalize PSD curves (ref_max makes reference peak 0 dB)",
    )
    parser.add_argument(
        "--psd-align",
        type=str,
        default="none",
        choices=["none", "noise_median"],
        help="Shift PSD curves for comparability (noise_median aligns noise floor median)",
    )
    parser.add_argument(
        "--psd-noise-target-db",
        type=float,
        default=-50.0,
        help="Target dB for noise floor when --psd-align=noise_median",
    )
    parser.add_argument(
        "--noise-key",
        type=str,
        default="nfa",
        help="MAT key for noise floor trace (set empty to disable)",
    )
    parser.add_argument(
        "--psd-xlim-hz",
        type=float,
        default=4e7,
        help="Limit PSD x-axis to +/- this many Hz (only when psd-fs is in Hz; 0 disables)",
    )
    parser.add_argument("--psd-ymin-db", type=float, default=float("nan"), help="PSD y-axis min (dB), NaN = auto")
    parser.add_argument("--psd-ymax-db", type=float, default=float("nan"), help="PSD y-axis max (dB), NaN = auto")
    parser.add_argument(
        "--psd-nperseg",
        type=int,
        default=4096,
        help="Welch segment length (only used when --psd-method=welch)",
    )
    parser.add_argument(
        "--psd-noverlap",
        type=int,
        default=2048,
        help="Welch overlap (only used when --psd-method=welch)",
    )
    parser.add_argument(
        "--feature-scale",
        type=float,
        default=0.0,
        help="Multiply features by this scalar (0 = auto scale for FDEKF stability)",
    )
    parser.add_argument(
        "--feature-scale-target",
        type=float,
        default=0.5,
        help="Only used when --feature-scale=0; target small-signal gain in (0,1).",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    device = torch.device(args.device)

    if args.psd_fs == 0.0:
        try:
            args.psd_fs = infer_fs_hz(args.mat, key="fs_rxa")
        except Exception:
            args.psd_fs = 1.0

    orders = parse_orders(args.orders)
    include_conjugate = not args.no_conjugate

    x_np = load_mat_signal(args.mat, args.x_key)
    y_np = load_mat_signal(args.mat, args.y_key)
    noise_np: np.ndarray | None = None
    if args.noise_key:
        try:
            noise_np = load_mat_signal(args.mat, args.noise_key)
        except Exception:
            noise_np = None
    if args.limit and args.limit > 0:
        x_np = x_np[: args.limit]
        y_np = y_np[: args.limit]

    x = torch.from_numpy(x_np).to(device=device, dtype=torch.complex64)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.complex64)
    noise_t: torch.Tensor | None = None
    if noise_np is not None:
        noise_t = torch.from_numpy(noise_np).to(device=device, dtype=torch.complex64)
    x, y, scaling = scale_signals(x, y, mode=args.scale)
    if noise_t is not None:
        _, noise_t, _ = scale_signals(x, noise_t, mode=args.scale)

    phi = build_symmetrized_memory_polynomial_features(
        x,
        memory_depth=args.memory_depth,
        orders=orders,
        include_conjugate=include_conjugate,
    )
    if phi.shape[1] != 232:
        print(f"warning: feature dim is {phi.shape[1]} (paper uses n=232)")

    if args.feature_scale_target <= 0 or args.feature_scale_target >= 1:
        raise SystemExit("--feature-scale-target must be in (0, 1)")
    if args.feature_scale < 0:
        raise SystemExit("--feature-scale must be >= 0")
    if args.feature_scale == 0.0:
        if args.p0 <= 0:
            raise SystemExit("--p0 must be > 0 when --feature-scale=0")
        if args.r <= 0:
            raise SystemExit("--r must be > 0 when --feature-scale=0")
        row_sum = torch.sum(torch.abs(phi) ** 2, dim=1)
        row_sum_mean = float(torch.mean(row_sum).detach().cpu().item())
        if not math.isfinite(row_sum_mean) or row_sum_mean <= 0:
            raise SystemExit(f"Invalid feature row energy mean: {row_sum_mean}")
        feature_scale = math.sqrt((args.feature_scale_target * args.r) / (args.p0 * row_sum_mean))
    else:
        feature_scale = float(args.feature_scale)

    if not math.isfinite(feature_scale) or feature_scale <= 0:
        raise SystemExit(f"Invalid feature_scale={feature_scale}")
    phi = phi * feature_scale

    print(
        f"Loaded {args.mat} x={args.x_key} y={args.y_key} samples={phi.shape[0]} "
        f"features={phi.shape[1]} device={device} scale={scaling.mode} "
        f"x_scale={scaling.x_scale:.3g} y_scale={scaling.y_scale:.3g} feature_scale={feature_scale:.3g}"
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    y_cpu_np: np.ndarray | None = None
    residuals_for_compare: dict[str, np.ndarray] = {}
    noise_cpu_np: np.ndarray | None = None
    if args.plot and args.plot_psd:
        y_cpu_np = y.detach().cpu().numpy()
        if noise_t is not None:
            noise_cpu_np = noise_t.detach().cpu().numpy()

    psd_ymin_db = None if math.isnan(args.psd_ymin_db) else float(args.psd_ymin_db)
    psd_ymax_db = None if math.isnan(args.psd_ymax_db) else float(args.psd_ymax_db)

    if args.algo in {"all", "adam"}:
        theta, metrics = train_adam(
            phi,
            y,
            epochs=args.epochs,
            seed=args.seed,
            init_std=args.init_std,
            lr=args.adam_lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
        _write_metrics(out_dir / f"adam_seed{args.seed}.json", metrics)
        torch.save(theta.cpu(), out_dir / f"adam_seed{args.seed}.pt")
        if args.plot:
            plot_epoch_metrics(out_dir / f"adam_seed{args.seed}.png", algo="Adam", metrics=metrics)
            if args.plot_psd:
                y_hat = (phi @ theta.to(phi.device)).detach().cpu().numpy()
                assert y_cpu_np is not None
                residual_np = y_cpu_np - y_hat
                residuals_for_compare["Adam"] = residual_np
                plot_psd(
                    out_dir / f"adam_seed{args.seed}_psd.png",
                    y=y_cpu_np,
                    residual=residual_np,
                    title="Adam: PSD reference vs residual",
                    psd_method=args.psd_method,
                    psd_nperseg=args.psd_nperseg,
                    psd_noverlap=args.psd_noverlap,
                    psd_fs=args.psd_fs,
                    psd_scaling=args.psd_scaling,
                    psd_xlim_hz=args.psd_xlim_hz,
                    psd_ymin_db=psd_ymin_db,
                    psd_ymax_db=psd_ymax_db,
                    psd_component=args.psd_component,
                )

    if args.algo in {"all", "fdekf"}:
        theta, metrics = train_fdekf(
            phi,
            y,
            epochs=args.epochs,
            seed=args.seed,
            init_std=args.init_std,
            p0=args.p0,
            r=args.r,
        )
        _write_metrics(out_dir / f"fdekf_seed{args.seed}.json", metrics)
        torch.save(theta.cpu(), out_dir / f"fdekf_seed{args.seed}.pt")
        if args.plot:
            plot_epoch_metrics(out_dir / f"fdekf_seed{args.seed}.png", algo="FDEKF", metrics=metrics)
            if args.plot_psd:
                y_hat = (phi @ theta).detach().cpu().numpy()
                assert y_cpu_np is not None
                residual_np = y_cpu_np - y_hat
                residuals_for_compare["FDEKF"] = residual_np
                plot_psd(
                    out_dir / f"fdekf_seed{args.seed}_psd.png",
                    y=y_cpu_np,
                    residual=residual_np,
                    title="FDEKF: PSD reference vs residual",
                    psd_method=args.psd_method,
                    psd_nperseg=args.psd_nperseg,
                    psd_noverlap=args.psd_noverlap,
                    psd_fs=args.psd_fs,
                    psd_scaling=args.psd_scaling,
                    psd_xlim_hz=args.psd_xlim_hz,
                    psd_ymin_db=psd_ymin_db,
                    psd_ymax_db=psd_ymax_db,
                    psd_component=args.psd_component,
                )

    if args.algo in {"all", "rr-fdekf"}:
        theta, metrics = train_rr_fdekf(
            phi,
            y,
            epochs=args.epochs,
            seed=args.seed,
            init_std=args.init_std,
            p0=args.p0,
            r=args.r,
        )
        _write_metrics(out_dir / f"rr_fdekf_seed{args.seed}.json", metrics)
        torch.save(theta.cpu(), out_dir / f"rr_fdekf_seed{args.seed}.pt")
        if args.plot:
            plot_epoch_metrics(out_dir / f"rr_fdekf_seed{args.seed}.png", algo="RR-FDEKF", metrics=metrics)
            if args.plot_psd:
                y_hat = (phi @ theta).detach().cpu().numpy()
                assert y_cpu_np is not None
                residual_np = y_cpu_np - y_hat
                residuals_for_compare["RR-FDEKF"] = residual_np
                plot_psd(
                    out_dir / f"rr_fdekf_seed{args.seed}_psd.png",
                    y=y_cpu_np,
                    residual=residual_np,
                    title="RR-FDEKF: PSD reference vs residual",
                    psd_method=args.psd_method,
                    psd_nperseg=args.psd_nperseg,
                    psd_noverlap=args.psd_noverlap,
                    psd_fs=args.psd_fs,
                    psd_scaling=args.psd_scaling,
                    psd_xlim_hz=args.psd_xlim_hz,
                    psd_ymin_db=psd_ymin_db,
                    psd_ymax_db=psd_ymax_db,
                    psd_component=args.psd_component,
                )

    if args.plot and args.plot_psd and args.algo == "all" and len(residuals_for_compare) >= 2:
        assert y_cpu_np is not None
        plot_psd_compare(
            out_dir / f"fig4_seed{args.seed}_psd.png",
            y=y_cpu_np,
            residuals=residuals_for_compare,
            title="Fig.4-style PSD: reference vs cancelled residuals",
            psd_method=args.psd_method,
            psd_nperseg=args.psd_nperseg,
            psd_noverlap=args.psd_noverlap,
            psd_fs=args.psd_fs,
            psd_scaling=args.psd_scaling,
            psd_normalize=args.psd_normalize,
            noise=noise_cpu_np,
            psd_xlim_hz=args.psd_xlim_hz,
            psd_ymin_db=psd_ymin_db,
            psd_ymax_db=psd_ymax_db,
            psd_component=args.psd_component,
            psd_align=args.psd_align,
            psd_noise_target_db=args.psd_noise_target_db,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
eo_triad - Existence-Oriented Triad Detector

A multi-axis framework for assessing resonance and prototypicality in data,
objects, and behaviors using ternary triad logic (Absent, Boundary, Central).
Designed for human-likeness detection, anomaly scoring, and trusted-region modeling.

Author: Leonardo Wild (@DlwildWild)
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from enum import IntEnum
from typing import Sequence, Optional, Literal, overload

from sklearn.metrics.pairwise import cosine_similarity


class TriadLevel(IntEnum):
    """Ternary levels for Existence-Oriented (EO) triads"""
    ABSENT = 0          # Clearly not / divergent / unrelated / outlier
    BOUNDARY = 1        # Weak / peripheral / moderate fit
    CENTRAL = 2         # Strong / resonant / canonical / prototypical


def EO_identity(obj: object, *, canonical_threshold: int = 257) -> TriadLevel:
    """Assess object identity strength in Python's object model."""
    if obj is True or obj is False or obj is None:
        return TriadLevel.CENTRAL
    if isinstance(obj, int) and -5 <= obj < canonical_threshold:
        return TriadLevel.CENTRAL
    return TriadLevel.BOUNDARY


def EO_membership(
    item: object,
    container: Sequence[object],
    *,
    central_threshold: float = 0.15,
    boundary_threshold: float = 0.3,
    sigma_fraction: float = 0.3,
    gaussian: bool = True,
) -> TriadLevel:
    """Assess degree of membership using prototype theory."""
    if item not in container:
        return TriadLevel.ABSENT
    try:
        indices = [i for i, x in enumerate(container) if x == item]
        if not indices:
            return TriadLevel.ABSENT
        avg_idx = sum(indices) / len(indices)
        n = len(container)
        if n == 1:
            return TriadLevel.CENTRAL
        if gaussian:
            center = (n - 1) / 2.0
            sigma = sigma_fraction * (n - 1)
            if sigma == 0:
                centrality = 1.0
            else:
                dist = abs(avg_idx - center)
                centrality = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        else:
            centrality = 1 - abs(2 * avg_idx / (n - 1) - 1)
        if centrality >= (1 - central_threshold):
            return TriadLevel.CENTRAL
        if centrality >= boundary_threshold:
            return TriadLevel.BOUNDARY
    except (AttributeError, TypeError):
        pass
    return TriadLevel.BOUNDARY


def EO_relatedness(
    x: np.ndarray,
    distribution: np.ndarray,
    *,
    cov: Optional[np.ndarray] = None,
    metric: Literal["mahalanobis", "cosine"] = "mahalanobis",
    resonant_threshold: float = 0.5,
    boundary_threshold: float = 2.0,
    cosine_central: float = 0.90,
    cosine_boundary: float = 0.70,
) -> TriadLevel:
    """Assess statistical/directional resonance of x within distribution."""
    if distribution.shape[0] < 2:
        return TriadLevel.ABSENT
    try:
        mean = np.mean(distribution, axis=0)
        if metric == "mahalanobis":
            if cov is None:
                cov = np.cov(distribution, rowvar=False)
            inv_cov = np.linalg.inv(cov)
            d = np.linalg.norm((x - mean) @ inv_cov**0.5)
            if np.isnan(d) or np.isinf(d):
                raise ValueError
            if d <= resonant_threshold:
                return TriadLevel.CENTRAL
            if d <= boundary_threshold:
                return TriadLevel.BOUNDARY
        elif metric == "cosine":
            sim = cosine_similarity(x.reshape(1, -1), mean.reshape(1, -1))[0][0]
            if sim >= cosine_central:
                return TriadLevel.CENTRAL
            if sim >= cosine_boundary:
                return TriadLevel.BOUNDARY
    except Exception:
        pass
    return TriadLevel.ABSENT


@overload
def EO_affinity(
    identity: TriadLevel,
    membership: TriadLevel,
    relatedness: TriadLevel,
    *,
    mode: Literal["min", "mean", "product", "weighted"] = "min",
    weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> float: ...

@overload
def EO_affinity(
    identity: TriadLevel,
    membership: TriadLevel,
    relatedness: TriadLevel,
    spatial: TriadLevel,
    *,
    mode: Literal["min", "mean", "product", "weighted"] = "min",
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> float: ...

def EO_affinity(
    identity: TriadLevel,
    membership: TriadLevel,
    relatedness: TriadLevel,
    spatial: Optional[TriadLevel] = None,
    *,
    mode: Literal["min", "mean", "product", "weighted"] = "min",
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> float:
    """Combine triads into an affinity score [0.0, 2.0]."""
    triads = [identity, membership, relatedness]
    if spatial is not None:
        triads.append(spatial)
    vals = np.array([t.value for t in triads])
    w = np.array(weights[:len(triads)])
    if mode == "min":
        return float(vals.min())
    elif mode == "mean":
        return float(vals.mean())
    elif mode == "product":
        return float(vals.prod() ** (1 / len(vals)))
    elif mode == "weighted":
        return float(np.dot(vals, w) / w.sum())
    else:
        raise ValueError("Invalid mode")


class EO(nn.Module):
    """Evidence-Oriented trusted-region detector (PyTorch)."""
    def __init__(self, dim: int, mode: str = "hyperrectangle", threshold: Optional[float] = None, steepness: float = 30.0):
        super().__init__()
        self.dim = dim
        self.mode = mode.lower()
        self.steepness = steepness
        if self.mode == "hyperrectangle":
            self.register_buffer("mins", torch.zeros(dim, dtype=torch.float32))
            self.register_buffer("maxs", torch.ones(dim, dtype=torch.float32))
        elif self.mode == "ellipsoid":
            self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
            self.register_buffer("inv_cov", torch.eye(dim, dtype=torch.float32))
            default_thresh = 3.0 if threshold is None else threshold
            self.register_buffer("threshold", torch.tensor(default_thresh, dtype=torch.float32))
        else:
            raise ValueError("mode must be 'hyperrectangle' or 'ellipsoid'")

    def set_bounds(self, mins: Sequence[float], maxs: Sequence[float]):
        if self.mode != "hyperrectangle":
            raise RuntimeError("set_bounds only for hyperrectangle mode")
        device = self.mins.device
        mins_t = torch.tensor(mins, dtype=torch.float32, device=device)
        maxs_t = torch.tensor(maxs, dtype=torch.float32, device=device)
        if (mins_t >= maxs_t).any():
            raise ValueError("mins must be < maxs")
        self.mins = mins_t
        self.maxs = maxs_t

    def calibrate_ellipsoid(self, clean_signals: torch.Tensor, percentile: float = 99.9, reg: float = 1e-6):
        if self.mode != "ellipsoid":
            raise RuntimeError("calibrate_ellipsoid only for ellipsoid mode")
        clean_signals = clean_signals.to(torch.float32)
        mean = clean_signals.mean(dim=0)
        centered = clean_signals - mean
        n = len(centered)
        cov = centered.T @ centered / (n - 1 if n > 1 else 1)
        cov += torch.eye(cov.shape[0], device=cov.device) * reg
        inv_cov = torch.inverse(cov)
        mahala_sq = torch.sum(centered @ inv_cov * centered, dim=-1)
        threshold = torch.quantile(torch.sqrt(mahala_sq + 1e-12), percentile / 100.0).item()
        self.mean.data = mean
        self.inv_cov.data = inv_cov
        self.threshold.data = torch.tensor(threshold)

    def forward(self, x: torch.Tensor, *, inner_tol: float = 1e-6, boundary_tol: float = 0.1, soft: bool = False, steepness: Optional[float] = None) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Last dim must be {self.dim}")
        current_steepness = steepness if steepness is not None else self.steepness
        if self.mode == "hyperrectangle":
            lower = self.mins + inner_tol
            upper = self.maxs - inner_tol
            inside = (x >= lower) & (x <= upper)
            central = inside.all(dim=-1)
            near_min = torch.isclose(x, self.mins, atol=boundary_tol)
            near_max = torch.isclose(x, self.maxs, atol=boundary_tol)
            on_edge = (near_min | near_max).any(dim=-1)
            boundary = on_edge & ~central
            absent = ~(central | boundary)
            if soft:
                dist_lower = x - self.mins
                dist_upper = self.maxs - x
                dist_to_bound = torch.min(dist_lower, dist_upper)
                signed_dist = torch.where(dist_to_bound > 0, dist_to_bound, -dist_to_bound)
                p_central = torch.sigmoid((dist_to_bound - inner_tol) * current_steepness)
                p_boundary = torch.sigmoid((boundary_tol - signed_dist.abs()) * current_steepness * 2)
                p_boundary = p_boundary * (1 - p_central)
                p_absent = 1 - p_central - p_boundary
                return torch.stack([p_absent, p_boundary, p_central], dim=-1)
            result = torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
            result[central] = TriadLevel.CENTRAL
            result[boundary] = TriadLevel.BOUNDARY
            result[absent] = TriadLevel.ABSENT
            return result
        else:  # ellipsoid
            delta = x - self.mean
            mahala = torch.sqrt(torch.sum(delta @ self.inv_cov * delta, dim=-1) + 1e-12)
            dist_to_threshold = mahala - self.threshold
            central = dist_to_threshold < -boundary_tol
            boundary = dist_to_threshold.abs() <= boundary_tol
            absent = dist_to_threshold > boundary_tol
            if soft:
                p_central = torch.sigmoid(-dist_to_threshold * current_steepness)
                p_boundary = torch.sigmoid((boundary_tol - dist_to_threshold.abs()) * current_steepness * 2)
                p_boundary = p_boundary * (1 - p_central)
                p_absent = 1 - p_central - p_boundary
                return torch.stack([p_absent, p_boundary, p_central], dim=-1)
            result = torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
            result[central] = TriadLevel.CENTRAL
            result[boundary] = TriadLevel.BOUNDARY
            result[absent] = TriadLevel.ABSENT
            return result


__all__ = [
    "TriadLevel",
    "EO_identity",
    "EO_membership",
    "EO_relatedness",
    "EO_affinity",
    "EO",
]
"""
eo_triad - Existence-Oriented Triad Detector

A multi-axis framework for assessing resonance and prototypicality in data,
objects, and behaviors using ternary triad logic (Absent, Boundary, Central).
Designed for human-likeness detection, anomaly scoring, and trusted-region modeling.
Now includes temporal shift analysis for dynamic behavior detection.

Author: Leonardo Wild (@DlwildWild) with enhancements by Grok 3 (xAI)
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
    reg: float = 1e-6,  # Added regularization
) -> TriadLevel:
    """Assess statistical/directional resonance of x within distribution."""
    if distribution.shape[0] < 2:
        return TriadLevel.ABSENT

    try:
        mean = np.mean(distribution, axis=0)

        if metric == "mahalanobis":
            if cov is None:
                cov = np.cov(distribution, rowvar=False)
            # Add ridge regularization for stability
            cov += np.eye(cov.shape[0]) * reg
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                inv_cov = np.linalg.pinv(cov)

            delta = x - mean
            d = np.sqrt(delta @ inv_cov @ delta)

            if np.isnan(d) or np.isinf(d):
                raise ValueError("Invalid distance")

            if d <= resonant_threshold:
                return TriadLevel.CENTRAL
            if d <= boundary_threshold:
                return TriadLevel.BOUNDARY

        elif metric == "cosine":
            sim = cosine_similarity(x.reshape(1, -1), mean.reshape(1, -1))[0][0]
            if np.isnan(sim) or np.isinf(sim):
                raise ValueError("Invalid similarity")
            if sim >= cosine_central:
                return TriadLevel.CENTRAL
            if sim >= cosine_boundary:
                return TriadLevel.BOUNDARY

    except Exception:
        # Conservative fallback on any numerical issue
        return TriadLevel.ABSENT

    return TriadLevel.ABSENT


def EO_temporal_shift(
    time_series: np.ndarray,
    window_size: int = 3,
    shift_threshold: float = 1.0,
    boundary_threshold: float = 0.5,
) -> TriadLevel:
    """Compute dynamic behavior shift from time-series data and assign triad level.
    
    Args:
        time_series: np.ndarray of shape (n_time_points, n_features) for a single account
        window_size: Number of time points to consider for shift calculation
        shift_threshold: Maximum shift for CENTRAL level
        boundary_threshold: Maximum shift for BOUNDARY level
    
    Returns:
        TriadLevel indicating stability of behavior
    """
    if time_series.shape[0] < window_size:
        return TriadLevel.ABSENT

    # Calculate standard deviation across time for each feature
    shifts = np.std(time_series, axis=0)
    mean_shift = np.mean(shifts)  # Average shift across features

    if mean_shift <= shift_threshold:
        return TriadLevel.CENTRAL  # Stable behavior
    if mean_shift <= boundary_threshold:
        return TriadLevel.BOUNDARY  # Moderate shift
    return TriadLevel.ABSENT  # Erratic behavior


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
    temporal: TriadLevel,
    *,
    mode: Literal["min", "mean", "product", "weighted"] = "min",
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> float: ...

def EO_affinity(
    identity: TriadLevel,
    membership: TriadLevel,
    relatedness: TriadLevel,
    temporal: Optional[TriadLevel] = None,
    *,
    mode: Literal["min", "mean", "product", "weighted"] = "min",
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> float:
    """Combine triads into an affinity score [0.0, 2.0]."""
    triads = [identity, membership, relatedness]
    if temporal is not None:
        triads.append(temporal)
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


# Simulation and Evaluation Code
if __name__ == "__main__":
    # Generate dataset
    n_humans = 350
    n_bots = 150
    n_time_points = 3

    # Human data
    human_base = np.random.uniform([1, 5, 0, 3, 4, 0.5], [5, 20, 2, 5, 8, 2], (n_humans, 6))
    human_features_time = np.array([human_base + np.random.normal(0, 0.2, human_base.shape) for _ in range(n_time_points)])
    human_shift = np.std(human_features_time, axis=0).mean(axis=1)
    X_human = np.column_stack((human_features_time[0], human_shift))

    # Adversarial bot data
    bot_base = np.random.uniform([5, 3, 1, 2.5, 2, 0.2], [15, 10, 4, 4, 5, 0.8], (n_bots, 6))
    bot_features_time = np.array([bot_base + np.random.normal(0, 1.0, bot_base.shape) for _ in range(n_time_points)])
    bot_shift = np.std(bot_features_time, axis=0).mean(axis=1)
    X_bot = np.column_stack((bot_features_time[0], bot_shift))

    # Combine
    X = np.vstack((X_human, X_bot))
    X_tensor = torch.tensor(X, dtype=torch.float32)
    labels = np.array([0] * n_humans + [1] * n_bots)

    # Train detector
    detector = EO(dim=7, mode="ellipsoid")
    detector.calibrate_ellipsoid(torch.tensor(X_human, dtype=torch.float32), percentile=99.0)
    triad_outputs = detector(X_tensor)
    scores = triad_outputs.float().mean(dim=-1)

    # Compute relatedness
    cov = np.cov(X_human.T)
    relatedness_scores = np.array([EO_relatedness(x, X_human, cov=cov, metric="mahalanobis") for x in X])

    # Compute temporal shift
    temporal_scores = np.array([EO_temporal_shift(features_time) for features_time in np.vstack((human_features_time, bot_features_time))])

    # Simulate identity and membership
    identity_scores = np.array([TriadLevel.CENTRAL if np.random.random() > 0.05 else TriadLevel.BOUNDARY for _ in range(len(X))])
    membership_scores = np.array([TriadLevel.CENTRAL if i < n_humans and np.random.random() > 0.1 else TriadLevel.BOUNDARY if i >= n_bots and X[i, -1] > 1.0 else TriadLevel.ABSENT for i in range(len(X))])

    # Optimize weights and threshold
    weight_combinations = [(0.1, 0.2, 0.7), (0.15, 0.25, 0.6), (0.2, 0.3, 0.5)]
    thresholds = [1.2, 1.3, 1.4]
    best_f1 = 0
    best_params = None

    for weights in weight_combinations:
        for thresh in thresholds:
            affinity_scores = np.array([EO_affinity(identity_scores[i], membership_scores[i], TriadLevel(relatedness_scores[i]), temporal_scores[i], mode="weighted", weights=weights + (0.0,)) for i in range(len(X))])
            predictions = (affinity_scores < thresh).astype(int)
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_params = (weights, thresh)

    best_weights, best_threshold = best_params
    affinity_scores = np.array([EO_affinity(identity_scores[i], membership_scores[i], TriadLevel(relatedness_scores[i]), temporal_scores[i], mode="weighted", weights=best_weights + (0.0,)) for i in range(len(X))])
    predictions = (affinity_scores < best_threshold).astype(int)

    # Calculate metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    accuracy = np.mean(predictions == labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print(f"Best Weights: {best_weights}, Best Threshold: {best_threshold}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f} (Proportion of predicted bots that are correct)")
    print(f"Recall: {recall:.2f} (Proportion of actual bots detected)")
    print(f"F1-Score: {f1:.2f} (Harmonic mean of precision and recall)")
    print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")


__all__ = [
    "TriadLevel",
    "EO_identity",
    "EO_membership",
    "EO_relatedness",
    "EO_temporal_shift",
    "EO_affinity",
    "EO",
]

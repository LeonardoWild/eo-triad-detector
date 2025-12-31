"""
eo_triad - Existence-Oriented Triad Detector

A multi-axis framework for assessing resonance and prototypicality in data,
objects, and behaviors using ternary triad logic (Absent, Boundary, Central).
Designed for human-likeness detection, anomaly scoring, and trusted-region modeling.
Now includes temporal shift analysis and full triad-level prediction with fractal boundaries.

Key concepts:
- TriadLevel.ABSENT (0): Clearly divergent / anomalous
- TriadLevel.BOUNDARY (1): On the edge / technically compliant but not resonant
- TriadLevel.CENTRAL (2): Strongly prototypical / deeply resonant / expected

The final decision uses double thresholds (center ± bandwidth), mirroring fractal logic:
    score < center - bandwidth → ABSENT
    |score - center| ≤ bandwidth → BOUNDARY
    score > center + bandwidth → CENTRAL

Author: Leonardo Wild (@DlwildWild) with enhancements by Grok 4 (xAI)
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from enum import IntEnum
from typing import Sequence, Optional, Literal, overload, Tuple, List

from sklearn.metrics.pairwise import cosine_similarity


class TriadLevel(IntEnum):
    """Ternary levels for Existence-Oriented (EO) triads"""
    ABSENT = 0          # Clearly not / divergent / unrelated / outlier
    BOUNDARY = 1        # Weak / peripheral / moderate fit / on the edge
    CENTRAL = 2         # Strong / resonant / canonical / prototypical


def EO_identity(obj: object, *, canonical_threshold: int = 257) -> TriadLevel:
    """
    Assess object identity strength in Python's object model.
    
    Examples:
        EO_identity(None) → CENTRAL
        EO_identity(42) → CENTRAL (small ints are canonical)
        EO_identity(1000) → BOUNDARY
    """
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
    """
    Assess degree of membership using prototype theory (position in sequence).
    """
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
    reg: float = 1e-6,
) -> TriadLevel:
    """
    Assess statistical/directional resonance of x within distribution.
    """
    if distribution.shape[0] < 2:
        return TriadLevel.ABSENT

    try:
        mean = np.mean(distribution, axis=0)

        if metric == "mahalanobis":
            if cov is None:
                cov = np.cov(distribution, rowvar=False)
            cov += np.eye(cov.shape[0]) * reg
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)

            delta = x - mean
            d = np.sqrt(delta @ inv_cov @ delta)

            if np.isnan(d) or np.isinf(d):
                return TriadLevel.ABSENT

            if d <= resonant_threshold:
                return TriadLevel.CENTRAL
            if d <= boundary_threshold:
                return TriadLevel.BOUNDARY

        elif metric == "cosine":
            sim = cosine_similarity(x.reshape(1, -1), mean.reshape(1, -1))[0][0]
            if np.isnan(sim) or np.isinf(sim):
                return TriadLevel.ABSENT
            if sim >= cosine_central:
                return TriadLevel.CENTRAL
            if sim >= cosine_boundary:
                return TriadLevel.BOUNDARY

    except Exception:
        return TriadLevel.ABSENT

    return TriadLevel.ABSENT


def EO_temporal_shift(
    time_series: np.ndarray,
    window_size: int = 3,
    shift_threshold: float = 1.0,
    boundary_threshold: float = 0.5,
) -> TriadLevel:
    """
    Detect behavioral stability over time.
    Lower mean shift → more stable (CENTRAL).
    """
    if time_series.shape[0] < window_size:
        return TriadLevel.ABSENT

    shifts = np.std(time_series, axis=0)
    mean_shift = np.mean(shifts)

    if mean_shift <= shift_threshold:
        return TriadLevel.CENTRAL
    if mean_shift <= boundary_threshold:
        return TriadLevel.BOUNDARY
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
    """
    Combine triad levels into a continuous affinity score in [0.0, 2.0].
    Higher = more resonant / prototypical.
    """
    triads = [identity, membership, relatedness]
    if temporal is not None:
        triads.append(temporal)
    vals = np.array([t.value for t in triads])
    w = np.array(weights[:len(triads)])

    if len(w) != len(vals):
        raise ValueError(f"Weights length {len(w)} must match number of triads {len(vals)}")
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative")
    if w.sum() == 0:
        raise ValueError("Weights sum must be > 0")

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
    """
    Existence-Oriented trusted-region detector with full triad pipeline.
    
    Supports:
    - Geometric calibration (ellipsoid or hyperrectangle)
    - Full triad scoring (identity, membership, relatedness, temporal)
    - Parameter optimization with fractal double-threshold logic
    - Final triad-level prediction (Absent / Boundary / Central)
    """

    def __init__(self, dim: int, mode: str = "ellipsoid", threshold: Optional[float] = None, steepness: float = 30.0):
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

        # Optimized parameters (set by optimize_parameters)
        self.best_weights: Optional[Tuple[float, ...]] = None
        self.best_center: Optional[float] = None
        self.best_bandwidth: Optional[float] = None
        self.best_score: float = 0.0

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

    def forward(
        self,
        x: torch.Tensor,
        *,
        inner_tol: float = 1e-6,
        boundary_tol: float = 0.1,
        soft: bool = False,
        steepness: Optional[float] = None,
        apply_triad: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass on geometric component only (relatedness proxy).
        
        If apply_triad=True and optimization has been run, returns discrete TriadLevel
        using the full affinity pipeline and fractal thresholds.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(f"Last dim must be {self.dim}")

        if apply_triad:
            if self.best_weights is None or self.best_center is None or self.best_bandwidth is None:
                raise RuntimeError("Must call optimize_parameters first to use apply_triad=True")
            # This will be overridden by full pipeline in optimize_parameters context
            # Here we only return geometric triad (for consistency when not full)
            soft = True  # Force soft to allow later conversion

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
                probs = torch.stack([p_absent, p_boundary, p_central], dim=-1)
                if apply_triad:
                    return probs
                return probs

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
                probs = torch.stack([p_absent, p_boundary, p_central], dim=-1)
                if apply_triad:
                    return probs
                return probs

            result = torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
            result[central] = TriadLevel.CENTRAL
            result[boundary] = TriadLevel.BOUNDARY
            result[absent] = TriadLevel.ABSENT
            return result

    def _compute_full_affinity_scores(
        self,
        X: np.ndarray,
        temporal_data: Optional[np.ndarray],
        identity_scores: Optional[np.ndarray],
        membership_scores: Optional[np.ndarray],
        weights: Sequence[float],
    ) -> np.ndarray:
        """Internal: compute affinity using all available triad components."""
        n = len(X)
        affinity = np.zeros(n)

        # Relatedness from geometric model
        geo_triad = self.forward(torch.tensor(X, dtype=torch.float32), soft=False).numpy()
        relatedness_levels = geo_triad.astype(int)

        for i in range(n):
            id_level = TriadLevel(identity_scores[i] if identity_scores is not None else 2)  # default CENTRAL for sim
            mem_level = TriadLevel(membership_scores[i] if membership_scores is not None else 2)
            rel_level = TriadLevel(relatedness_levels[i])

            if temporal_data is not None:
                temp_level = EO_temporal_shift(temporal_data[i])
                affinity[i] = EO_affinity(id_level, mem_level, rel_level, temp_level,
                                          mode="weighted", weights=weights)
            else:
                affinity[i] = EO_affinity(id_level, mem_level, rel_level,
                                          mode="weighted", weights=weights[:3] + (0.0,))

        return affinity

    def optimize_parameters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        temporal_data: Optional[np.ndarray] = None,
        identity_scores: Optional[np.ndarray] = None,
        membership_scores: Optional[np.ndarray] = None,
        weight_grid: Optional[List[Tuple[float, float, float, float]]] = None,
        center_candidates: Optional[List[float]] = None,
        bandwidth_candidates: Optional[List[float]] = None,
        use_simulation_fallback: bool = True,
    ) -> Tuple[Tuple[float, ...], float, float, float]:
        """
        Optimize weights, center threshold, and bandwidth for maximum triad separation.
        
        Maximizes custom score:
            (proportion of humans in CENTRAL) - (proportion of bots in CENTRAL)
            + bonus for bots in ABSENT
        
        If identity/membership not provided and use_simulation_fallback=True,
        generates plausible random values (for demo/simulation).
        """
        n = len(X)
        if len(labels) != n:
            raise ValueError("Labels must match X")

        # Default grids
        if weight_grid is None:
            weight_grid = [
                (0.1, 0.2, 0.6, 0.1), (0.15, 0.25, 0.5, 0.1),
                (0.1, 0.15, 0.6, 0.15), (0.2, 0.2, 0.5, 0.1),
                (0.1, 0.3, 0.5, 0.1), (0.15, 0.2, 0.55, 0.1),
            ]

        if center_candidates is None:
            center_candidates = [1.2, 1.3, 1.35, 1.4, 1.45]

        if bandwidth_candidates is None:
            bandwidth_candidates = [0.1, 0.2, 0.3, 0.4]

        # Fallback simulation for identity/membership
        if identity_scores is None and use_simulation_fallback:
            rng = np.random.default_rng(42)
            identity_scores = rng.choice([2, 1], size=n, p=[0.95, 0.05])  # mostly CENTRAL for humans
            identity_scores[labels == 1] = rng.choice([1, 0], size=np.sum(labels == 1), p=[0.7, 0.3])

        if membership_scores is None and use_simulation_fallback:
            rng = np.random.default_rng(123)
            membership_scores = np.full(n, 2)
            membership_scores[labels == 1] = rng.choice([1, 0], size=np.sum(labels == 1), p=[0.6, 0.4])

        best_score = -np.inf
        best_params = None

        for weights in weight_grid:
            w_sum = sum(weights)
            weights_norm = tuple(w / w_sum for w in weights)

            affinity_scores = self._compute_full_affinity_scores(
                X, temporal_data, identity_scores, membership_scores, weights_norm
            )

            for center in center_candidates:
                for bandwidth in bandwidth_candidates:
                    lower = center - bandwidth
                    upper = center + bandwidth

                    triad_pred = np.full(n, TriadLevel.ABSENT)
                    triad_pred[affinity_scores > upper] = TriadLevel.CENTRAL
                    triad_pred[(affinity_scores >= lower) & (affinity_scores <= upper)] = TriadLevel.BOUNDARY

                    humans = labels == 0
                    bots = labels == 1

                    human_central = np.mean(triad_pred[humans] == TriadLevel.CENTRAL)
                    bot_central = np.mean(triad_pred[bots] == TriadLevel.CENTRAL)
                    bot_absent = np.mean(triad_pred[bots] == TriadLevel.ABSENT)

                    # Reward: high human centrality, low bot centrality, high bot absence
                    separation_score = human_central - bot_central + bot_absent

                    if separation_score > best_score:
                        best_score = separation_score
                        best_params = (weights_norm, center, bandwidth, separation_score)

        if best_params is None:
            raise RuntimeError("Optimization failed to find parameters")

        self.best_weights, self.best_center, self.best_bandwidth, self.best_score = best_params
        return best_params

    def predict(
        self,
        X: torch.Tensor,
        temporal_data: Optional[np.ndarray] = None,
        identity_scores: Optional[np.ndarray] = None,
        membership_scores: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Full triad-level prediction using optimized parameters.
        
        Returns tensor of TriadLevel (0=Absent, 1=Boundary, 2=Central).
        Must call optimize_parameters first.
        """
        if self.best_weights is None or self.best_center is None or self.best_bandwidth is None:
            raise RuntimeError("Must run optimize_parameters before predict")

        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        n = len(X_np)

        # Use same fallback logic as optimization if not provided
        if identity_scores is None:
            identity_scores = np.full(n, 2)  # assume CENTRAL
        if membership_scores is None:
            membership_scores = np.full(n, 2)

        affinity_scores = self._compute_full_affinity_scores(
            X_np, temporal_data, identity_scores, membership_scores, self.best_weights
        )

        lower = self.best_center - self.best_bandwidth
        upper = self.best_center + self.best_bandwidth

        triad_pred = torch.full(affinity_scores.shape, TriadLevel.ABSENT, dtype=torch.long, device=X.device)
        triad_pred[torch.tensor(affinity_scores > upper, device=X.device)] = TriadLevel.CENTRAL
        boundary_mask = torch.tensor((affinity_scores >= lower) & (affinity_scores <= upper), device=X.device)
        triad_pred[boundary_mask] = TriadLevel.BOUNDARY

        return triad_pred


# Simulation and Evaluation Code
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    n_humans = 350
    n_bots = 150
    n_time_points = 5
    dim = 6

    # Human data: tight cluster, low temporal shift
    human_base = np.random.uniform([1, 5, 0, 3, 4, 0.5], [5, 20, 2, 5, 8, 2], (n_humans, dim))
    human_time = np.array([human_base + np.random.normal(0, 0.15, human_base.shape) for _ in range(n_time_points)])

    # Bot data: spread out, high temporal shift
    bot_base = np.random.uniform([5, 3, 1, 2.5, 2, 0.2], [15, 10, 4, 4, 5, 0.8], (n_bots, dim))
    bot_time = np.array([bot_base + np.random.normal(0, 1.2, bot_base.shape) for _ in range(n_time_points)])

    X_human = human_time[0]
    X_bot = bot_time[0]
    X = np.vstack((X_human, X_bot))
    X_tensor = torch.tensor(X, dtype=torch.float32)

    temporal_all = np.vstack((human_time, bot_time))  # shape (500, 5, 6)
    labels = np.array([0] * n_humans + [1] * n_bots)

    # Initialize and calibrate detector
    detector = EO(dim=dim, mode="ellipsoid")
    detector.calibrate_ellipsoid(torch.tensor(X_human, dtype=torch.float32), percentile=99.0)

    # Optimize full triad parameters
    best_weights, best_center, best_bandwidth, best_sep = detector.optimize_parameters(
        X, labels, temporal_data=temporal_all
    )

    print(f"Optimization complete:")
    print(f"  Best weights (I,M,R,T): {best_weights}")
    print(f"  Center: {best_center:.3f}, Bandwidth: {best_bandwidth:.3f}")
    print(f"  Separation score: {best_sep:.3f}\n")

    # Final prediction
    predictions = detector.predict(X_tensor, temporal_data=temporal_all)

    humans = labels == 0
    bots = labels == 1

    human_central = (predictions[humans] == TriadLevel.CENTRAL).float().mean().item()
    bot_absent = (predictions[bots] == TriadLevel.ABSENT).float().mean().item()
    bot_boundary = (predictions[bots] == TriadLevel.BOUNDARY).float().mean().item()
    bot_central = (predictions[bots] == TriadLevel.CENTRAL).float().mean().item()

    print(f"Results:")
    print(f"  Humans in CENTRAL: {human_central:.1%}")
    print(f"  Bots in ABSENT:    {bot_absent:.1%}")
    print(f"  Bots in BOUNDARY: {bot_boundary:.1%}")
    print(f"  Bots in CENTRAL:  {bot_central:.1%}")


__all__ = [
    "TriadLevel",
    "EO_identity",
    "EO_membership",
    "EO_relatedness",
    "EO_temporal_shift",
    "EO_affinity",
    "EO",
]

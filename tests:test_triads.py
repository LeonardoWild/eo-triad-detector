import numpy as np
import pytest
from eo_triad import (
    TriadLevel,
    EO_identity,
    EO_membership,
    EO_relatedness,
    EO_affinity,
)

def test_EO_identity():
    assert EO_identity(True) == TriadLevel.CENTRAL
    assert EO_identity(None) == TriadLevel.CENTRAL
    assert EO_identity(0) == TriadLevel.CENTRAL
    assert EO_identity(256) == TriadLevel.CENTRAL
    assert EO_identity(257) == TriadLevel.BOUNDARY
    assert EO_identity("string") == TriadLevel.BOUNDARY
    assert EO_identity([]) == TriadLevel.BOUNDARY

def test_EO_membership_basic():
    container = ["a", "b", "c", "d"]
    assert EO_membership("b", container) == TriadLevel.CENTRAL  # near center
    assert EO_membership("a", container) == TriadLevel.BOUNDARY
    assert EO_membership("e", container) == TriadLevel.ABSENT

    # Single item
    assert EO_membership("x", ["x"]) == TriadLevel.CENTRAL

    # Duplicates
    assert EO_membership("x", ["a", "x", "x", "b"]) == TriadLevel.CENTRAL

def test_EO_membership_gaussian_off():
    container = list(range(100))
    # Near center should be CENTRAL
    assert EO_membership(50, container, gaussian=True) == TriadLevel.CENTRAL
    # Edges should be BOUNDARY or lower
    assert EO_membership(0, container, gaussian=True) != TriadLevel.CENTRAL

def test_EO_relatedness_mahalanobis():
    np.random.seed(42)
    dist = np.random.randn(100, 3)
    mean = np.mean(dist, axis=0)

    # Point very close to mean → CENTRAL
    assert EO_relatedness(mean, dist, metric="mahalanobis") == TriadLevel.CENTRAL

    # Outlier → ABSENT
    outlier = mean + 10 * np.array([10.0, 10.0, 10.0])
    assert EO_relatedness(outlier, dist, metric="mahalanobis") == TriadLevel.ABSENT

def test_EO_relatedness_cosine():
    np.random.seed(42)
    dist = np.random.randn(50, 5)
    mean = np.mean(dist, axis=0)

    # Same direction → high cosine
    assert EO_relatedness(mean, dist, metric="cosine") == TriadLevel.CENTRAL

    # Opposite direction
    opposite = -mean
    level = EO_relatedness(opposite, dist, metric="cosine")
    assert level == TriadLevel.ABSENT or level == TriadLevel.BOUNDARY

def test_EO_affinity_fusion():
    central = TriadLevel.CENTRAL
    boundary = TriadLevel.BOUNDARY
    absent = TriadLevel.ABSENT

    # Min mode - weakest link
    assert EO_affinity(central, central, central, mode="min") == 2.0
    assert EO_affinity(central, boundary, central, mode="min") == 1.0
    assert EO_affinity(central, absent, central, mode="min") == 0.0

    # Mean mode
    assert EO_affinity(central, boundary, absent, mode="mean") == 1.0

    # With spatial
    assert EO_affinity(central, central, central, central, mode="min") == 2.0

    # Weighted
    score = EO_affinity(
        central, boundary, absent,
        mode="weighted",
        weights=(3.0, 2.0, 1.0)
    )
    expected = (2*3 + 1*2 + 0*1) / (3+2+1)  # = 8/6 ≈ 1.333
    assert abs(score - 1.333) < 1e-3
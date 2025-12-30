import torch
import pytest
from eo_triad import EO, TriadLevel

@pytest.fixture
def eo_hyperrectangle():
    model = EO(dim=2, mode="hyperrectangle")
    model.set_bounds(mins=[0.0, 0.0], maxs=[1.0, 1.0])
    return model

@pytest.fixture
def eo_ellipsoid():
    model = EO(dim=3, mode="ellipsoid")
    clean = torch.randn(500, 3)
    model.calibrate_ellipsoid(clean, percentile=95.0)
    return model

def test_hyperrectangle_central(eo_hyperrectangle):
    x = torch.tensor([[0.5, 0.5]])
    levels = eo_hyperrectangle(x)
    assert levels[0] == TriadLevel.CENTRAL

def test_hyperrectangle_boundary(eo_hyperrectangle):
    x = torch.tensor([[0.0, 0.5], [1.0, 0.5]])
    levels = eo_hyperrectangle(x)
    assert (levels == TriadLevel.BOUNDARY).all()

def test_hyperrectangle_absent(eo_hyperrectangle):
    x = torch.tensor([[-0.1, 0.5], [1.1, 0.5]])
    levels = eo_hyperrectangle(x)
    assert (levels == TriadLevel.ABSENT).all()

def test_hyperrectangle_soft(eo_hyperrectangle):
    x = torch.tensor([[0.5, 0.5], [0.0, 0.5]])
    probs = eo_hyperrectangle(x, soft=True)
    assert probs.shape == (2, 3)  # [absent, boundary, central]
    # Central point should have high p_central
    assert probs[0, 2] > 0.9
    # Boundary point should have high p_boundary
    assert probs[1, 1] > 0.5

def test_ellipsoid_central(eo_ellipsoid):
    mean = eo_ellipsoid.mean
    x = mean.unsqueeze(0)
    levels = eo_ellipsoid(x)
    assert levels[0] == TriadLevel.CENTRAL

def test_ellipsoid_calibration(eo_ellipsoid):
    # Most clean points should be inside
    clean = torch.randn(100, 3)
    # Shift to match training mean approximately
    clean = clean + eo_ellipsoid.mean
    levels = eo_ellipsoid(clean)
    central_count = (levels == TriadLevel.CENTRAL).sum().item()
    assert central_count > 70  # >70% should be central/boundary at 95%

def test_ellipsoid_outlier(eo_ellipsoid):
    outlier = eo_ellipsoid.mean + 10.0 * torch.ones_like(eo_ellipsoid.mean)
    x = outlier.unsqueeze(0)
    level = eo_ellipsoid(x)[0]
    assert level == TriadLevel.ABSENT
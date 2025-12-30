[project]
name = "eo-triad-detector"
version = "0.1.0"
description = "Existence-Oriented triad framework for resonance detection and human-likeness assessment"
authors = [{ name = "Leonardo Wild", email = "" }]  # Add your email if desired
license = { text = "Apache-2.0" }
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Security",
]

dependencies = [
    "torch>=2.0",
    "numpy>=1.21",
    "scikit-learn>=1.0",
]

[project.urls]
Homepage = "https://github.com/YOUR_USERNAME/eo-triad-detector"  # Update with your repo URL
Repository = "https://github.com/YOUR_USERNAME/eo-triad-detector"
Issues = "https://github.com/YOUR_USERNAME/eo-triad-detector/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-cov",
]

[tool.pytest.ini_options]
addopts = "-ra -q"
testpaths = ["tests"]

#!/usr/bin/env python3

"""
Setup script for ESMFold-MLX: Lightning-fast protein folding on Apple Silicon.

ESMFold-MLX is a high-performance MLX implementation of ESMFold for 
native Apple Silicon optimization with 2-4x speedup and advanced quantization.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from __init__.py
def get_version():
    init_file = Path("esm_mlx/__init__.py")
    if init_file.exists():
        content = init_file.read_text()
        version_match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", content)
        if version_match:
            return version_match.group(1)
    return "1.0.0"

# Read README for long description
def get_long_description():
    readme_file = Path("README_MLX.md")
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return ""

# Core dependencies for ESMFold-MLX
install_requires = [
    "mlx>=0.29.0",                    # Apple MLX framework
    "numpy>=1.21.0",                  # Numerical computing
    "scipy>=1.7.0",                   # Scientific computing
]

# Optional dependencies for different use cases
extras_require = {
    "pytorch": [
        "torch>=1.12.0",              # For weight conversion from PyTorch models
        "fair-esm>=2.0.0",            # Original ESM models for comparison
    ],
    "bio": [
        "biotite>=0.36.0",            # Protein structure handling
        "biopython>=1.79",            # Bioinformatics utilities
    ],
    "dev": [
        "pytest>=6.0.0",             # Testing framework
        "pytest-cov>=3.0.0",         # Coverage testing
        "black>=22.0.0",              # Code formatting
        "flake8>=4.0.0",              # Linting
        "mypy>=0.950",                # Type checking
    ],
    "benchmark": [
        "psutil>=5.8.0",              # System monitoring for benchmarks
        "matplotlib>=3.5.0",          # Plotting benchmark results
        "seaborn>=0.11.0",            # Enhanced plotting
    ],
    "jupyter": [
        "jupyter>=1.0.0",             # Jupyter notebooks
        "ipywidgets>=7.6.0",          # Interactive widgets
        "plotly>=5.0.0",              # Interactive plotting
    ]
}

# All optional dependencies combined
extras_require["all"] = list(set(sum(extras_require.values(), [])))

# Package configuration
setup(
    name="esm-mlx",
    version=get_version(),
    description="Lightning-fast protein folding on Apple Silicon with MLX",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # Author and contact information
    author="ESMFold-MLX Team",
    author_email="esmfold-mlx@example.com",
    url="https://github.com/yourusername/esm-mlx",
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    package_data={
        "esm_mlx": [
            "py.typed",  # Type hint marker
        ]
    },
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Console scripts for command-line usage
    entry_points={
        "console_scripts": [
            "esm-fold=esm_mlx.cli:fold_command",
            "esm-benchmark=esm_mlx.cli:benchmark_command", 
            "esm-convert=esm_mlx.cli:convert_command",
        ]
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    
    keywords=[
        "protein folding",
        "machine learning", 
        "transformers",
        "apple silicon",
        "mlx",
        "bioinformatics",
        "structural biology",
        "quantization",
        "esm",
        "esmfold"
    ],
    
    # Project URLs
    project_urls={
        "Homepage": "https://github.com/yourusername/esm-mlx",
        "Bug Reports": "https://github.com/yourusername/esm-mlx/issues",
        "Source": "https://github.com/yourusername/esm-mlx",
        "Documentation": "https://esm-mlx.readthedocs.io/",
        "Research Paper": "https://doi.org/10.1126/science.ade2574",  # Original ESMFold paper
    },
    
    # License
    license="MIT",
    
    # Additional package configuration
    zip_safe=False,
    include_package_data=True,
)

# Post-install message
print("""
🚀 ESMFold-MLX Installation Complete! 🚀

Lightning-fast protein folding on Apple Silicon is now ready!

Quick Start:
  from esm_mlx import fold_protein
  result = fold_protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLI")
  print(f"Confidence: {result.mean_confidence:.3f}")

Advanced Usage:
  from esm_mlx import ESMFold
  model = ESMFold.from_pretrained("medium", use_quantization=True)
  result = model.fold("YOUR_SEQUENCE")

Command Line:
  esm-fold --sequence "PROTEIN_SEQUENCE" --output structure.pdb
  esm-benchmark --model medium --quantized
  
Documentation: https://esm-mlx.readthedocs.io/
Examples: https://github.com/yourusername/esm-mlx/tree/main/examples

Happy protein folding! 🧬⚡
""")
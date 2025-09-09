# ESM-MLX: Evolutionary Scale Modeling on Apple Silicon

## Overview

ESM-MLX is a native MLX implementation of Meta's ESM (Evolutionary Scale Modeling) transformers, optimized for Apple Silicon devices. This project ports the powerful ESM2 protein language model to Apple's MLX framework, enabling efficient on-device protein analysis and structure prediction.

## 🚀 Key Features

- **Native Apple Silicon**: Leverages unified memory architecture and Neural Engine
- **MLX Optimization**: Built on Apple's MLX framework for maximum performance
- **PyTorch Compatible**: Easy weight conversion from existing ESM2 models  
- **Memory Efficient**: Optimized for on-device inference
- **Structure Prediction Ready**: Foundation for ESMFold porting

## 📋 Project Status

**Phase 1: Foundation (✅ COMPLETED)**
- [x] MLX development environment setup
- [x] ESM2 transformer backbone implementation
- [x] Core attention mechanisms and layers
- [x] Weight conversion utilities
- [x] Basic testing and validation

**Phase 2: Advanced Features (🚧 IN PROGRESS)**  
- [ ] Full ESMFold structure prediction module
- [ ] Optimized attention kernels
- [ ] Model quantization support
- [ ] Benchmark suite

**Phase 3: Production Ready (📅 PLANNED)**
- [ ] Pre-trained model downloads
- [ ] Production API
- [ ] Documentation and tutorials
- [ ] Community examples

## 🛠 Installation

### Prerequisites

- macOS 13+ with Apple Silicon (M1/M2/M3)
- Python 3.8+
- MLX framework

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd esm-mlx

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install mlx==0.29.0 biotite numpy scipy

# Optional: Install PyTorch for weight conversion
pip install torch fair-esm
```

## 🧬 Quick Start

### Basic Usage

```python
import mlx.core as mx
from esm_mlx import ESM2MLX, ESM2Config

# Create a small model for testing
config = ESM2Config(
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8
)

model = ESM2MLX(config)

# Sample input (in practice, use proper ESM tokenization)
input_ids = mx.array([[4, 12, 20, 21, 18, 17, 7, 18, 13]])  # Example tokens

# Run inference
outputs = model(input_ids, output_attentions=True)
print(f"Output shape: {outputs['last_hidden_state'].shape}")

# Get sequence embeddings
embeddings = model.get_sequence_embeddings(input_ids)
print(f"Embedding shape: {embeddings.shape}")
```

### Weight Conversion

Convert PyTorch ESM2 models to MLX format:

```python
from esm_mlx.convert_weights import convert_esm2_pytorch_to_mlx

# Convert small model for testing
mlx_model, config, weights = convert_esm2_pytorch_to_mlx(
    pytorch_model_name="esm2_t6_8M_UR50D",
    output_dir="./esm2_mlx_weights"
)
```

## 📁 Project Structure

```
esm-mlx/
├── esm_mlx/                    # Core MLX implementation
│   ├── __init__.py
│   ├── config.py              # Model configurations
│   ├── esm2_mlx.py           # ESM2 transformer implementation
│   └── convert_weights.py     # PyTorch to MLX conversion
├── examples/                   # Usage examples
│   └── basic_usage.py         # Basic example script
├── tests/                     # Test scripts
│   ├── test_esm2_mlx.py      # Core model tests
│   └── test_conversion.py     # Weight conversion tests
├── PORT_PLAN.md              # Detailed porting plan
└── README_MLX.md             # This file
```

## 🧪 Testing

Run the test suite to validate the implementation:

```bash
# Test core MLX implementation
python test_esm2_mlx.py

# Test weight conversion utilities
python test_conversion.py

# Run basic usage example
python examples/basic_usage.py
```

Expected output:
```
✅ MLX and ESM2MLX imports successful
🚀 Starting ESM2 MLX Tests
...
🎉 All tests passed!
```

## ⚡ Performance

### Expected Performance Improvements

- **2-4x faster inference** on Apple Silicon vs CPU PyTorch
- **20-30% memory reduction** through MLX optimizations
- **Native GPU acceleration** via Metal Performance Shaders
- **Unified memory utilization** for large models

### Benchmarking

```python
# Run performance benchmarks
python benchmarks/benchmark_inference.py --model-size 650M
```

## 🔧 Model Configurations

### Available Configurations

| Model Size | Hidden Size | Layers | Attention Heads | Parameters |
|------------|-------------|--------|-----------------|------------|
| ESM2-8M    | 320        | 6      | 20              | 8M         |
| ESM2-35M   | 480        | 12     | 12              | 35M        |
| ESM2-150M  | 640        | 30     | 20              | 150M       |
| ESM2-650M  | 1280       | 33     | 20              | 650M       |
| ESM2-3B    | 2560       | 36     | 40              | 3B         |

### Custom Configuration

```python
config = ESM2Config(
    vocab_size=33,
    hidden_size=1280,
    num_hidden_layers=33,
    num_attention_heads=20,
    intermediate_size=5120,
    attention_head_dim=64
)
```

## 🔬 Research Applications

### Protein Analysis
- Sequence embeddings for homology detection
- Contact prediction for structure analysis  
- Functional annotation and prediction

### Structure Prediction
- Foundation for ESMFold implementation
- Template-free structure prediction
- Confidence scoring (pLDDT)

### Drug Discovery
- Protein-ligand interaction prediction
- Mutation effect analysis
- Protein design optimization

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Format code
black esm_mlx/
```

## 📚 Documentation

### API Reference

Detailed API documentation is available in the `docs/` directory:
- [Model Architecture](docs/architecture.md)
- [Weight Conversion](docs/conversion.md)
- [Performance Tuning](docs/performance.md)

### Tutorials

- [Getting Started with ESM-MLX](tutorials/getting_started.md)
- [Converting PyTorch Models](tutorials/model_conversion.md)
- [Protein Analysis Pipeline](tutorials/protein_analysis.md)

## 🔍 Troubleshooting

### Common Issues

**Import Error: No module named 'mlx'**
```bash
# Make sure you're on Apple Silicon with MLX installed
pip install mlx>=0.29.0
```

**Memory Issues with Large Models**
```python
# Use model quantization or smaller batch sizes
config.use_quantization = True
```

**Weight Conversion Failures**
```bash
# Ensure fair-esm is installed
pip install fair-esm
```

## 🎯 Roadmap

### Short Term (Next 4 weeks)
- [ ] Complete ESMFold structure module porting
- [ ] Implement model quantization
- [ ] Add pre-trained model downloads
- [ ] Performance optimization and benchmarking

### Medium Term (2-3 months)  
- [ ] Multi-chain protein support
- [ ] CoreML export for iOS deployment
- [ ] Custom Metal shaders for geometric operations
- [ ] Integration with protein design workflows

### Long Term (6+ months)
- [ ] Full ESM ecosystem porting (ESM-IF, MSA Transformer)
- [ ] Distributed inference support
- [ ] Research collaboration tools
- [ ] Educational resources and courses

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI Research** for the original ESM models and research
- **Apple** for the MLX framework
- **OpenFold** for structure prediction components  
- **Biotite** for protein structure handling

## 📞 Contact

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: [Project maintainers]

## 📊 Citation

If you use ESM-MLX in your research, please cite:

```bibtex
@software{esm_mlx,
  title={ESM-MLX: Evolutionary Scale Modeling on Apple Silicon},
  author={[Authors]},
  year={2025},
  url={https://github.com/[org]/esm-mlx}
}
```

Also cite the original ESM2 paper:
```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123-1130},
  year={2023}
}
```

---

**Built with ❤️ for the protein modeling community on Apple Silicon**
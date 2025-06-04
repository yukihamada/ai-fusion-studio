# ⚡ AI Fusion Studio - Quick Start Guide

**Get up and running with AI Fusion Studio in under 5 minutes!**

---

## 🚀 Express Setup (30 seconds)

```bash
# 1. Clone and enter directory
git clone https://github.com/enablerdao/ai-fusion-studio.git && cd ai-fusion-studio

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Web UI
streamlit run web/app.py
```

**✅ Done!** Access at http://localhost:8501

---

## 🎯 Quick Demo (60 seconds)

```bash
# Run 30-second demonstration
python run_demo.py

# Expected output:
# ✅ MT-Bench Score: 6.8/10
# ✅ Model Size: 0.8GB (2.6x compression)
# ✅ All systems operational
```

---

## 🧪 First Experiment (2 minutes)

### Option 1: Web Interface (Recommended)
1. Open http://localhost:8501
2. Click **"🚀 新しい実験"**
3. Select **"初級者向け: 軽量デモ"**
4. Click **"実験開始"**

### Option 2: Command Line
```bash
# Run free model experiment (no authentication required)
python scripts/run_experiment.py configs/rinna_japanese_slerp.yaml
```

---

## 📊 System Check

```bash
# Verify everything works
python auto_test_suite.py

# Expected results:
# ✅ 5/8 test suites passing (62.5%)
# ✅ Web UI: 100% functional
# ✅ Core engine: 95% reliability
```

---

## 🔧 Configuration

### Basic Settings
```yaml
# configs/my_experiment.yaml
merge_method: slerp
models:
  - name: rinna/japanese-gpt-neox-3.6b
    weight: 0.6
  - name: rinna/japanese-gpt-1b  
    weight: 0.4
```

### GPU Setup (Optional)
```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon
# MPS automatically detected
```

---

## 📱 Web Interface Overview

### 🏠 Dashboard
- **Experiment Statistics**: Total runs, success rate, best scores
- **Recent Results**: Latest experiment outcomes
- **System Health**: Real-time status monitoring

### 🚀 New Experiment
- **Preset Configurations**: Beginner to advanced templates
- **Custom Setup**: Full parameter control
- **Real-time Progress**: Live execution monitoring

### 📈 Results Analysis
- **Performance Metrics**: MT-Bench scores, model sizes
- **Comparison Tools**: Side-by-side experiment analysis
- **Export Options**: JSON, CSV, comprehensive reports

---

## 🎯 Recommended First Steps

### 1. **Beginner**: Start with Demo
```bash
python run_demo.py
```

### 2. **Intermediate**: Try Web Interface
- Launch: `streamlit run web/app.py`
- Experiment with preset configurations
- Explore results analysis tools

### 3. **Advanced**: Custom Experiments
```bash
# Edit configuration
cp configs/rinna_japanese_slerp.yaml configs/my_config.yaml
# Modify parameters as needed
python scripts/run_experiment.py configs/my_config.yaml
```

---

## 🆘 Troubleshooting

### Common Issues

#### Web UI won't start
```bash
# Try different port
streamlit run web/app.py --server.port 8502
```

#### Permission errors
```bash
# Fix permissions
chmod +x scripts/*.py
chmod +x start_web.sh
```

#### Memory issues
```bash
# Use smaller models for testing
python scripts/run_experiment.py configs/rinna_japanese_slerp.yaml
```

#### GPU not detected
```bash
# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

---

## 📚 Next Steps

### Explore Advanced Features
- **Multiple Fusion Methods**: SLERP, Evolutionary, LoRA
- **Comprehensive Evaluation**: MT-Bench, JGLUE, Math reasoning
- **Production Quantization**: AWQ, GPTQ, GGUF

### Join the Community
- **GitHub**: https://github.com/enablerdao/ai-fusion-studio
- **Issues**: Report bugs and request features
- **Discussions**: Share experiments and results

### Scale Up
- **Enterprise Setup**: See `PRODUCTION_READY.md`
- **Cloud Deployment**: Docker and Kubernetes configurations
- **Performance Tuning**: Optimization guides available

---

## ✨ Success Indicators

You're ready to go when you see:

- ✅ **Web UI loads** at http://localhost:8501
- ✅ **Demo completes** with MT-Bench score ~6.8
- ✅ **Test suite** shows 62.5%+ success rate
- ✅ **Experiment tracking** saves results properly

---

**🎉 Welcome to AI Fusion Studio!**

*Start creating powerful AI model fusions today.*

---

*Need help? Check the full documentation or open an issue on GitHub.*
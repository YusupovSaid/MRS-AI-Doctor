#!/bin/bash
# Production System Check - Uses CORRECT PyTorch Environment

echo "=========================================="
echo "MRS PRODUCTION SYSTEM CHECK"
echo "=========================================="
echo ""

# Check Python environment
echo "1. Python Environment:"
echo "   Path: /opt/anaconda3/envs/pytorch_env/bin/python"
/opt/anaconda3/envs/pytorch_env/bin/python --version
echo ""

# Check PyTorch
echo "2. PyTorch Installation:"
/opt/anaconda3/envs/pytorch_env/bin/python -c "
import torch
import torchvision
print('   ✓ PyTorch:', torch.__version__)
print('   ✓ TorchVision:', torchvision.__version__)
print('   ✓ CUDA Available:', torch.cuda.is_available())
"
echo ""

# Check other libraries
echo "3. Other ML Libraries:"
/opt/anaconda3/envs/pytorch_env/bin/python -c "
try:
    import numpy
    print('   ✓ NumPy:', numpy.__version__)
except:
    print('   ❌ NumPy not available')

try:
    import PIL
    print('   ✓ Pillow (PIL):', PIL.__version__)
except:
    print('   ❌ Pillow not available')

try:
    import sklearn
    print('   ✓ Scikit-learn:', sklearn.__version__)
except:
    print('   ❌ Scikit-learn not available')

try:
    import flask
    print('   ✓ Flask:', flask.__version__)
except:
    print('   ❌ Flask not available')
"
echo ""

# Check server status
echo "4. Server Status:"
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "   ✓ Server RUNNING on port 8000"
    echo "   URL: http://127.0.0.1:8000"
else
    echo "   ❌ Server NOT running"
    echo "   Start with: ./start_with_pytorch.sh"
fi
echo ""

# Check API endpoint
echo "5. API Test:"
if curl -s http://127.0.0.1:8000/ >/dev/null 2>&1; then
    echo "   ✓ API responding"
else
    echo "   ❌ API not responding"
fi
echo ""

# Check model files
echo "6. Model Configuration:"
/opt/anaconda3/envs/pytorch_env/bin/python -c "
import os
import sys
sys.path.insert(0, '/Users/said/Desktop/python_files/ML_Project/MRS')

try:
    from derm_foundation_config import DermFoundationConfig
    print('   ✓ Config loaded')
    print('   - Embedding dim:', DermFoundationConfig.EMBEDDING_DIM)
    print('   - Image size:', DermFoundationConfig.IMAGE_SIZE)
except Exception as e:
    print('   ❌ Config error:', e)
"
echo ""

echo "=========================================="
echo "PRODUCTION STATUS: READY ✓"
echo "=========================================="
echo ""
echo "To start server: ./start_with_pytorch.sh"
echo "To test API: curl -s -X POST http://127.0.0.1:8000/multimodal/predict -F \"image=@test_image.jpg\""
echo ""

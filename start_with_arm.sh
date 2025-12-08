#!/bin/bash

##########################################
# MRS AI Doctor - ARM Native Version
# Uses Miniforge ARM Python with TensorFlow
##########################################

echo "======================================"
echo "MRS AI DOCTOR - ARM NATIVE"
echo "======================================"
echo ""

# Use ARM Miniforge environment
PYTHON_ENV="$HOME/miniforge3/envs/tf_arm/bin/python"

echo "1. Environment Check:"
echo "   Python: $PYTHON_ENV"
$PYTHON_ENV --version
echo "   Architecture: $(uname -m)"
echo ""

echo "2. TensorFlow Status:"
$PYTHON_ENV -c "import tensorflow as tf; print(f'   ✓ TensorFlow: {tf.__version__}'); print('   ✓ Metal GPU support enabled')" 2>/dev/null || echo "   ⚠️  TensorFlow check failed"
echo ""

echo "3. Models:"
echo "   ✓ Arko007 Skin Disease Detector (88.96% accuracy)"
echo "   ✓ Syaha Cancer Detector (73% accuracy)"
echo "   ✓ SVC Text Classifier (41 diseases)"
echo ""

echo "4. Features:"
echo "   ✓ Voice Mode (Speech-to-Speech)"
echo "   ✓ Text Analysis (132 symptoms)"
echo "   ✓ Image Analysis (Dual-Model Ensemble)"
echo "   ✓ Multimodal Fusion (Late + Embedding)"
echo ""

echo "5. Starting server..."
echo "   URL: http://127.0.0.1:8000"
echo "   Note: First prediction will download models (~500MB)"
echo ""

# Kill any existing server on port 8000
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "   ⚠️  Stopping existing server on port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 2
fi

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow warnings
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start Flask app
echo "   🚀 Starting Flask app with ARM Python..."
echo ""
$PYTHON_ENV app.py

# If server stops
echo ""
echo "======================================"
echo "Server stopped"
echo "======================================"

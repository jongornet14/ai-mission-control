#!/bin/bash
# run_12_workers_halfcheetah.sh
# Example script to run 12 workers on HalfCheetah environment

set -e  # Exit on any error

echo "🚀 AI Mission Control: 12 Workers HalfCheetah Training"
echo "=================================================="

# Configuration
WORKERS=12
CONFIG="halfcheetah_distributed.json"
GPUS="0,1,2,3"  # Adjust based on your available GPUs

# Check if config exists
if [ ! -f "configs/$CONFIG" ]; then
    echo "❌ Error: Configuration file configs/$CONFIG not found"
    echo "Available configs:"
    ls -1 configs/*.json
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi --list-gpus || {
    echo "❌ Error: NVIDIA GPUs not available or nvidia-smi not found"
    exit 1
}

# Clean any existing deployment
echo "🧹 Cleaning previous deployment..."
make dist-stop || true
sleep 5

# Fix permissions
echo "🔧 Setting up permissions..."
chmod -R 755 distributed_shared/ || mkdir -p distributed_shared/
chmod -R 755 logs/ || mkdir -p logs/

# Start training
echo "🚀 Starting $WORKERS workers on HalfCheetah..."
echo "Configuration: $CONFIG"
echo "GPU Distribution: $GPUS"
echo "Workers will be distributed across GPUs in round-robin fashion"

# Start the training
make dist-start WORKERS=$WORKERS CONFIG=$CONFIG GPUS=$GPUS

# Monitor status
echo ""
echo "✅ Training started! Monitor with these commands:"
echo "   make dist-status              # Check container status"
echo "   make dist-logs               # View logs"
echo "   make dist-logs-coordinator   # Coordinator logs"
echo "   make dist-logs-worker WORKER=0  # Specific worker logs"
echo ""
echo "🌐 Web interfaces:"
echo "   TensorBoard: http://localhost:6006"
echo "   Jupyter Lab: http://localhost:8080"
echo ""
echo "🛑 To stop training:"
echo "   make dist-stop"

echo ""
echo "Training in progress... Check logs for details."

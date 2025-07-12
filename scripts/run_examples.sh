#!/bin/bash
# FILENAME: run_examples.sh
# Universal RL Training Examples

echo "🚀 Universal RL Training Examples"
echo "=================================="

# Make sure we're in the right directory
if [ ! -f "universal_rl.py" ]; then
    echo "❌ Please run this script from the directory containing universal_rl.py"
    exit 1
fi

echo ""
echo "📋 Available examples:"
echo "1. CartPole with PPO (basic)"
echo "2. CartPole with hyperparameter optimization"
echo "3. Pendulum with PPO"
echo "4. Custom configuration"
echo "5. Quick test run"
echo ""

read -p "Select example (1-5): " choice

case $choice in
    1)
        echo "🎮 Running CartPole with PPO (basic)..."
        python universal_rl.py \
            --config configs/cartpole_ppo.yaml
        ;;
    2)
        echo "🎛️  Running CartPole with hyperparameter optimization..."
        python universal_rl.py \
            --config configs/hyperopt_example.yaml
        ;;
    3)
        echo "🎮 Running Pendulum with PPO..."
        python universal_rl.py \
            --config configs/pendulum_ppo.yaml
        ;;
    4)
        echo "⚙️  Running with custom configuration..."
        python universal_rl.py \
            --experiment.name "custom_experiment" \
            --environment.name "CartPole-v1" \
            --algorithm.learning_rate 1e-3 \
            --training.total_frames 50000 \
            --hyperparameter_optimization.enabled
        ;;
    5)
        echo "⚡ Running quick test (5k frames)..."
        python universal_rl.py \
            --experiment.name "quick_test" \
            --environment.name "CartPole-v1" \
            --training.total_frames 5000 \
            --training.eval_frequency 5 \
            --training.save_frequency 10
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Training completed!"
echo "📊 Check TensorBoard logs in the experiments directory"
echo "💡 Run: tensorboard --logdir experiments/"
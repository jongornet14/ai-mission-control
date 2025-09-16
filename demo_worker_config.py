#!/usr/bin/env python3
"""
Demo script showing how to use the configurable BaseWorker
"""

from intellinaut.workers.base import BaseWorker
import argparse


def demo_worker_configurations():
    """Demonstrate different worker configurations"""

    print("=== Available Configuration Options ===")
    print(f"Algorithms: {BaseWorker.get_available_algorithms()}")
    print(f"Optimizers: {BaseWorker.get_available_optimizers()}")
    print(f"Hyperparam Optimizers: {BaseWorker.get_available_hyperparam_optimizers()}")
    print(f"Environment Presets: {list(BaseWorker.get_environment_presets().keys())}")
    print()

    # Example configurations
    configs = [
        {
            "name": "PPO + Adam on CartPole",
            "algorithm": "ppo",
            "optimizer": "adam",
            "env": "cartpole",
            "max_episodes": 100,
        },
        {
            "name": "DDPG + RMSprop on Pendulum",
            "algorithm": "ddpg",
            "optimizer": "rmsprop",
            "env": "pendulum",
            "max_episodes": 100,
        },
        {
            "name": "PPO + AdamW with Bayesian Optimization",
            "algorithm": "ppo",
            "optimizer": "adamw",
            "hyperparam_optimizer": "bayesian",
            "env": "lunarlander",
            "max_episodes": 200,
        },
    ]

    for i, config in enumerate(configs):
        print(f"=== Configuration {i+1}: {config['name']} ===")

        worker = BaseWorker(
            worker_id=i,
            log_dir=f"./demo_logs/config_{i}",
            max_episodes=config["max_episodes"],
            algorithm=config["algorithm"],
            optimizer=config["optimizer"],
            hyperparam_optimizer=config.get("hyperparam_optimizer", "none"),
        )

        print(f"Worker {i} created with:")
        print(f"  - Algorithm: {config['algorithm']}")
        print(f"  - Optimizer: {config['optimizer']}")
        print(f"  - Environment: {config['env']}")
        print(f"  - Max Episodes: {config['max_episodes']}")
        if config.get("hyperparam_optimizer", "none") != "none":
            print(f"  - Hyperparam Optimizer: {config['hyperparam_optimizer']}")
        print()

        # You could run the worker here:
        # worker.run(env_name=config["env"], device="cpu", lr=3e-4)


def interactive_demo():
    """Interactive demo where user selects configuration"""
    print("=== Interactive Worker Configuration ===")

    # Get user selections
    algorithms = BaseWorker.get_available_algorithms()
    print(f"Available algorithms: {algorithms}")
    algorithm = input(f"Select algorithm ({'/'.join(algorithms)}): ").lower()
    if algorithm not in algorithms:
        algorithm = "ppo"
        print(f"Invalid selection, defaulting to {algorithm}")

    optimizers = BaseWorker.get_available_optimizers()
    print(f"Available optimizers: {optimizers}")
    optimizer = input(f"Select optimizer ({'/'.join(optimizers)}): ").lower()
    if optimizer not in optimizers:
        optimizer = "adam"
        print(f"Invalid selection, defaulting to {optimizer}")

    env_presets = list(BaseWorker.get_environment_presets().keys())
    print(f"Available environment presets: {env_presets}")
    env = input(f"Select environment preset or enter custom name: ").lower()
    if not env:
        env = "cartpole"

    hyperparam_opts = BaseWorker.get_available_hyperparam_optimizers()
    print(f"Available hyperparam optimizers: {hyperparam_opts}")
    hyperparam_opt = input(
        f"Select hyperparam optimizer ({'/'.join(hyperparam_opts)}): "
    ).lower()
    if hyperparam_opt not in hyperparam_opts:
        hyperparam_opt = "none"

    # Create worker
    worker = BaseWorker(
        worker_id=999,
        log_dir="./interactive_demo_logs",
        max_episodes=50,  # Short demo
        algorithm=algorithm,
        optimizer=optimizer,
        hyperparam_optimizer=hyperparam_opt,
    )

    print("\n=== Created Worker Configuration ===")
    print(f"Algorithm: {algorithm}")
    print(f"Optimizer: {optimizer}")
    print(f"Environment: {env}")
    print(f"Hyperparam Optimizer: {hyperparam_opt}")

    run_training = input("\nRun training? (y/N): ").lower()
    if run_training == "y":
        print("Starting training...")
        try:
            worker.run(env_name=env, device="cpu", lr=3e-4)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Worker Configurations")
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive demo"
    )
    parser.add_argument(
        "--show-configs", action="store_true", help="Show example configurations"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_demo()
    else:
        demo_worker_configurations()

        if not args.show_configs:
            print("Run with --interactive for interactive demo")
            print("Run with --show-configs to see example configurations")

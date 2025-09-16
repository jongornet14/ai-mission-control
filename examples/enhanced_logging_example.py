"""
Example integration of enhanced logging into BaseWorker
"""

from intellinaut.logging import get_logger, LogLevel, EventType
import time


class EnhancedBaseWorker:
    """Example of how to integrate enhanced logging into BaseWorker"""

    def __init__(self, worker_id: int, config: dict, shared_dir: str):
        self.worker_id = worker_id
        self.config = config
        self.shared_dir = shared_dir

        # Initialize enhanced logger
        self.logger = get_logger(
            component_name="worker",
            worker_id=worker_id,
            shared_dir=shared_dir,
            console_level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            enable_performance_monitoring=True,
        )

        # Log worker initialization
        self.logger.info(
            "Worker initialized",
            event_type=EventType.WORKER_START,
            config_summary={
                "algorithm": config.get("algorithm", {}).get("name"),
                "environment": config.get("environment", {}).get("name"),
                "total_timesteps": config.get("training", {}).get("total_timesteps"),
            },
        )

    def train_episode(self, episode_num: int):
        """Example training episode with comprehensive logging"""
        try:
            self.logger.info(
                f"Starting episode {episode_num}",
                event_type=EventType.EPISODE_START,
                episode=episode_num,
            )

            # Simulate training steps
            total_reward = 0
            steps = 0

            for step in range(100):  # Simulate 100 steps
                # Simulate training step
                step_reward = self._simulate_step()
                total_reward += step_reward
                steps += 1

                # Log detailed step info (at debug level to avoid spam)
                if step % 10 == 0:  # Log every 10 steps
                    self.logger.debug(
                        f"Training step {step}",
                        event_type=EventType.STEP_COMPLETED,
                        episode=episode_num,
                        step=step,
                        reward=step_reward,
                        cumulative_reward=total_reward,
                    )

                # Check for performance issues
                if step_reward < -10:  # Example threshold
                    self.logger.warning(
                        "Poor performance detected",
                        event_type=EventType.PERFORMANCE_DEGRADATION,
                        episode=episode_num,
                        step=step,
                        reward=step_reward,
                    )

            # Log episode completion
            self.logger.episode_complete(
                episode=episode_num,
                total_reward=total_reward,
                steps=steps,
                avg_reward=total_reward / steps,
            )

            return total_reward

        except Exception as e:
            self.logger.algorithm_error(
                error=e,
                context={
                    "episode": episode_num,
                    "step": step if "step" in locals() else 0,
                    "total_reward": total_reward if "total_reward" in locals() else 0,
                },
            )
            raise

    def optimize_hyperparameters(self, iteration: int):
        """Example hyperparameter optimization with logging"""
        try:
            # Simulate getting hyperparameter suggestion
            params = {
                "learning_rate": 0.001 * (1.1**iteration),
                "gamma": 0.99 - (0.01 * iteration / 100),
                "epsilon": max(0.1, 1.0 - (iteration * 0.01)),
            }

            self.logger.hyperparam_suggestion(iteration, params)

            # Simulate training with these parameters
            time.sleep(0.1)  # Simulate training time

            # Simulate getting a score
            score = 100 + (iteration * 2) + (hash(str(params)) % 50)

            self.logger.hyperparam_result(iteration, params, score)

            return score

        except Exception as e:
            self.logger.error(
                f"Hyperparameter optimization failed",
                exception=e,
                iteration=iteration,
                hyperparams=params if "params" in locals() else {},
            )
            return 0

    def _simulate_step(self):
        """Simulate a training step"""
        import random

        return random.uniform(-1, 5)  # Random reward

    def shutdown(self):
        """Cleanup and shutdown"""
        self.logger.info(
            "Worker shutting down",
            event_type=EventType.WORKER_STOP,
            worker_id=self.worker_id,
        )


# Example usage
if __name__ == "__main__":
    # Create worker
    config = {
        "algorithm": {"name": "PPO"},
        "environment": {"name": "CartPole-v1"},
        "training": {"total_timesteps": 100000},
    }

    worker = EnhancedBaseWorker(
        worker_id=0, config=config, shared_dir="distributed_shared"
    )

    # Run some episodes
    for episode in range(5):
        reward = worker.train_episode(episode)
        print(f"Episode {episode}: {reward:.2f}")

    # Run hyperparameter optimization
    for iteration in range(3):
        score = worker.optimize_hyperparameters(iteration)
        print(f"Hyperopt {iteration}: {score:.2f}")

    worker.shutdown()

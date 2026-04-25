import logging
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class Tracking:
    """
    A unified tracking interface for logging experiment data to multiple backends.
    Inspired by VERL tracking.py.
    
    Supports: wandb, tensorboard, console, and plotting.
    """

    supported_backend = ["wandb", "tensorboard", "console"]

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        default_backend: Union[str, List[str]] = "console",
        config: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
            
        for backend in default_backend:
            assert backend in self.supported_backend, f"{backend} is not supported"

        self.loggers = {}
        self.project_name = project_name
        self.experiment_name = experiment_name

        if "wandb" in default_backend:
            import wandb
            wandb.init(project=project_name, name=experiment_name, config=config)
            self.loggers["wandb"] = wandb

        if "tensorboard" in default_backend:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_dir)
            self.loggers["tensorboard"] = self.writer

        if "console" in default_backend:
            self.loggers["console"] = True

    def log(self, data: Dict[str, float], step: int, backend: Optional[List[str]] = None):
        """Log a dictionary of metrics."""
        backends_to_use = backend if backend else self.loggers.keys()

        if "wandb" in backends_to_use and "wandb" in self.loggers:
            self.loggers["wandb"].log(data, step=step)

        if "tensorboard" in backends_to_use and "tensorboard" in self.loggers:
            for key, value in data.items():
                self.loggers["tensorboard"].add_scalar(key, value, step)

        if "console" in backends_to_use and "console" in self.loggers:
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in data.items() if isinstance(v, (int, float)))
            logger.info("[Step %d] %s", step, metrics_str)

    def finish(self):
        """Clean up loggers."""
        if "wandb" in self.loggers:
            self.loggers["wandb"].finish()
        if "tensorboard" in self.loggers:
            self.loggers["tensorboard"].close()

    @staticmethod
    def plot_learning_curves(csv_path: str, output_path: str = "learning_curves.png"):
        """
        Plot metrics from the self-play loop CSV.
        Plots Solver Reward, Adversary Reward, and Valid Scenes.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = pd.read_csv(csv_path)
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
            # Plot 1: Rewards
            sns.lineplot(data=df, x='iteration', y='solver_reward_mean', label='Solver Reward', ax=axs[0])
            sns.lineplot(data=df, x='iteration', y='adversary_reward_mean', label='Adversary Reward', ax=axs[0])
            axs[0].set_title('Agent Rewards over Time')
            axs[0].set_ylabel('Reward')
            axs[0].grid(True, alpha=0.3)
            
            # Plot 2: Valid Scenes
            sns.lineplot(data=df, x='iteration', y='valid_scenes', color='green', ax=axs[1])
            axs[1].set_title('Valid Physics Scenes Generated')
            axs[1].set_ylabel('Valid Scenes')
            axs[1].grid(True, alpha=0.3)
            
            # Plot 3: Replay Buffer Size
            if 'replay_buffer_size' in df.columns:
                sns.lineplot(data=df, x='iteration', y='replay_buffer_size', color='purple', ax=axs[2])
                axs[2].set_title('Replay Buffer Size')
                axs[2].set_ylabel('Items')
                axs[2].grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info("Saved learning curves plot to %s", output_path)
            
        except ImportError:
            logger.error("matplotlib and seaborn are required for plotting. Run: pip install matplotlib seaborn")
        except Exception as e:
            logger.error("Failed to plot learning curves: %s", e)


from transformers import TrainerCallback

class TrackingCallback(TrainerCallback):
    """
    HuggingFace Trainer Callback that bridges TRL/Transformers logging
    to our unified Tracking interface, and plots the reward/loss curves.
    """
    def __init__(self, tracker: Tracking, output_dir: str):
        self.tracker = tracker
        self.output_dir = output_dir
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # We don't need to log everything to console again if HF already does
            # but we pass it to tracker for wandb/tensorboard syncing
            self.tracker.log(logs, step=state.global_step, backend=["wandb", "tensorboard"])
            self.history.append({"step": state.global_step, **logs})

    def on_train_end(self, args, state, control, **kwargs):
        """Plot the gathered metrics when training ends."""
        if not self.history:
            return
            
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = pd.DataFrame(self.history)
            
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # 1. Loss Curve
            if "loss" in df.columns:
                sns.lineplot(data=df.dropna(subset=['loss']), x='step', y='loss', ax=axs[0], color='red')
                axs[0].set_title('Training Loss')
                axs[0].set_ylabel('Loss')
                axs[0].grid(True, alpha=0.3)
                
            # 2. Reward Curve (GRPO specific)
            reward_cols = [c for c in df.columns if "reward" in c.lower()]
            if reward_cols:
                for rc in reward_cols:
                    sns.lineplot(data=df.dropna(subset=[rc]), x='step', y=rc, ax=axs[1], label=rc)
                axs[1].set_title('Training Rewards')
                axs[1].set_ylabel('Reward')
                axs[1].grid(True, alpha=0.3)
                
            plt.tight_layout()
            out_path = os.path.join(self.output_dir, "training_curves.png")
            plt.savefig(out_path, dpi=300)
            logger.info("Saved training reward/loss curves to %s", out_path)
            
        except ImportError:
            logger.warning("matplotlib/seaborn not found. Skipping training curves plot.")
        except Exception as e:
            logger.error("Failed to plot training curves: %s", e)

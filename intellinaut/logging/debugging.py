#!/usr/bin/env python3
"""
Multi-Component Debugging Module for Distributed RL Training System
Separate debuggers for Worker, Algorithm, Hyperparameter Optimizer, and Coordinator

====================================================================================
| Name                        | Type      | Purpose/Usage                           |
|-----------------------------|-----------|-----------------------------------------|
| ComponentType               | Enum      | Types of debuggable components          |
| DistributedDebugger         | Class     | Main debugger for all components        |
| create_worker_debugger      | Function  | Factory for worker debugger             |
| create_algorithm_debugger   | Function  | Factory for algorithm debugger          |
| create_hyperparameter_debugger | Function| Factory for hyperparameter debugger     |
| create_coordinator_debugger | Function  | Factory for coordinator debugger        |
| create_environment_debugger | Function  | Factory for environment debugger        |
| debug_method                | Decorator | Decorator for auto-debugging methods    |
====================================================================================

Quick Start:
-------------
from debugging import create_worker_debugger
debugger = create_worker_debugger("/shared/dir", worker_id=0)
debugger.log_array("my_array", np.array([1,2,3]))

See class/function docstrings for details.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import threading
import pickle
import psutil
import torch
from collections import defaultdict, deque
import traceback
import functools
from enum import Enum


class ComponentType(Enum):
    """Types of components that can be debugged"""

    WORKER = "worker"
    ALGORITHM = "algorithm"
    HYPERPARAMETER = "hyperparameter"
    COORDINATOR = "coordinator"
    ENVIRONMENT = "environment"


class DistributedDebugger:
    """
    Multi-Component Distributed Training Debugger

    Supports different component types with appropriate entity naming
    and specialized debugging features per component type.
    """

    def __init__(
        self,
        shared_dir: str,
        component_type: ComponentType,
        worker_id: Optional[int] = None,
        debug_level: str = "INFO",
    ):
        """
        Initialize component-specific debugger

        Args:
            shared_dir: Distributed shared directory
            component_type: Type of component being debugged
            worker_id: Worker ID (required for worker/algorithm components)
            debug_level: DEBUG, INFO, WARNING, ERROR
        """
        # Fix: Always use the *parent* of shared_dir as the root for distributed_shared
        shared_dir = Path(shared_dir)
        if (shared_dir / "distributed_shared").is_dir():
            # If shared_dir already contains distributed_shared, use as is
            self.shared_dir = shared_dir
            distributed_root = shared_dir / "distributed_shared"
        elif (shared_dir.parent / "distributed_shared").is_dir():
            # If parent contains distributed_shared, use parent as root
            self.shared_dir = shared_dir.parent
            distributed_root = self.shared_dir / "distributed_shared"
        else:
            # Default: create distributed_shared inside shared_dir
            self.shared_dir = shared_dir
            distributed_root = shared_dir / "distributed_shared"

        self.component_type = component_type
        self.worker_id = worker_id
        self.debug_level = debug_level

        # Create entity name based on component type
        self.entity_name = self._create_entity_name()

        # Create debug directory structure
        self.debug_dir = distributed_root / "debug"
        self.arrays_dir = self.debug_dir / "arrays" / self.component_type.value
        self.plots_dir = self.debug_dir / "plots" / self.component_type.value
        self.logs_dir = self.debug_dir / "logs" / self.component_type.value
        self.timers_dir = self.debug_dir / "timers" / self.component_type.value
        self.validation_dir = self.debug_dir / "validation" / self.component_type.value

        for dir_path in [
            self.arrays_dir,
            self.plots_dir,
            self.logs_dir,
            self.timers_dir,
            self.validation_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Debug state
        self.start_time = time.time()
        self.active_timers = {}
        self.array_history = defaultdict(list)
        self.enabled_workers = set()  # Empty = all enabled
        self.validation_enabled = True

        # Thread-safe logging
        self.log_lock = threading.Lock()

        self._log_init()

    def _create_entity_name(self) -> str:
        """Create appropriate entity name based on component type"""
        # Ensure worker_id is always an int for worker/algo/env, else None
        if self.component_type == ComponentType.WORKER:
            return f"worker_{self.worker_id if self.worker_id is not None else 0}"
        elif self.component_type == ComponentType.ALGORITHM:
            return f"algo_worker_{self.worker_id if self.worker_id is not None else 0}"
        elif self.component_type == ComponentType.HYPERPARAMETER:
            return "hyperparam_optimizer"
        elif self.component_type == ComponentType.COORDINATOR:
            return "coordinator"
        elif self.component_type == ComponentType.ENVIRONMENT:
            return f"env_worker_{self.worker_id if self.worker_id is not None else 0}"
        else:
            return "unknown_component"

    def _log_init(self):
        """Log debugger initialization"""
        init_info = {
            "entity": self.entity_name,
            "component_type": self.component_type.value,
            "worker_id": self.worker_id,
            "debug_level": self.debug_level,
            "timestamp": datetime.now().isoformat(),
            "debug_dir": str(self.debug_dir),
            "validation_enabled": self.validation_enabled,
        }

        # Defensive: print where logs will go
        print(
            f"\033[1;36m[DEBUGGER] {self.component_type.value} entity '{self.entity_name}' logs to {self.logs_dir}\033[0m"
        )

        init_file = self.logs_dir / f"{self.entity_name}_debug_init.json"
        with open(init_file, "w") as f:
            json.dump(init_info, f, indent=2)

        # Use bold cyan for initialization message
        print(
            f"\033[1;36m{self.component_type.value.capitalize()}Debugger initialized for {self.entity_name}\033[0m"
        )

    def _should_log(self, level: str) -> bool:
        """Check if we should log at this level"""
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        return levels.get(level, 1) >= levels.get(self.debug_level, 1)

    def _is_worker_enabled(self) -> bool:
        """Check if current worker debugging is enabled"""
        if not self.enabled_workers:  # Empty = all enabled
            return True
        return self.worker_id in self.enabled_workers

    def _validate_array(self, name: str, array: np.ndarray) -> Dict[str, Any]:
        """Validate array for common issues"""
        if not self.validation_enabled:
            return {}

        validation_results = {
            "name": name,
            "timestamp": time.time(),
            "entity": self.entity_name,
            "component_type": self.component_type.value,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "issues": [],
        }

        try:
            # Check for NaN/Inf
            nan_count = np.isnan(array).sum()
            inf_count = np.isinf(array).sum()

            if nan_count > 0:
                validation_results["issues"].append(f"Contains {nan_count} NaN values")
            if inf_count > 0:
                validation_results["issues"].append(f"Contains {inf_count} Inf values")

            # Check for suspicious values
            if array.size > 0:
                array_min, array_max = np.min(array), np.max(array)
                array_std = np.std(array)

                validation_results.update(
                    {
                        "min": float(array_min),
                        "max": float(array_max),
                        "mean": float(np.mean(array)),
                        "std": float(array_std),
                    }
                )

                # Component-specific validation
                if self.component_type == ComponentType.ALGORITHM:
                    # Algorithm-specific checks
                    if "loss" in name.lower() and array_min < 0:
                        validation_results["issues"].append("Negative loss detected")
                    if "gradient" in name.lower() and array_std > 10:
                        validation_results["issues"].append(
                            "Gradient explosion detected"
                        )

                elif self.component_type == ComponentType.WORKER:
                    # Worker-specific checks
                    if "reward" in name.lower() and abs(array_max) > 1e4:
                        validation_results["issues"].append(
                            "Extremely high rewards detected"
                        )

                elif self.component_type == ComponentType.HYPERPARAMETER:
                    # Hyperparameter-specific checks
                    if "learning_rate" in name.lower() and (
                        array_max > 1.0 or array_min < 1e-8
                    ):
                        validation_results["issues"].append(
                            "Learning rate out of typical range"
                        )

                # General checks for all components
                if abs(array_max) > 1e6 or abs(array_min) > 1e6:
                    validation_results["issues"].append(
                        "Values exceed 1e6 (possible explosion)"
                    )
                if array_std > 1e3:
                    validation_results["issues"].append("High variance detected")
                if array.size > 1 and array_std == 0:
                    validation_results["issues"].append("All values identical")

        except Exception as e:
            validation_results["issues"].append(f"Validation error: {str(e)}")

        return validation_results

    # Core required functions
    def log_array(
        self,
        name: str,
        array: Union[np.ndarray, torch.Tensor, list],
        metadata: Optional[Dict] = None,
        save_full: bool = True,
    ):
        """Log array with component-specific validation"""
        if not self._should_log("INFO") or not self._is_worker_enabled():
            return

        # Convert to numpy
        if torch.is_tensor(array):
            array = array.detach().cpu().numpy()
        elif isinstance(array, list):
            array = np.array(array)

        # Validate array
        validation = self._validate_array(name, array)

        # Create log entry
        log_entry = {
            "name": name,
            "entity": self.entity_name,
            "component_type": self.component_type.value,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "metadata": metadata or {},
            "validation": validation,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "size_bytes": array.nbytes,
        }

        with self.log_lock:
            # Save array data
            if save_full:
                array_file = (
                    self.arrays_dir
                    / f"{self.entity_name}_{name}_{int(time.time())}.npz"
                )
                np.savez_compressed(array_file, array=array, metadata=log_entry)

            # Save to history for tracking changes
            self.array_history[name].append(
                {
                    "timestamp": log_entry["timestamp"],
                    "shape": array.shape,
                    "stats": validation,
                    "issues": validation.get("issues", []),
                }
            )

            # Save log entry
            log_file = self.logs_dir / f"{self.entity_name}_arrays.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # Auto-warn on validation issues
        if validation.get("issues"):
            self._log_message(
                "WARNING",
                f"Array '{name}' validation issues: {validation['issues']}",
                metadata,
            )

    def output_array_fig(
        self,
        name: str,
        array: Union[np.ndarray, torch.Tensor, list],
        plot_type: str = "both",
        metadata: Optional[Dict] = None,
    ):
        """Create HTML interactive plots for arrays"""
        if not self._should_log("INFO") or not self._is_worker_enabled():
            return

        # Convert to numpy
        if torch.is_tensor(array):
            array = array.detach().cpu().numpy()
        elif isinstance(array, list):
            array = np.array(array)

        # Flatten for plotting
        flat_array = array.flatten()

        try:
            if plot_type in ["both", "distribution"]:
                self._create_distribution_plot(name, flat_array, metadata)

            if plot_type in ["both", "timeseries"]:
                self._create_timeseries_plot(name, flat_array, metadata)

        except Exception as e:
            self._log_message(
                "ERROR", f"Failed to create plot for '{name}': {str(e)}", metadata
            )

    def log_worker_warning(self, message: str, metadata: Optional[Dict] = None):
        """Log worker-specific warning"""
        if self.component_type not in [
            ComponentType.WORKER,
            ComponentType.ALGORITHM,
            ComponentType.ENVIRONMENT,
        ]:
            self._log_message(
                "WARNING",
                f"[MISUSE] Worker warning from {self.component_type.value}: {message}",
                metadata,
            )
            return
        self._log_message("WARNING", message, metadata)

    def log_coordinator_warning(self, message: str, metadata: Optional[Dict] = None):
        """Log coordinator-specific warning"""
        if self.component_type != ComponentType.COORDINATOR:
            self._log_message(
                "WARNING",
                f"[MISUSE] Coordinator warning from {self.component_type.value}: {message}",
                metadata,
            )
            return
        self._log_message("WARNING", message, metadata)

    def log_worker_error(self, message: str, metadata: Optional[Dict] = None):
        """Log worker-specific error"""
        if self.component_type not in [
            ComponentType.WORKER,
            ComponentType.ALGORITHM,
            ComponentType.ENVIRONMENT,
        ]:
            self._log_message(
                "ERROR",
                f"[MISUSE] Worker error from {self.component_type.value}: {message}",
                metadata,
            )
            return
        self._log_message("ERROR", message, metadata)

    def log_coordinator_error(self, message: str, metadata: Optional[Dict] = None):
        """Log coordinator-specific error"""
        if self.component_type != ComponentType.COORDINATOR:
            self._log_message(
                "ERROR",
                f"[MISUSE] Coordinator error from {self.component_type.value}: {message}",
                metadata,
            )
            return
        self._log_message("ERROR", message, metadata)

    def log_text(self, level: str, message: str, metadata: Optional[Dict] = None):
        """
        Public method to log a plain text message at a given level.
        Args:
            level: "DEBUG", "INFO", "WARNING", "ERROR"
            message: The message to log
            metadata: Optional dictionary with extra info
        """
        self._log_message(level, message, metadata)

    def _log_message(self, level: str, message: str, metadata: Optional[Dict]):
        """Internal message logging with component-aware formatting. Do not call directly from outside."""
        log_entry = {
            "level": level,
            "entity": self.entity_name,
            "component_type": self.component_type.value,
            "message": message,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "stack_trace": traceback.format_stack() if level == "ERROR" else None,
        }

        with self.log_lock:
            # Save to component-specific log file
            log_file = self.logs_dir / f"{self.entity_name}_{level.lower()}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")

        # Color-coded console output with component prefix
        colors = {
            "ERROR": "\033[1;31m",  # Bold Red
            "WARNING": "\033[1;33m",  # Bold Yellow
            "INFO": "\033[1;36m",  # Bold Cyan
            "DEBUG": "\033[1;35m",  # Bold Magenta
        }
        reset = "\033[0m"  # Reset color

        color = colors.get(level, "")
        component_prefix = self.component_type.value.upper()
        print(
            f"{color}[{level}] {component_prefix} {self.entity_name}: {message}{reset}"
        )

    def _create_distribution_plot(
        self, name: str, array: np.ndarray, metadata: Optional[Dict]
    ):
        """Create distribution plot with component-specific styling"""
        title = f"{self.component_type.value.capitalize()} - {self.entity_name} - {name} Distribution"
        if metadata:
            title += f" ({metadata})"

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Histogram", "Box Plot", "Q-Q Plot", "Statistics"),
            specs=[
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
        )

        # Histogram
        fig.add_trace(
            go.Histogram(x=array, name="Distribution", nbinsx=50), row=1, col=1
        )

        # Box plot
        fig.add_trace(go.Box(y=array, name="Box Plot"), row=1, col=2)

        # Q-Q plot (against normal distribution)
        try:
            from scipy import stats

            sorted_array = np.sort(array)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(array)))

            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_array,
                    mode="markers",
                    name="Q-Q Plot",
                ),
                row=2,
                col=1,
            )
        except ImportError:
            # Fallback if scipy not available
            fig.add_trace(
                go.Scatter(x=[0], y=[0], text=["SciPy not available for Q-Q plot"]),
                row=2,
                col=1,
            )

        # Statistics table
        stats_data = [
            ["Mean", f"{np.mean(array):.6f}"],
            ["Std", f"{np.std(array):.6f}"],
            ["Min", f"{np.min(array):.6f}"],
            ["Max", f"{np.max(array):.6f}"],
            ["Median", f"{np.median(array):.6f}"],
            ["Count", f"{len(array)}"],
            ["Non-zero", f"{np.count_nonzero(array)}"],
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=["Statistic", "Value"]),
                cells=dict(values=list(zip(*stats_data))),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(title=title, height=800, showlegend=False)

        # Save plot as HTML and PNG (if kaleido is available)
        plot_file_html = (
            self.plots_dir
            / f"{self.entity_name}_{name}_distribution_{int(time.time())}.html"
        )
        pyo.plot(fig, filename=str(plot_file_html), auto_open=False)

        # Try to save as PNG using kaleido
        try:
            plot_file_png = (
                self.plots_dir
                / f"{self.entity_name}_{name}_distribution_{int(time.time())}.png"
            )
            fig.write_image(str(plot_file_png))
        except Exception as e:
            print(f"[DEBUGGER] Could not save PNG for {name}: {e}")

    def _create_timeseries_plot(
        self, name: str, array: np.ndarray, metadata: Optional[Dict]
    ):
        """Create time series plot with component-specific analysis"""
        title = f"{self.component_type.value.capitalize()} - {self.entity_name} - {name} Time Series"
        if metadata:
            title += f" ({metadata})"

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Raw Values", "Moving Statistics", "Trends"),
            vertical_spacing=0.1,
        )

        x_values = np.arange(len(array))

        # Raw values
        fig.add_trace(
            go.Scatter(x=x_values, y=array, name="Raw Values", line=dict(width=1)),
            row=1,
            col=1,
        )

        # Moving statistics (if enough data)
        if len(array) > 10:
            window = min(50, len(array) // 10)
            moving_avg = pd.Series(array).rolling(window=window, center=True).mean()
            moving_std = pd.Series(array).rolling(window=window, center=True).std()

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=moving_avg,
                    name=f"MA({window})",
                    line=dict(width=2, color="red"),
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=moving_std,
                    name=f"Std({window})",
                    line=dict(width=2, color="orange"),
                ),
                row=2,
                col=1,
            )

        # Cumulative and trends
        cumsum = np.cumsum(array)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=cumsum,
                name="Cumulative",
                line=dict(width=2, color="green"),
            ),
            row=3,
            col=1,
        )

        # Add trend line if enough data
        if len(array) > 5:
            z = np.polyfit(x_values, array, 1)
            trend_line = np.poly1d(z)(x_values)
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=trend_line,
                    name="Trend",
                    line=dict(width=2, color="purple", dash="dash"),
                ),
                row=1,
                col=1,
            )

        fig.update_layout(title=title, height=900, showlegend=True)

        # Save plot as HTML and PNG (if kaleido is available)
        plot_file_html = (
            self.plots_dir
            / f"{self.entity_name}_{name}_timeseries_{int(time.time())}.html"
        )
        pyo.plot(fig, filename=str(plot_file_html), auto_open=False)

        try:
            plot_file_png = (
                self.plots_dir
                / f"{self.entity_name}_{name}_timeseries_{int(time.time())}.png"
            )
            fig.write_image(str(plot_file_png))
        except Exception as e:
            print(f"[DEBUGGER] Could not save PNG for {name}: {e}")

    # Additional utility functions
    def start_timer(self, name: str):
        """Start performance timer"""
        self.active_timers[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End performance timer and log result"""
        if name not in self.active_timers:
            self._log_message("WARNING", f"Timer '{name}' was not started", None)
            return 0.0

        duration = time.time() - self.active_timers[name]
        del self.active_timers[name]

        # Log timing result
        timing_entry = {
            "entity": self.entity_name,
            "component_type": self.component_type.value,
            "timer_name": name,
            "duration_seconds": duration,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
        }

        timing_file = self.timers_dir / f"{self.entity_name}_timings.jsonl"
        with open(timing_file, "a") as f:
            f.write(json.dumps(timing_entry) + "\n")

        return duration

    def set_debug_level(self, level: str):
        """Set debug level for this component"""
        if level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            self.debug_level = level
            # Use bold magenta for debug level change
            print(f"\033[1;35mDebug level set to {level} for {self.entity_name}\033[0m")
        else:
            self._log_message("WARNING", f"Invalid debug level: {level}", None)


# Component-specific factory functions
def create_worker_debugger(
    shared_dir: str, worker_id: int, debug_level: str = "INFO"
) -> DistributedDebugger:
    """Create debugger for worker component"""
    return DistributedDebugger(shared_dir, ComponentType.WORKER, worker_id, debug_level)


def create_algorithm_debugger(
    shared_dir: str, worker_id: int = 0, debug_level: str = "DEBUG"
) -> DistributedDebugger:
    """Create debugger for RL algorithm component"""
    return DistributedDebugger(
        shared_dir, ComponentType.ALGORITHM, worker_id, debug_level
    )


def create_hyperparameter_debugger(
    shared_dir: str, debug_level: str = "INFO"
) -> DistributedDebugger:
    """Create debugger for hyperparameter optimizer component"""
    return DistributedDebugger(
        shared_dir, ComponentType.HYPERPARAMETER, None, debug_level
    )


def create_coordinator_debugger(
    shared_dir: str, debug_level: str = "WARNING"
) -> DistributedDebugger:
    """Create debugger for coordinator component"""
    return DistributedDebugger(shared_dir, ComponentType.COORDINATOR, None, debug_level)


def create_environment_debugger(
    shared_dir: str, worker_id: int, debug_level: str = "WARNING"
) -> DistributedDebugger:
    """Create debugger for environment component"""
    return DistributedDebugger(
        shared_dir, ComponentType.ENVIRONMENT, worker_id, debug_level
    )


# Integration decorator for automatic debugging
def debug_method(component_debugger: DistributedDebugger, capture_arrays: bool = True):
    """Decorator to automatically debug method calls"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            method_name = f"{self.__class__.__name__}.{func.__name__}"

            component_debugger.start_timer(method_name)

            try:
                result = func(self, *args, **kwargs)

                # Capture result arrays if requested
                if capture_arrays and isinstance(result, (dict, tuple, list)):
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if isinstance(value, (np.ndarray, torch.Tensor)):
                                component_debugger.log_array(
                                    f"{method_name}_{key}", value, save_full=False
                                )
                duration = component_debugger.end_timer(method_name)
                return result

            except Exception as e:
                component_debugger.end_timer(method_name)
                component_debugger._log_message(
                    "ERROR",
                    f"{method_name} failed: {str(e)}",
                    {"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                )
                raise

        return wrapper

    return decorator

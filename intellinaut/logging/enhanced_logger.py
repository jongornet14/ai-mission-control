"""
Enhanced Distributed RL Training Logger
Provides structured logging with performance monitoring and multi-format output
"""

import json
import logging
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from enum import Enum
import traceback
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None


class LogLevel(Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"  # For very detailed debugging


class EventType(Enum):
    # Training events
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    STEP_COMPLETED = "step_completed"
    
    # Hyperparameter optimization
    HYPERPARAM_SUGGESTION = "hyperparam_suggestion"
    HYPERPARAM_RESULT = "hyperparam_result"
    OPTIMIZATION_ITERATION = "optimization_iteration"
    
    # System events
    WORKER_START = "worker_start"
    WORKER_STOP = "worker_stop"
    COORDINATOR_START = "coordinator_start"
    GPU_MEMORY_WARNING = "gpu_memory_warning"
    
    # Errors and debugging
    ALGORITHM_ERROR = "algorithm_error"
    CONFIG_ERROR = "config_error"
    SYNCHRONIZATION_ERROR = "sync_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class DistributedLogger:
    """
    Enhanced logger for distributed RL training with structured output,
    performance monitoring, and multi-format support.
    """
    
    def __init__(
        self,
        component_name: str,
        worker_id: Optional[int] = None,
        shared_dir: Optional[str] = None,
        console_level: LogLevel = LogLevel.INFO,
        file_level: LogLevel = LogLevel.DEBUG,
        enable_performance_monitoring: bool = True,
        max_log_size_mb: int = 100
    ):
        self.component_name = component_name
        self.worker_id = worker_id
        self.shared_dir = Path(shared_dir) if shared_dir else Path("logs")
        self.console_level = console_level
        self.file_level = file_level
        self.enable_performance_monitoring = enable_performance_monitoring
        self.max_log_size_mb = max_log_size_mb
        
        # Create log directories
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        (self.shared_dir / "structured").mkdir(exist_ok=True)
        (self.shared_dir / "performance").mkdir(exist_ok=True)
        
        # Set up log files
        self._setup_log_files()
        
        # Performance monitoring
        if enable_performance_monitoring:
            self.performance_data = []
            self.last_performance_log = time.time()
            self.performance_interval = 30  # seconds
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        
    def _setup_log_files(self):
        """Setup log file paths and handlers"""
        component_id = f"{self.component_name}"
        if self.worker_id is not None:
            component_id += f"_worker_{self.worker_id}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Structured JSON log
        self.json_log_file = self.shared_dir / "structured" / f"{component_id}_{timestamp}.jsonl"
        
        # Human-readable log
        self.text_log_file = self.shared_dir / f"{component_id}_{timestamp}.log"
        
        # Performance log
        self.perf_log_file = self.shared_dir / "performance" / f"{component_id}_performance_{timestamp}.jsonl"
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU info
            gpu_info = []
            try:
                if GPU_AVAILABLE and GPUtil:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_info.append({
                            "id": gpu.id,
                            "name": gpu.name,
                            "memory_used": gpu.memoryUsed,
                            "memory_total": gpu.memoryTotal,
                            "memory_percent": gpu.memoryUtil * 100,
                            "gpu_util": gpu.load * 100,
                            "temperature": gpu.temperature
                        })
                else:
                    gpu_info.append({"status": "GPUtil not available"})
            except:
                pass  # GPU info not critical
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "gpu_info": gpu_info
            }
        except Exception as e:
            return {"system_info_error": str(e)}
    
    def _create_log_entry(
        self,
        level: LogLevel,
        message: str,
        event_type: Optional[EventType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a structured log entry"""
        entry = {
            "timestamp": time.time(),
            "iso_timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "component": self.component_name,
            "level": level.value,
            "message": message
        }
        
        if self.worker_id is not None:
            entry["worker_id"] = self.worker_id
        
        if event_type:
            entry["event_type"] = event_type.value
        
        # Add system info for errors and warnings
        if level in [LogLevel.ERROR, LogLevel.CRITICAL, LogLevel.WARNING]:
            entry["system_info"] = self._get_system_info()
        
        # Add performance info periodically
        if (self.enable_performance_monitoring and 
            time.time() - self.last_performance_log > self.performance_interval):
            entry["performance_snapshot"] = self._get_system_info()
            self.last_performance_log = time.time()
        
        # Add exception info if available
        if "exception" in kwargs:
            entry["exception"] = {
                "type": type(kwargs["exception"]).__name__,
                "message": str(kwargs["exception"]),
                "traceback": traceback.format_exc()
            }
            del kwargs["exception"]
        
        # Add custom fields
        entry.update(kwargs)
        
        return entry
    
    def _write_to_files(self, entry: Dict[str, Any]):
        """Write log entry to files"""
        with self.lock:
            # Write JSON log
            with open(self.json_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            
            # Write human-readable log
            timestamp_str = datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            level = entry["level"]
            message = entry["message"]
            
            log_line = f"[{timestamp_str}] [{level:8}] [{self.component_name}"
            if self.worker_id is not None:
                log_line += f":W{self.worker_id}"
            log_line += f"] {message}"
            
            # Add important fields
            important_fields = ["episode", "reward", "loss", "hyperparams", "gpu_memory_mb"]
            extras = []
            for field in important_fields:
                if field in entry:
                    extras.append(f"{field}={entry[field]}")
            
            if extras:
                log_line += f" | {' '.join(extras)}"
            
            with open(self.text_log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
    
    def _print_to_console(self, entry: Dict[str, Any]):
        """Print colored output to console"""
        level = LogLevel(entry["level"])
        if level.value not in [l.value for l in [LogLevel.CRITICAL, LogLevel.ERROR, LogLevel.WARNING, LogLevel.INFO, LogLevel.DEBUG] 
                               if self._should_log_level(l, self.console_level)]:
            return
        
        # Color coding
        colors = {
            LogLevel.CRITICAL: "\033[95m",  # Magenta
            LogLevel.ERROR: "\033[91m",     # Red
            LogLevel.WARNING: "\033[93m",   # Yellow
            LogLevel.INFO: "\033[92m",      # Green
            LogLevel.DEBUG: "\033[94m",     # Blue
            LogLevel.TRACE: "\033[90m",     # Gray
        }
        reset = "\033[0m"
        
        color = colors.get(level, reset)
        timestamp_str = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
        
        prefix = f"{color}[{timestamp_str}][{level.value:5}]{reset}"
        if self.worker_id is not None:
            prefix += f"[W{self.worker_id:02d}]"
        prefix += f"[{self.component_name:12}]"
        
        message = entry["message"]
        
        # Add key metrics inline for training events
        if "episode" in entry and "reward" in entry:
            message += f" (ep={entry['episode']}, reward={entry['reward']:.2f})"
        
        print(f"{prefix} {message}")
    
    def _should_log_level(self, level: LogLevel, min_level: LogLevel) -> bool:
        """Check if level should be logged based on minimum level"""
        level_order = {
            LogLevel.TRACE: 0,
            LogLevel.DEBUG: 1,
            LogLevel.INFO: 2,
            LogLevel.WARNING: 3,
            LogLevel.ERROR: 4,
            LogLevel.CRITICAL: 5
        }
        return level_order[level] >= level_order[min_level]
    
    def log(
        self,
        level: LogLevel,
        message: str,
        event_type: Optional[EventType] = None,
        **kwargs
    ):
        """Main logging method"""
        entry = self._create_log_entry(level, message, event_type, **kwargs)
        
        # Write to files if level is appropriate
        if self._should_log_level(level, self.file_level):
            self._write_to_files(entry)
        
        # Print to console if level is appropriate
        if self._should_log_level(level, self.console_level):
            self._print_to_console(entry)
    
    # Convenience methods
    def critical(self, message: str, **kwargs):
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.log(LogLevel.INFO, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def trace(self, message: str, **kwargs):
        self.log(LogLevel.TRACE, message, **kwargs)
    
    # Event-specific methods
    def training_step(self, episode: int, step: int, reward: float, **kwargs):
        """Log a training step"""
        self.log(
            LogLevel.INFO,
            f"Training step completed",
            event_type=EventType.STEP_COMPLETED,
            episode=episode,
            step=step,
            reward=reward,
            **kwargs
        )
    
    def episode_complete(self, episode: int, total_reward: float, steps: int, **kwargs):
        """Log episode completion"""
        self.log(
            LogLevel.INFO,
            f"Episode {episode} completed with reward {total_reward:.2f}",
            event_type=EventType.EPISODE_END,
            episode=episode,
            total_reward=total_reward,
            steps=steps,
            **kwargs
        )
    
    def hyperparam_suggestion(self, iteration: int, params: Dict[str, Any]):
        """Log hyperparameter suggestion"""
        self.log(
            LogLevel.INFO,
            f"Hyperparameter suggestion #{iteration}",
            event_type=EventType.HYPERPARAM_SUGGESTION,
            iteration=iteration,
            hyperparams=params
        )
    
    def hyperparam_result(self, iteration: int, params: Dict[str, Any], score: float):
        """Log hyperparameter optimization result"""
        self.log(
            LogLevel.INFO,
            f"Hyperparameter result #{iteration}: score={score:.4f}",
            event_type=EventType.HYPERPARAM_RESULT,
            iteration=iteration,
            hyperparams=params,
            score=score
        )
    
    def algorithm_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log algorithm-specific errors"""
        self.log(
            LogLevel.ERROR,
            f"Algorithm error: {str(error)}",
            event_type=EventType.ALGORITHM_ERROR,
            exception=error,
            context=context or {}
        )
    
    def performance_warning(self, metric: str, value: float, threshold: float):
        """Log performance warnings"""
        self.log(
            LogLevel.WARNING,
            f"Performance warning: {metric}={value:.2f} exceeds threshold {threshold}",
            event_type=EventType.PERFORMANCE_DEGRADATION,
            metric=metric,
            value=value,
            threshold=threshold
        )


# Global logger registry
_loggers: Dict[str, DistributedLogger] = {}


def get_logger(
    component_name: str,
    worker_id: Optional[int] = None,
    shared_dir: Optional[str] = None,
    **kwargs
) -> DistributedLogger:
    """Get or create a logger instance"""
    key = f"{component_name}_{worker_id or 'main'}"
    
    if key not in _loggers:
        _loggers[key] = DistributedLogger(
            component_name=component_name,
            worker_id=worker_id,
            shared_dir=shared_dir,
            **kwargs
        )
    
    return _loggers[key]


def close_all_loggers():
    """Close all logger instances"""
    global _loggers
    _loggers.clear()

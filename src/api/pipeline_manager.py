"""
Pipeline Manager for Real-time Model Training.

Manages the execution of training pipeline steps with real-time log streaming:
- Step 1: Preprocessing & Feature Engineering
- Step 2: EDA Analysis
- Step 3: Model Training

Provides WebSocket streaming of terminal output and progress tracking.
"""

import asyncio
import subprocess
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class PipelineStep(Enum):
    """Pipeline execution steps."""
    IDLE = "idle"
    UPLOADING = "uploading"
    PREPROCESSING = "preprocessing"
    EDA = "eda"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    # Step 1: Preprocessing
    feature_mode: str = "minimal"  # 'minimal' or 'full'
    
    # Step 3: Training
    optuna_trials: int = 10
    models: List[str] = field(default_factory=lambda: ['linear_trend', 'xgboost', 'random_forest', 'prophet', 'sarima'])
    test_size: float = 0.2
    use_holdout: bool = False  # False = Option 2 (full data), True = Option 1
    skip_eda: bool = False
    skip_training: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_mode": self.feature_mode,
            "optuna_trials": self.optuna_trials,
            "models": self.models,
            "test_size": self.test_size,
            "use_holdout": self.use_holdout,
            "skip_eda": self.skip_eda,
            "skip_training": self.skip_training
        }


@dataclass
class PipelineStatus:
    """Current status of the pipeline."""
    step: PipelineStep = PipelineStep.IDLE
    progress: int = 0  # 0-100
    current_step_name: str = ""
    current_step_number: int = 0
    total_steps: int = 3
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step.value,
            "progress": self.progress,
            "current_step_name": self.current_step_name,
            "current_step_number": self.current_step_number,
            "total_steps": self.total_steps,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "is_running": self.step in [PipelineStep.PREPROCESSING, PipelineStep.EDA, PipelineStep.TRAINING, PipelineStep.UPLOADING]
        }


class PipelineManager:
    """
    Manages the ML training pipeline execution with real-time log streaming.
    
    Features:
    - Sequential execution of pipeline steps
    - Real-time stdout/stderr streaming via WebSocket
    - Progress tracking
    - Cancellation support
    - Configuration options for each step
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.status = PipelineStatus()
        self.config = PipelineConfig()
        self.current_process: Optional[subprocess.Popen] = None
        self.is_cancelled = False
        self.log_callback: Optional[Callable[[str, str], None]] = None  # (log_line, log_type)
        self.status_callback: Optional[Callable[[PipelineStatus], None]] = None
        self.uploaded_csv_path: Optional[Path] = None
        
        # Get Python executable
        self.python_exe = sys.executable
        
    def set_log_callback(self, callback: Callable[[str, str], None]):
        """Set callback for log streaming."""
        self.log_callback = callback
        
    def set_status_callback(self, callback: Callable[[PipelineStatus], None]):
        """Set callback for status updates."""
        self.status_callback = callback
        
    def _emit_log(self, message: str, log_type: str = "info"):
        """Emit a log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self.status.logs.append(formatted)
        
        # Keep only last 1000 logs
        if len(self.status.logs) > 1000:
            self.status.logs = self.status.logs[-1000:]
        
        if self.log_callback:
            self.log_callback(formatted, log_type)
            
    def _update_status(self, **kwargs):
        """Update status and emit callback."""
        for key, value in kwargs.items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)
        
        if self.status_callback:
            self.status_callback(self.status)
    
    async def upload_csv(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Upload and validate CSV file.
        
        Args:
            file_content: Raw CSV file content
            filename: Original filename
            
        Returns:
            Dict with upload status and file info
        """
        self._update_status(step=PipelineStep.UPLOADING, progress=10, current_step_name="Uploading CSV")
        self._emit_log(f"Receiving file: {filename}", "info")
        
        try:
            # Create temp directory for uploaded files
            upload_dir = self.project_root / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"upload_{timestamp}_{filename}"
            file_path = upload_dir / safe_filename
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            self._emit_log(f"File saved: {file_path}", "info")
            self._update_status(progress=50)
            
            # Validate CSV
            import pandas as pd
            try:
                df = pd.read_csv(file_path)
                
                # Check required columns
                required = ['date', 'daily_sales']
                missing = [col for col in required if col not in df.columns]
                
                if missing:
                    self._emit_log(f"ERROR: Missing required columns: {missing}", "error")
                    file_path.unlink()  # Remove invalid file
                    self._update_status(step=PipelineStep.FAILED, error_message=f"Missing columns: {missing}")
                    return {
                        "success": False,
                        "error": f"Missing required columns: {missing}",
                        "required_columns": required
                    }
                
                # Parse dates
                df['date'] = pd.to_datetime(df['date'])
                
                self._emit_log(f"CSV validated: {len(df)} rows, {len(df.columns)} columns", "success")
                self._emit_log(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}", "info")
                
                self.uploaded_csv_path = file_path
                self._update_status(step=PipelineStep.IDLE, progress=100)
                
                return {
                    "success": True,
                    "file_path": str(file_path),
                    "filename": safe_filename,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "date_range": {
                        "start": df['date'].min().strftime("%Y-%m-%d"),
                        "end": df['date'].max().strftime("%Y-%m-%d")
                    }
                }
                
            except Exception as e:
                self._emit_log(f"ERROR: Invalid CSV format: {e}", "error")
                file_path.unlink()
                self._update_status(step=PipelineStep.FAILED, error_message=str(e))
                return {
                    "success": False,
                    "error": f"Invalid CSV: {str(e)}"
                }
                
        except Exception as e:
            self._emit_log(f"ERROR: Upload failed: {e}", "error")
            self._update_status(step=PipelineStep.FAILED, error_message=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def run_pipeline(self, config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Dict with pipeline results
        """
        if config:
            self.config = config
            
        if not self.uploaded_csv_path or not self.uploaded_csv_path.exists():
            return {
                "success": False,
                "error": "No CSV file uploaded. Please upload a file first."
            }
        
        self.is_cancelled = False
        self.status = PipelineStatus(
            started_at=datetime.now(),
            total_steps=3 if not self.config.skip_eda else 2
        )
        
        self._emit_log("=" * 60, "info")
        self._emit_log("STARTING TRAINING PIPELINE", "info")
        self._emit_log("=" * 60, "info")
        self._emit_log(f"Input file: {self.uploaded_csv_path}", "info")
        self._emit_log(f"Configuration: {json.dumps(self.config.to_dict(), indent=2)}", "info")
        
        results = {
            "step1_preprocessing": None,
            "step2_eda": None,
            "step3_training": None
        }
        
        try:
            # Step 1: Preprocessing
            self._update_status(
                step=PipelineStep.PREPROCESSING,
                current_step_number=1,
                current_step_name="Preprocessing & Feature Engineering",
                progress=0
            )
            
            success = await self._run_step1_preprocessing()
            results["step1_preprocessing"] = {"success": success}
            
            if not success or self.is_cancelled:
                if self.is_cancelled:
                    self._update_status(step=PipelineStep.CANCELLED)
                return {"success": False, "results": results, "cancelled": self.is_cancelled}
            
            # Step 2: EDA (optional)
            if not self.config.skip_eda:
                self._update_status(
                    step=PipelineStep.EDA,
                    current_step_number=2,
                    current_step_name="Exploratory Data Analysis",
                    progress=33
                )
                
                success = await self._run_step2_eda()
                results["step2_eda"] = {"success": success}
                
                if not success or self.is_cancelled:
                    if self.is_cancelled:
                        self._update_status(step=PipelineStep.CANCELLED)
                    return {"success": False, "results": results, "cancelled": self.is_cancelled}
            
            # Step 3: Training (optional)
            if not self.config.skip_training:
                self._update_status(
                    step=PipelineStep.TRAINING,
                    current_step_number=3 if not self.config.skip_eda else 2,
                    current_step_name="Model Training",
                    progress=66
                )
                
                success = await self._run_step3_training()
                results["step3_training"] = {"success": success}
                
                if not success or self.is_cancelled:
                    if self.is_cancelled:
                        self._update_status(step=PipelineStep.CANCELLED)
                    return {"success": False, "results": results, "cancelled": self.is_cancelled}
            
            # Complete
            self._update_status(
                step=PipelineStep.COMPLETED,
                progress=100,
                current_step_name="Completed",
                completed_at=datetime.now()
            )
            
            self._emit_log("=" * 60, "success")
            self._emit_log("PIPELINE COMPLETED SUCCESSFULLY!", "success")
            self._emit_log("=" * 60, "success")
            
            return {"success": True, "results": results}
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._emit_log(f"PIPELINE ERROR: {e}", "error")
            self._update_status(
                step=PipelineStep.FAILED,
                error_message=str(e)
            )
            return {"success": False, "error": str(e), "results": results}
    
    async def _run_step1_preprocessing(self) -> bool:
        """Run Step 1: Preprocessing & Feature Engineering."""
        self._emit_log("", "info")
        self._emit_log("=" * 60, "info")
        self._emit_log("STEP 1: PREPROCESSING & FEATURE ENGINEERING", "info")
        self._emit_log("=" * 60, "info")
        
        # Copy uploaded file to expected location (for compatibility)
        target_csv = self.project_root / "ecommerce_sales_data (1).csv"
        shutil.copy(self.uploaded_csv_path, target_csv)
        self._emit_log(f"Using uploaded CSV: {self.uploaded_csv_path.name}", "info")
        
        # Build command with --input argument for non-interactive execution
        script_path = self.project_root / "step_1_run_pipeline(preprocessing_Feature_engineering).py"
        cmd = [self.python_exe, str(script_path), "--input", str(target_csv)]
        
        return await self._run_script(cmd, "Step 1")
    
    async def _run_step2_eda(self) -> bool:
        """Run Step 2: EDA Analysis."""
        self._emit_log("", "info")
        self._emit_log("=" * 60, "info")
        self._emit_log("STEP 2: EXPLORATORY DATA ANALYSIS", "info")
        self._emit_log("=" * 60, "info")
        
        script_path = self.project_root / "step_2_eda_analysis.py"
        cmd = [self.python_exe, str(script_path)]
        
        return await self._run_script(cmd, "Step 2")
    
    async def _run_step3_training(self) -> bool:
        """Run Step 3: Model Training."""
        self._emit_log("", "info")
        self._emit_log("=" * 60, "info")
        self._emit_log("STEP 3: MODEL TRAINING", "info")
        self._emit_log("=" * 60, "info")
        
        script_path = self.project_root / "step_3_train_models.py"
        cmd = [
            self.python_exe, 
            str(script_path),
            "--trials", str(self.config.optuna_trials),
            "--test-size", str(self.config.test_size),
            "--models"
        ] + self.config.models
        
        if self.config.use_holdout:
            cmd.append("--holdout")
        
        self._emit_log(f"Training models: {', '.join(self.config.models)}", "info")
        self._emit_log(f"Optuna trials: {self.config.optuna_trials}", "info")
        self._emit_log(f"Training mode: {'Holdout (Option 1)' if self.config.use_holdout else 'Full Data (Option 2)'}", "info")
        
        return await self._run_script(cmd, "Step 3")
    
    async def _run_script(self, cmd: List[str], step_name: str) -> bool:
        """
        Run a script and stream output.
        
        Args:
            cmd: Command to run
            step_name: Name for logging
            
        Returns:
            True if successful
        """
        self._emit_log(f"Running: {' '.join(cmd)}", "info")
        
        try:
            # Create process
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self.project_root),
                bufsize=1,
                universal_newlines=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            # Stream output
            while True:
                if self.is_cancelled:
                    self.current_process.terminate()
                    self._emit_log(f"{step_name} cancelled by user", "warning")
                    return False
                
                line = self.current_process.stdout.readline()
                if not line and self.current_process.poll() is not None:
                    break
                    
                if line:
                    line = line.rstrip()
                    # Determine log type
                    log_type = "info"
                    if "ERROR" in line.upper() or "FAIL" in line.upper():
                        log_type = "error"
                    elif "WARNING" in line.upper() or "WARN" in line.upper():
                        log_type = "warning"
                    elif "SUCCESS" in line.upper() or "COMPLETE" in line.upper():
                        log_type = "success"
                    
                    self._emit_log(line, log_type)
                
                # Allow other async tasks to run
                await asyncio.sleep(0.01)
            
            # Check return code
            return_code = self.current_process.wait()
            
            if return_code == 0:
                self._emit_log(f"{step_name} completed successfully", "success")
                return True
            else:
                self._emit_log(f"{step_name} failed with return code: {return_code}", "error")
                return False
                
        except Exception as e:
            self._emit_log(f"{step_name} error: {e}", "error")
            return False
        finally:
            self.current_process = None
    
    def cancel(self):
        """Cancel the current pipeline execution."""
        self.is_cancelled = True
        if self.current_process:
            try:
                self.current_process.terminate()
            except:
                pass
        self._emit_log("Pipeline cancellation requested", "warning")
        self._update_status(step=PipelineStep.CANCELLED)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return self.status.to_dict()
    
    def get_logs(self, limit: int = 100) -> List[str]:
        """Get recent logs."""
        return self.status.logs[-limit:]
    
    def reset(self):
        """Reset pipeline state."""
        self.status = PipelineStatus()
        self.is_cancelled = False
        self.current_process = None


# Global pipeline manager
_pipeline_manager: Optional[PipelineManager] = None


def get_pipeline_manager(project_root: Path = None) -> PipelineManager:
    """Get or create the pipeline manager."""
    global _pipeline_manager
    if _pipeline_manager is None:
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        _pipeline_manager = PipelineManager(project_root)
    return _pipeline_manager


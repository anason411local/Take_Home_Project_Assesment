"""
Terminal Logger - Captures all terminal output to log files.

This module provides functionality to capture all stdout/stderr output
and save it to timestamped log files in the logs/ directory.
"""
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import io


class TeeOutput:
    """
    Tee output stream - writes to both original stream and log file.
    
    This allows capturing all terminal output while still displaying it.
    """
    
    def __init__(self, original_stream, log_file, encoding='utf-8'):
        self.original_stream = original_stream
        self.log_file = log_file
        self.encoding = encoding
    
    def write(self, message):
        """Write to both original stream and log file."""
        # Write to original stream (terminal)
        try:
            self.original_stream.write(message)
        except Exception:
            pass
        
        # Write to log file
        try:
            if isinstance(message, bytes):
                message = message.decode(self.encoding, errors='replace')
            self.log_file.write(message)
            self.log_file.flush()  # Ensure immediate write
        except Exception:
            pass
    
    def flush(self):
        """Flush both streams."""
        try:
            self.original_stream.flush()
        except Exception:
            pass
        try:
            self.log_file.flush()
        except Exception:
            pass
    
    def isatty(self):
        """Check if original stream is a tty."""
        try:
            return self.original_stream.isatty()
        except Exception:
            return False
    
    def fileno(self):
        """Return file descriptor of original stream."""
        return self.original_stream.fileno()


class TerminalLogger:
    """
    Context manager for capturing all terminal output to log files.
    
    Usage:
        with TerminalLogger("step_1_pipeline"):
            # All print statements and logging will be captured
            print("This goes to both terminal and log file")
    
    Or:
        logger = TerminalLogger("step_2_training")
        logger.start()
        # ... your code ...
        logger.stop()
    """
    
    def __init__(
        self, 
        script_name: str,
        logs_dir: str = "logs",
        include_timestamp: bool = True,
        capture_stderr: bool = True
    ):
        """
        Initialize TerminalLogger.
        
        Args:
            script_name: Name of the script (used in log filename)
            logs_dir: Directory to save log files
            include_timestamp: Whether to include timestamp in filename
            capture_stderr: Whether to also capture stderr
        """
        self.script_name = script_name
        self.logs_dir = Path(logs_dir)
        self.include_timestamp = include_timestamp
        self.capture_stderr = capture_stderr
        
        self.log_file = None
        self.original_stdout = None
        self.original_stderr = None
        self.tee_stdout = None
        self.tee_stderr = None
        self.log_path = None
        self.is_active = False
    
    def _get_log_filename(self) -> str:
        """Generate log filename with timestamp."""
        if self.include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{self.script_name}_{timestamp}.log"
        else:
            return f"{self.script_name}.log"
    
    def start(self) -> Path:
        """
        Start capturing terminal output.
        
        Returns:
            Path to the log file
        """
        if self.is_active:
            return self.log_path
        
        # Create logs directory
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        self.log_path = self.logs_dir / self._get_log_filename()
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        # Write header
        self._write_header()
        
        # Save original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create tee streams
        self.tee_stdout = TeeOutput(self.original_stdout, self.log_file)
        sys.stdout = self.tee_stdout
        
        if self.capture_stderr:
            self.tee_stderr = TeeOutput(self.original_stderr, self.log_file)
            sys.stderr = self.tee_stderr
        
        self.is_active = True
        return self.log_path
    
    def stop(self):
        """Stop capturing and restore original streams."""
        if not self.is_active:
            return
        
        # Write footer
        self._write_footer()
        
        # Restore original streams
        sys.stdout = self.original_stdout
        if self.capture_stderr:
            sys.stderr = self.original_stderr
        
        # Close log file
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        
        self.is_active = False
    
    def _write_header(self):
        """Write log file header."""
        header = [
            "=" * 80,
            f"TERMINAL LOG: {self.script_name}",
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Log file: {self.log_path}",
            "=" * 80,
            ""
        ]
        self.log_file.write("\n".join(header))
        self.log_file.flush()
    
    def _write_footer(self):
        """Write log file footer."""
        footer = [
            "",
            "=" * 80,
            f"TERMINAL LOG ENDED",
            f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ]
        self.log_file.write("\n".join(footer))
        self.log_file.flush()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            # Log exception info
            try:
                self.log_file.write(f"\n\nEXCEPTION: {exc_type.__name__}: {exc_val}\n")
            except Exception:
                pass
        self.stop()
        return False  # Don't suppress exceptions
    
    def get_log_path(self) -> Optional[Path]:
        """Get the path to the current log file."""
        return self.log_path


def setup_terminal_logging(script_name: str, logs_dir: str = "logs") -> TerminalLogger:
    """
    Convenience function to setup terminal logging.
    
    Args:
        script_name: Name of the script
        logs_dir: Directory for log files
        
    Returns:
        TerminalLogger instance (already started)
    """
    logger = TerminalLogger(script_name, logs_dir)
    log_path = logger.start()
    print(f"Terminal logging enabled. Log file: {log_path}")
    return logger



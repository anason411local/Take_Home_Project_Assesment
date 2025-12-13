"""
Sales Forecasting Application - Main Entry Point

This script starts the FastAPI server and automatically opens the web interface.

Usage:
    python app.py
    python app.py --port 8000
    python app.py --no-browser

Web Interface:
    - Dashboard: http://localhost:8000/app/
    
API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
"""

import sys
import argparse
import threading
import time
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sales Forecasting Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py                    # Start server and open browser
    python app.py --port 8080        # Use custom port
    python app.py --no-browser       # Don't open browser automatically
    python app.py --reload           # Enable auto-reload for development
        """
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Port to bind to (default: 8000)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    return parser.parse_args()


def open_browser(url: str, delay: float = 2.0):
    """
    Open the browser after a delay to allow server to start.
    
    Args:
        url: URL to open
        delay: Seconds to wait before opening
    """
    def _open():
        time.sleep(delay)
        print(f"\n>>> Opening browser: {url}\n")
        webbrowser.open(url)
    
    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def print_banner(host: str, port: int, open_browser_flag: bool):
    """Print startup banner."""
    app_url = f"http://{host}:{port}/app/"
    api_url = f"http://{host}:{port}"
    
    print()
    print("=" * 65)
    print("  ____        _             _____                              _   ")
    print(" / ___|  __ _| | ___  ___  |  ___|__  _ __ ___  ___ __ _ ___| |_ ")
    print(" \\___ \\ / _` | |/ _ \\/ __| | |_ / _ \\| '__/ _ \\/ __/ _` / __| __|")
    print("  ___) | (_| | |  __/\\__ \\ |  _| (_) | | |  __/ (_| (_| \\__ \\ |_ ")
    print(" |____/ \\__,_|_|\\___||___/ |_|  \\___/|_|  \\___|\\___\\__,_|___/\\__|")
    print()
    print("=" * 65)
    print()
    print(f"  Server running at: {api_url}")
    print()
    print("  WEB INTERFACE:")
    print(f"    Dashboard:    {app_url}")
    print(f"    Forecast:     {app_url}forecast.html")
    print(f"    Data View:    {app_url}data.html")
    print()
    print("  API DOCUMENTATION:")
    print(f"    Swagger UI:   {api_url}/docs")
    print(f"    ReDoc:        {api_url}/redoc")
    print()
    print("  API ENDPOINTS:")
    print(f"    POST /api/forecast          - Generate sales forecast")
    print(f"    GET  /api/historical        - Get historical data")
    print(f"    GET  /api/metrics           - Get model metrics")
    print(f"    GET  /api/feature-importance - Get feature importance")
    print(f"    POST /api/upload            - Upload new data")
    print(f"    POST /api/session           - Create session (WebSocket)")
    print(f"    WS   /ws/{{session_id}}       - WebSocket connection")
    print()
    print("=" * 65)
    
    if open_browser_flag:
        print(f"  >>> Browser will open automatically...")
    else:
        print(f"  >>> Open {app_url} in your browser")
    
    print("  >>> Press Ctrl+C to stop the server")
    print("=" * 65)
    print()


def main():
    """Run the application."""
    args = parse_args()
    
    # Print banner
    print_banner(args.host, args.port, not args.no_browser)
    
    # Open browser if not disabled
    if not args.no_browser:
        app_url = f"http://{args.host}:{args.port}/app/"
        open_browser(app_url, delay=2.5)
    
    # Start server
    import uvicorn
    
    try:
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Sales Forecasting Application - Ngrok Public URL Generator

This script starts the Sales Forecasting application and creates a public URL
using Ngrok, allowing you to share the application with anyone on the internet.

Usage:
    python start_with_ngrok.py --token YOUR_NGROK_TOKEN
    python start_with_ngrok.py --token YOUR_NGROK_TOKEN --port 8000

Example:
    python start_with_ngrok.py --token 2fyUzTYxWg0JNbJkl_2Jw6oMBQ7BjDWo4Q5xKsE

To get your Ngrok token:
    1. Sign up at https://ngrok.com/
    2. Go to https://dashboard.ngrok.com/get-started/your-authtoken
    3. Copy your authtoken
"""

import os
import sys
import time
import argparse
from contextlib import closing
import socket
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_and_install_pyngrok():
    """Check if pyngrok is installed, and install it if not"""
    try:
        import pyngrok
        print("✓ pyngrok is already installed")
        return True
    except ImportError:
        print("📦 pyngrok not found. Installing it now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
            print("✓ pyngrok installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install pyngrok: {e}")
            return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    required = ['uvicorn', 'fastapi']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✓ All required dependencies are installed")
    return True


def find_free_port(default_port=8000, max_attempts=10):
    """Find a free port starting from the default port"""
    for port in range(default_port, default_port + max_attempts):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('127.0.0.1', port))
                return port
            except socket.error:
                continue
    raise RuntimeError(f"Could not find a free port after {max_attempts} attempts")


def print_ngrok_banner(public_url: str, local_port: int):
    """Print the banner with ngrok URLs"""
    print()
    print("=" * 70)
    print("  ____        _             _____                              _   ")
    print(" / ___|  __ _| | ___  ___  |  ___|__  _ __ ___  ___ __ _ ___| |_ ")
    print(" \\___ \\ / _` | |/ _ \\/ __| | |_ / _ \\| '__/ _ \\/ __/ _` / __| __|")
    print("  ___) | (_| | |  __/\\__ \\ |  _| (_) | | |  __/ (_| (_| \\__ \\ |_ ")
    print(" |____/ \\__,_|_|\\___||___/ |_|  \\___/|_|  \\___|\\___\\__,_|___/\\__|")
    print()
    print("                    🌐 PUBLIC ACCESS ENABLED 🌐")
    print("=" * 70)
    print()
    print("  🎉 YOUR APPLICATION IS NOW PUBLICLY ACCESSIBLE!")
    print()
    print("  📤 SHARE THESE URLs WITH ANYONE:")
    print("  ─" * 35)
    print(f"    🌐 Public Dashboard:  {public_url}/app/")
    print(f"    📊 Public Forecast:   {public_url}/app/forecast.html")
    print(f"    📁 Public Data View:  {public_url}/app/data.html")
    print(f"    📚 API Docs:          {public_url}/docs")
    print()
    print("  🏠 LOCAL ACCESS:")
    print("  ─" * 35)
    print(f"    Dashboard:    http://localhost:{local_port}/app/")
    print(f"    API Docs:     http://localhost:{local_port}/docs")
    print()
    print("  📋 COPY THIS LINK TO SHARE:")
    print("  ─" * 35)
    print(f"    {public_url}/app/")
    print()
    print("=" * 70)
    print("  ⚠️  IMPORTANT: Keep this script running to maintain the connection!")
    print("  🛑 Press Ctrl+C to stop the server and close the tunnel")
    print("=" * 70)
    print()


def start_app_with_ngrok(ngrok_token: str, port: int = 8000):
    """Start the Sales Forecasting app with Ngrok tunnel"""
    
    # Check and install pyngrok if needed
    if not check_and_install_pyngrok():
        print("❌ Cannot continue without pyngrok.")
        print("   Please install it manually: pip install pyngrok")
        return False
    
    # Check other dependencies
    if not check_dependencies():
        return False
    
    # Import pyngrok after ensuring it's installed
    from pyngrok import ngrok
    
    public_tunnel = None
    
    try:
        # Set the ngrok auth token
        print(f"\n🔑 Setting up Ngrok with your token...")
        ngrok.set_auth_token(ngrok_token)
        
        # Find a free port
        try:
            free_port = find_free_port(port)
            if free_port != port:
                print(f"⚠️  Port {port} is in use, using port {free_port} instead")
            else:
                print(f"✓ Using port {free_port}")
        except RuntimeError as e:
            print(f"✗ Error finding free port: {e}")
            return False
        
        # Start ngrok tunnel
        print("🚀 Starting Ngrok tunnel...")
        public_tunnel = ngrok.connect(free_port)
        public_url = public_tunnel.public_url
        
        # Print the banner with URLs
        print_ngrok_banner(public_url, free_port)
        
        # Start the FastAPI server
        print("🔧 Starting Sales Forecasting server...")
        print()
        
        import uvicorn
        
        uvicorn.run(
            "src.api.main:app",
            host="127.0.0.1",
            port=free_port,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False
    finally:
        # Clean up ngrok tunnels
        if public_tunnel:
            try:
                print("🧹 Cleaning up Ngrok tunnel...")
                ngrok.disconnect(public_tunnel.public_url)
                ngrok.kill()
                print("✓ Ngrok tunnel closed")
            except Exception:
                pass
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Start Sales Forecasting Application with Ngrok Public URL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start_with_ngrok.py --token YOUR_NGROK_TOKEN
    python start_with_ngrok.py --token YOUR_NGROK_TOKEN --port 8080

To get your Ngrok token:
    1. Sign up at https://ngrok.com/
    2. Go to https://dashboard.ngrok.com/get-started/your-authtoken
    3. Copy your authtoken and use it with --token
        """
    )
    parser.add_argument(
        '--token', '-t',
        type=str,
        required=True,
        help='Your Ngrok auth token (required)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Port to run the application on (default: 8000)'
    )
    
    args = parser.parse_args()
    
    if not args.token or len(args.token) < 10:
        print("❌ Invalid Ngrok token!")
        print("\nTo get your token:")
        print("  1. Go to https://dashboard.ngrok.com/get-started/your-authtoken")
        print("  2. Copy your authtoken")
        print("  3. Run: python start_with_ngrok.py --token YOUR_TOKEN")
        return
    
    print()
    print("=" * 70)
    print("  🚀 SALES FORECASTING - NGROK PUBLIC URL GENERATOR")
    print("=" * 70)
    print(f"  📁 Project Root: {PROJECT_ROOT}")
    print(f"  🔑 Ngrok Token:  {args.token[:8]}{'*' * (len(args.token) - 12)}{args.token[-4:]}")
    print(f"  🔌 Target Port:  {args.port}")
    print("=" * 70)
    
    # Check if we're in the right directory
    if not (PROJECT_ROOT / "src" / "api" / "main.py").exists():
        print("\n❌ Error: Cannot find src/api/main.py")
        print("   Make sure you're running this script from the project root directory.")
        return
    
    success = start_app_with_ngrok(args.token, args.port)
    
    if not success:
        print("\n❌ Failed to start application with Ngrok")
        print("   Please check the errors above and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Test WebSocket and Session Management for Sales Forecasting API.

This script tests:
1. Session creation and management
2. WebSocket connection with session
3. Real-time forecast streaming
4. Multiple concurrent connections

Run with: python test_websocket.py
(Make sure API server is running: python run_api.py)
"""

import asyncio
import json
import requests
import websockets
import time
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000"


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(success: bool, message: str):
    """Print a formatted result."""
    status = "[PASS]" if success else "[FAIL]"
    print(f"  {status} {message}")


# ==================== SESSION TESTS ====================

def test_create_session():
    """Test session creation."""
    print_header("TEST: Create Session")
    
    try:
        response = requests.post(f"{BASE_URL}/api/session")
        data = response.json()
        
        success = (
            response.status_code == 200 and
            data.get("success") == True and
            "session_id" in data and
            "websocket_url" in data
        )
        
        if success:
            print_result(True, f"Session created: {data['session_id'][:8]}...")
            print(f"       WebSocket URL: {data['websocket_url']}")
            return data['session_id']
        else:
            print_result(False, f"Failed: {data}")
            return None
            
    except Exception as e:
        print_result(False, f"Error: {e}")
        return None


def test_get_session(session_id: str):
    """Test getting session info."""
    print_header("TEST: Get Session Info")
    
    try:
        response = requests.get(f"{BASE_URL}/api/session/{session_id}")
        data = response.json()
        
        success = (
            response.status_code == 200 and
            data.get("success") == True and
            "session" in data
        )
        
        if success:
            session = data['session']
            print_result(True, "Session retrieved successfully")
            print(f"       Created: {session.get('created_at', 'N/A')}")
            print(f"       Model: {session.get('preferred_model', 'N/A')}")
            print(f"       Horizon: {session.get('default_horizon', 'N/A')}")
        else:
            print_result(False, f"Failed: {data}")
            
    except Exception as e:
        print_result(False, f"Error: {e}")


def test_update_session(session_id: str):
    """Test updating session preferences."""
    print_header("TEST: Update Session Preferences")
    
    try:
        response = requests.put(
            f"{BASE_URL}/api/session/{session_id}",
            params={"preferred_model": "sarima", "default_horizon": 60}
        )
        data = response.json()
        
        success = (
            response.status_code == 200 and
            data.get("success") == True
        )
        
        if success:
            session = data.get('session', {})
            print_result(True, "Session updated successfully")
            print(f"       New Model: {session.get('preferred_model', 'N/A')}")
            print(f"       New Horizon: {session.get('default_horizon', 'N/A')}")
        else:
            print_result(False, f"Failed: {data}")
            
    except Exception as e:
        print_result(False, f"Error: {e}")


def test_list_sessions():
    """Test listing all sessions."""
    print_header("TEST: List All Sessions")
    
    try:
        response = requests.get(f"{BASE_URL}/api/sessions")
        data = response.json()
        
        success = (
            response.status_code == 200 and
            data.get("success") == True
        )
        
        if success:
            print_result(True, "Sessions listed successfully")
            print(f"       Active Sessions: {data.get('active_sessions', 0)}")
            print(f"       WebSocket Connections: {data.get('websocket_connections', 0)}")
        else:
            print_result(False, f"Failed: {data}")
            
    except Exception as e:
        print_result(False, f"Error: {e}")


# ==================== WEBSOCKET TESTS ====================

async def test_websocket_connection(session_id: str):
    """Test WebSocket connection."""
    print_header("TEST: WebSocket Connection")
    
    try:
        uri = f"{WS_URL}/ws/{session_id}"
        
        async with websockets.connect(uri) as websocket:
            # Wait for connection message
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            
            success = data.get("type") == "connected"
            
            if success:
                print_result(True, "WebSocket connected successfully")
                print(f"       Message: {data.get('data', {}).get('message', 'N/A')}")
            else:
                print_result(False, f"Unexpected response: {data}")
                
            return True
            
    except asyncio.TimeoutError:
        print_result(False, "Connection timeout")
        return False
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False


async def test_websocket_heartbeat(session_id: str):
    """Test WebSocket heartbeat."""
    print_header("TEST: WebSocket Heartbeat")
    
    try:
        uri = f"{WS_URL}/ws/{session_id}"
        
        async with websockets.connect(uri) as websocket:
            # Skip connection message
            await asyncio.wait_for(websocket.recv(), timeout=5.0)
            
            # Send heartbeat
            await websocket.send(json.dumps({"action": "heartbeat"}))
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            
            success = data.get("type") == "heartbeat"
            
            if success:
                print_result(True, "Heartbeat received")
                print(f"       Status: {data.get('data', {}).get('status', 'N/A')}")
            else:
                print_result(False, f"Unexpected response: {data}")
                
            return True
            
    except asyncio.TimeoutError:
        print_result(False, "Heartbeat timeout")
        return False
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False


async def test_websocket_forecast(session_id: str):
    """Test WebSocket forecast with real-time streaming."""
    print_header("TEST: WebSocket Forecast Streaming")
    
    try:
        uri = f"{WS_URL}/ws/{session_id}"
        
        async with websockets.connect(uri) as websocket:
            # Skip connection message
            await asyncio.wait_for(websocket.recv(), timeout=5.0)
            
            # Send forecast request
            forecast_request = {
                "action": "forecast",
                "payload": {
                    "horizon": 7,
                    "model": "best"
                }
            }
            await websocket.send(json.dumps(forecast_request))
            print(f"  [INFO] Forecast request sent (horizon=7)")
            
            # Collect all messages until completion or error
            messages = []
            start_time = time.time()
            
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    data = json.loads(response)
                    messages.append(data)
                    
                    msg_type = data.get("type", "")
                    
                    if msg_type == "forecast_start":
                        print(f"  [STREAM] Forecast started...")
                        
                    elif msg_type == "forecast_progress":
                        progress = data.get("data", {}).get("progress", 0)
                        step = data.get("data", {}).get("current_step", "")
                        print(f"  [STREAM] Progress: {progress}% - {step}")
                        
                    elif msg_type == "forecast_complete":
                        elapsed = time.time() - start_time
                        predictions = data.get("data", {}).get("predictions", [])
                        model_used = data.get("data", {}).get("model_used", "N/A")
                        summary = data.get("data", {}).get("summary", {})
                        
                        print_result(True, f"Forecast completed in {elapsed:.2f}s")
                        print(f"       Model: {model_used}")
                        print(f"       Predictions: {len(predictions)} days")
                        print(f"       Total: ${summary.get('total', 0):,.2f}")
                        print(f"       Mean: ${summary.get('mean', 0):,.2f}")
                        break
                        
                    elif msg_type == "forecast_error":
                        error = data.get("data", {}).get("message", "Unknown error")
                        print_result(False, f"Forecast error: {error}")
                        break
                        
                    elif msg_type == "error":
                        error = data.get("data", {}).get("message", "Unknown error")
                        print_result(False, f"Error: {error}")
                        break
                        
                except asyncio.TimeoutError:
                    print_result(False, "Forecast timeout (30s)")
                    break
                    
            return True
            
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False


async def test_multiple_connections():
    """Test multiple concurrent WebSocket connections."""
    print_header("TEST: Multiple Concurrent Connections")
    
    try:
        # Create 3 sessions
        sessions = []
        for i in range(3):
            response = requests.post(f"{BASE_URL}/api/session")
            if response.status_code == 200:
                sessions.append(response.json()['session_id'])
        
        if len(sessions) < 3:
            print_result(False, f"Could only create {len(sessions)} sessions")
            return False
        
        print(f"  [INFO] Created {len(sessions)} sessions")
        
        # Connect all simultaneously
        async def connect_and_ping(session_id: str, idx: int):
            uri = f"{WS_URL}/ws/{session_id}"
            async with websockets.connect(uri) as ws:
                # Get connection message
                await ws.recv()
                
                # Send ping
                await ws.send(json.dumps({"action": "ping"}))
                response = await ws.recv()
                data = json.loads(response)
                
                return data.get("data", {}).get("message") == "pong"
        
        # Run all connections concurrently
        tasks = [connect_and_ping(sid, i) for i, sid in enumerate(sessions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successes = sum(1 for r in results if r == True)
        
        if successes == len(sessions):
            print_result(True, f"All {len(sessions)} connections successful")
        else:
            print_result(False, f"Only {successes}/{len(sessions)} connections succeeded")
        
        # Cleanup sessions
        for sid in sessions:
            requests.delete(f"{BASE_URL}/api/session/{sid}")
        
        return successes == len(sessions)
        
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False


def test_delete_session(session_id: str):
    """Test session deletion."""
    print_header("TEST: Delete Session")
    
    try:
        response = requests.delete(f"{BASE_URL}/api/session/{session_id}")
        data = response.json()
        
        success = (
            response.status_code == 200 and
            data.get("success") == True
        )
        
        if success:
            print_result(True, "Session deleted successfully")
        else:
            print_result(False, f"Failed: {data}")
            
    except Exception as e:
        print_result(False, f"Error: {e}")


# ==================== MAIN ====================

async def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("  SALES FORECASTING API - SESSION & WEBSOCKET TESTS")
    print("="*60)
    print(f"\n  Server: {BASE_URL}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test Session Management
    session_id = test_create_session()
    
    if session_id:
        test_get_session(session_id)
        test_update_session(session_id)
        test_list_sessions()
        
        # Test WebSocket
        await test_websocket_connection(session_id)
        await test_websocket_heartbeat(session_id)
        await test_websocket_forecast(session_id)
        
        # Test Multiple Connections
        await test_multiple_connections()
        
        # Cleanup
        test_delete_session(session_id)
    
    print("\n" + "="*60)
    print("  TESTS COMPLETED")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"ERROR: Server not healthy. Status: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to server at {BASE_URL}")
        print("Make sure the API server is running: python run_api.py")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Run async tests
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()


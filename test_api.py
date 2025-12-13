"""
Test script for Sales Forecasting API.

Run with: python test_api.py
"""

import requests
import json

base_url = 'http://127.0.0.1:8000'

print('='*60)
print('TESTING API ENDPOINTS')
print('='*60)

# Test 1: Root endpoint
print('\n1. GET / (Health Check)')
print('-'*40)
r = requests.get(f'{base_url}/')
data = r.json()
print(f'Status: {r.status_code}')
print(f'Response: {json.dumps(data, indent=2)}')

# Test 2: POST /api/forecast
print('\n2. POST /api/forecast')
print('-'*40)
r = requests.post(f'{base_url}/api/forecast', json={'horizon': 7})
data = r.json()
print(f'Status: {r.status_code}')
print(f'Success: {data.get("success")}')
print(f'Model Used: {data.get("model_used")}')
print(f'Predictions Count: {len(data.get("predictions", []))}')
print(f'Summary: {data.get("summary")}')

# Test 3: GET /api/historical
print('\n3. GET /api/historical')
print('-'*40)
r = requests.get(f'{base_url}/api/historical?limit=5')
data = r.json()
print(f'Status: {r.status_code}')
print(f'Success: {data.get("success")}')
print(f'Total Records: {data.get("total_records")}')
print(f'Date Range: {data.get("date_range")}')
print(f'Records Returned: {len(data.get("data", []))}')

# Test 4: GET /api/metrics
print('\n4. GET /api/metrics')
print('-'*40)
r = requests.get(f'{base_url}/api/metrics')
data = r.json()
print(f'Status: {r.status_code}')
print(f'Success: {data.get("success")}')
print(f'Best Model: {data.get("best_model")}')
print(f'Target MAPE: {data.get("target_mape")}%')
print(f'Target Met: {data.get("target_met")}')
print(f'Models Count: {len(data.get("models", []))}')
for m in data.get("models", []):
    print(f'  - {m["model_name"]}: MAPE={m["test_mape"]}%, MAE=${m["test_mae"]:,.0f}')

# Test 5: GET /api/feature-importance
print('\n5. GET /api/feature-importance')
print('-'*40)
r = requests.get(f'{base_url}/api/feature-importance?model=xgboost')
data = r.json()
print(f'Status: {r.status_code}')
print(f'Success: {data.get("success")}')
print(f'Model: {data.get("model_name")}')
print(f'Features Count: {len(data.get("features", []))}')
if data.get('features'):
    print('Top 5 Features:')
    for f in data["features"][:5]:
        print(f'  {f["rank"]}. {f["feature"]}: {f["importance"]:.4f}')

# Test 6: GET /api/models
print('\n6. GET /api/models')
print('-'*40)
r = requests.get(f'{base_url}/api/models')
data = r.json()
print(f'Status: {r.status_code}')
print(f'Success: {data.get("success")}')
print(f'Loaded Models: {data.get("loaded_count")}')
print(f'Trained Models: {data.get("trained_count")}')

print('\n' + '='*60)
print('ALL TESTS COMPLETED!')
print('='*60)
print('\nAPI Documentation:')
print(f'  Swagger UI: {base_url}/docs')
print(f'  ReDoc: {base_url}/redoc')


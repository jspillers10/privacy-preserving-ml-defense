#!/bin/bash
# HTB Privacy Defense - Submission Script

# Usage: ./submit.sh <server_ip:port>

if [ -z "$1" ]; then
    echo "Usage: ./submit.sh <server_ip:port>"
    echo "Example: ./submit.sh IP:PORT"
    exit 1
fi

BASE_URL="http://$1"

echo "=================================================="
echo "HTB Privacy Defense Challenge - Submission"
echo "=================================================="
echo "Server: $BASE_URL"
echo ""

# Check if model exists
if [ ! -f "defended_model.safetensors" ]; then
    echo "Error: defended_model.safetensors not found"
    echo "   Run: python3 train.py first"
    exit 1
fi

# Check server health
echo "1. Checking server health..."
health=$(curl -s "$BASE_URL/health" | jq -r '.status')
if [ "$health" != "ok" ]; then
    echo "Server not healthy"
    exit 1
fi
echo "✓ Server is healthy"
echo ""

# Get baseline metrics
echo "2. Baseline metrics:"
curl -s "$BASE_URL/baseline" | jq .
echo ""

# Submit model
echo "3. Submitting model..."
echo ""
result=$(curl -s -X POST "$BASE_URL/submit" -F "defended_model=@defended_model.safetensors")
echo "$result" | jq .

# Extract and display flag if successful
flag=$(echo "$result" | jq -r '.flag')
if [ "$flag" != "null" ]; then
    echo ""
    echo "=================================================="
    echo "SUCCESS!"
    echo "=================================================="
    echo "Flag: $flag"
    echo "=================================================="
else
    echo ""
    echo "❌ Submission failed - check requirements"
fi

#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Testing GenOCR API Server${NC}"

API_URL="http://localhost:8080"

# Test health endpoint
echo -e "\n${YELLOW}1. Testing health endpoint...${NC}"
curl -s ${API_URL}/api/v1/health | jq .

# Test metrics endpoint
echo -e "\n${YELLOW}2. Testing metrics endpoint...${NC}"
curl -s ${API_URL}/api/v1/metrics | jq .

# Test OCR endpoint (if test image exists)
if [ -f "tests/test_assets/sample_document.png" ]; then
    echo -e "\n${YELLOW}3. Testing OCR processing...${NC}"
    curl -s -X POST ${API_URL}/api/v1/ocr/process \
        -F "image=@tests/test_assets/sample_document.png" | jq .
else
    echo -e "\n${YELLOW}3. Skipping OCR test (no test image found)${NC}"
fi

echo -e "\n${GREEN}✓ All tests completed${NC}"

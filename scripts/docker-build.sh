#!/bin/bash
set -e

# GenOCR API Server Docker Build Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GenOCR API Server - Docker Build${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Parse arguments
BUILD_TYPE=${1:-cpu}
TAG=${2:-latest}

if [ "$BUILD_TYPE" = "gpu" ]; then
    echo -e "${YELLOW}Building GPU-enabled image...${NC}"
    docker build -f api_server/Dockerfile.gpu -t genocr-api:${TAG}-gpu .
    echo -e "${GREEN}✓ GPU image built: genocr-api:${TAG}-gpu${NC}"
elif [ "$BUILD_TYPE" = "cpu" ]; then
    echo -e "${YELLOW}Building CPU-only image...${NC}"
    docker build -f api_server/Dockerfile -t genocr-api:${TAG} .
    echo -e "${GREEN}✓ CPU image built: genocr-api:${TAG}${NC}"
else
    echo -e "${RED}Invalid build type. Use 'cpu' or 'gpu'${NC}"
    exit 1
fi

# Show image size
echo -e "\n${YELLOW}Image size:${NC}"
docker images genocr-api:${TAG}*

echo -e "\n${GREEN}Build complete!${NC}"
echo -e "Run with: ${YELLOW}docker-compose up -d${NC}"

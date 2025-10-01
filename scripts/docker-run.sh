#!/bin/bash
set -e

# GenOCR API Server Docker Run Script

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting GenOCR API Server...${NC}"

# Check if docker-compose is installed
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}Error: docker-compose not found${NC}"
    exit 1
fi

# Parse arguments
MODE=${1:-cpu}

if [ "$MODE" = "gpu" ]; then
    echo -e "${YELLOW}Starting with GPU support...${NC}"
    $COMPOSE_CMD -f docker-compose.gpu.yml up -d
else
    echo -e "${YELLOW}Starting with CPU only...${NC}"
    $COMPOSE_CMD up -d
fi

# Wait for container to start
echo -e "\n${YELLOW}Waiting for container to start...${NC}"
sleep 3

# Check if container is running
if ! docker ps | grep -q genocr-api-server; then
    echo -e "${RED}Container failed to start!${NC}"
    echo -e "Check logs with: ${YELLOW}docker logs genocr-api-server${NC}"
    exit 1
fi

echo -e "${YELLOW}Loading OCR models (this takes ~30-40 seconds)...${NC}"

# Wait for health check with progress indicator
MAX_ATTEMPTS=60
ATTEMPT=0
SLEEP_TIME=2

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -sf http://localhost:8080/api/v1/health > /dev/null 2>&1; then
        echo -e "\n${GREEN}✓ Server is running and healthy!${NC}"
        
        # Show server info
        echo -e "\n${YELLOW}API Endpoints:${NC}"
        echo "  Health:  http://localhost:8080/api/v1/health"
        echo "  Metrics: http://localhost:8080/api/v1/metrics"
        echo "  OCR:     POST http://localhost:8080/api/v1/ocr/process"
        
        echo -e "\n${YELLOW}Example usage:${NC}"
        echo "  curl -X POST http://localhost:8080/api/v1/ocr/process \\"
        echo "    -F \"image=@your_image.png\" | jq ."
        
        echo -e "\n${YELLOW}View logs:${NC}"
        echo "  docker logs -f genocr-api-server"
        
        exit 0
    fi
    
    # Show progress
    printf "."
    
    ATTEMPT=$((ATTEMPT + 1))
    sleep $SLEEP_TIME
done

# Timeout reached
echo -e "\n${RED}Health check timed out after $((MAX_ATTEMPTS * SLEEP_TIME)) seconds${NC}"
echo -e "The server might still be loading models. Check logs:"
echo -e "${YELLOW}docker logs genocr-api-server${NC}"
exit 1

#!/bin/bash
# Deployment script for DQN Stock Analysis API

set -e

echo "🚀 Deploying DQN Stock Analysis API..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t dqn-stock-api:latest .

# Check if container is running
if docker ps | grep -q dqn-stock-api; then
    echo "🛑 Stopping existing container..."
    docker stop dqn-stock-api
    docker rm dqn-stock-api
fi

# Run container
echo "▶️  Starting container..."
docker run -d \
    --name dqn-stock-api \
    -p 8000:8000 \
    -v $(pwd)/rl-enhanced-agentic-investment:/app/rl-enhanced-agentic-investment \
    -v ~/Downloads:/app/models \
    --restart unless-stopped \
    dqn-stock-api:latest

echo "✅ Deployment complete!"
echo ""
echo "API available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""
echo "To view logs: docker logs -f dqn-stock-api"
echo "To stop: docker stop dqn-stock-api"


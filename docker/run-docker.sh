#!/bin/bash

docker build docker/ --tag="genesis-agentic-demo:latest"

if [ $? -eq 0 ]; then
  echo "Docker build successful."
else
  echo "Docker build failed. Please check the messages above. Exiting..."
  exit 4
fi

# remove old container if it exists
docker container inspect genesis-agentic-demo &>/dev/null && docker rm -f genesis-agentic-demo

# Run docker container
docker run -p 8000:8000 --name genesis-agentic-demo genesis-agentic-demo:latest

if [ $? -eq 0 ]; then
  echo "Success! genesis-agentic simple agent is running."
  echo "Go to http://localhost:8001 to access the genesis-agentic simple agent."
else
  echo "Genesis-agentic container failed to start. Please check the messages above."
fi
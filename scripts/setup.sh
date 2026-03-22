#!/bin/bash
set -e

# One-time server setup script
# Run as root on a fresh Ubuntu server

# Install Docker
curl -fsSL https://get.docker.com | sh
usermod -aG docker "$USER"

# Required for Elasticsearch
sysctl -w vm.max_map_count=262144
echo "vm.max_map_count=262144" >> /etc/sysctl.conf

# Clone the repo
git clone https://github.com/semyonmynko/mlops-spring-2026.git /opt/mlops-spring-2026
cd /opt/mlops-spring-2026

cp .env.example .env
echo ""
echo "Edit .env and fill in your credentials, then continue."
echo "Press Enter when ready..."
read -r

# Start databases first, wait for them to be ready
docker compose up -d mongo elasticsearch redis
echo "Waiting for DBs to start..."
sleep 30

# Start ClearML server
docker compose up -d apiserver fileserver webserver async_delete
sleep 15

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "========================================"
echo "ClearML UI: http://$IP:8080"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Open the UI, log in (admin / clearml1234)"
echo "  2. Settings → Workspace → Create new credentials"
echo "  3. Copy access_key and secret_key into .env"
echo "  4. Run: clearml-serving create --name 'Sentiment Serving'"
echo "     Copy the service ID into .env as CLEARML_SERVING_TASK_ID"
echo "  5. Run: docker compose up -d serving ui"
echo ""
echo "Serving will be at: http://$IP:8082/serve/sentiment"
echo "UI will be at:      http://$IP:8501"

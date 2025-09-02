#!/usr/bin/env bash
set -e

# Start all microservices using docker compose
docker compose -f infra/docker-compose.yml up -d

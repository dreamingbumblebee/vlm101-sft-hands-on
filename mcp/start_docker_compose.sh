#!/bin/bash

# 기본설정 (Open WebUI만 실행)
# docker compose up -d

# Open WebUI & MCP Proxy Server & MCP Servers 실행
docker compose --profile tools up -d --build
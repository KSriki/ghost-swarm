#!/bin/bash
# AgentGateway Diagnostic Script

echo "🔍 Ghost Swarm AgentGateway Diagnostics"
echo "========================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Container running?
echo "1️⃣  Checking if AgentGateway container is running..."
if docker ps | grep -q ghost-agentgateway; then
    echo -e "${GREEN}✅ Container is running${NC}"
    docker ps | grep ghost-agentgateway | awk '{print "   Container: "$2" (Status: "$7")"}'
else
    echo -e "${RED}❌ Container is NOT running${NC}"
    echo "   Run: docker compose -f docker-compose-AGENT-MESH.yml up -d agentgateway"
    exit 1
fi
echo ""

# Check 2: Port mappings
echo "2️⃣  Checking port mappings..."
PORTS=$(docker port ghost-agentgateway 2>/dev/null)
if [ -n "$PORTS" ]; then
    echo -e "${GREEN}✅ Ports are mapped:${NC}"
    echo "$PORTS" | sed 's/^/   /'
else
    echo -e "${RED}❌ No ports mapped${NC}"
fi
echo ""

# Check 3: Health endpoint
echo "3️⃣  Checking health endpoint (port 3000)..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/health 2>/dev/null)
if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ Health endpoint responding (HTTP $HTTP_CODE)${NC}"
    HEALTH=$(curl -s http://localhost:3000/health 2>/dev/null)
    echo "   Response: $HEALTH"
elif [ "$HTTP_CODE" = "406" ]; then
    echo -e "${YELLOW}⚠️  Port 3000 responding but expects SSE headers (HTTP $HTTP_CODE)${NC}"
    echo "   This is normal - the gateway is running!"
else
    echo -e "${RED}❌ Health endpoint not responding (HTTP $HTTP_CODE)${NC}"
fi
echo ""

# Check 4: Port 15000 (UI)
echo "4️⃣  Checking UI port (15000)..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15000 2>/dev/null)
if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ UI is accessible at http://localhost:15000${NC}"
elif [ "$HTTP_CODE" = "000" ]; then
    echo -e "${RED}❌ Port 15000 is not reachable${NC}"
    echo "   The UI might not be included in this AgentGateway version"
else
    echo -e "${YELLOW}⚠️  Port 15000 responding with HTTP $HTTP_CODE${NC}"
fi
echo ""

# Check 5: Test inference endpoint
echo "5️⃣  Testing LLM inference endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "test"}], "max_tokens": 5}' 2>/dev/null)

if echo "$RESPONSE" | grep -q "error\|Error"; then
    echo -e "${YELLOW}⚠️  Endpoint responded with error:${NC}"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
elif echo "$RESPONSE" | grep -q "content\|response"; then
    echo -e "${GREEN}✅ Inference endpoint working!${NC}"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null | head -20 || echo "$RESPONSE"
else
    echo -e "${RED}❌ Unexpected response from inference endpoint${NC}"
    echo "   Response: $RESPONSE"
fi
echo ""

# Check 6: Container logs
echo "6️⃣  Recent container logs (last 10 lines)..."
docker logs ghost-agentgateway --tail 10 2>&1 | sed 's/^/   /'
echo ""

# Check 7: Agent connectivity
echo "7️⃣  Checking if agents are connecting to gateway..."
ORCH_LOGS=$(docker logs ghost-orchestrator 2>&1 | grep -i "gateway_client_initialized" | tail -1)
if [ -n "$ORCH_LOGS" ]; then
    echo -e "${GREEN}✅ Orchestrator connected to gateway${NC}"
    echo "$ORCH_LOGS" | sed 's/^/   /'
else
    echo -e "${YELLOW}⚠️  No gateway connection logs from orchestrator${NC}"
fi

WORKER_LOGS=$(docker logs ghost-worker-1 2>&1 | grep -i "gateway_client_initialized" | tail -1)
if [ -n "$WORKER_LOGS" ]; then
    echo -e "${GREEN}✅ Workers connected to gateway${NC}"
    echo "$WORKER_LOGS" | sed 's/^/   /'
else
    echo -e "${YELLOW}⚠️  No gateway connection logs from workers${NC}"
fi
echo ""

# Summary
echo "========================================"
echo "📊 Summary"
echo "========================================"
echo ""
echo "Gateway Status: Gateway is running ✅"
echo "API Endpoint:   http://localhost:3000"
echo ""

if [ "$HTTP_CODE" = "000" ]; then
    echo -e "${YELLOW}⚠️  UI Port 15000 not accessible${NC}"
    echo ""
    echo "The built-in UI may not be available in this AgentGateway version."
    echo ""
    echo "📊 Alternative Monitoring Options:"
    echo "   1. Use container logs: docker logs -f ghost-agentgateway"
    echo "   2. Use metrics endpoint: curl http://localhost:3000/metrics"
    echo "   3. Add Grafana for visualization (recommended)"
    echo ""
    echo "See: GATEWAY_ACCESS_TROUBLESHOOTING.md for details"
else
    echo -e "${GREEN}✅ UI should be accessible at http://localhost:15000${NC}"
fi

echo ""
echo "🧪 Quick Test Commands:"
echo "   # Test inference"
echo "   curl -X POST http://localhost:3000/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 20}'"
echo ""
echo "   # Watch logs"
echo "   docker logs -f ghost-agentgateway"
echo ""
echo "   # Check metrics"
echo "   curl http://localhost:3000/metrics"
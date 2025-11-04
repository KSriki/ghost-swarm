#!/bin/bash
# Ghost Swarm Service Testing Script (FIXED - no early exit)

COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m'

echo -e "${COLOR_BLUE}========================================${COLOR_NC}"
echo -e "${COLOR_BLUE}Ghost Swarm Service Testing Suite${COLOR_NC}"
echo -e "${COLOR_BLUE}========================================${COLOR_NC}"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

test_pass() {
    echo -e "${COLOR_GREEN}✓ PASS${COLOR_NC}: $1"
    ((PASSED++))
}

test_fail() {
    echo -e "${COLOR_RED}✗ FAIL${COLOR_NC}: $1"
    ((FAILED++))
}

test_warn() {
    echo -e "${COLOR_YELLOW}⚠ WARN${COLOR_NC}: $1"
    ((WARNINGS++))
}

echo "=== 1. Container Status ==="
echo ""

check_container() {
    local container=$1
    local name=$2
    
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
        status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "unknown")
        if [ "$status" == "running" ]; then
            test_pass "$name is running"
            return 0
        else
            test_fail "$name is $status"
            return 1
        fi
    else
        test_fail "$name not found"
        return 1
    fi
}

check_container "ghost-redis" "Redis"
check_container "ghost-slm-server" "SLM Server (llama.cpp)" || check_container "ghost-slm-ollama" "SLM Server (Ollama)"
check_container "ghost-mcp-filesystem" "MCP Filesystem"
check_container "ghost-mcp-agents" "MCP Agents"
check_container "ghost-orchestrator" "Orchestrator"
check_container "ghost-worker-1" "Worker 1"
check_container "ghost-worker-2" "Worker 2"
check_container "ghost-worker-3" "Worker 3"

echo ""
echo "=== 2. Redis Service ==="
echo ""

if docker exec ghost-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
    test_pass "Redis responds to PING"
    
    docker exec ghost-redis redis-cli SET test_key "hello" >/dev/null 2>&1
    result=$(docker exec ghost-redis redis-cli GET test_key 2>/dev/null)
    if [ "$result" == "hello" ]; then
        test_pass "Redis SET/GET works"
    else
        test_fail "Redis SET/GET failed"
    fi
    docker exec ghost-redis redis-cli DEL test_key >/dev/null 2>&1
else
    test_fail "Redis not responding"
fi

echo ""
echo "=== 3. SLM Server ==="
echo ""

if docker ps --format '{{.Names}}' | grep -q "ghost-slm-server"; then
    echo "Detected: llama.cpp"
    if timeout 5 curl -sf http://localhost:8080/health >/dev/null 2>&1; then
        test_pass "llama.cpp health check OK"
    else
        test_warn "llama.cpp not responding (may be loading model)"
    fi
elif docker ps --format '{{.Names}}' | grep -q "ghost-slm-ollama"; then
    echo "Detected: Ollama"
    if timeout 5 curl -sf http://localhost:11434/ >/dev/null 2>&1; then
        test_pass "Ollama health check OK"
        models=$(docker exec ghost-slm-ollama ollama list 2>/dev/null | tail -n +2)
        if [ -n "$models" ]; then
            test_pass "Ollama has models"
        else
            test_warn "Ollama has no models"
        fi
    else
        test_fail "Ollama not responding"
    fi
else
    test_warn "No SLM server found"
fi

echo ""
echo "=== 4. MCP Servers ==="
echo ""

if docker ps --format '{{.Names}}' | grep -q "ghost-mcp-filesystem"; then
    if docker logs ghost-mcp-filesystem 2>&1 | tail -20 | grep -qi "critical\|fatal"; then
        test_fail "MCP Filesystem has critical errors"
    else
        test_pass "MCP Filesystem OK"
    fi
else
    test_fail "MCP Filesystem not found"
fi

if docker ps --format '{{.Names}}' | grep -q "ghost-mcp-agents"; then
    if docker logs ghost-mcp-agents 2>&1 | tail -20 | grep -qi "critical\|fatal"; then
        test_fail "MCP Agents has critical errors"
    else
        test_pass "MCP Agents OK"
    fi
else
    test_fail "MCP Agents not found"
fi

echo ""
echo "=== 5. Orchestrator ==="
echo ""

if docker ps --format '{{.Names}}' | grep -q "ghost-orchestrator"; then
    if docker logs ghost-orchestrator 2>&1 | grep -q "orchestrator_initialized"; then
        test_pass "Orchestrator initialized"
    else
        test_warn "Orchestrator may not have initialized"
    fi

    if docker logs ghost-orchestrator 2>&1 | grep -q "hybrid_inference"; then
        test_pass "Hybrid inference initialized"
    else
        test_warn "Hybrid inference may not be initialized"
    fi

    if docker logs ghost-orchestrator 2>&1 | grep -qi "executor"; then
        test_pass "ExecutorPool found"
    else
        test_warn "ExecutorPool not found in logs"
    fi
    
    # Check if agent started successfully
    if docker logs ghost-orchestrator 2>&1 | grep -q "agent_started"; then
        test_pass "Orchestrator agent started"
    elif docker logs ghost-orchestrator 2>&1 | grep -q "a2a_server_started"; then
        test_pass "Orchestrator A2A server started"
    fi
else
    test_fail "Orchestrator not found"
fi

echo ""
echo "=== 6. Workers ==="
echo ""

for i in 1 2 3; do
    worker="ghost-worker-$i"
    if docker ps --format '{{.Names}}' | grep -q "^${worker}$"; then
        if docker logs "$worker" 2>&1 | grep -qi "agent_initialized\|started"; then
            test_pass "Worker $i started"
        else
            test_warn "Worker $i may not have started"
        fi
    else
        test_fail "Worker $i not found"
    fi
done

echo ""
echo "=== 7. Service Discovery ==="
echo ""

agent_keys=$(docker exec ghost-redis redis-cli KEYS "agent:*" 2>/dev/null | grep -c "agent:")
if [ "$agent_keys" -gt 0 ]; then
    test_pass "Found $agent_keys agents in Redis"
else
    test_warn "No agents in Redis"
fi

echo ""
echo "=== 8. Environment ==="
echo ""

if docker exec ghost-orchestrator printenv ANTHROPIC_API_KEY >/dev/null 2>&1; then
    api_key=$(docker exec ghost-orchestrator printenv ANTHROPIC_API_KEY 2>/dev/null)
    if [ -n "$api_key" ] && [ "$api_key" != "your-key-here" ]; then
        test_pass "ANTHROPIC_API_KEY is set"
    else
        test_fail "ANTHROPIC_API_KEY not set"
    fi
else
    test_fail "ANTHROPIC_API_KEY not found"
fi

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "${COLOR_GREEN}Passed:${COLOR_NC} $PASSED"
echo -e "${COLOR_RED}Failed:${COLOR_NC} $FAILED"
echo -e "${COLOR_YELLOW}Warnings:${COLOR_NC} $WARNINGS"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${COLOR_GREEN}✓ All critical tests passed!${COLOR_NC}"
    exit 0
else
    echo -e "${COLOR_YELLOW}⚠ Some tests failed${COLOR_NC}"
    echo "Run 'docker-compose logs <service>' for details"
    exit 1
fi
# Ghost Swarm - Agent Mesh Architecture with AgentGateway

## 🎯 Architecture Vision

Based on Solo.io's Agent Mesh pattern, Ghost Swarm implements:

1. **Agent to LLM** - Intelligent routing with guardrails and caching
2. **Agent to Tools (MCP)** - Federated MCP tool registry with security
3. **Agent to Agent (A2A)** - Secure agent mesh with discovery and observability

## 🏗️ Agent Mesh Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Mesh                                │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    AgentGateway                           │   │
│  │                                                            │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │   │
│  │  │ LLM Gateway │  │  MCP Proxy   │  │  A2A Mesh     │  │   │
│  │  │             │  │              │  │                │  │   │
│  │  │ - Routing   │  │ - Federation │  │ - Discovery   │  │   │
│  │  │ - Caching   │  │ - Security   │  │ - Routing     │  │   │
│  │  │ - Guardrails│  │ - Observ.    │  │ - Tracing     │  │   │
│  │  └──────┬──────┘  └──────┬───────┘  └────────┬───────┘  │   │
│  └─────────┼────────────────┼───────────────────┼──────────┘   │
│            │                │                   │              │
│            │                │                   │              │
└────────────┼────────────────┼───────────────────┼──────────────┘
             │                │                   │
             ▼                ▼                   ▼
    ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
    │   Claude    │  │  MCP Servers │  │   Agents        │
    │   SLM       │  │  - FileSystem│  │   - Orchestrator│
    │   OpenAI    │  │  - Database  │  │   - Workers     │
    │   Ollama    │  │  - APIs      │  │   - Specialists │
    └─────────────┘  └──────────────┘  └─────────────────┘
```

## 🎯 Key Benefits

### 1. Unified LLM Access
- **Intelligent routing** based on complexity, cost, latency
- **Caching** for repeated queries
- **Guardrails** for safety and compliance
- **Rate limiting** per user, tenant, agent
- **Failover** across providers (Claude → OpenAI → Ollama)

### 2. MCP Tool Federation
- **Centralized tool registry** - Single catalog of all tools
- **Discovery** - Agents find tools at runtime
- **Security** - Authentication, authorization per tool
- **Version management** - Decoupled tool evolution
- **Observability** - Trace tool calls end-to-end

### 3. A2A Agent Mesh
- **Agent discovery** - Find agents by capability
- **AgentCard publication** - Declare skills and capabilities
- **Secure routing** - mTLS between agents
- **Task orchestration** - Multi-agent workflows
- **Tracing** - End-to-end observability

### 4. Enterprise Ready
- **Zero-trust** - mTLS, SPIFFE identity, fine-grained authz
- **Multi-tenancy** - Isolation per user/tenant
- **Observability** - Unified tracing, metrics, logs
- **Resilience** - Health checks, failover, circuit breakers
- **GitOps** - Declarative configuration

## 📦 Components

### 1. AgentGateway (Data Plane)
- **Image**: `ghcr.io/agentgateway/agentgateway:latest`
- **Ports**:
  - 3000: Main gateway (LLM, MCP, A2A)
  - 15000: Built-in UI
- **Capabilities**:
  - LLM routing with caching and guardrails
  - MCP protocol proxy (stateful, multiplexing)
  - A2A protocol mesh (agent discovery, routing)
  - Semantic observability (OTel)
  - Security (mTLS, authz, SPIFFE)

### 2. MCP Servers
- **Filesystem**: File operations, data access
- **Database**: SQL queries, data retrieval
- **APIs**: REST/GraphQL tool wrappers
- **Custom**: Your domain-specific tools

### 3. Agents
- **Orchestrator**: Task decomposition, agent coordination
- **Workers**: Specialized capabilities (coding, analysis, research)
- **Each agent**:
  - Publishes AgentCard (capabilities)
  - Discovers tools via MCP
  - Communicates via A2A
  - Routes LLM calls through gateway

### 4. LLM Backends
- **Claude API**: Complex reasoning (via gateway)
- **SLM Server**: Simple tasks, local inference (via gateway)
- **OpenAI**: Alternative LLM (via gateway)
- **Ollama**: Self-hosted models (via gateway)

## 🔧 Configuration

### AgentGateway Config

The gateway needs comprehensive configuration for all three interaction patterns:

```yaml
# agentgateway-config.yaml
config:
  logging:
    format: json
    level: info
    fields:
      add:
        service: ghost-swarm-gateway
        environment: ${ENVIRONMENT}

binds:
  # Main gateway port - handles LLM, MCP, and A2A
  - port: 3000
    listeners:
      # =================================================================
      # LLM ROUTING - Agent to LLM Communication
      # =================================================================
      - name: llm-routing
        routes:
          - policies:
              # CORS for web clients
              cors:
                allowOrigins: ["*"]
                allowHeaders: ["*"]
              
              # Rate limiting per agent/user
              rateLimit:
                requestsPerUnit: 100
                unit: MINUTE
              
              # Caching for repeated queries
              cache:
                ttl: 3600
                keyBy: [prompt_hash]
              
              # Guardrails
              guardrails:
                inputValidation: true
                outputFiltering: true
                promptInjectionDetection: true
            
            backends:
              # Primary: Claude API
              - llm:
                  provider: anthropic
                  model: ${LLM_MODEL:-claude-sonnet-4-5-20250929}
                  apiKey: ${ANTHROPIC_API_KEY}
                  priority: 1
                  retries: 3
              
              # Secondary: SLM (local, free)
              - llm:
                  provider: openai_compatible
                  baseUrl: http://slm-server:8080/v1
                  model: ${SLM_MODEL:-phi-3-mini}
                  priority: 2
                  costMultiplier: 0.0  # Free!
              
              # Tertiary: OpenAI (fallback)
              - llm:
                  provider: openai
                  apiKey: ${OPENAI_API_KEY}
                  priority: 3
      
      # =================================================================
      # MCP FEDERATION - Agent to Tool Communication
      # =================================================================
      - name: mcp-federation
        routes:
          - policies:
              cors:
                allowOrigins: ["*"]
                allowHeaders:
                  - mcp-protocol-version
                  - content-type
                  - authorization
              
              # Tool authentication
              auth:
                type: composite
                userIdentity: jwt
                agentIdentity: spiffe
              
              # Fine-grained authz
              authorization:
                policy: opa
                rulesPath: /config/policies/mcp-authz.rego
              
              # Semantic observability
              observability:
                tracing: true
                semanticContext: true
                toolMetrics: true
            
            backends:
              # Filesystem MCP Server
              - mcp:
                  targets:
                    - name: filesystem
                      description: "File system operations"
                      stdio:
                        cmd: python
                        args:
                          - "-m"
                          - "mcp_server.servers.filesystem"
                        env:
                          MCP_ALLOWED_DIRECTORIES: "/app/data"
                          MCP_READONLY: "false"
                      health:
                        enabled: true
                        interval: 30s
                      metadata:
                        category: "filesystem"
                        capabilities:
                          - "read_file"
                          - "write_file"
                          - "list_directory"
              
              # Database MCP Server (future)
              - mcp:
                  targets:
                    - name: database
                      description: "Database queries and operations"
                      stdio:
                        cmd: python
                        args:
                          - "-m"
                          - "mcp_server.servers.database"
                        env:
                          DB_URL: ${DATABASE_URL}
                      metadata:
                        category: "database"
                        capabilities:
                          - "query"
                          - "insert"
                          - "update"
              
              # Custom API MCP Server (future)
              - mcp:
                  targets:
                    - name: api-wrapper
                      description: "REST API wrapper"
                      stdio:
                        cmd: python
                        args:
                          - "-m"
                          - "mcp_server.servers.api_wrapper"
                      metadata:
                        category: "api"
      
      # =================================================================
      # A2A MESH - Agent to Agent Communication
      # =================================================================
      - name: a2a-mesh
        routes:
          - policies:
              # Mark as A2A traffic
              a2a: {}
              
              cors:
                allowOrigins: ["*"]
                allowHeaders: ["*"]
              
              # Agent identity and auth
              auth:
                type: spiffe
                validateAgentCard: true
              
              # Agent discovery
              discovery:
                enabled: true
                registryType: redis
                registryUrl: redis://redis:6379
              
              # Semantic routing
              routing:
                type: semantic
                matchOnCapabilities: true
              
              # Observability
              observability:
                tracing: true
                agentMetrics: true
                taskTracking: true
            
            backends:
              # Orchestrator
              - a2a:
                  target:
                    host: ghost-orchestrator:8765
                    agentCard:
                      id: orchestrator
                      capabilities:
                        - task_decomposition
                        - agent_coordination
                        - workflow_management
              
              # Workers (dynamic discovery preferred)
              - a2a:
                  target:
                    host: ghost-worker-1:8766
                    agentCard:
                      id: worker-1
                      capabilities:
                        - llm_inference
                        - data_processing
                        - analysis
              
              - a2a:
                  target:
                    host: ghost-worker-2:8766
                    agentCard:
                      id: worker-2
                      capabilities:
                        - llm_inference
                        - data_processing
                        - analysis
              
              - a2a:
                  target:
                    host: ghost-worker-3:8766
                    agentCard:
                      id: worker-3
                      capabilities:
                        - llm_inference
                        - data_processing
                        - analysis

# Security policies
policies:
  # mTLS for all inter-agent communication
  mtls:
    enabled: true
    mode: STRICT
    certificateAuthority: /config/certs/ca.pem
  
  # SPIFFE identity
  spiffe:
    enabled: true
    trustDomain: ghost-swarm.local
  
  # Zero-trust
  zeroTrust:
    enabled: true
    defaultAction: DENY
    allowedPaths: /config/policies/allowed-paths.yaml

# Observability
observability:
  tracing:
    enabled: true
    backend: jaeger
    endpoint: http://jaeger:14268/api/traces
    sampleRate: 1.0
  
  metrics:
    enabled: true
    backend: prometheus
    port: 9464
  
  logging:
    level: info
    format: json
    includeSemanticContext: true
```

## 🎯 Agent Changes

Agents now interact with AgentGateway for ALL three patterns:

### 1. LLM Calls → Gateway

```python
# Before: Direct client
result = await self.llm_client.messages.create(...)

# After: Via gateway
result = await self.gateway_client.chat(
    messages=[{"role": "user", "content": prompt}],
    complexity="simple",  # Gateway routes to appropriate LLM
)
```

### 2. Tool Calls → Gateway (MCP)

```python
# Before: Direct MCP client
tools = await self.mcp_client.list_tools()

# After: Via gateway
tools = await self.gateway_client.list_mcp_tools()
result = await self.gateway_client.call_mcp_tool(
    tool="filesystem.read_file",
    arguments={"path": "/data/file.txt"}
)
```

### 3. Agent Calls → Gateway (A2A)

```python
# Before: Direct WebSocket
await self.peer_client.send_message(...)

# After: Via gateway
result = await self.gateway_client.send_agent_task(
    agent_capability="data_processing",
    task=task_request
)
```

## 📊 Benefits Over Direct Approach

| Feature | Direct | Agent Mesh | Improvement |
|---------|--------|------------|-------------|
| **LLM Routing** | Manual | Intelligent | Automatic failover |
| **Caching** | None | Built-in | 50-80% cost reduction |
| **Tool Discovery** | Static | Dynamic | Runtime flexibility |
| **Agent Discovery** | Manual | Automatic | True mesh |
| **Security** | Custom | Zero-trust | Enterprise-grade |
| **Observability** | Basic | Semantic | Full traceability |
| **Guardrails** | None | Built-in | Safety & compliance |
| **Multi-tenancy** | None | Built-in | Isolation |

## 🚀 Deployment

### Architecture Layers

```
┌────────────────────────────────────────────────────────┐
│  Layer 1: Agent Mesh (AgentGateway)                   │
│  - Unified entry point                                 │
│  - LLM routing, caching, guardrails                    │
│  - MCP federation                                      │
│  - A2A mesh                                            │
└────────────────────────────────────────────────────────┘
                           │
┌────────────────────────────────────────────────────────┐
│  Layer 2: Agents                                       │
│  - Orchestrator                                        │
│  - Workers (1-N)                                       │
│  - Specialists (future)                                │
└────────────────────────────────────────────────────────┘
                           │
┌────────────────────────────────────────────────────────┐
│  Layer 3: Backends                                     │
│  - LLMs (Claude, SLM, OpenAI)                         │
│  - MCP Servers (filesystem, db, apis)                 │
│  - Infrastructure (Redis, monitoring)                  │
└────────────────────────────────────────────────────────┘
```

## 🎓 Next Steps

1. **Review** this architecture document
2. **Decide** on MCP servers needed (filesystem, database, apis?)
3. **Configure** AgentGateway for your use case
4. **Update** agents to use gateway for all interactions
5. **Deploy** with full observability
6. **Monitor** metrics and optimize

---

**This is the full Agent Mesh architecture you want!** 🎯
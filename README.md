# Claude Code 中文插件市场

> **⚡ 为中国用户本地化的 Claude Code 插件市场** — 基于 Opus 4.5, Sonnet 4.5 & Haiku 4.5 优化

> **🎯 Agent Skills 支持** — 47 个专业化技能通过渐进式披露扩展 Claude 的能力

一个全面的生产就绪系统，包含 **91 个专业化 AI agents**、**15 个多代理工作流编排器**、**47 个 agent skills** 和 **45 个开发工具**，组织成 **65 个专注的、单一目的的插件**，专为 [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) 打造。

> **注意**：这是 [wshobson/agents](https://github.com/wshobson/agents) 的中文本地化版本，专注于为中国开发者提供优化的工具集。原始英文插件在 `plugins/` 目录中，中文插件在 `plugins-zh/` 目录中。

## 概述

本仓库为现代软件开发的智能自动化和多代理编排提供了一切所需：

- **65 个专注插件** - 优化的单一目的插件，最小化 token 使用，最大化可组合性
- **91 个专业 Agents** - 架构、语言、基础设施、质量、数据/AI、文档、业务运营和 SEO 领域的专家
- **47 个 Agent Skills** - 模块化知识包，通过渐进式披露提供专业专业知识
- **15 个工作流编排器** - 多代理协调系统，用于全栈开发、安全加固、ML 流程和事件响应等复杂操作
- **45 个开发工具** - 优化的实用程序，包括项目脚手架、安全扫描、测试自动化和基础设施设置

### 核心特性

- **细粒度插件架构**：65 个专注插件，优化 token 使用
- **全面的工具集**：45 个开发工具，包括测试生成、脚手架和安全扫描
- **100% Agent 覆盖**：所有插件都包含专业 agents
- **Agent Skills**：47 个专业技能，遵循渐进式披露和 token 效率原则
- **清晰的组织**：23 个类别，每个类别 1-6 个插件，易于发现
- **高效设计**：每个插件平均 3.4 个组件（遵循 Anthropic 的 2-8 模式）

### 工作原理

每个插件都是完全隔离的，拥有自己的 agents、commands 和 skills：

- **只安装你需要的** - 每个插件只加载其特定的 agents、commands 和 skills
- **最小 token 使用** - 不会加载不必要的资源到上下文中
- **混合搭配** - 组合多个插件用于复杂工作流
- **清晰的边界** - 每个插件都有单一的、专注的目的
- **渐进式披露** - Skills 只在激活时加载知识

**示例**：安装 `python-development` 会加载 3 个 Python agents、1 个脚手架工具，并使 5 个 skills 可用（约 300 tokens），而不是整个市场。

### 中文插件

当前已本地化的插件：

1. **python-development** - Python 3.12+ 现代开发，包含 3 个 agents 和 5 个 skills
2. **mcp_builder** - MCP 服务器开发指南和最佳实践
3. **multi-platform-apps** - React 19、Next.js 15 跨平台应用开发
4. **myhron** - 专属工具集和命令

## 快速开始

### 步骤 1：添加市场

将此市场添加到 Claude Code：

```bash
/plugin marketplace add moyueheng/cc-plugin-zh
```

这使得所有 65 个插件可用于安装，但**不会将任何 agents 或工具加载到你的上下文中**。

### 步骤 2：安装插件

浏览可用插件：

```bash
/plugin
```

安装你需要的插件：

```bash
# 中文插件（推荐）
/plugin install python-development          # Python，5 个专业技能
/plugin install mcp_builder                 # MCP 服务器开发
/plugin install multi-platform-apps         # 跨平台应用开发

# 英文插件
/plugin install javascript-typescript       # JS/TS，4 个专业技能
/plugin install backend-development         # Backend APIs，3 个架构技能

# 基础设施和运维
/plugin install kubernetes-operations       # K8s，4 个部署技能
/plugin install cloud-infrastructure        # AWS/Azure/GCP，4 个云技能

# 安全和质量
/plugin install security-scanning           # SAST，安全技能
/plugin install code-review-ai             # AI 驱动的代码审查

# 全栈编排
/plugin install full-stack-orchestration   # 多代理工作流
```

每个已安装的插件**仅将其特定的 agents、commands 和 skills** 加载到 Claude 的上下文中。

## 文档

### 核心指南

- **[插件参考](docs/plugins.md)** - 所有 65 个插件的完整目录
- **[Agent 参考](docs/agents.md)** - 按类别组织的所有 91 个 agents
- **[Agent Skills](docs/agent-skills.md)** - 47 个专业技能，支持渐进式披露
- **[使用指南](docs/usage.md)** - 命令、工作流和最佳实践
- **[架构](docs/architecture.md)** - 设计原则和模式
- **[CLAUDE.md](CLAUDE.md)** - 中文插件开发指南和规范

### 快速链接

- [安装](#快速开始) - 2 步快速开始
- [核心插件](docs/plugins.md#quick-start---essential-plugins) - 立即提高生产力的顶级插件
- [命令参考](docs/usage.md#command-reference-by-category) - 按类别组织的所有斜杠命令
- [多代理工作流](docs/usage.md#multi-agent-workflow-examples) - 预配置的编排示例
- [模型配置](docs/agents.md#model-configuration) - Haiku/Sonnet 混合编排

## 最新动态

### Agent Skills（14 个插件中的 47 个技能）

遵循 Anthropic 渐进式披露架构的专业知识包：

**语言开发：**
- **Python**（5 个技能）：async 模式、测试、打包、性能、UV 包管理器
- **JavaScript/TypeScript**（4 个技能）：高级类型、Node.js 模式、测试、现代 ES6+

**基础设施和 DevOps：**
- **Kubernetes**（4 个技能）：manifests、Helm charts、GitOps、安全策略
- **云基础设施**（4 个技能）：Terraform、多云、混合网络、成本优化
- **CI/CD**（4 个技能）：流水线设计、GitHub Actions、GitLab CI、密钥管理

**开发和架构：**
- **Backend**（3 个技能）：API 设计、架构模式、微服务
- **LLM 应用**（4 个技能）：LangChain、提示工程、RAG、评估

**区块链和 Web3**（4 个技能）：DeFi 协议、NFT 标准、Solidity 安全、Web3 测试

**以及更多**：框架迁移、可观测性、支付处理、ML 运维、安全扫描

[→ 查看完整的技能文档](docs/agent-skills.md)

### 三层模型策略

为最优性能和成本进行战略性模型分配：

| Tier | Model | Agents | Use Case |
|------|-------|--------|----------|
| **Tier 1** | Opus 4.5 | 42 | 关键架构、安全、所有代码审查、生产编码（语言专家、框架） |
| **Tier 2** | Inherit | 42 | 复杂任务 - 用户选择模型（AI/ML、后端、前端/移动、专业） |
| **Tier 3** | Sonnet | 51 | 智能支持（文档、测试、调试、网络、API 文档、DX、遗留、支付） |
| **Tier 4** | Haiku | 18 | 快速操作任务（SEO、部署、简单文档、销售、内容、搜索） |

**为什么关键 Agents 使用 Opus 4.5？**
- SWE-bench 上 80.9% 的分数（行业领先）
- 复杂任务减少 65% 的 token
- 最适合架构决策和安全审计

**Tier 2 灵活性（`inherit`）：**
标记为 `inherit` 的 agents 使用你会话的默认模型，让你平衡成本和能力：
- 通过 `claude --model opus` 或 `claude --model sonnet` 在启动会话时设置
- 如果未指定默认值，则回退到 Sonnet 4.5
- 非常适合想要成本控制的前端/移动开发者
- AI/ML 工程师可以选择 Opus 进行复杂的模型工作

**成本考虑：**
- **Opus 4.5**：每百万输入/输出 token $5/$25 - 关键工作的 premium
- **Sonnet 4.5**：每百万 token $3/$15 - 平衡的性能/成本
- **Haiku 4.5**：每百万 token $1/$5 - 快速、成本效益高的操作
- Opus 在复杂任务上减少 65% 的 token 通常抵消更高的费率
- 使用 `inherit` 层级来控制高量用例的成本

编排模式组合模型以提高效率：
```
Opus（架构）→ Sonnet（开发）→ Haiku（部署）
```

[→ 查看模型配置详情](docs/agents.md#model-configuration)

## 流行用例

### 全栈功能开发

```bash
/full-stack-orchestration:full-stack-feature "user authentication with OAuth2"
```

协调 7+ 个 agents：backend-architect → database-architect → frontend-developer → test-automator → security-auditor → deployment-engineer → observability-engineer

[→ 查看所有工作流示例](docs/usage.md#multi-agent-workflow-examples)

### 安全加固

```bash
/security-scanning:security-hardening --level comprehensive
```

多代理安全评估，包括 SAST、依赖扫描和代码审查。

### 使用现代工具进行 Python 开发

```bash
/python-development:python-scaffold fastapi-microservice
```

创建生产就绪的 FastAPI 项目，具有 async 模式，激活技能：
- `async-python-patterns` - AsyncIO 和并发
- `python-testing-patterns` - pytest 和 fixtures
- `uv-package-manager` - 快速依赖管理

### Kubernetes 部署

```bash
# 自动激活 k8s 技能
"Create production Kubernetes deployment with Helm chart and GitOps"
```

使用 kubernetes-architect agent 和 4 个专业技能进行生产级配置。

[→ 查看完整使用指南](docs/usage.md)

## 插件类别

**23 个类别，65 个插件：**

- 🎨 **开发**（4）- debugging、backend、frontend、multi-platform
- 📚 **文档**（3）- 代码文档、API 规范、图表、C4 架构
- 🔄 **工作流**（3）- git、full-stack、TDD
- ✅ **测试**（2）- 单元测试、TDD 工作流
- 🔍 **质量**（3）- 代码审查、综合审查、性能
- 🤖 **AI & ML**（4）- LLM 应用、agent 编排、context、MLOps
- 📊 **数据**（2）- 数据工程、数据验证
- 🗄️ **数据库**（2）- 数据库设计、迁移
- 🚨 **运维**（4）- 事件响应、诊断、分布式调试、可观测性
- ⚡ **性能**（2）- 应用性能、数据库/云优化
- ☁️ **基础设施**（5）- 部署、验证、Kubernetes、云、CI/CD
- 🔒 **安全**（4）- 扫描、合规、后端/API、前端/移动
- 💻 **语言**（7）- Python、JS/TS、系统、JVM、脚本、函数式、嵌入式
- 🔗 **区块链**（1）- 智能合约、DeFi、Web3
- 💰 **金融**（1）- 量化交易、风险管理
- 💳 **支付**（1）- Stripe、PayPal、计费
- 🎮 **游戏**（1）- Unity、Minecraft 插件
- 📢 **营销**（4）- SEO 内容、技术 SEO、SEO 分析、内容营销
- 💼 **商业**（3）- 分析、HR/法律、客户/销售
- 以及更多...

[→ 查看完整的插件目录](docs/plugins.md)

## 架构亮点

### 细粒度设计

- **单一职责** - 每个插件只做一件事
- **最小 token 使用** - 每个插件平均 3.4 个组件
- **可组合** - 混合搭配用于复杂工作流
- **100% 覆盖** - 所有 91 个 agents 在插件间可访问

### 渐进式披露（Skills）

三层架构以提高 token 效率：
1. **元数据** - 名称和激活条件（始终加载）
2. **指令** - 核心指导（激活时加载）
3. **资源** - 示例和模板（按需加载）

### 仓库结构

```
claude-agents/
├── .claude-plugin/
│   └── marketplace.json          # 65 个插件
├── plugins/                      # 英文插件（原始）
│   ├── python-development/
│   │   ├── agents/               # 3 个 Python 专家
│   │   ├── commands/             # 脚手架工具
│   │   └── skills/               # 5 个专业技能
│   └── ... (63 more plugins)
├── plugins-zh/                   # 中文插件（本地化）
│   ├── python-development/
│   │   ├── agents/               # 3 个 Python 专家（中文）
│   │   ├── commands/             # 脚手架工具（中文）
│   │   └── skills/               # 5 个专业技能（中文）
│   └── ... (3 more plugins)
├── docs/                          # 全面的文档
└── README.md                      # 本文件
```

[→ 查看架构详情](docs/architecture.md)

## 贡献

要添加新的 agents、skills 或 commands：

1. 在 `plugins-zh/` 中识别或创建适当的插件目录
2. 在适当的子目录中创建 `.md` 文件：
   - `agents/` - 用于专业 agents
   - `commands/` - 用于工具和工作流
   - `skills/` - 用于模块化知识包
3. 遵循命名约定（小写、连字符分隔）
4. 编写清晰的激活条件和全面的内容
5. 在 `.claude-plugin/marketplace.json` 中更新插件定义

有关详细指南，请参阅 [架构文档](docs/architecture.md) 和 [CLAUDE.md](CLAUDE.md)。

## 资源

### 文档
- [Claude Code 文档](https://docs.claude.com/en/docs/claude-code/overview)
- [插件指南](https://docs.claude.com/en/docs/claude-code/plugins)
- [子代理指南](https://docs.claude.com/en/docs/claude-code/sub-agents)
- [Agent Skills 指南](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview)
- [斜杠命令参考](https://docs.claude.com/en/docs/claude-code/slash-commands)

### 本仓库
- [插件参考](docs/plugins.md)
- [Agent 参考](docs/agents.md)
- [Agent Skills 指南](docs/agent-skills.md)
- [使用指南](docs/usage.md)
- [架构](docs/architecture.md)
- [CLAUDE.md](CLAUDE.md) - 中文插件开发指南

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=moyueheng/cc-plugin-zh&type=date&legend=top-left)](https://www.star-history.com/#moyueheng/cc-plugin-zh&type=date&legend=top-left)

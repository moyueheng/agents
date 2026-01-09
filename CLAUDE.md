# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

这是一个 Claude Code 插件市场，为中文用户提供本地化、生产级的工作流编排系统。本仓库从原始 `plugins/` 目录（65 个英文插件）中提取并翻译核心插件到 `plugins-zh/`，专注于为中国开发者提供优化的工具集。

**重要原则：**
- **`plugins/` 目录只作为参考和追踪来源，不应被修改**
- **所有变更和新增内容都在 `plugins-zh/` 目录中进行**
- 中文插件基于英文插件进行本地化和优化

## Architecture

### Marketplace Configuration

`.claude-plugin/marketplace.json` 定义了所有可用的中文插件。每个插件必须在此注册才能被安装。

插件配置结构：
```json
{
  "name": "python-development",
  "source": "./plugins-zh/python-development",
  "description": "中文描述",
  "category": "languages",
  "strict": false,
  "agents": ["./agents/python-pro.md", "./agents/python-pro_zh.md"],
  "commands": ["./commands/python-scaffold.md"],
  "skills": ["./skills/async-python-patterns", "./skills/python-testing-patterns"]
}
```

### Plugin Directory Structure

每个插件遵循标准的三层结构：

```
plugins-zh/<plugin-name>/
├── agents/           # 专业化 AI agent 定义（.md 文件）
├── commands/         # 斜杠命令工具和工作流（.md 文件）
└── skills/           # 渐进式知识包（SKILL.md 文件）
```

**关键概念：**
- **Agents** - 领域专家，使用特定的模型（opus、sonnet、haiku 或 inherit）
- **Commands** - 可通过 `/plugin-name:command` 直接调用的工具
- **Skills** - 模块化知识包，只在激活时加载内容以节省 token

### Agent Frontmatter

所有 agent 文件使用 YAML frontmatter 定义元数据：

```yaml
---
name: python-pro
description: 精通 Python 3.12+ 的现代特性...
model: opus  # 或 sonnet、haiku、inherit
---
```

`model` 字段控制 agent 使用的模型层级：
- `opus` - 最强能力，用于关键架构、安全审查、代码评审
- `sonnet` - 平衡性能和成本，用于开发、调试、文档
- `haiku` - 快速操作，用于部署、简单文档、SEO
- `inherit` - 继承用户会话的默认模型

### Skill Structure

Skills 遵循 Anthropic 的渐进式披露架构：

```
skills/<skill-name>/
├── SKILL.md          # 元数据和激活条件（始终加载）
└── reference/        # 示例和模板（按需加载）
    ├── example1.md
    └── example2.md
```

SKILL.md 文件格式：
```yaml
---
name: async-python-patterns
description: Master Python asyncio...
---
```

## Translation Workflow

当需要从 `plugins/` 添加新插件到 `plugins-zh/` 时：

1. 在 `plugins-zh/` 创建对应的插件目录
2. 翻译 agent 文件，保持技术术语英文（如 "async/await"、"FastAPI"）
3. 翻译 skills，保留代码示例不变
4. 翻译 commands 的描述和说明
5. 在 `.claude-plugin/marketplace.json` 中注册新插件

**翻译原则：**
- 技术术语保持英文（如 "async/await"、"middleware"、"endpoint"）
- 代码注释保持英文或双语
- 描述性内容使用中文
- 保留代码示例和配置文件原样

## Current Plugins

当前已本地化的插件：

1. **python-development** - Python 3.12+ 现代开发，包含 3 个 agents 和 5 个 skills
2. **mcp_builder** - MCP 服务器开发指南和最佳实践
3. **multi-platform-apps** - React 19、Next.js 15 跨平台应用开发
4. **myhron** - 专属工具集和命令

## Adding New Content

### 添加新的 Agent

1. 在对应插件的 `agents/` 目录创建 `.md` 文件
2. 包含 YAML frontmatter（name、description、model）
3. 在 marketplace.json 的 `agents` 数组中注册

### 添加新的 Command

1. 在 `commands/` 目录创建 `.md` 文件
2. 文件内容即为命令的说明文档
3. 在 marketplace.json 的 `commands` 数组中注册

### 添加新的 Skill

1. 在 `skills/` 创建子目录
2. 创建 `SKILL.md` 包含元数据和激活条件
3. 可选：在 `reference/` 添加示例和模板
4. 在 marketplace.json 的 `skills` 数组中注册

## Naming Conventions

- **插件目录**：小写，连字符分隔（`python-development`）
- **Agent 文件**：小写，连字符分隔（`python-pro.md`、`backend-architect.md`）
- **Agent 名称**：小写，连字符分隔（`name: python-pro`）
- **Skill 目录**：小写，连字符分隔（`async-python-patterns`）
- **Command 文件**：小写，连字符分隔（`python-scaffold.md`）

## File Paths in marketplace.json

所有路径必须是相对于插件 `source` 目录的相对路径：

```json
{
  "source": "./plugins-zh/python-development",
  "agents": [
    "./agents/python-pro.md",      // 相对于 plugins-zh/python-development/
    "./agents/python-pro_zh.md"
  ],
  "skills": [
    "./skills/async-python-patterns"  // 指向 skills/async-python-patterns/ 目录
  ]
}
```

## Documentation

详细文档位于 `docs/` 目录：
- `plugins.md` - 所有插件的完整目录
- `agents.md` - Agent 参考和模型配置
- `agent-skills.md` - Skills 参考和渐进式披露
- `usage.md` - 命令、工作流和最佳实践
- `architecture.md` - 设计原则和模式

## Git Workflow

- 主开发分支：`moyueheng`
- 远程仓库：`github.com/moyueheng/agents.git`
- 提交规范：使用中文描述，保留英文 type（feat、fix、docs 等）

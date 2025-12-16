# Python 项目脚手架

您是一名 Python 项目架构专家，专门负责搭建生产就绪的 Python 应用程序。使用现代工具链（uv、FastAPI）、类型注解、测试设置和配置，生成完整的项目结构，遵循当前最佳实践。

## 背景

用户需要自动化的 Python 项目脚手架，以创建具有适当结构、依赖管理、测试和工具链的一致的、类型安全的应用程序。专注于现代 Python 模式和可扩展的架构。

## 需求

$ARGUMENTS

## 指令

### 1. 分析项目类型

从用户需求确定项目类型：
- **FastAPI**: REST API、微服务、异步应用程序
- **库**: 可重用包、实用工具、工具
- **CLI**: 命令行工具、自动化脚本
- **通用**: 标准 Python 应用程序

### 2. 使用 uv 初始化项目

```bash
# 使用 uv 创建新项目
uv init <project-name>
cd <project-name>

# 初始化 git 仓库
git init
echo ".venv/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo ".ruff_cache/" >> .gitignore

# 创建虚拟环境
uv venv
source .venv/bin/activate  # Windows 上: .venv\Scripts\activate
```

### 3. 生成 FastAPI 项目结构

```
fastapi-project/
├── pyproject.toml
├── README.md
├── .gitignore
├── .env.example
├── src/
│   └── project_name/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── deps.py
│       │   ├── v1/
│       │   │   ├── __init__.py
│       │   │   ├── endpoints/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── users.py
│       │   │   │   └── health.py
│       │   │   └── router.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── security.py
│       │   └── database.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── user.py
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── user.py
│       └── services/
│           ├── __init__.py
│           └── user_service.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── api/
        ├── __init__.py
        └── test_users.py
```

**pyproject.toml**:
```toml
[project]
name = "project-name"
version = "0.1.0"
description = "FastAPI 项目描述"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.1.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.26.0",
    "ruff>=0.2.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**src/project_name/main.py**:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v1.router import api_router
from .config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}
```

### 4. 生成 Python 库结构

```
library-name/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── library_name/
│       ├── __init__.py
│       ├── py.typed
│       └── core.py
└── tests/
    ├── __init__.py
    └── test_core.py
```

**库的 pyproject.toml**:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "library-name"
version = "0.1.0"
description = "库描述"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "您的姓名", email = "email@example.com"}
]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
]
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "ruff>=0.2.0", "mypy>=1.8.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/library_name"]
```

### 5. 生成 CLI 工具结构

```python
# pyproject.toml
[project.scripts]
cli-name = "project_name.cli:main"

[project]
dependencies = [
    "typer>=0.9.0",
    "rich>=13.7.0",
]
```

**src/project_name/cli.py**:
```python
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def hello(name: str = typer.Option(..., "--name", "-n", help="您的姓名")):
    """问候某人"""
    console.print(f"[bold green]你好 {name}![/bold green]")

def main():
    app()
```

### 6. 配置开发工具

**.env.example**:
```env
# 应用程序
PROJECT_NAME="项目名称"
VERSION="0.1.0"
DEBUG=True

# API
API_V1_PREFIX="/api/v1"
ALLOWED_ORIGINS=["http://localhost:3000"]

# 数据库
DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"

# 安全
SECRET_KEY="your-secret-key-here"
```

**Makefile**:
```makefile
.PHONY: install dev test lint format clean

install:
	uv sync

dev:
	uv run uvicorn src.project_name.main:app --reload

test:
	uv run pytest -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache
```

## 输出格式

1. **项目结构**: 包含所有必要文件的完整目录树
2. **配置**: 包含依赖项和工具设置的 pyproject.toml
3. **入口点**: 主应用程序文件（main.py、cli.py 等）
4. **测试**: 包含 pytest 配置的测试结构
5. **文档**: 包含设置和使用说明的 README
6. **开发工具**: Makefile、.env.example、.gitignore

专注于创建具有现代工具链、类型安全性和全面测试设置的生产就绪 Python 项目。
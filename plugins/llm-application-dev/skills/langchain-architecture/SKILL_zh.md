---
name: langchain-architecture
description: 使用 LangChain 框架设计 LLM 应用程序，包括 agents、memory 和工具集成模式。用于构建 LangChain 应用程序、实现 AI agents 或创建复杂的 LLM 工作流。
---

# LangChain 架构

掌握 LangChain 框架，用于构建包含 agents、chains、memory 和工具集成的高级 LLM 应用程序。

## 何时使用此技能

- 构建具有工具访问能力的自主 AI agents
- 实现复杂的多步骤 LLM 工作流
- 管理对话 memory 和状态
- 集成 LLM 与外部数据源和 APIs
- 创建模块化、可重用的 LLM 应用程序组件
- 实现文档处理管道
- 构建生产级 LLM 应用程序

## 核心概念

### 1. Agents
使用 LLM 决定采取何种行动的自主系统。

**Agent 类型：**
- **ReAct**: 以交错方式进行推理和行动
- **OpenAI Functions**: 利用函数调用 API
- **Structured Chat**: 处理多输入工具
- **Conversational**: 为聊天界面优化
- **Self-Ask with Search**: 分解复杂查询

### 2. Chains
对 LLM 或其他实用程序进行调用的序列。

**Chain 类型：**
- **LLMChain**: 基本的 prompt + LLM 组合
- **SequentialChain**: 多个 chains 顺序执行
- **RouterChain**: 将输入路由到专门的 chains
- **TransformChain**: 步骤之间的数据转换
- **MapReduceChain**: 带聚合的并行处理

### 3. Memory
在交互之间维护上下文的系统。

**Memory 类型：**
- **ConversationBufferMemory**: 存储所有消息
- **ConversationSummaryMemory**: 总结旧消息
- **ConversationBufferWindowMemory**: 保留最后 N 条消息
- **EntityMemory**: 跟踪实体信息
- **VectorStoreMemory**: 语义相似性检索

### 4. 文档处理
加载、转换和存储文档以供检索。

**组件：**
- **Document Loaders**: 从各种源加载
- **Text Splitters**: 智能分块文档
- **Vector Stores**: 存储和检索 embeddings
- **Retrievers**: 获取相关文档
- **Indexes**: 组织文档以便高效访问

### 5. Callbacks
用于 logging、monitoring 和 debugging 的钩子。

**使用场景：**
- 请求/响应 logging
- Token 使用跟踪
- 延迟监控
- 错误处理
- 自定义指标收集

## 快速开始

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# 初始化 LLM
llm = OpenAI(temperature=0)

# 加载工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 添加 memory
memory = ConversationBufferMemory(memory_key="chat_history")

# 创建 agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 运行 agent
result = agent.run("What's the weather in SF? Then calculate 25 * 4")
```

## 架构模式

### 模式 1: 使用 LangChain 的 RAG
```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 加载和处理文档
loader = TextLoader('documents.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 创建 vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# 创建 retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 查询
result = qa_chain({"query": "What is the main topic?"})
```

### 模式 2: 带工具的自定义 Agent
```python
from langchain.agents import Tool, AgentExecutor
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.tools import tool

@tool
def search_database(query: str) -> str:
    """在内部数据库中搜索信息。"""
    # 你的数据库搜索逻辑
    return f"Results for: {query}"

@tool
def send_email(recipient: str, content: str) -> str:
    """向指定收件人发送电子邮件。"""
    # 邮件发送逻辑
    return f"Email sent to {recipient}"

tools = [search_database, send_email]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

### 模式 3: 多步骤 Chain
```python
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# 步骤 1: 提取关键信息
extract_prompt = PromptTemplate(
    input_variables=["text"],
    template="Extract key entities from: {text}\n\nEntities:"
)
extract_chain = LLMChain(llm=llm, prompt=extract_prompt, output_key="entities")

# 步骤 2: 分析实体
analyze_prompt = PromptTemplate(
    input_variables=["entities"],
    template="Analyze these entities: {entities}\n\nAnalysis:"
)
analyze_chain = LLMChain(llm=llm, prompt=analyze_prompt, output_key="analysis")

# 步骤 3: 生成摘要
summary_prompt = PromptTemplate(
    input_variables=["entities", "analysis"],
    template="Summarize:\nEntities: {entities}\nAnalysis: {analysis}\n\nSummary:"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# 组合成 sequential chain
overall_chain = SequentialChain(
    chains=[extract_chain, analyze_chain, summary_chain],
    input_variables=["text"],
    output_variables=["entities", "analysis", "summary"],
    verbose=True
)
```

## Memory 管理最佳实践

### 选择正确的 Memory 类型
```python
# 对于短对话（< 10 条消息）
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# 对于长对话（总结旧消息）
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)

# 对于滑动窗口（最后 N 条消息）
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=5)

# 对于实体跟踪
from langchain.memory import ConversationEntityMemory
memory = ConversationEntityMemory(llm=llm)

# 对于相关历史的语义检索
from langchain.memory import VectorStoreRetrieverMemory
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

## Callback 系统

### 自定义 Callback Handler
```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM ended with response: {response}")

    def on_llm_error(self, error, **kwargs):
        print(f"LLM error: {error}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain started with inputs: {inputs}")

    def on_agent_action(self, action, **kwargs):
        print(f"Agent taking action: {action}")

# 使用 callback
agent.run("query", callbacks=[CustomCallbackHandler()])
```

## 测试策略

```python
import pytest
from unittest.mock import Mock

def test_agent_tool_selection():
    # Mock LLM 返回特定的工具选择
    mock_llm = Mock()
    mock_llm.predict.return_value = "Action: search_database\nAction Input: test query"

    agent = initialize_agent(tools, mock_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    result = agent.run("test query")

    # 验证选择了正确的工具
    assert "search_database" in str(mock_llm.predict.call_args)

def test_memory_persistence():
    memory = ConversationBufferMemory()

    memory.save_context({"input": "Hi"}, {"output": "Hello!"})

    assert "Hi" in memory.load_memory_variables({})['history']
    assert "Hello!" in memory.load_memory_variables({})['history']
```

## 性能优化

### 1. 缓存
```python
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()
```

### 2. 批处理
```python
# 并行处理多个文档
from langchain.document_loaders import DirectoryLoader
from concurrent.futures import ThreadPoolExecutor

loader = DirectoryLoader('./docs')
docs = loader.load()

def process_doc(doc):
    return text_splitter.split_documents([doc])

with ThreadPoolExecutor(max_workers=4) as executor:
    split_docs = list(executor.map(process_doc, docs))
```

### 3. 流式响应
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
```

## 资源

- **references/agents.md**: Agent 架构深入探讨
- **references/memory.md**: Memory 系统模式
- **references/chains.md**: Chain 组合策略
- **references/document-processing.md**: 文档加载和索引
- **references/callbacks.md**: 监控和可观测性
- **assets/agent-template.py**: 生产就绪的 agent 模板
- **assets/memory-config.yaml**: Memory 配置示例
- **assets/chain-example.py**: 复杂 chain 示例

## 常见陷阱

1. **Memory 溢出**: 不管理对话历史长度
2. **工具选择错误**: 工具描述不清导致 agent 混淆
3. **上下文窗口超出**: 超出 LLM token 限制
4. **无错误处理**: 未捕获和处理 agent 故障
5. **检索效率低**: 未优化 vector store 查询

## 生产环境检查清单

- [ ] 实现适当的错误处理
- [ ] 添加请求/响应 logging
- [ ] 监控 token 使用和成本
- [ ] 设置 agent 执行的超时限制
- [ ] 实现速率限制
- [ ] 添加输入验证
- [ ] 使用边缘情况测试
- [ ] 设置可观测性（callbacks）
- [ ] 实现回退策略
- [ ] 对 prompts 和配置进行版本控制
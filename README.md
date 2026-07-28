# 🤖 多工具智能助手（Multi-Tool AI Assistant）

> 基于 LangGraph 构建的智能 Agent，具备自主决策、多工具调用、多轮对话记忆与持久化存储能力。

---

## 📖 项目简介

本项目是一个基于 **LangGraph** 框架开发的命令行智能助手，能够根据用户问题的类型**自主决策**调用最合适的工具：

- **RAG 文档检索**：基于公司内部文档进行精准问答（支持私有知识库）
- **Wikipedia 百科查询**：获取人物、概念、历史事件等事实信息
- **数学计算器**：支持基础四则运算

同时，系统具备**多轮对话记忆**能力，并能通过 **JSON 持久化** 存储对话历史，即使重启程序也能恢复上下文。

---

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| 🧠 **自主决策** | 使用 ReAct 模式，Agent 自动判断问题类型并选择工具 |
| 📄 **RAG 检索** | 基于 Chroma 向量数据库 + 智谱 Embedding，精准检索私有文档 |
| 📚 **百科查询** | 集成 Wikipedia API，支持中英文百科知识检索 |
| 🧮 **数学计算** | 支持基础四则运算表达式（如 `25*4+10`） |
| 💬 **多轮记忆** | 维护完整对话历史，支持上下文连续对话 |
| 💾 **持久化存储** | 对话历史自动保存到 `memory.json`，重启后自动加载 |
| 🔄 **滑动窗口** | 自动裁剪超长对话，避免上下文溢出（默认保留最近 10 轮） |
| 🧹 **会话管理** | 支持 `clear` 命令清空当前记忆 |

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **框架** | LangChain, LangGraph |
| **大模型** | 智谱 AI（ChatGLM）`glm-4-flash` |
| **嵌入模型** | 智谱 `embedding-2` |
| **向量数据库** | Chroma（本地持久化） |
| **工具集成** | Wikipedia API, RAG 检索, 计算器 |
| **开发语言** | Python 3.12 |
| **环境管理** | Python venv + pip |

---

## 📦 安装与运行

### 1. 克隆项目
```bash
git clone git@github.com:2730266616/multi-tool-ai-assistant.git
cd multi-tool-ai-assistant
```

### 2. 创建并激活虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置 API 密钥
在项目根目录创建 `.env` 文件：
```bash
ZHIPU_API_KEY=你的智谱API密钥
```

> ⚠️ **注意**：`.env` 文件已加入 `.gitignore`，不会上传到 GitHub，请勿在代码中硬编码密钥。

### 5. 准备测试文档
确保 `../company.txt` 文件存在（默认路径），内容示例：
```
公司名称：云创科技
成立时间：2018年
主营业务：人工智能与大数据解决方案
员工人数：120人
总部地址：上海市浦东新区
创始人：李明
融资情况：2022年完成B轮融资1亿元
```

> 如需修改文档路径，请修改 `smart_agent.py` 中的 `DOC_PATH` 变量。

### 6. 运行项目
```bash
python smart_agent.py
```

---

## 🧪 使用示例

```text

```

---

## 📁 项目结构

```
.
├── smart_agent.py          # 主程序（Agent 核心逻辑）
├── .gitignore              # Git 忽略配置
├── .env                    # API 密钥（不提交）
├── requirements.txt        # Python 依赖清单
├── README.md               # 项目说明文档
├── memory.json             # 记忆持久化文件（自动生成）
└── chroma_db_agent/        # Chroma 向量数据库（自动生成）
    └── ...
```

---

## 🔧 核心设计亮点

### 1. 工具封装（Tool Abstraction）
每个工具使用 `@tool` 装饰器封装，包含清晰的 **名称、描述、输入参数类型**，方便 Agent 理解并调用。

### 2. ReAct Agent 架构
使用 LangGraph 的 `create_react_agent` 预置构建器，实现 **思考 → 行动 → 观察** 的自主循环，Agent 能够：
- 分析用户问题
- 决定是否调用工具及调用哪个工具
- 根据工具返回结果生成最终回答

### 3. 对话记忆与持久化
- 维护 `messages` 列表，存储完整对话历史
- 启动时从 `memory.json` 加载历史
- 退出时自动保存到 `memory.json`
- 支持 `clear` 命令清空记忆

### 4. 滑动窗口机制
当对话超过 **10 轮（20 条消息）** 时，自动裁剪早期历史，防止上下文溢出（Context Overflow），确保模型在有限的上下文窗口内始终保持最佳性能。

---

## 🔮 未来优化方向

- [ ] 接入更多工具（搜索引擎、天气 API、新闻 API）
- [ ] 使用 LangGraph `checkpointer` 实现更优雅的持久化（支持时间旅行）
- [ ] 支持多 Agent 协作（规划 Agent + 执行 Agent）
- [ ] 添加 Web 界面（Streamlit / Gradio）
- [ ] 支持多种文档格式（PDF、Word、Markdown）

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

## 👨‍💻 作者

- GitHub: [2730266616](https://github.com/2730266616)
- 邮箱: 2730266616@qq.com

---

## 🙏 致谢

- [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- [智谱 AI](https://open.bigmodel.cn/) 提供大模型与嵌入 API
- [Wikipedia](https://www.wikipedia.org/) 提供百科数据

---

> 📌 **项目演示**：如有任何问题或建议，欢迎提交 Issue 或 Pull Request！



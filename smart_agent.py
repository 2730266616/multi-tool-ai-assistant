import os
import re
import json
import sys
from pathlib import Path
from dotenv import load_dotenv


# LangChain 相关组件
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# LangGraph 相关：预置的 ReAct Agent 构建器
from langgraph.prebuilt import create_react_agent
# LangChain 核心工具与消息
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
# 使用 LangChain 内置的 Wikipedia 工具
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# ========== 配置 ==========
load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY", "你的智谱API密钥")
if not API_KEY:
    print("❌ 错误：未找到 ZHIPU_API_KEY，请在 .env 文件中设置")
    sys.exit(1)

# 模型配置
MODEL_NAME = "glm-4-flash"
TEMPERATURE = 0


# ========== 路径处理 ==========
SCRIPT_DIR = Path(__file__).parent.resolve()
PERSIST_DIR = str(SCRIPT_DIR / "chroma_db_agent")
MEMORY_FILE = SCRIPT_DIR / "memory.json"          # 记忆存储文件
DOC_PATH = (SCRIPT_DIR / "../company.txt").resolve()

if not DOC_PATH.exists():
    print(f"❌ 错误：文档文件不存在：{DOC_PATH}")
    print("请确保 company.txt 放在正确位置（默认在脚本的上一级目录）")
    sys.exit(1)

print(f"📄 使用文档：{DOC_PATH}")
print(f"💾 记忆文件：{MEMORY_FILE}")


# ========== 初始化模型 ==========
# 创建智谱 AI 的聊天模型实例，使用 glm-4-flash 模型，温度设为0使输出更确定
model = ChatZhipuAI(api_key=API_KEY, model= MODEL_NAME, temperature=TEMPERATURE)


# ========== 准备 RAG 检索器（加载已有向量库或新建） ==========
print("📂 加载文档并创建向量库...")
# 1. 加载本地公司文档（假设存在 ../company.txt）
loader = TextLoader(str(DOC_PATH), encoding="utf-8")
documents = loader.load()
# 2. 将长文档切分成小块（chunk），便于检索
test_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks = test_splitter.split_documents(documents)

# 3. 使用智谱 Embedding 模型将文本块向量化，并存入 Chroma 向量库（持久化到本地目录）
embeddings = ZhipuAIEmbeddings(api_key=API_KEY,model="embedding-2")
# 创建 Chroma 向量数据库，持久化到本地目录，以便后续复用
vectorstore = Chroma.from_documents(chunks,embedding=embeddings,persist_directory=PERSIST_DIR)
# 创建检索器，每次检索返回最相关的3个文档块
retriever = vectorstore.as_retriever(search_kwargs={"k":3})


# ========== 工具1：公司文档检索（RAG） ==========
@tool
def search_company_docs(query: str) -> str:
    """
    从公司文档中检索相关信息。
    适用于公司介绍、业务、人员等问题。
    """
    docs = retriever.invoke(query)  # 调用检索器获取相关文档
    if not docs:
        return "未找到相关信息。"
    # 将多个文档内容拼接返回
    return "\n\n".join([d.page_content for d in docs])


# ========== 工具2：Wikipedia（增强异常处理） ==========
# 实例化原始工具
_wikipedia_raw = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def search_wikipedia(query: str) -> str:
    """
    搜索 Wikipedia 获取百科知识，适用于人物、概念、历史事件等。
    网络不通时会返回友好提示，不会抛出异常。
    """
    try:
        return _wikipedia_raw.run(query)
    except Exception as e:
        # 捕获网络错误、超时等
        return f"⚠️ 无法访问 Wikipedia（可能是网络问题），错误信息：{str(e)}。请尝试其他问题。"


# ========== 工具3：计算器 ==========
@tool
def calculate(expression: str) -> str:
    """
    执行简单数学计算，输入为数学表达式，如 '2+3' 或 '10*5'。
    """
    try:
        # 简单校验表达式只包含数字、运算符、括号、小数点、空格
        if not re.match(r'^[\d+\-*/().\s]+$', expression):
            return "仅支持基础四则运算。"
        # 使用 eval 计算（注意安全风险，但仅限基础运算可接受）
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


# ========== 工具列表 ==========
tools = [search_company_docs, search_wikipedia, calculate]


# ========== 创建 ReAct Agent ==========
# 使用 LangGraph 的预置函数，将模型、工具列表和系统提示组装成 Agent
agent = create_react_agent(
    model,
    tools,
    prompt=SystemMessage(
        content="你是一个智能助手，可以使用以下工具：\n"
        "1. search_company_docs：查询公司内部文档\n"
        "2. search_wikipedia：查询百科知识（即 Wikipedia 工具）\n"
        "3. calculate：执行数学计算\n"
        "根据用户问题，自主选择最合适的工具。如果问题可以直接回答，不需要工具。"
    )
)


# ========== 记忆持久化函数 ==========
def save_memory(messages):
    """将消息列表保存到 JSON 文件"""
    serializable = []
    for role, content in messages:
        serializable.append({"role": role, "content": content})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

def load_memory():
    """从 JSON 文件加载消息列表"""
    if not MEMORY_FILE.exists():
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(item["role"], item["content"]) for item in data]
    except (json.JSONDecodeError, KeyError):
        print("⚠️ 记忆文件损坏，将开始新对话")
        return []


# ========== 交互循环 ==========
def main():
    """主交互函数，启动命令行对话循环"""
    messages = load_memory()
    print(f"📂 已加载 {len(messages)} 条历史消息\n")

    print("\n" + "="*50)
    print("🤖 多工具智能助手（RAG + Wikipedia + 计算器 + 记忆）")
    print("="*50)
    print("支持的问题类型：")
    print("  📄 公司文档问答（如：公司主营业务是什么？）")
    print("  📚 百科知识（如：什么是人工智能？）")
    print("  🧮 数学计算（如：25*4+10 等于多少？）")
    print("\n命令：")
    print("  exit  → 退出并保存记忆")
    print("  clear → 清空当前会话记忆\n")

    # 滑动窗口配置：最多保留 10 轮对话（即 20 条消息）
    MAX_HISTORY_PAIRS = 10

    while True:
        # 获取用户输入
        user_input = input("你: ")
        if user_input.lower() == "exit":
            save_memory(messages)
            print("💾 记忆已保存，再见！")
            break
        if user_input.lower() == "clear":
            messages = []  # 清空记忆
            print("🧹 记忆已清空（文件未删除，下次启动不会加载）\n")
            continue
        if not user_input.strip():
            continue

        # 将用户消息加入历史
        messages.append(("human", user_input))

        # 【核心改动】滑动窗口截断：如果消息数超过阈值，只保留最近 N 轮
        # 例如：10 轮 = 20 条消息，保留最后 20 条
        max_messages = MAX_HISTORY_PAIRS * 2
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
            print(f"🔄 记忆已滑动至最近 {MAX_HISTORY_PAIRS} 轮")

        # 调用 Agent，传入完整历史
        result = agent.invoke({"messages": messages})
        final_message = result["messages"][-1]

        # 将 AI 回复加入历史
        messages.append(("ai", final_message.content))

        print(f"AI: {final_message.content}\n")

if __name__ == "__main__":
    main()
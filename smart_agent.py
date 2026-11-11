import os
import ast
import operator
import json
import sys
from pathlib import Path
from dotenv import load_dotenv


# LangChain 相关组件
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# LangGraph 相关：预置的 ReAct Agent 构建器
from langgraph.prebuilt import create_react_agent
# LangChain 核心工具与消息
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
# 从子模块直接导入，避免 Pylance 对包根懒加载导出的误报
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


# ========== 路径处理 ==========
SCRIPT_DIR = Path(__file__).parent.resolve()
PERSIST_DIR = str(SCRIPT_DIR / "chroma_db_agent")
INDEX_META = SCRIPT_DIR / "index_meta.json"
MEMORY_FILE = SCRIPT_DIR / "memory.json"          # 记忆存储文件
PDF_PATH = (SCRIPT_DIR / "hr_policy.pdf").resolve()

# ========== 配置 ==========
# 显式指定仓库根目录的 .env，无论从哪个目录启动脚本都能读到密钥
load_dotenv(SCRIPT_DIR.parent / ".env")
API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    print("❌ 错误：未找到 ZHIPU_API_KEY，请在 .env 文件中设置")
    sys.exit(1)

# 模型配置
MODEL_NAME = "glm-4-flash"
TEMPERATURE = 0


if not PDF_PATH.exists():
    print(f"❌ 错误：文档文件不存在：{PDF_PATH}")
    print("请确保 hr_policy.pdf 与 smart_agent.py 位于同一目录")
    sys.exit(1)

print(f"📄 使用文档：{PDF_PATH}")
print(f"💾 记忆文件：{MEMORY_FILE}")


# ========== 初始化模型 ==========
# 创建智谱 AI 的聊天模型实例，使用 glm-4-flash 模型，温度设为0使输出更确定
model = ChatZhipuAI(api_key=API_KEY, model= MODEL_NAME, temperature=TEMPERATURE)


# ========== 准备 RAG 检索器（加载已有向量库或新建） ==========
print("📂 正在加载企业级 PDF 文档...")
# 1. 加载 PDF 文档
loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()

suspicious = []
for d in documents:
    text = d.page_content or ""
    bad_ratio = text.count("\ufffd") / max(len(text), 1)
    if len(text.strip()) < 50 or bad_ratio > 0.02:
        suspicious.append({"page": d.metadata.get("page", -1) + 1,
                           "chars": len(text), "bad_ratio": round(bad_ratio, 3)})
print(f"[PDF质量] 共 {len(documents)} 页，可疑页: {suspicious}")

# 2. 将长文档切分成小块（chunk），便于检索
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)
chunks = text_splitter.split_documents(documents)

# 3. 使用智谱 Embedding 模型将文本块向量化，并存入 Chroma 向量库（持久化到本地目录）
embeddings = ZhipuAIEmbeddings(api_key=API_KEY,model="embedding-2")
# 索引版本管理：用 PDF mtime/size + 切分参数做签名，签名变化自动重建，避免旧索引污染
def _index_signature():
    st = PDF_PATH.stat()
    return {"pdf_mtime": st.st_mtime, "pdf_size": st.st_size,
            "chunk_size": 400, "chunk_overlap": 100}

def _need_rebuild():
    if not os.path.exists(PERSIST_DIR) or not INDEX_META.exists():
        return True
    try:
        return json.loads(INDEX_META.read_text(encoding="utf-8")) != _index_signature()
    except json.JSONDecodeError:
        return True
    
if _need_rebuild():
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)
        print("   ⚠️ 检测到 PDF/切分参数变化，重建向量库")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    INDEX_META.write_text(json.dumps(_index_signature(), ensure_ascii=False), encoding="utf-8")
    print("   已创建新向量库")
else:
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    print("   已加载现有向量库")
# 检索器：每次召回 Top5。k=3 时会漏掉被表格块挤掉的条款，扩到 5 兜住召回
retriever = vectorstore.as_retriever(search_kwargs={"k":5})


# ========== 工具1：公司文档检索（RAG） ==========
@tool
def search_company_docs(query: str) -> str:
    """
    从公司内部文档中检索信息，适用于社保政策、缴纳比例、报销流程、HR 制度等。
    """
    docs = retriever.invoke(query)  # 调用检索器获取相关文档
    if not docs:
        return "未找到相关信息。"
    # 将多个文档内容拼接返回
    return "\n\n".join(
        f"【来源·第{d.metadata.get('page', 0) + 1}页】\n{d.page_content}" for d in docs
    )


# ========== 工具2：Wikipedia（增强异常处理） ==========
# 实例化原始工具
_wikipedia_raw = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()) # type: ignore

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
# 仅允许基础四则运算 + 取模 + 幂运算的白名单
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval_node(node):
    """递归求值 AST 节点，只接受数字和基础运算符，拒绝任何其他语法（变量、函数调用等）"""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        # 限制幂运算的指数，防止 9**9**9 之类的天文数字把程序卡死
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("指数过大")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError("不支持的表达式")

@tool
def calculate(expression: str) -> str:
    """
    执行简单数学计算，输入为数学表达式，如 '2+3' 或 '10*5'。
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree)
        return f"计算结果：{result}"
    except ZeroDivisionError:
        return "计算错误：除数不能为 0"
    except Exception as e:
        return f"计算错误：{str(e)}"


# ========== 工具列表 ==========
tools = [search_company_docs, search_wikipedia, calculate]


# ========== 创建 ReAct Agent ==========
# 使用 LangGraph 预置的 ReAct Agent 构建器
# TODO(Day3): create_react_agent 在 LangGraph V1.0 已弃用，计划迁移到 langchain.agents.create_agent
agent = create_react_agent(
    model,
    tools,
    prompt=SystemMessage(
        content="你是一个智能助手，可以使用以下工具：\n"
        "1. search_company_docs：查询公司内部文档\n"
        "2. search_wikipedia：查询百科知识（即 Wikipedia 工具）\n"
        "3. calculate：执行数学计算\n"
        "根据用户问题，自主选择最合适的工具。如果问题可以直接回答，不需要工具。"
        "\n当用户询问社保比例/缴费标准时，必须覆盖：养老、医疗、失业、工伤、生育五险及住房公积金，不得遗漏。"
        "\n引用 search_company_docs 的内容时，每一条事实后面必须紧跟来源标注，格式为【来源·第X页】。"
        "\n示例：基本养老保险：单位16%，个人8%【来源·第1页】"
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
            messages = []  # 清空当前记忆
            MEMORY_FILE.unlink(missing_ok=True)  # 同时删除记忆文件，下次启动才会真正从新对话开始
            print("🧹 记忆已清空（会话与文件均已清除），下次启动从新对话开始\n")
            continue
        if not user_input.strip():
            continue

        # 将用户消息加入历史
        messages.append(("human", user_input))

        # 滑动窗口截断：消息数超过阈值时只保留最近 N 轮，防止上下文溢出
        # 例如：10 轮 = 20 条消息，保留最后 20 条
        max_messages = MAX_HISTORY_PAIRS * 2
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
            while messages and messages[0][0] != "human":
                messages = messages[1:]
            print(f"🔄 记忆已滑动至最近 {MAX_HISTORY_PAIRS} 轮")

        # 调用 Agent，传入完整历史
        result = agent.invoke({"messages": messages})
        final_message = result["messages"][-1]

        # 将 AI 回复加入历史
        messages.append(("ai", final_message.content))

        print(f"AI: {final_message.content}\n")

if __name__ == "__main__":
    main()
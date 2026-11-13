import time
import streamlit as st

st.set_page_config(page_title="企业级智能文档解析 Agent", layout="wide")

@st.cache_resource
def load_agent():
    from smart_agent import agent, extract_sources
    return agent, extract_sources

agent, extract_sources = load_agent()

st.title("📄 基于 RAG 的企业级复杂文档解析 Agent")
st.caption("LangGraph 多工具路由 | L2 阈值拦截防幻觉 | 源码级页码溯源")

# 侧边栏展示系统配置
with st.sidebar:
    st.header("系统运行参数")
    st.write("🔹 **检索模型**: 智谱 embedding-2")
    st.write("🔹 **向量库**: Chroma (L2 距离)")
    st.write("🔹 **阈值拦截**: > 1.10 自动转人工")
    st.write("🔹 **防幻觉层**: Prompt软约束 + 代码硬兜底")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📌 溯源定位：第 {'、'.join(map(str, msg['sources']))} 页")

if user_input := st.chat_input("请输入查询需求，如：社保缴纳比例 / 什么是人工智能"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在拆解任务并检索..."):
            t0 = time.time()
            history = [(m["role"].replace("assistant", "ai"), m["content"])
                       for m in st.session_state.messages[-20:]]
            result = agent.invoke({"messages": history})
            
            answer = result["messages"][-1].content
            if not answer or not answer.strip():
                answer = "抱歉，系统未能生成有效回复。"
            
            # 模拟代码层硬兜底（网页端展示）
            _tool_refused = any("请拨打" in (m.content if isinstance(m.content, str) else "") for m in result["messages"] if getattr(m, "type", "") == "tool")
            if _tool_refused and answer.count("\n") > 3:
                answer = "【系统拦截】未在库中找到足够相关的内容，已阻断大模型生成，请转人工客服咨询。"

            sources = extract_sources(result)
            elapsed = time.time() - t0

            st.markdown(answer)
            if sources:
                st.caption(f"📌 溯源定位：第 {'、'.join(map(str, sources))} 页")
            st.caption(f"⏱️ 检索与生成总耗时 {elapsed:.2f}s")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
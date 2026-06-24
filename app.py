"""Module 5 -- Streamlit UI.

Query input -> answer panel -> clause source -> optional graph visualisation.
Zero LLM dependency: every answer is produced by graph traversal over a
NetworkX knowledge graph built from the parsed policy text.
"""
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from parser import parse_policy          # noqa: E402
from entity_extractor import extract     # noqa: E402
from graph_builder import build_graph    # noqa: E402
from query_engine import answer_query    # noqa: E402

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "data", "sample_motor_policy.pdf")

st.set_page_config(page_title="PolicyLens", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Parsing policy and building knowledge graph...")
def load_pipeline(pdf_path: str, _cache_key: str):
    policy = parse_policy(pdf_path)
    extracted = extract(policy)
    graph = build_graph(extracted)
    return policy, extracted, graph


def render_graph_html(graph, highlighted_nodes):
    from pyvis.network import Network

    net = Network(height="500px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "clause":
            label = data["citation"]
            color = "#ffb703" if node in highlighted_nodes else "#3a86ff"
        else:
            label = data["label"]
            color = "#fb5607" if node in highlighted_nodes else "#8ecae6"
        net.add_node(node, label=label, color=color, title=data.get("text", label))
    for u, v, data in graph.edges(data=True):
        net.add_edge(u, v, label=data["relation"], color="#adb5bd")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_file.name)
    with open(tmp_file.name, "r", encoding="utf-8") as f:
        return f.read()


st.title("🔍 PolicyLens")
st.caption("Rule-based NLP knowledge graph for insurance policy querying — symbolic AI, zero LLM dependency.")

with st.sidebar:
    st.header("Policy Document")
    uploaded = st.file_uploader("Upload a text-layer motor policy PDF (optional)", type=["pdf"])
    if uploaded is not None:
        tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        pdf_path, cache_key = tmp_path, uploaded.name
        st.success(f"Using uploaded policy: {uploaded.name}")
    else:
        pdf_path, cache_key = SAMPLE_PDF, "sample"
        st.info("Using bundled sample policy (synthetic specimen).")

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. **Parse** policy PDF into sections/clauses\n"
        "2. **Extract** entities via rule-based keyword/spaCy matching\n"
        "3. **Build** a NetworkX knowledge graph\n"
        "4. **Traverse** the graph to answer your query\n"
        "5. Every answer cites its **exact source clause**"
    )

policy, extracted, graph = load_pipeline(pdf_path, cache_key)

tab_query, tab_contrast, tab_graph_full = st.tabs(["💬 Query", "⚖️ Symbolic vs RAG", "🕸️ Full Graph"])

with tab_query:
    st.subheader("Ask a question about this policy")
    sample_queries = [
        "Is theft covered?",
        "What is excluded?",
        "What is the deductible?",
        "Is engine flood damage covered?",
    ]
    cols = st.columns(len(sample_queries))
    clicked = None
    for col, sq in zip(cols, sample_queries):
        if col.button(sq, use_container_width=True):
            clicked = sq

    query = st.text_input("Your query", value=clicked or "", placeholder="e.g. Is theft covered?")

    if query:
        result = answer_query(graph, query)

        status_color = {"YES": "green", "NO": "red", "PARTIAL": "orange", "LIST": "blue", "UNKNOWN": "gray"}
        color = status_color.get(result.status, "gray")
        st.markdown(f"### Answer  :{color}[{result.status}]")
        st.write(result.answer)

        if result.sources:
            st.subheader("📄 Source Clause(s)")
            for s in result.sources:
                with st.container(border=True):
                    st.markdown(f"**{s.citation}**  ·  relation: `{s.relation}`")
                    st.write(s.text)

        if result.graph_path:
            st.subheader("🧭 Graph Traversal Path")
            for hop in result.graph_path:
                st.code(hop, language=None)

        highlighted = set()
        for s in result.sources:
            highlighted.add(f"clause:{s.clause_id}")
        for key in result.matched_entities:
            highlighted.add(f"entity:{key}")

        if highlighted:
            with st.expander("🕸️ Show traversal subgraph"):
                html = render_graph_html(graph, highlighted)
                st.components.v1.html(html, height=520)
    else:
        st.info("Type a query above or click a sample question to see a fully auditable, source-cited answer.")

with tab_contrast:
    st.subheader("Symbolic AI (PolicyLens) vs. Neural RAG (Insurebot-AI)")
    st.markdown(
        "PolicyLens is the symbolic counterpart to a RAG + LLM chatbot. The same query, "
        "answered two ways, illustrates the **flexibility vs. explainability** tradeoff in "
        "human-centric AI design. The RAG-side response below is a static illustrative example "
        "only — no LLM is called anywhere in this application."
    )
    example_query = st.selectbox("Example query", sample_queries)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔎 PolicyLens (Symbolic)")
        result = answer_query(graph, example_query)
        st.success(result.answer if len(result.answer) < 300 else result.answer[:300] + "...")
        st.caption("Fully traceable: entity match -> graph edge -> exact clause. Every word is auditable.")
    with col2:
        st.markdown("#### 🤖 Insurebot-AI (RAG + LLM) -- illustrative")
        st.warning(
            "Yes, that's generally covered under your policy, though there may be some "
            "conditions that apply depending on the circumstances."
        )
        st.caption(
            "Fluent and conversational, but no clause citation, no graph path, and the exact "
            "reasoning is opaque inside the model's weights -- not auditable."
        )

with tab_graph_full:
    st.subheader("Full Policy Knowledge Graph")
    st.caption(f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    html = render_graph_html(graph, highlighted_nodes=set())
    st.components.v1.html(html, height=600)

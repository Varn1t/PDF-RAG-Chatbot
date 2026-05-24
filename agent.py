from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────

class RAGState(TypedDict):
    original_query: str          # Never changes — the user's actual question
    current_query: str           # May be rewritten each iteration
    chunks: List[str]            # Raw text of retrieved chunks
    answer: str                  # Generated answer
    is_grounded: bool            # Hallucination check result
    is_relevant: bool            # Answer relevance check result
    iterations: int              # How many attempts so far
    max_iterations: int          # Cap (default 3)
    final_answer: str            # What gets returned to the caller
    failed: bool                 # True if we exhausted retries
    run_history: List[dict]      # Diagnostic history of the agentic loops
    chat_history: List[dict]     # Historical conversation messages for contextualization

# ─────────────────────────────────────────────
# Build the graph (called once per session)
# ─────────────────────────────────────────────

def build_rag_graph(vectorstore: FAISS, llm: OllamaLLM, k: int = 3):
    """
    Returns a compiled LangGraph app.
    vectorstore and llm are injected so the graph is stateless/reusable.
    """

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # ── Node 0: Contextualize Query ──────────────────────────────────
    def contextualize_query(state: RAGState) -> dict:
        chat_history = state.get("chat_history", [])
        if not chat_history:
            return {"current_query": state["original_query"]}
            
        # Format chat history (limit to last 3 conversations, which is 6 messages)
        history_str = ""
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
            
        prompt = f"""Given the following chat history and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the chat history.
Do NOT answer the question, just rephrase it if needed to refer to the correct subject, otherwise return it exactly as is.

Chat History:
{history_str}

Follow-up Question: {state['original_query']}

Standalone Rephrased Question (write ONLY the rephrased question):"""

        standalone_query = llm.invoke(prompt).strip()
        # Strip potential wrapping quotes
        if (standalone_query.startswith('"') and standalone_query.endswith('"')) or (standalone_query.startswith("'") and standalone_query.endswith("'")):
            standalone_query = standalone_query[1:-1]
            
        return {"current_query": standalone_query}

    # ── Node 1: Retrieve ──────────────────────────────────────────────
    def retrieve(state: RAGState) -> dict:
        docs = retriever.invoke(state["current_query"])
        return {"chunks": [d.page_content for d in docs]}

    # ── Node 2: Generate ─────────────────────────────────────────────
    def generate(state: RAGState) -> dict:
        context = "\n\n---\n\n".join(state["chunks"])
        prompt = f"""You are a helpful assistant. You must answer the specific question asked using ONLY the context provided below.

Strict Instructions:
- If the question asks for examples, provide examples from the context.
- If the question asks for limitations, provide limitations from the context.
- Do not cherry-pick or answer a different aspect of the topic than what was specifically asked.
- Do not make up facts or assumptions not present in the context.
- If the context does not contain enough information to answer the specific question, say so explicitly.

Context:
{context}

Question: {state['current_query']}

Answer:"""
        answer = llm.invoke(prompt)
        return {"answer": answer.strip()}

    # ── Node 3: Fact-Checker & Relevance Grader ──────────────────────
    def check_hallucination(state: RAGState) -> dict:
        context = "\n\n---\n\n".join(state["chunks"])
        prompt = f"""You are an expert fact-checker and QA relevance grader. Your job is to evaluate a generated answer based on the retrieved context and the user query.

Provide two binary grades (YES or NO) in the exact format shown below:
1. GROUNDED: Is the generated answer fully supported by and traceable to the context, without making assumptions or hallucinating outside facts? (YES or NO)
2. RELEVANT: Does the generated answer directly address and resolve what the question specifically asked for? 
   - Output YES ONLY if the answer specifically addresses the core intent of the question.
   - Output NO if the answer describes a different aspect of the topic (e.g., if the question asks for EXAMPLES and the answer lists LIMITATIONS instead, that is NOT RELEVANT even if the general topic is correct). 
   - (YES or NO)

Context:
{context}

User Query:
{state['current_query']}

Generated Answer:
{state['answer']}

Verdict format (MUST write exactly this and nothing else):
GROUNDED: <YES or NO>
RELEVANT: <YES or NO>"""

        verdict_raw = llm.invoke(prompt).strip().upper()
        
        is_grounded = "GROUNDED: YES" in verdict_raw or "GROUNDED:YES" in verdict_raw
        is_relevant = "RELEVANT: YES" in verdict_raw or "RELEVANT:YES" in verdict_raw
        new_iterations = state.get("iterations", 0) + 1
        
        # Capture telemetry history
        history = list(state.get("run_history", []))
        history.append({
            "iteration": new_iterations,
            "query": state["current_query"],
            "answer": state["answer"],
            "verdict": f"Grounded: {'YES' if is_grounded else 'NO'} | Relevant: {'YES' if is_relevant else 'NO'}",
            "is_grounded": is_grounded,
            "is_relevant": is_relevant,
            "chunks": state["chunks"]
        })
        
        return {
            "is_grounded": is_grounded,
            "is_relevant": is_relevant,
            "iterations": new_iterations,
            "run_history": history
        }

    # ── Node 4: Query Rewriter ───────────────────────────────────────
    def rewrite_query(state: RAGState) -> dict:
        grounded = state.get("is_grounded", False)
        relevant = state.get("is_relevant", False)

        if not grounded and not relevant:
            failure_reason = "the generated answer was neither grounded in the retrieved context nor relevant to the query"
        elif not grounded:
            failure_reason = "the generated answer contained information not supported by the retrieved context chunks (hallucination)"
        else:
            failure_reason = "the generated answer was grounded but did not actually address or resolve the user's question (off-topic)"

        prompt = f"""The following question was asked and the generated answer failed verification because {failure_reason}.
Rewrite the question to be more specific, keyword-rich, and targeted so better context chunks can be retrieved from the vector index.

Original question: {state['original_query']}
Previous (failed) query: {state['current_query']}
Failed answer: {state['answer']}

Write ONLY the rewritten question, nothing else:"""

        new_query = llm.invoke(prompt).strip()
        
        # Strip potential wrapping quotes
        if (new_query.startswith('"') and new_query.endswith('"')) or (new_query.startswith("'") and new_query.endswith("'")):
            new_query = new_query[1:-1]
            
        return {"current_query": new_query}

    # ── Node 5: Finalize (success) ───────────────────────────────────
    def finalize_success(state: RAGState) -> dict:
        return {
            "final_answer": state["answer"],
            "failed": False
        }

    # ── Node 6: Finalize (failure) ───────────────────────────────────
    def finalize_failure(state: RAGState) -> dict:
        fallback_msg = (
            "❌ The information you're looking for does not appear to be in the loaded document. "
            f"I tried {state['iterations']} time(s) with different query formulations but could not "
            "generate a grounded answer from the available chunks."
        )
        return {
            "final_answer": fallback_msg,
            "failed": True
        }

    # ── Routing logic ────────────────────────────────────────────────
    def route_after_check(state: RAGState) -> str:
        # Route to success ONLY if the response is fully grounded AND answers the question (is relevant)
        if state["is_grounded"] and state["is_relevant"]:
            return "finalize_success"
        elif state["iterations"] >= state["max_iterations"]:
            return "finalize_failure"
        else:
            return "rewrite_query"

    # ── Assemble Graph ───────────────────────────────────────────────
    graph = StateGraph(RAGState)

    graph.add_node("contextualize_query", contextualize_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("finalize_success", finalize_success)
    graph.add_node("finalize_failure", finalize_failure)

    graph.set_entry_point("contextualize_query")
    graph.add_edge("contextualize_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "check_hallucination")
    
    graph.add_conditional_edges(
        "check_hallucination",
        route_after_check,
        {
            "finalize_success": "finalize_success",
            "finalize_failure": "finalize_failure",
            "rewrite_query": "rewrite_query",
        }
    )
    graph.add_edge("rewrite_query", "retrieve")   # Loop back
    graph.add_edge("finalize_success", END)
    graph.add_edge("finalize_failure", END)

    return graph.compile()

# ─────────────────────────────────────────────
# Helper: build vectorstore + LLM from docs
# (shared between app.py and main.py)
# ─────────────────────────────────────────────

def build_pipeline(docs: List[Document], chunk_size: int, chunk_overlap: int,
                   k: int, model_name: str):
    """
    Splits docs, builds FAISS vectorstore, wires up the LangGraph RAG.
    Returns (rag_app, num_chunks).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = OllamaLLM(model=model_name)
    rag_app = build_rag_graph(vectorstore, llm, k=k)

    return rag_app, len(chunks)

def run_query(rag_app, query: str, max_iterations: int = 3, chat_history: List[dict] = None) -> dict:
    """
    Runs a query through the self-correcting RAG graph.
    Returns a dict with: final_answer, failed, iterations, run_history.
    """
    if chat_history is None:
        chat_history = []
        
    initial_state: RAGState = {
        "original_query": query,
        "current_query": query,
        "chunks": [],
        "answer": "",
        "is_grounded": False,
        "is_relevant": False,
        "iterations": 0,
        "max_iterations": max_iterations,
        "final_answer": "",
        "failed": False,
        "run_history": [],
        "chat_history": chat_history
    }
    result = rag_app.invoke(initial_state)
    return {
        "final_answer": result["final_answer"],
        "failed": result["failed"],
        "iterations": result["iterations"],
        "run_history": result.get("run_history", [])
    }

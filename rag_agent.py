import os
import random
import pickle
import re
import time
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import List, Dict
from collections import defaultdict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import logging
from config import (
    EMBEDDING_MODEL, VECTOR_DB_PATH, TOP_K_NODES, OLLAMA_MODEL, OLLAMA_BASE_URL,
    TEMPERATURE, MAX_CTX, MAX_SESSION_HISTORY, MAX_EPISODIC_MEMORY, MAX_GRAPH_NODES,
    USE_HEBBIAN_MEMORY, QUERY_MEMORY_FAISS_PATH, MEMORY_GRAPH_PATH, AGENT_STATE_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64}
)

try:
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=False)
except Exception as e:
    logging.warning(f"Safe load failed: {e}")
    logging.warning("Falling back to allow_dangerous_deserialization (DEMO/LOCAL USE ONLY!)")
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

all_docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in db.docstore._dict.values()]
bm25_retriever = BM25Retriever.from_documents(all_docs, k=TOP_K_NODES * 2)

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=TEMPERATURE,
    num_ctx=MAX_CTX,
    base_url=OLLAMA_BASE_URL
)

def infer_relevant_node_types(q: str) -> List[str]:
    q = q.lower()
    types = set()
    if any(w in q for w in ["disease", "cancer", "disorder", "syndrome"]): types.add("Disease")
    if any(w in q for w in ["symptom", "sign", "pain", "fever"]): types.add("Symptom")
    if any(w in q for w in ["gene", "mutation", "variant"]): types.add("Gene")
    if any(w in q for w in ["drug", "compound", "medication"]): types.add("Compound")
    if any(w in q for w in ["anatomy", "organ", "tissue", "cell"]): types.add("Anatomy")
    if any(w in q for w in ["pathway", "process", "mechanism"]): types.add("Biological Process")
    return list(types) or ["Disease", "Symptom", "Gene", "Compound", "Anatomy"]


def hybrid_retrieve(query: str, max_k: int = TOP_K_NODES * 3) -> List[Document]:
    types = infer_relevant_node_types(query)
    sem_docs = db.similarity_search(
        query,
        k=max_k,
        filter={"node_type": lambda x: x in types}
    )
    bm25_docs = bm25_retriever.invoke(query) if len(sem_docs) < max_k // 2 else []

    seen = set()
    scored = []
    for doc in sem_docs + bm25_docs:
        key = doc.page_content[:200].strip()
        if key in seen: continue
        seen.add(key)
        score = 1.0 if doc in sem_docs else 0.65
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in scored][:max_k]


def graph_aware_retrieve(query: str, max_docs: int = TOP_K_NODES) -> str:
    docs = hybrid_retrieve(query, max_k=max_docs * 2)
    if not docs:
        return ""

    lines = []
    for doc in docs[:max_docs]:
        m = doc.metadata
        name = m.get("name") or "Unknown"
        typ = m.get("node_type") or m.get("kind") or "Unknown"
        lines.append(f"→ {name} ({typ})")

        clean = re.sub(r'(?i)(DOID|GO|MESH|UMLS|CHEBI|ENTREZ|HGNC|HP|PC):[^\s:]+', '', doc.page_content)
        clean = re.sub(r'(?i)(symptom|disease|gene|compound|anatomy|pathway)::[^\s]+', '', clean)
        clean = re.sub(r'\bid:?\s*[a-z0-9:]+\b', '', clean, flags=re.I)
        clean = clean.strip()[:700]
        if clean and clean != name:
            lines.append(clean)
        lines.append("─" * 70)

    return "\n".join(lines)


def ultra_clean_answer(text: str) -> str:
    text = re.sub(r'(?is)(node id|this node|id:|::|doid:|go:|mesh:|umls:|Node).*?(?=\n\n|\Z)', '', text)
    text = re.sub(r'(?i)\b(disease|symptom|gene|compound)\s*:\s*[a-z0-9:]+\b', '', text)
    text = re.sub(r'\b(DOID|GO|MESH|UMLS|CHEBI|ENTREZ|HGNC):[^\s]+\b', '', text, flags=re.I)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^\s*[\*\-•]\s*', '', text, flags=re.MULTILINE)
    lines = [l.rstrip() for l in text.splitlines() if
             l.strip() and not re.search(r'(Node ID|ID:|::|DOID|GO:|MESH)', l, re.I)]
    return '\n\n'.join(lines).strip()

class HebbianNeuroGraphBrain:
    def __init__(self):
        self.hebbian_eta = 0.14
        self.threshold = 0.48
        self.memory_graph = nx.DiGraph()
        self.episodic_memory: List[Dict] = []
        self.k_adjust = 1.0
        self.use_memory = USE_HEBBIAN_MEMORY
        self.last_answer = ""
        self.session_id = None
        self.session_history = []
        self.current_query = ""
        self.current_step = 0
        self.prev_reward = 0.0
        self.query_cache: Dict[str, str] = {}
        self.query_memory_db = None
        self._init_query_memory()
        self._load_state()
        self.base_temperature = TEMPERATURE

    def _init_query_memory(self):
        if os.path.exists(QUERY_MEMORY_FAISS_PATH):
            try:
                self.query_memory_db = FAISS.load_local(
                    QUERY_MEMORY_FAISS_PATH, embeddings, allow_dangerous_deserialization=False
                )
                logging.info(f"Loaded query memory with {self.query_memory_db.index.ntotal} entries")
            except Exception as e:
                logging.warning(f"Query memory safe load failed: {e}")
                self.query_memory_db = FAISS.load_local(
                    QUERY_MEMORY_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
                )
        else:
            self.query_memory_db = FAISS.from_texts(["init"], embeddings)

    def start_new_session(self):
        self.session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
        self.session_history = []
        self.current_query = ""
        self.current_step = 0
        self.prev_reward = 0.0
        self.last_answer = ""
        logging.info(f"New session started: {self.session_id}")

    def add_to_session_history(self, role: str, content: str):
        if len(self.session_history) >= MAX_SESSION_HISTORY:
            self.session_history.pop(0)
        self.session_history.append({"role": role, "content": content})

    def get_session_context(self) -> str:
        if not self.session_history:
            return ""
        parts = [
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in self.session_history
        ]
        return "\n".join(parts) + "\n\n(Conversation history - for context only)"

    def _embed(self, text: str) -> torch.Tensor:
        prefixed = f"query: {text}"
        emb = embeddings.embed_query(prefixed)
        return torch.tensor(emb, dtype=torch.float32, device="cpu")

    def _load_state(self):
        if os.path.exists(MEMORY_GRAPH_PATH):
            try:
                with open(MEMORY_GRAPH_PATH, "rb") as f:
                    self.memory_graph = pickle.load(f)
                logging.info(f"Loaded Hebbian graph: {len(self.memory_graph.nodes)} nodes")
            except Exception as e:
                logging.warning(f"Graph load failed: {e}")
                self.memory_graph = nx.DiGraph()  

        if os.path.exists(AGENT_STATE_PATH):
            try:
                with open(AGENT_STATE_PATH, "rb") as f:
                    state = pickle.load(f)
                self.k_adjust = state.get("k_adjust", 1.0)
                self.use_memory = state.get("use_memory", True)
                self.episodic_memory = state.get("episodic_memory", [])

                for entry in self.episodic_memory:
                    if "emb" not in entry:
                        entry["emb"] = self._embed(entry["query"])

            except Exception as e:
                logging.warning(f"Agent state load failed: {e}")

    def save_state(self):
        try:
            with open(MEMORY_GRAPH_PATH, "wb") as f:
                pickle.dump(self.memory_graph, f)

            if self.query_memory_db:
                self.query_memory_db.save_local(QUERY_MEMORY_FAISS_PATH)

            state = {
                "k_adjust": self.k_adjust,
                "use_memory": self.use_memory,
                "episodic_memory": self.episodic_memory[-MAX_EPISODIC_MEMORY:],
            }
            with open(AGENT_STATE_PATH, "wb") as f:
                pickle.dump(state, f)

            logging.info("Agent state and graph saved successfully")

        except Exception as e:
            logging.warning(f"Save failed: {e}")

    def _prune_graph(self):
        if len(self.memory_graph.nodes) > MAX_GRAPH_NODES:
            weak_edges = [
                (u, v) for u, v, d in self.memory_graph.edges(data=True)
                if d.get('weight', 0) < 0.18
            ]
            for u, v in weak_edges:
                self.memory_graph.remove_edge(u, v)

            low_degree = [
                n for n in self.memory_graph.nodes
                if self.memory_graph.degree(n) < 2 and n.startswith("Q_")
            ]
            for n in low_degree[:300]:
                if n in self.memory_graph:
                    self.memory_graph.remove_node(n)
            logging.info(f"Pruned graph → {len(self.memory_graph.nodes)} nodes")

    def _add_query_to_memory(self, query: str, action: int, reward: float):
        try:
            self.query_memory_db.add_texts(
                [query],
                metadatas=[{"action": action, "reward": reward}]
            )
        except Exception:
            pass

    def _select_action(self) -> int:
        if self.query_memory_db and self.query_memory_db.index.ntotal > 8:
            try:
                similar = self.query_memory_db.similarity_search(self.current_query, k=10)
                votes = defaultdict(float)
                for doc in similar:
                    act = doc.metadata.get("action", 4)
                    rew = doc.metadata.get("reward", 0.5)
                    votes[act] += rew
                if votes:
                    best = max(votes, key=votes.get)
                    if best != 4 or len(votes) > 2:
                        return best
            except:
                pass

        if self.memory_graph.nodes:
            q_emb = self._embed(self.current_query)
            best_score = -1.0
            best_action = 4
            candidate_nodes = [n for n in list(self.memory_graph.nodes) if n.startswith("Q_")][:400]
            for node in candidate_nodes:
                if "emb" in self.memory_graph.nodes[node]:
                    n_emb = self.memory_graph.nodes[node]["emb"]
                    sim = F.cosine_similarity(q_emb.unsqueeze(0), n_emb.unsqueeze(0)).item()
                    if sim > self.threshold and sim > best_score:
                        best_score = sim
                        for neigh in self.memory_graph.successors(node):
                            if neigh.startswith("A_"):
                                act = int(neigh[2:])
                                if 0 <= act <= 7:
                                    best_action = act
                                    break
            return best_action

        return 4

    def _retrieve(self) -> str:
        parts = []
        if self.use_memory and self.episodic_memory:
            q_emb = self._embed(self.current_query)
            scored = []
            for entry in self.episodic_memory[-120:]:
                sim = F.cosine_similarity(q_emb.unsqueeze(0), entry["emb"].unsqueeze(0)).item()
                if sim > 0.63:
                    scored.append((sim, entry))
            scored.sort(reverse=True, key=lambda x: x[0])
            for _, entry in scored[:4]:
                parts.append(f"Similar past experience:\n{entry['context'][:680]}")

        parts.append(graph_aware_retrieve(self.current_query, max_docs=int(TOP_K_NODES * self.k_adjust)))
        return "\n\n".join([p for p in parts if p.strip()])

    def _adaptive_llm(self, confidence: float):
        if confidence < 0.45:
            temp = min(0.9, self.base_temperature + 0.3)
        elif confidence > 0.8:
            temp = max(0.1, self.base_temperature - 0.2)
        else:
            temp = self.base_temperature

        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=temp,
            num_ctx=MAX_CTX,
            base_url=OLLAMA_BASE_URL
        )

    def _llm_secondary_judge(self, answer: str, context: str) -> float:
        try:
            prompt = f"""
    You are a second independent biomedical reviewer.

    Context:
    {context[:1200]}

    Answer:
    {answer[:1200]}

    Score from 0 to 1 for factual grounding only.
    Return only a number.
    Score:
    """
            out = llm.invoke(prompt).content.strip()
            score = float(re.findall(r"\d*\.?\d+", out)[0])
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def _bayesian_uncertainty(self, scores: List[float]) -> float:
        if not scores:
            return 0.5

        mean = np.mean(scores)
        var = np.var(scores)
        uncertainty = var * 3 + (1 - mean) * 0.2

        return max(0.0, min(1.0, uncertainty))

    def _hebbian_update(self, action: int, reward: float):
        if reward < 0.42:
            return

        q_node = f"Q_{hash(self.current_query) % 9999999}"
        a_node = f"A_{action}"

        if q_node not in self.memory_graph:
            self.memory_graph.add_node(q_node, emb=self._embed(self.current_query))

        if a_node not in self.memory_graph:
            self.memory_graph.add_node(a_node)

        w = reward * 0.8
        if self.memory_graph.has_edge(q_node, a_node):
            w = self.memory_graph[q_node][a_node]['weight'] + reward * self.hebbian_eta

        self.memory_graph.add_edge(q_node, a_node, weight=w)
        self._prune_graph()
        self._add_query_to_memory(self.current_query, action, reward)

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        try:
            e1 = self._embed(text1)
            e2 = self._embed(text2)
            sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
            return max(0.0, min(1.0, (sim + 1) / 2))
        except:
            return 0.0

    def _coverage_score(self, answer: str, context: str) -> float:
        if not context.strip():
            return 0.5
        return self._semantic_similarity(answer[:1200], context[:1200])

    def _hallucination_penalty(self, answer: str, context: str) -> float:
        if not context.strip():
            return 0.0

        context_emb = self._embed(context)
        sentences = re.split(r'(?<=[.!?]) +', answer)

        penalty = 0.0
        for s in sentences:
            if len(s.strip()) < 30:
                continue
            sim = F.cosine_similarity(
                self._embed(s).unsqueeze(0),
                context_emb.unsqueeze(0)
            ).item()
            if sim < 0.35:
                penalty += 0.05

        return min(0.4, penalty)


    def _llm_self_critique(self, answer: str, context: str) -> float:
        try:
            critique_prompt = f"""
You are a strict biomedical reviewer.

Context:
{context[:1500]}

Answer:
{answer[:1500]}

Score from 0 to 1 based on:
- Scientific accuracy
- Clarity
- Completeness
- No hallucinations

Return ONLY a number.
Score:
"""
            critique = llm.invoke(critique_prompt).content.strip()
            score = float(re.findall(r"\d*\.?\d+", critique)[0])
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def _faithfulness_score(self, answer: str, context: str) -> float:
        if not context.strip():
            return 0.5

        sim = self._semantic_similarity(answer, context)
        return sim

    def _confidence_score(self, reward: float, coverage: float, halluc_penalty: float) -> float:

        conf = reward * 0.6 + coverage * 0.3 + (1 - halluc_penalty) * 0.1
        return max(0.0, min(1.0, conf))

    def evaluate_answer(self, answer: str, context: str) -> Dict:

        if not answer.strip():
            return {"final_score": 0.0}

        length_score = min(0.15, len(answer) / 1800)

        coverage = self._coverage_score(answer, context)
        halluc_penalty = self._hallucination_penalty(answer, context)
        faithfulness = self._faithfulness_score(answer, context)

        llm_score_primary = self._llm_self_critique(answer, context)
        llm_score_secondary = self._llm_secondary_judge(answer, context)

        uncertainty = self._bayesian_uncertainty(
            [coverage, faithfulness, llm_score_primary, llm_score_secondary]
        )

        final_score = (
                0.25 +
                length_score +
                coverage * 0.2 +
                faithfulness * 0.2 +
                llm_score_primary * 0.15 +
                llm_score_secondary * 0.15 -
                halluc_penalty
        )

        final_score = max(0.0, min(1.0, final_score))
        confidence = max(0.0, min(1.0, final_score * (1 - uncertainty)))

        return {
            "final_score": final_score,
            "length_score": length_score,
            "coverage": coverage,
            "faithfulness": faithfulness,
            "llm_score_primary": llm_score_primary,
            "llm_score_secondary": llm_score_secondary,
            "hallucination_penalty": halluc_penalty,
            "uncertainty": uncertainty,
            "confidence": confidence
        }

    def answer_query(self, query: str, max_steps: int = 4) -> str:
        if query in self.query_cache:
            return self.query_cache[query]

        self.current_query = query.strip()
        self.current_step = 0
        self.prev_reward = 0.0
        reward_sum = 0.0
        session_context = self.get_session_context()
        final_answer = ""

        for step in range(max_steps):
            action = self._select_action()

            if action == 0:
                prompt = f"Rephrase this query to get better biomedical results:\n{self.current_query}"
                new_q = llm.invoke(prompt).content.strip()
                if new_q and new_q != self.current_query:
                    self.current_query = new_q
                    reward_sum += 0.15
                continue

            elif action == 1:
                self.k_adjust = min(3.0, self.k_adjust + 0.45)
                reward_sum += 0.08
                continue

            elif action == 2:
                self.k_adjust = max(0.4, self.k_adjust - 0.3)
                reward_sum += 0.10
                continue

            elif action == 3:
                self.use_memory = True
                reward_sum += 0.06
                continue

            elif action == 5:
                self.save_state()
                reward_sum += 0.04
                continue

            raw_context = self._retrieve()
            generator = llm
            prompt = f"""You are a world-class medical expert. Answer in natural, flowing paragraphs.
Previous conversation (for context only, never mention it):
{session_context}

Biomedical reference information (use the facts, never quote IDs, nodes, technical codes):
{raw_context}

Current question: {self.current_query}

Rules:
- Write ONLY continuous paragraphs. No bullets, no lists, no markdown, no * or **.
- Never mention sources, databases, nodes, IDs, DOID, GO, MESH, etc.
- Start directly with the answer.
- Be precise, scientific but very readable.
Answer:"""

            raw_answer = generator.invoke(prompt).content.strip()
            final_answer = ultra_clean_answer(raw_answer)

            eval_metrics = self.evaluate_answer(final_answer, raw_context)
            if eval_metrics["confidence"] < 0.55:
                print("Low confidence detected → Self-reflection triggered\n")

                reflection_prompt = f"""
            Improve the following biomedical answer.
            Fix hallucinations.
            Increase grounding in context.

            Context:
            {raw_context[:1500]}

            Original Answer:
            {final_answer}

            Improved Answer:
            """
                improved = generator.invoke(reflection_prompt).content.strip()
                improved = ultra_clean_answer(improved)

                new_metrics = self.evaluate_answer(improved, raw_context)

                if new_metrics["final_score"] > eval_metrics["final_score"]:
                    final_answer = improved
                    eval_metrics = new_metrics
            generator = self._adaptive_llm(eval_metrics["confidence"])
            reward = eval_metrics["final_score"]
            confidence = eval_metrics["confidence"]
            reward_sum += reward
            print("\n=========== Evaluation Report ===========")
            for k, v in eval_metrics.items():
                print(f"{k:25s}: {round(v, 4)}")
            print("=========================================\n")

            self.add_to_session_history("user", self.current_query)
            self.add_to_session_history("assistant", final_answer)

            self.episodic_memory.append({
                "query": self.current_query,
                "context": raw_context,
                "answer": final_answer,
                "reward": reward,
                "emb": self._embed(self.current_query)
            })

            self.last_answer = final_answer
            self._hebbian_update(action, reward)
            break

        if not final_answer:
            final_answer = "Sorry! iwas not able to generate appropriate answer! Ask the question in a better way!"

        self.query_cache[query] = final_answer
        if len(self.query_cache) > 50:
            self.query_cache.pop(next(iter(self.query_cache)))

        self.save_state()
        return final_answer



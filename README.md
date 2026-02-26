<!-- Badges Section -->
<p  align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  </a>
  <a href="https://github.com/your-repo">
    <img src="https://img.shields.io/badge/Agentic-AI-green.svg" alt="Agentic AI Project">
  </a>
</p>

## 🧠 Adaptive Biomedical AI Engine (MediChat Demo Edition)

Hebbian-NeuroGraph is a **self-evolving biomedical AI agent** that integrates hybrid retrieval, Hebbian structural memory, episodic memory, multi-LLM evaluation, and Bayesian confidence modeling to deliver reliable, context-grounded medical reasoning.

Designed for **research**, **clinical deployment**, and **startup applications**, offering transparency, trustworthiness, and adaptive intelligence.

---

## 📌 Table of Contents

1. [Problem Statement](#problem-statement)  
2. [Solution Overview](#solution-overview)  
3. [Architecture](#architecture)  
4. [Core Capabilities](#core-capabilities)  
5. [Evaluation Engine](#evaluation-engine)  
6. [Memory Systems](#memory-systems)  
7. [Action & Learning Loops](#action--learning-loops)  
8. [Startup Use Case & Market Fit](#startup-use-case--market-fit)  
9. [Tech Stack](#tech-stack)  
10. [Running the Agent](#running-the-agent)  
11. [Evaluation Results](#evaluation-results)  
---

## 🛑 Problem Statement

Modern LLM-powered biomedical assistants often suffer from:

- Hallucinated medical facts  
- Inconsistent reasoning  
- Stateless interactions  
- Weak uncertainty estimation  
- Poor clinical-grade evaluation  

Healthcare environments require:

- Transparent, measurable confidence  
- Persistent memory & learning  
- Risk-aware reasoning  
- Continuous improvement  

---

## 💡 Solution Overview

Hebbian-NeuroGraph provides a **self-correcting, memory-reinforced biomedical AI system**:

- Grounds answers in curated biomedical knowledge  
- Quantifies confidence using multi-metric scoring + Bayesian uncertainty  
- Self-corrects low-quality responses (Reflexion Loop)  
- Reinforces successful reasoning in the Hebbian memory graph  
- Adapts over time through episodic memory and query-vector reinforcement  

This system is suitable for **research, clinical, and commercial deployment**.

---

## 🏗 Architecture
<p align="center">
   <img width="1280" height="720" alt="Agent Architecture" src="https://github.com/user-attachments/assets/cc06d4e8-ecbc-4bf1-8582-f74438e3256b" />
</p>

**Core Modules:**

- **Hybrid Retrieval:** Dense (FAISS/HuggingFace) + Sparse (BM25)  
- **Graph-Aware Context:** Removes ontology/technical codes for clean context  
- **Generation Engine:** LLM + adaptive temperature  
- **Multi-Metric Evaluation:** Coverage, faithfulness, hallucination, LLM self-critique, Bayesian uncertainty  
- **Reflexion Loop:** Self-corrects low-confidence answers  
- **Memory Systems:** Episodic memory, query vector memory, Hebbian structural graph  
- **Action Selector:** Policy-driven query transformations & retrieval adjustments  

---

## 🧠 Core Capabilities

### 1️⃣ Hybrid Retrieval

- Dense + sparse combination  
- Node-type filtering (Disease, Symptom, Gene, Compound, Anatomy)  
- Graph-aware context building  

### 2️⃣ Multi-Layer Memory

| Memory | Role |
|--------|------|
| **Episodic Memory** | Stores past queries, context, answer, reward, embedding |
| **Query Vector Memory** | Stores query-action-reward embeddings for policy reinforcement |
| **Hebbian Graph** | Q→A nodes with reward-weighted edges for adaptive action selection |

### 3️⃣ Reflexion & Self-Correction

- Triggered if confidence < threshold  
- Improves answer using the same LLM with enhanced context  
- Keeps the higher-scoring version  

### 4️⃣ Adaptive Generation

- Temperature adapts with confidence  
- Balances exploration vs. exploitation  
- Reduces hallucinations  

---

## 🔬 Evaluation Engine

Metrics used for each answer:

1. **Length Score** – penalizes too-short answers  
2. **Coverage Score** – semantic similarity with context  
3. **Faithfulness Score** – global semantic similarity  
4. **Hallucination Penalty** – sentence-level semantic mismatches  
5. **LLM Primary Critique** – self-review  
6. **LLM Secondary Judge** – independent evaluation  
7. **Bayesian Uncertainty** – variance-based confidence adjustment  
8. **Final Score → Confidence** – used for Reflexion & Hebbian update  

---

## 🔁 Action & Learning Loops

### Actions

| Action | Behavior |
|--------|----------|
| 0 | Rephrase query |
| 1 | Increase retrieval depth (k) |
| 2 | Decrease retrieval depth (k) |
| 3 | Enable episodic memory usage |
| 4 | Standard retrieval & generation |
| 5 | Save state & memory |

### Learning Loops

- **Reflexion Loop:** Low-confidence → regenerate → evaluate → select better answer  
- **Reinforcement Loop:** Evaluation → Hebbian graph → update future action selection  

---

## 🏥 Startup Use Case & Market Fit

**Targets:**

- Private clinics & hospitals  
- Biotech & pharma research teams  
- Digital health platforms  
- Medical education & research labs  

**Value Proposition:**

- Clinically grounded, measurable confidence  
- Self-learning, improving AI over time  
- Configurable & deployable in secure on-premise environments  
- Scalable for multiple clinical domains  

**Revenue Opportunities:**

- API licensing for AI-assisted diagnostics  
- White-labeled clinical assistants  
- Research-grade AI subscription for hospitals & labs  
- AI-backed drug repurposing & literature insights  

---

## 🧱 Tech Stack

- FAISS + HuggingFace embeddings (dense retrieval)  
- BM25 (sparse retrieval)  
- Ollama LLM backend (on-premise)  
- PyTorch (similarity scoring)  
- NetworkX (memory graph)  
- Python 3.10+  
---
## ⚙️ Running the Agent

```bash
python app.py
```
---
## Automatic features
- Memory updates (episodic, query vector, Hebbian graph)
- Adaptive policy & temperature
- Persistent state across sessions
---
## 📊 Evaluation Results (Sample)

Benchmark: 20 representative biomedical queries

|Metric|Value|
|------|-----|
|**Final Score**|	1.0|
|**Length Score**|0.15|
|**Coverage**|0.896|
|**Faithfulness**|0.907|
|**LLM Primary Score**|0.95|
|**LLM Secondary Score**|0.9|
|**Hallucination Penalty**|0.0|
|**Uncertainty**|0.019|
|**Confidence**|0.981|

---
## 📬 Contact
For questions, feel free to reach out:

📧 alaeddini.zahra@gmail.com

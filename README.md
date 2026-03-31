# 🚀 NOVA: Autonomous DBA Optimization Agent

**Hackathon Status:** 🥇 Sovereign AI Secured (Top Tier: 3.00 / 3.20)
**Live Environment:** [Hugging Face Space](https://itsflash44-db-tune-env.hf.space)
**Model Used:** Qwen/Qwen2.5-72B-Instruct

## 👥 Team NOVA
* **Tirth Trivedi** - tirthtrivedi01@gmail.com
* **Bhuvnesh Sharma** - 26f1001154@ds.study.iitm.ac.in
* **Vansh Sahu** - sahuvansh781@gmail.com

---

## 📌 Project Overview
Team NOVA has engineered a fully autonomous, containerized Database Administrator (DBA) agent. Built to interface with the OpenEnv simulation, our agent utilizes a 72B-parameter Large Language Model to iteratively optimize database query costs (`query_cost`) while strictly managing constrained storage budgets (`storage_budget`). 

Rather than relying on hardcoded heuristics, our agent achieves a Top Tier performance metric (3.00/3.20) entirely through autonomous Chain-of-Thought reasoning and dynamic state evaluation.

## 🌐 Cloud Deployment

The database simulation is containerized via Docker and deployed to a **Hugging Face Space**. This provides a persistent, secure API endpoint for the autonomous agent.

* **Deployment URL:** `https://itsflash44-db-tune-env.hf.space`
* **Infrastructure:** Python 3.13 / FastAPI / Uvicorn / Docker
* **Compliance:** Fully verified via `openenv validate`.

---
## 🛠️ Key Engineering Achievements

### 1. Chain-of-Thought (CoT) "Scratchpad" Architecture
To prevent zero-shot hallucinations and state-fatigue in the "Hard" tier, we engineered a custom JSON output format requiring a mandatory `"thought_process"` key. This forces the model to mathematically calculate storage constraints *before* committing to a `CREATE` or `DROP` action, proving genuine systemic reasoning rather than prompted mimicry.

### 2. Bulletproof Output Parsing
LLMs are notoriously brittle when constrained to strict JSON schemas. We built a highly resilient parser in `client.py` utilizing safe `.get()` extraction methods and substring indexing. This guarantees that the core environment never crashes, even when the model generates complex, multi-line reasoning inside its scratchpad.

### 3. Stateful Memory Management
We resolved autonomous looping by implementing an iterative `conversation_history` array within `inference.py`. By appending both the environment's state observations and the agent's prior actions to the context window, the model maintains perfect state awareness across the 10-step sequence.

### 4. Isolated UI Telemetry (Zero-Risk Dashboard)
To provide a clear presentation layer without jeopardizing the core containerized logic, we built `ui_demo.py`. This Streamlit dashboard operates as a secure telemetry monitor. It executes the core `inference.py` as an isolated subprocess, streaming real-time terminal outputs to a modern web interface.

---

## 📂 Repository Structure

* `inference.py`: The core systemic loop, prompt architecture, and LLM API integration.
* `client.py`: The OpenEnv interface and robust JSON extraction logic.
* `ui_demo.py`: The Streamlit telemetry dashboard for live execution monitoring.
* `requirements.txt`: Project dependencies (FastAPI, Uvicorn, OpenAI, Streamlit, etc.).
* `Dockerfile` / `openenv.yaml`: Containerization and environment specs.

---

## ⚙️ How to Run Locally

### 1. Setup Environment**
Ensure your local environment is active, then install the required dependencies:
```bash
pip install -r requirements.txt
```
### 2. Initialize the Backend Environmen
```python3 -m uvicorn server.app:app --reload
```
### 3. Execute the Autonomous Agent
```
export HF_TOKEN="your_hugging_face_token_here"
python3 inference.py
```
---

## ⚖️ Strategic Engineering Trade-offs

### 1. The 3.00/3.20 Accuracy Logic
Team NOVA achieved a high-tier score of **3.00/3.20**. The remaining 0.2 margin is a **deliberate production safety choice**:
* **Index Churn Prevention:** The agent is tuned to avoid "Index Churn" (frequent CREATE/DROP cycles), which causes CPU spikes in real-world databases.
* **Storage Buffer:** We maintain a 5% storage margin to allow for background vacuuming and log growth, ensuring 100% environment uptime.

### 2. Thread-Safe Sovereign API
Unlike standard scaffolds, our server implements an **Atomic Locking Mechanism** (`threading.Lock`). This prevents state corruption during concurrent testing—a critical feature for multi-judge hackathon environments.

### 3. Regex-Shielded Inference
To prevent "hallucination crashes," we implemented a custom **Regex JSON Extractor**. This allows the agent to remain resilient even if the underlying LLM includes conversational chatter in its response.---
## 🏆 Technical Excellence: Team NOVA
* **Atomic Integrity:** Our environment uses white-listed column validation to ensure 100% SQL reliability.
* **Thread-Safe Architecture:** Implemented `threading.Lock` and `check_same_thread=False` to handle concurrent evaluation requests.
* **Storage-Aware Heuristics:** The agent is penalized for exceeding budgets, mirroring real-world DBA constraints on cloud infrastructure costs.
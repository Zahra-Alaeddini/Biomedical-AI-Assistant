EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

VECTOR_DB_PATH          = "./vector_db"
QUERY_MEMORY_FAISS_PATH = "./query_memory_faiss"
MEMORY_GRAPH_PATH       = "./hebbian_memory_graph.pkl"
AGENT_STATE_PATH        = "./hebbian_agent_state.pkl"

TOP_K_NODES    = 10
K_HOP          = 2

OLLAMA_MODEL    = "gemma3:1b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
TEMPERATURE     = 0.0
MAX_CTX         = 16384

MAX_SESSION_HISTORY   = 12
MAX_EPISODIC_MEMORY   = 500
MAX_GRAPH_NODES       = 1500

USE_HEBBIAN_MEMORY    = True

HEBBIAN_ETA    = 0.14
SIMILARITY_THRESHOLD = 0.48
EPISODIC_SIM_THRESHOLD = 0.63
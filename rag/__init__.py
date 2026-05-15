# Lazy package — do not eagerly import chromadb-dependent modules here.
# Streamlit Cloud (Python 3.14) fails if chromadb/opentelemetry protobuf
# generated files are imported at package-load time before the env is ready.
# app.py imports directly from rag.ingestion / rag.retriever / rag.prompts.

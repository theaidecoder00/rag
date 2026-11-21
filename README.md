# RAG from scratch in Python

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key_here
    python rag.py

Put your own .txt files in `docs/` (three sample company policies are included).
First run downloads the embedding model (all-MiniLM-L6-v2). Press Enter on an empty prompt to quit.
Swap `MODEL` for any Claude model, or replace the `client.messages.create` call with another provider.

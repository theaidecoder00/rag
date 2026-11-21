import glob
import os

import numpy as np
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = Anthropic()  # reads ANTHROPIC_API_KEY
MODEL = "claude-sonnet-5-5"


def chunk(text, size=120, overlap=20):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size])
            for i in range(0, len(words), step)]


def load_chunks(folder):
    items = []
    for path in glob.glob(f"{folder}/*.txt"):
        text = open(path, encoding="utf-8").read()
        for piece in chunk(text):
            items.append({"text": piece,
                          "source": os.path.basename(path)})
    return items


def build_index(items):
    texts = [it["text"] for it in items]
    return embedder.encode(texts, normalize_embeddings=True)


def retrieve(question, items, index, k=3):
    q = embedder.encode([question],
                        normalize_embeddings=True)[0]
    scores = index @ q
    best = np.argsort(scores)[::-1][:k]
    return [items[i] for i in best]


PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}"""


def answer(question, items, index, k=3):
    hits = retrieve(question, items, index, k)
    context = "\n\n".join(f"[{h['source']}] {h['text']}"
                          for h in hits)
    prompt = PROMPT.format(context=context,
                           question=question)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    sources = sorted({h["source"] for h in hits})
    return msg.content[0].text, sources


if __name__ == "__main__":
    items = load_chunks("docs")
    index = build_index(items)
    while True:
        q = input("\nAsk: ")
        if not q.strip():
            break
        text, sources = answer(q, items, index)
        print(f"\n{text}\n\nSources: {', '.join(sources)}")

import chromadb
import ollama

DB_DIR = "./chorma_db"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name="pydantic_docs")

def retrieve(question: str, top_k: int = 5):
    q_embed = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    results = collection.query(query_embeddings=[q_embed], n_results=top_k)

    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return chunks, sources

def ask(question: str, top_k: int = 5) -> dict:
    chunks, sources = retrieve(question, top_k=top_k)
    context = "\n\n---\n\n".join(
        f"\n{chunk}" for src, chunk in zip(sources, chunks)
    )

    prompt = f"""You are a helpful Pydantic documentation assistant.
Answer the user's question using ONLY the context below.
If the context does not contain the answer, say "I don't know based on the provided docs."
Always cite the source filename(s) you used.

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": response["message"]["content"],
        "sources": list(dict.fromkeys(sources)),
    }

if __name__ == "__main__":
    result = ask("How do I create a custom field validator in Pydantic?")
    print(result["answer"])
    print("\nSources:", result["sources"])

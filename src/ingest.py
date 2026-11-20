from pathlib import Path
import chromadb
import ollama

DB_DIR = './chorma_db'
DATA_DIR = Path("data")
EMBED_MODEL = "nomic-embed-text"

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Spilliting the text into chunks with an overlap so context isn't cut half"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = start + chunk_size - overlap
        
    return chunks


def main():
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    try:
        chroma_client.delete_collection("pydantic_docs")
        
    except Exception:
        pass
    
    collection = chroma_client.create_collection(name="pydantic_docs")
    
    md_files = list(DATA_DIR.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    doc_id = 0
    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            response = ollama.embed(model=EMBED_MODEL, input=chunk)
            embedding = response["embeddings"][0]

            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": file_path.name, "chunk_index": i}],
            )
            doc_id += 1

    print(f"Indexed {doc_id} chunks into ChromaDB.")

if __name__ == "__main__":
    main()
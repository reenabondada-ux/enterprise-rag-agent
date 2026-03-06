# services/ingestor/ingest.py
import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from splitters import fixed_split
from services.shared import embed_texts
from services.shared import DATABASE_URL, EMBEDDING_DIM

load_dotenv()


def ensure_schema():
    connection = psycopg2.connect(DATABASE_URL)
    cursor = connection.cursor()

    # Create vector extension (pgvector) and documents table
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # register pgvector adapter after extension exists
    register_vector(connection)
    cursor.execute(
        f"""    
        CREATE TABLE IF NOT EXISTS documents (        
            id SERIAL PRIMARY KEY,        
            title TEXT,        
            text TEXT,        
            embedding vector({EMBEDDING_DIM})    
        );
        """
    )

    connection.commit()
    cursor.close()
    connection.close()
    print("Schema ensured (extension + table)")


def upsert_doc(title, text, embedding):
    connection = psycopg2.connect(DATABASE_URL)
    register_vector(connection)
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO documents (title, text, embedding) VALUES (%s, %s, %s)",
        (title, text, embedding),
    )
    connection.commit()
    cursor.close()
    connection.close()


def ingest_text_file(path, title=None):
    """Ingest a single text file, splitting into chunks."""
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    title = title or os.path.basename(path)
    # simple fixed splitter; replace with semantic chunker later
    chunks = fixed_split(text, chunk_size=1000, overlap=200)
    embeddings = embed_texts(chunks)
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        upsert_doc(f"{title}_chunk_{i}", chunk, emb)
    print(f"✓ Ingested {len(chunks)} chunks from {title}")


def ingest_directory(dir_path, extension=".txt"):
    """Ingest all files with given extension from a directory."""
    if not os.path.isdir(dir_path):
        raise ValueError(f"Directory not found: {dir_path}")

    files = [f for f in os.listdir(dir_path) if f.endswith(extension)]
    if not files:
        print(f"No {extension} files found in {dir_path}")
        return

    print(f"Found {len(files)} files to ingest...")
    for filename in files:
        file_path = os.path.join(dir_path, filename)
        try:
            ingest_text_file(file_path)
        except Exception as e:
            print(f"✗ Failed to ingest {filename}: {e}")

    print(f"\n✓ Completed ingestion of {len(files)} files")


if __name__ == "__main__":
    ensure_schema()

    # Ingest sample documents from data/samples/ directory
    samples_dir = os.path.join(os.path.dirname(__file__), "../../data/samples")
    if os.path.isdir(samples_dir):
        print(f"Ingesting documents from {samples_dir}...")
        ingest_directory(samples_dir, extension=".txt")
    else:
        print(f"Sample directory not found: {samples_dir}")
        print("Creating a single example document...")
        sample_text = (
            "This is a small sample document used as an example for the RAG ingest "
            "flow."
        )
        sample_embedding = embed_texts([sample_text])[0]
        upsert_doc("Sample doc", sample_text, sample_embedding)
        print("✓ Inserted sample doc")

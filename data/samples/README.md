# Sample Documents

This directory contains sample documents for testing the Enterprise RAG Agent.

## Files

- **product_overview.txt** - Overview of the RAG system features and use cases
- **technical_architecture.txt** - System architecture and component details
- **deployment_guide.txt** - Instructions for deploying the system
- **troubleshooting.txt** - Common issues and solutions
- **api_reference.txt** - API endpoint documentation

## Ingesting Documents

To ingest these documents into the system:

```bash
cd /path/to/enterprise-rag-agent
PYTHONPATH=. python services/ingestor/ingest.py
```

The ingestor will automatically:
1. Find all supported files in this directory
2. Split them into semantic chunks
3. Generate embeddings using local SBERT
4. Store them in the PostgreSQL database with pgvector

## Adding Your Own Documents

To add your own documents:

1. Place text files (.txt) in this directory
2. Run the ingestor script
3. Query the system via the API

For other formats (PDF, Word, etc.), you'll need to:
- Add document parsers to the ingestor
- Convert to text before chunking
- Or use libraries like pypdf, python-docx, etc.

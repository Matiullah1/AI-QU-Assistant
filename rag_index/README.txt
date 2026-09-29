UNIVERSITY ACADEMIC RAG INDEX
==============================

Embedding model:
BAAI/bge-m3

Embedding dimension:
1024

Vector database:
FAISS

FAISS index:
IndexFlatIP

Similarity:
Cosine similarity

Normalized embeddings:
Yes

Chunk size:
700

Chunk overlap:
120

Documents:
7

Pages:
1265

Chunks:
5348


FILES
-----

index.faiss
    FAISS vector database.

metadata.json
    Text chunks and source metadata.

embedding_config.json
    Embedding configuration.

document_manifest.json
    Original PDF information and SHA-256 hashes.

README.txt
    This file.


VECTOR / METADATA MAPPING
--------------------------

FAISS vector 0
    -> metadata records[0]

FAISS vector 1
    -> metadata records[1]

FAISS vector 2
    -> metadata records[2]

...


SOURCE TRACEABILITY
-------------------

Every chunk contains:

source_file
page_number
total_pages
chunk_id
chunk_number_on_page
chunk_characters
document_type
text
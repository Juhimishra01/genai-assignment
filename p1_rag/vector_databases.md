# Vector Databases and Cost

A vector database stores embeddings, which are lists of numbers that represent the meaning of a piece of text. The all-MiniLM-L6-v2 embedding model produces vectors with 384 dimensions, while many larger models produce 768 or 1536 dimensions.

## Storage arithmetic

Each dimension is usually stored as a 32-bit float, which takes 4 bytes. A 384-dimension vector therefore needs 1,536 bytes before any index overhead. One million such vectors need roughly 1.5 gigabytes of raw storage, and graph indexes such as HNSW add about fifty percent on top.

## Managed versus self-hosted

Managed vector databases charge for always-on capacity, often called pods, so the monthly bill grows with the number of stored vectors even when nobody is querying. Self-hosted options such as ChromaDB, Qdrant, LanceDB, FAISS and sqlite-vec run on a single virtual machine whose price depends mostly on RAM. The trade-off is that the team must handle backups, upgrades and monitoring itself.

## Idempotent ingestion

Re-running ingestion should never create duplicate vectors. A reliable approach is to derive each chunk's identifier from a hash of its source file, position and text, and to use an upsert operation. Deleting the old vectors of a file before re-adding them also handles edited files.

## Metadata filters

Vectors can carry metadata such as the source file name, document type and page number. A metadata filter restricts similarity search to a subset, for example only PDF documents.

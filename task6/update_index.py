import os
import json
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

SOURCE_PATH = os.getenv("SOURCE_PATH", "../task2/knowledge_base/star_wars_planets_dataset.json")
CHROMA_DIR = os.getenv("CHROMA_DIR", "../task3/chroma_starwars")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "starwars_planets")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))         # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))   # characters

# Update behavior
DELETE_STALE = os.getenv("DELETE_STALE", "true").lower() in ("1", "true", "yes")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "/var/log/indexer.log")


@dataclass
class Doc:
    doc_id: str
    text: str
    meta: dict[str, Any]
    doc_hash: str


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("indexer")
    logger.setLevel(LOG_LEVEL)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # If file logging fails, continue with stdout logging
        pass

    return logger


LOGGER = setup_logger()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks


def coerce_doc_text(item: Any) -> tuple[str, dict[str, Any]]:
    """
    Robust text extraction:
    - if item is str -> text=item
    - if dict -> tries common fields; else join all values as text
    """
    if isinstance(item, str):
        return item, {}

    if isinstance(item, dict):
        # Try common text fields
        for key in ("rag_text", "text", "description", "summary", "content"):
            if key in item and item[key] is not None:
                return str(item[key]), item

        # Fallback: join values
        joined = " | ".join(str(v) for v in item.values() if v is not None)
        return joined, item

    # Fallback: stringify anything
    return str(item), {}


def load_source_docs(path: str) -> list[Doc]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: list[Doc] = []

    # JSON could be: list[str] OR list[dict] OR dict(...)
    if isinstance(data, list):
        for i, item in enumerate(data):
            text, meta = coerce_doc_text(item)
            doc_id = str(meta.get("id", i))  # allow stable id if present
            doc_hash = sha256_text(text)
            docs.append(Doc(doc_id=doc_id, text=text, meta=meta, doc_hash=doc_hash))

    elif isinstance(data, dict):
        # If dict of items: keys are ids
        for k, v in data.items():
            text, meta = coerce_doc_text(v)
            doc_id = str(k)
            doc_hash = sha256_text(text)
            # merge key into meta if helpful
            if isinstance(meta, dict):
                meta = dict(meta)
                meta.setdefault("id", doc_id)
            docs.append(Doc(doc_id=doc_id, text=text, meta=meta, doc_hash=doc_hash))

    else:
        # Rare fallback
        text = str(data)
        docs.append(Doc(doc_id="0", text=text, meta={}, doc_hash=sha256_text(text)))

    return docs


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection


def get_existing_doc_hashes(collection) -> dict[str, str]:
    """
    Build doc_id -> doc_hash map from existing chunks.
    We assume each chunk has metadata: doc_id, doc_hash.
    """
    existing: dict[str, str] = {}
    # Pull metadatas and ids; for large collections, you'd paginate.
    res = collection.get(include=["metadatas"], limit=10_000_000)
    metas = res.get("metadatas") or []

    for m in metas:
        if not isinstance(m, dict):
            continue
        doc_id = m.get("doc_id")
        doc_hash = m.get("doc_hash")
        if doc_id and doc_hash:
            # all chunks of same doc share same doc_hash; last write wins
            existing[str(doc_id)] = str(doc_hash)

    return existing


def delete_doc_chunks(collection, doc_id: str):
    # delete all chunks for doc_id
    collection.delete(where={"doc_id": doc_id})


def add_chunks(
    collection,
    embed_model: SentenceTransformer,
    doc: Doc,
    chunks: list[str]
):
    if not chunks:
        return 0

    # deterministic chunk IDs
    # (doc_id + doc_hash prefix keeps IDs stable across reruns, but changes on doc change)
    hash_prefix = doc.doc_hash[:12]
    ids = [f"{doc.doc_id}:{hash_prefix}:{i}" for i in range(len(chunks))]

    # keep metadatas lightweight (Chroma metadata should be JSON-serializable primitives)
    metadatas = []
    for i in range(len(chunks)):
        metadatas.append({
            "doc_id": doc.doc_id,
            "doc_hash": doc.doc_hash,
            "chunk_index": i,
            "source_path": SOURCE_PATH,
        })

    # embeddings
    embs = embed_model.encode(chunks, convert_to_numpy=True).tolist()

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embs
    )
    return len(chunks)


def main():
    start_ts = time.time()
    LOGGER.info("Starting index update")
    LOGGER.info(f"SOURCE_PATH={SOURCE_PATH}")
    LOGGER.info(f"CHROMA_DIR={CHROMA_DIR} COLLECTION_NAME={COLLECTION_NAME}")
    LOGGER.info(f"EMBED_MODEL_NAME={EMBED_MODEL_NAME}")
    LOGGER.info(f"CHUNK_SIZE={CHUNK_SIZE} CHUNK_OVERLAP={CHUNK_OVERLAP} DELETE_STALE={DELETE_STALE}")

    # Load docs
    docs = load_source_docs(SOURCE_PATH)
    LOGGER.info(f"Loaded {len(docs)} documents from source")

    # Init embedder + chroma
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    _, collection = get_chroma_collection()

    # Determine changes
    existing_doc_hashes = get_existing_doc_hashes(collection)
    current_doc_ids = {d.doc_id for d in docs}
    existing_doc_ids = set(existing_doc_hashes.keys())

    changed_docs: list[Doc] = []
    new_docs = 0
    updated_docs = 0

    for d in docs:
        old_hash = existing_doc_hashes.get(d.doc_id)
        if old_hash is None:
            changed_docs.append(d)
            new_docs += 1
        elif old_hash != d.doc_hash:
            changed_docs.append(d)
            updated_docs += 1

    removed_doc_ids = list(existing_doc_ids - current_doc_ids) if DELETE_STALE else []

    LOGGER.info(f"Detected changes: new={new_docs}, updated={updated_docs}, removed={len(removed_doc_ids)}")

    # Delete stale chunks (removed docs)
    if DELETE_STALE and removed_doc_ids:
        for doc_id in removed_doc_ids:
            try:
                delete_doc_chunks(collection, doc_id)
                LOGGER.info(f"Deleted chunks for removed doc_id={doc_id}")
            except Exception as e:
                LOGGER.warning(f"Failed to delete chunks for removed doc_id={doc_id}: {e}")

    # Update changed docs: delete old chunks then add new chunks
    total_added_chunks = 0
    for d in changed_docs:
        try:
            if DELETE_STALE:
                delete_doc_chunks(collection, d.doc_id)

            chunks = chunk_text(d.text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                LOGGER.warning(f"Doc doc_id={d.doc_id} produced 0 chunks (empty text?)")
                continue

            added = add_chunks(collection, embed_model, d, chunks)
            total_added_chunks += added

            LOGGER.info(f"Upserted doc_id={d.doc_id} hash={d.doc_hash[:12]} chunks={added}")

        except Exception as e:
            LOGGER.exception(f"Failed processing doc_id={d.doc_id}: {e}")

    duration = time.time() - start_ts
    LOGGER.info(f"Index update finished. Added_chunks={total_added_chunks}. Collection_count={collection.count()}. Took={duration:.2f}s")


if __name__ == "__main__":
    main()

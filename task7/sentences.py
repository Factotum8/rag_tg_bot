import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import chromadb

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "../task2/knowledge_base/star_wars_planets_dataset.json"
CHROMA_DIR = "./chroma_starwars"
COLLECTION_NAME = "starwars_planets"

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_json(path)
    df.rename(columns={0: "text"}, inplace=True)

    # DELETE important terms
    df = df[~df["text"].str.contains("carrot")]
    df = df[~df["text"].str.contains("carrots")]

    return df


def build_chroma_index(df: pd.DataFrame):
    print("Инициализируем модель эмбеддингов:", EMBED_MODEL_NAME)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Если коллекция уже была — переиспользуем
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Удалим старые данные, чтобы не дублировать
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Коллекция уже содержит {existing_count} объектов, очищаем...")
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    docs = df["text"].astype(str).tolist()
    ids = [str(i) for i in range(len(docs))]
    metadatas = df.to_dict(orient="records")

    print("Считаем эмбеддинги документов...")
    embeddings = embed_model.encode(docs, convert_to_numpy=True).tolist()

    print("Кладём в Chroma...")
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("Готово! В коллекции документов:", collection.count())
    return client, collection, embed_model

if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    client, collection, embed_model = build_chroma_index(df)

    query = "Which carrot is the most common in hole pieces"
    query_emb = embed_model.encode([query], convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=5
    )

    print("\nQuery:", query)
    print("\nTop matches:")
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"dist={dist:.4f} | {doc[:200]}")

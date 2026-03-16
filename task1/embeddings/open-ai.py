import numpy as np
from openai import OpenAI
client = OpenAI()

# Те же предложения
sentences = [
    "RAG — это способ комбинировать поиск и генерацию.",
    "Многорукий бандит — простая задача обучения с подкреплением.",
    "Цифровые двойники используются в промышленности.",
    "SaaS-платформа для моделирования промышленных объектов «Digital Twin». Расскажи историю создания таких систем.",
    "Что такое цифровые двойники в SCADA-системах."
]

# --- 1. Считаем эмбеддинги всех предложений ---
emb_res = client.embeddings.create(
    model="text-embedding-3-small",
    input=sentences
)

sentence_embeddings = np.array([e.embedding for e in emb_res.data])

# --- 2. Считаем эмбеддинг пользовательского запроса ---
query = sentences[0]
query_emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

query_emb = np.array(query_emb)

# --- 3. Косинусное сходство ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine_similarity(query_emb, emb) for emb in sentence_embeddings]

# --- 4. Сортируем по убыванию ---
sorted_idx = np.argsort(scores)[::-1]

print(f"\nЗапрос: {query}\n")
print("Похожие предложения:\n")

for idx in sorted_idx:
    print(f"score={scores[idx]:.3f} | {sentences[idx]}")

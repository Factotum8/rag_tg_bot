import faiss
from sentence_transformers import SentenceTransformer

# Наши документы
docs = [
    "RAG — это способ комбинировать поиск и генерацию.",
    "Многорукий бандит — простая задача обучения с подкреплением.",
    "Цифровые двойники используются в промышленности.",
    "SaaS-платформа для моделирования промышленных объектов «Digital Twin». Расскажи историю создания таких систем.",
    "Что такое цифровые двойники в SCADA-системах."
]

# 1. Загружаем модель эмбеддингов
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Считаем эмбеддинги документов
doc_embeddings = model.encode(docs, convert_to_numpy=True).astype("float32")

# Размерность вектора
d = doc_embeddings.shape[1]

# 3. Создаём FAISS-индекс (L2, можно потом использовать cosine через нормировку)
index = faiss.IndexFlatL2(d)

# 4. Добавляем эмбеддинги документов в индекс
index.add(doc_embeddings)

print("В индексе векторов:", index.ntotal)

# 5. Запрос пользователя
query = "Что такое цифровые двойники?"
query_vec = model.encode([query], convert_to_numpy=True).astype("float32")

# 6. Ищем k наиболее похожих документов
k = 3
distances, indices = index.search(query_vec, k)

print(f"\nЗапрос: {query}\n")
print("Топ-результаты:\n")

for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"{rank}) dist={dist:.4f} | {docs[idx]}")

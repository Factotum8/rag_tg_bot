from sentence_transformers import SentenceTransformer, util

# 1. Загружаем модель
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Набор предложений
sentences = [
    "RAG — это способ комбинировать поиск и генерацию.",
    "Многорукий бандит — простая задача обучения с подкреплением.",
    "Цифровые двойники используются в промышленности.",
    "SaaS-платформа для моделирования промышленных объектов «Digital Twin». Расскажи историю создания таких систем.",
    "Что такое цифровые двойники в SCADA-системах."
]

# 3. Считаем эмбеддинги
embeddings = model.encode(sentences, convert_to_tensor=True)

# 4. Считаем сходство между первым предложением и остальными
cos_sim = util.cos_sim(embeddings[0], embeddings)

print("Базовое предложение:")
print(sentences[0], "\n")

print("Сходство с другими:")
for i, s in enumerate(sentences):
    print(f"{i}: {s}  —  cos_sim = {cos_sim[0][i]:.3f}")

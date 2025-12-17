import chromadb

# 1. Создаём клиент (in-memory)
client = chromadb.Client()

# 2. Создаём коллекцию
collection = client.create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"}  # метрика сходства
)
collection.add(
    ids=["1", "2", "3", "4", "5"],
    documents=[
        "RAG — это способ комбинировать поиск и генерацию.",
        "Многорукий бандит — простая задача обучения с подкреплением.",
        "Цифровые двойники используются в промышленности.",
        "SaaS-платформа для моделирования промышленных объектов «Digital Twin». Расскажи историю создания таких систем.",
        "Что такое цифровые двойники в SCADA-системах." ],
    metadatas=[
        {"topic": "RAG"},
        {"topic": "RL"},
        {"topic": "Digital Twin"},
        {"topic": "Digital Twin"},
        {"topic": "Digital Twin"},
    ]
)

# 3. Добавляем документы + метаданные

# 4. Делаем запрос: найдём похожие документы
results = collection.query(
    query_texts=["Что такое цифровые двойники?"],
    n_results=2
)

print(results)

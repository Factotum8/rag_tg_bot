import json

from rdflib import Graph, Namespace, RDFS, Literal


def build_docs_from_graph(g: Graph) -> list[dict]:
    """
    Строим документы вида:
    {
      "id": "<URI субъекта>",
      "rag_text": "Имя — описание ...",
      "all_literals": [...],
    }
    """
    SCHEMA = Namespace("http://schema.org/")

    subjects = set()

    # 1) Берём все субъекты, у которых есть label / name — чаще всего это и есть сущности (планеты)
    for s, p, o in g.triples((None, RDFS.label, None)):
        subjects.add(s)
    for s, p, o in g.triples((None, SCHEMA.name, None)):
        subjects.add(s)

    print(f"Найдено сущностей с именами/label: {len(subjects)}")

    docs = []

    for s in subjects:
        labels = set()
        descs = set()
        extra_literals = set()

        # Имена
        for o in g.objects(s, RDFS.label):
            if isinstance(o, Literal):
                labels.add(str(o))
        for o in g.objects(s, SCHEMA.name):
            if isinstance(o, Literal):
                labels.add(str(o))

        # Описания
        for o in g.objects(s, RDFS.comment):
            if isinstance(o, Literal):
                descs.add(str(o))
        for o in g.objects(s, SCHEMA.description):
            if isinstance(o, Literal):
                descs.add(str(o))

        # Все остальные литералы (на всякий случай)
        for p, o in g.predicate_objects(s):
            if isinstance(o, Literal):
                extra_literals.add(str(o))

        # Формируем текст для RAG
        label_part = " / ".join(sorted(labels)) if labels else str(s)
        if descs:
            desc_part = " ".join(sorted(descs))
        else:
            # если нет явных описаний — используем все литералы
            desc_part = " ".join(sorted(extra_literals))

        rag_text = f"{label_part} — {desc_part}" if desc_part else label_part

        docs.append({
            "id": str(s),
            "rag_text": rag_text,
            "all_literals": list(extra_literals),
        })

    print(f"Всего документов сформировано: {len(docs)}")
    return docs

def replace_words(docs: list[dict], map_: dict[str, str]) -> list[dict]:
    map_ = {"star": "hole", "starts": "holes", "war": "piece", "wars": "pieces", "planet": "carrot", "planets": "carrots"}
    for k, v in map_.items():
        for doc in docs:
            if k in doc["rag_text"].lower():
                doc["rag_text"] = doc["rag_text"].lower().replace(k, v)

    # return [doc["rag_text"].split("—") for doc in docs]
    return [doc["rag_text"] for doc in docs]


if __name__ == "__main__":
    graph = Graph()

    TTL_PATH = "star_wars_planets_dataset.ttl"
    NT_PATH = "star_wars_planets_dataset.nt"

    graph.parse(TTL_PATH, format="turtle")
    graph.parse(NT_PATH, format="nt")

    docs = build_docs_from_graph(graph)
    rag_texts = replace_words(docs, NT_PATH)

    with open("star_wars_planets_dataset.json", "w", encoding="utf-8") as f:
        # for fname, describe in rag_texts:
        #     f.write(name, "," + describe + "\n")

        json.dump(rag_texts, f)

    # client, collection, embed_model = build_chroma_index(docs)

    # Пример запроса
    # user_query = "Расскажи о планете Корускант и её роли во вселенной Star Wars."
    # rag_query(collection, embed_model, user_query, top_k=5)

    # build_docs_from_graph()
    # graph.serialize(destination=TTL_PATH, format="turtle")
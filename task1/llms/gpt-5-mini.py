from openai import OpenAI

client = OpenAI()

messages = [
    {"role": "system", "content": "Ты дружелюбный чат-бот."}
]

def ask(user_message: str):
    messages.append({"role": "user", "content": user_message})

    response = client.responses.create(
        model="gpt-5-mini",
        input=messages
    )

    answer = response.output_text
    messages.append({"role": "assistant", "content": answer})
    return answer

# Пример:
print(ask("Объясни простыми словами, что такое RAG."))
print(ask("SaaS-платформа для моделирования промышленных объектов «Digital Twin». Расскажи историю создания таких систем."))
print(ask("Что такое цифровые двойники в SCADA-системах."))
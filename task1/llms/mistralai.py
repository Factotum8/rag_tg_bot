from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def chat(user_message: str):
    messages = [
        {"role": "user", "content": user_message}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)

chat("Объясни простыми словами, что такое RAG.")
chat("SaaS-платформа для моделирования промышленных объектов «Digital Twin». Расскажи историю создания таких систем.")
chat("Что такое цифровые двойники в SCADA-системах.")

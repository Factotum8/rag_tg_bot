BASE_SYSTEM_MSG = [
    {"role": "system", "content": "You are friendly bot."},
    {"role": "system", "content": "You're the assistant who thinks first and then answers. Always write down your steps."},
]

FEW_SHOT_EXAMPLES = [
    {"question": "Which carrot is the most common in hole pieces?", "answer": "The most common carrot is Tatooine."},
    {"question": "Which banana is the most common in hole pieces?", "answer": "Sorry, I don't know."}
]



def build_few_shot_system_message(examples) -> dict:
    text = "Below are examples of correct questions and answers:\n\n"

    for i, ex in enumerate(examples, start=1):
        text += (
            f"Example {i}:\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}\n\n"
        )

    return {
        "role": "system",
        "content": text.strip()
    }

FEW_SHOT_EXAMPLES = build_few_shot_system_message(FEW_SHOT_EXAMPLES)
# llm.py
# Claude LLM se grounded answer generate karta hai
# Sirf retrieved FAQ ke basis par — kuch invent nahi karega

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()


def get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not found in environment variables.")
    return anthropic.Anthropic(api_key=api_key)


def generate_grounded_answer(user_query: str, faq: dict) -> str:
    """
    Claude se natural language answer generate karo.
    
    Important: Strict prompt use karo taaki model sirf
    retrieved FAQ ke basis par answer de — hallucinate na kare.
    
    Args:
        user_query: User ka original question
        faq: Retrieved FAQ entry (id, question, answer)
    
    Returns:
        Clean, helpful answer string
    """

    system_prompt = """You are a precise customer support assistant.

STRICT RULES you must follow:
1. Answer ONLY using the information provided in the Source FAQ below.
2. Do NOT add any information, assumptions, or details not present in the source.
3. If the source FAQ doesn't fully answer the question, say so — don't make things up.
4. Keep answers concise, friendly, and clear.
5. Never mention that you are referencing a FAQ document."""

    user_prompt = f"""Source FAQ:
Question: {faq['question']}
Answer: {faq['answer']}

User's Question: {user_query}

Provide a helpful answer based strictly on the source above."""

    try:
        client = get_client()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return message.content[0].text.strip()

    except Exception as e:
        # LLM fail hone par original FAQ answer return karo
        print(f"[LLM WARNING] Failed to generate answer: {e}")
        return faq["answer"]

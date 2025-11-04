import logging
import os
from openai import OpenAI
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
CATEGORIZATION_OPENAI_BASE_URL = os.environ.get(
    "CATEGORIZATION_OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL")
)
CATEGORIZATION_OPENAI_API_KEY = os.environ.get(
    "CATEGORIZATION_OPENAI_API_KEY",
    os.environ.get("OPENAI_API_KEY"),
)
CATEGORIZATION_OPENAI_MODEL = os.environ.get(
    "CATEGORIZATION_OPENAI_MODEL", os.environ.get("OPENAI_MODEL")
)
openai_client = OpenAI(
    base_url=CATEGORIZATION_OPENAI_BASE_URL, api_key=CATEGORIZATION_OPENAI_API_KEY
)

print(f"✅ Categorization OpenAI Client initialized:")
print(f"   📍 Categorizatoin Base URL: {CATEGORIZATION_OPENAI_BASE_URL}")
print(f"   🤖 Categorizatoin Model: {CATEGORIZATION_OPENAI_MODEL}")
print(
    f"   🔑 Categorizatoin API Key: {'***' + CATEGORIZATION_OPENAI_API_KEY[-4:] if len(CATEGORIZATION_OPENAI_API_KEY) > 4 else '***'}"
)

class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory}
        ]

        # Let OpenAI handle the pydantic parsing directly
        completion = openai_client.beta.chat.completions.parse(
            model=CATEGORIZATION_OPENAI_MODEL,
            messages=messages,
            response_format=MemoryCategories,
            temperature=0
        )

        parsed: MemoryCategories = completion.choices[0].message.parsed
        return [cat.strip().lower() for cat in parsed.categories]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        try:
            logging.debug(f"[DEBUG] Raw response: {completion.choices[0].message.content}")
        except Exception as debug_e:
            logging.debug(f"[DEBUG] Could not extract raw response: {debug_e}")
        raise

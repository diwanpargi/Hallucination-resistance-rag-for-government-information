 
import time

from google import genai
from google.genai import types

import config

NOT_FOUND = "Information not found in the provided documents."


def get_gemini_client():
    if not config.GEMINI_API_KEY:
        raise ValueError("set api key for llm (environment variable).")
    return genai.Client(api_key=config.GEMINI_API_KEY)


from pathlib import Path


def build_context(hits):
    """Group chunks by PDF and display only PDF filenames."""
    groups = {}

    for p in hits:
        d = p.payload or {}

        raw_source = d.get("source", "Unknown")
        source = Path(str(raw_source).replace("\\", "/")).name

        groups.setdefault(source, []).append(p)

    parts = []

    for source, points in groups.items():
        lines = [f"=== DOCUMENT: {source} ==="]

        for p in points:
            d = p.payload or {}

            lines.append(
                f"PAGE: {d.get('page_number') or 'Unknown'}\n"
                f"SECTION: {d.get('chunk_id', 'Unknown')}\n"
                f"TEXT:\n{d.get('text', '')}"
            )

        parts.append("\n\n".join(lines))

    return "\n\n--------------------\n\n".join(parts)


def make_prompt(context, question):
    return f"""You are a reliable government scholarship information assistant.

Answer the user's question using ONLY the retrieved context.

Rules:

1. Do not use outside knowledge or invent facts. You may combine
information from multiple relevant chunks if the information is
supported by the retrieved documents.

2. Each DOCUMENT represents a different scholarship scheme.
Never combine eligibility, scholarship amounts, dates, required
documents, renewal rules or other conditions from different schemes.

3. Identify the scheme relevant to the user's question.
If the question clearly refers to one scheme, answer for that scheme
only. Do not include unrelated schemes just because their documents
appear in the retrieved context.

4. If the question is general and multiple schemes are relevant,
answer separately for each relevant scheme. Clearly identify each
scheme using its PDF filename.

5. Read all relevant chunks of the same document together and produce
ONE complete, concise answer. Do not answer chunk by chunk.

6. Remove duplicate information before generating the final answer.
If two or more chunks state the same fact using identical or different
wording, include that fact only once. Preserve additional details
or conditions that provide genuinely new information.

7. For questions about required documents, eligibility or renewal,
combine all relevant information from the same scheme into one
organized answer. Preserve important conditions, including whether
they apply to fresh candidates, renewal candidates or both.

8. Do not reject information merely because it is expressed using
different wording. If the retrieved text clearly supports the meaning
of a statement, you may answer using simple paraphrasing.

9. If the retrieved context contains enough relevant information to
answer the question, provide the answer. Do not return NOT_FOUND
merely because some minor details are missing.

10. If only part of the answer is available, provide the supported
information and clearly state which specific detail is not available.
Do not guess the missing detail.

11. If the retrieved context contains no relevant information that
answers the question, respond exactly:
{NOT_FOUND}

12. After the answer, provide the source in this format:
Source: <pdf filename>, Page: <number>

Use only PDF filenames and page numbers present in the retrieved
context. Never display full directory paths.

13. If the answer uses information from multiple pages of the same
document, include the relevant page numbers in the source citation.
Do not cite a page unless it supports the answer.

14. Use simple plain text and numbered lists when needed.
Do not use Markdown formatting, asterisks or heading symbols.

15. Include only information relevant to the user's question.
Avoid unnecessary explanations, repeated facts and raw document
extracts.

Retrieved Context:
{context}

User Question:
{question}
"""



def call_gemini(gemini, prompt):
    """Calls Gemini. If Gemini is busy (429 / 5xx) wait a little and try again."""
    wait = 5
    for attempt in range(1, config.LLM_RETRIES + 1):
        try:
            response = gemini.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),   # temperature 0 = less guessing
            )
            text = (response.text or "").strip()
            if not text:
                raise ValueError("Gemini returned an empty answer")
            return text

        except Exception as e:
            code = getattr(e, "code", None)
            busy = code in (429, 500, 503, 504)
            if not busy or attempt == config.LLM_RETRIES:
                raise
            print(f"  Gemini is busy (error {code}). Waiting {wait}s, try {attempt}/{config.LLM_RETRIES}...")
            time.sleep(wait)
            wait *= 2


def generate_answer(gemini, context, question):
    return call_gemini(gemini, make_prompt(context, question))

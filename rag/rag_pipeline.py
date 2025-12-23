# rag/rag_pipeline.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import os
import time
import re
from datetime import datetime
import ollama

# Modèle ollama (tu peux changer via variable d'env OLLAMA_MODEL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","qwen2.5:3b")





def build_prompt(query: str, retrieved: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Build a RAG prompt that INCLUDES metadata (date, filename, chunk_id).
    This is crucial because many questions ask for dates, and the date often
    lives in metadata (not in the chunk text).
    """
    sources: List[str] = []
    context_blocks: List[str] = []

    for i, c in enumerate(retrieved, start=1):
        filename = c.get("filename", "unknown")
        chunk_id = c.get("chunk_id", "unknown")
        date = c.get("date", "unknown")
        parl = c.get("parliament", "unknown")

        ref = f"{filename} | {chunk_id} | {date} | {parl}"
        sources.append(ref)

        # IMPORTANT: we put the date INSIDE the context block
        context_blocks.append(
            f"[Doc {i}] (source: {ref})\n"
            f"[Document date: {date}]\n"
            f"{c.get('text','')}"
        )

    prompt = (
        "You are an assistant for parliamentary debates.\n"
        "Answer using ONLY the context below.\n"
        "If the answer is not in the context, say: \"The context does not contain this information.\".\n"
        "\n"
        "IMPORTANT RULES:\n"
        "- If the question asks for a DATE (e.g., \"On what date(s)\") and the date is provided in the metadata\n"
        "  (Document date / source line), you MUST use those dates.\n"
        "- When listing dates, output unique dates in chronological order.\n"
        "- Cite the Doc number(s) after each claim.\n"
        "\n"
        "CONTEXT:\n"
        + "\n\n".join(context_blocks)
        + f"\n\nQUESTION: {query}\nANSWER:"
    )

    # unique sources
    return prompt, sorted(set(sources))

def generate_with_ollama(prompt: str) -> str:
    """
    Génération locale via Ollama.
    Pour les questions de dates: réponse déterministe depuis les métadonnées du contexte
    (évite hallucinations).
    """
    # ✅ Si question de date → on répond depuis les [Document date: ...]
    # On récupère la QUESTION depuis le prompt
    q_match = re.search(r"QUESTION:\s*(.*)\nANSWER:", prompt, re.DOTALL)
    query = (q_match.group(1).strip() if q_match else "")

    if _is_date_question(query):
        dates = _extract_doc_dates_from_prompt(prompt)
        if not dates:
            return "The context does not contain this information."

        # Cite Doc numbers (simplement Doc 1..k)
        # (même si plusieurs docs ont la même date, on reste propre)
        doc_nums = _extract_doc_refs_from_prompt(prompt)
        cite = ", ".join([f"Doc {n}" for n in doc_nums[:min(len(doc_nums), 10)]])  # limite

        if len(dates) == 1:
            return f"The speech date is {dates[0]}. ({cite})"
        else:
            return f"The speech dates are: " + ", ".join(dates) + f". ({cite})"

    # Sinon → LLM normal
    time.sleep(0.2)
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp["message"]["content"] or "").strip()


def _is_date_question(q: str) -> bool:
    q = q.lower()
    return (
        "on what date" in q
        or "on what dates" in q
        or "what date" in q
        or "what dates" in q
        or "when did" in q
        or "date did" in q
    )

def _extract_doc_dates_from_prompt(prompt: str):
    # récupère toutes les lignes "[Document date: YYYY-MM-DD]"
    dates = re.findall(r"\[Document date:\s*(\d{4}-\d{2}-\d{2})\]", prompt)
    # unique + tri chrono
    uniq = sorted(set(dates), key=lambda d: datetime.strptime(d, "%Y-%m-%d"))
    return uniq

def _extract_doc_refs_from_prompt(prompt: str):
    # récupère les refs "[Doc i] (source: ...)"
    # on va juste citer Doc 1, Doc 2 etc si on répond via métadonnées
    doc_nums = re.findall(r"\[Doc\s+(\d+)\]", prompt)
    # unique dans l'ordre
    seen = set()
    ordered = []
    for n in doc_nums:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


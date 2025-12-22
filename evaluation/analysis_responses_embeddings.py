# # analysis_responses_embeddings.py

# import os
# import json
# import numpy as np
# from pathlib import Path
# from sentence_transformers import SentenceTransformer

# # 🔒 OFFLINE SAFE
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# RESULTS_PATHS = [
#     Path("evaluation/results/run_required_20251222_194723.jsonl"),
#     Path("evaluation/results/run_custom_20251222_201857.jsonl"),
# ]

# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# def load_answers(paths):
#     answers = []
#     meta = []

#     for path in paths:
#         if not path.exists():
#             print("⚠️ Fichier introuvable :", path)
#             continue

#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 row = json.loads(line)
#                 if row.get("answer"):
#                     answers.append(row["answer"])
#                     meta.append({
#                         "query_id": row.get("query_id"),
#                         "config": row.get("config"),
#                         "k": row.get("k"),
#                         "source_file": path.name,
#                     })

#     return answers, meta


# def main():
#     answers, meta = load_answers(RESULTS_PATHS)
#     print("Nb answers:", len(answers))

#     if not answers:
#         print("❌ Aucune réponse chargée, arrêt.")
#         return

#     model = SentenceTransformer(MODEL_NAME)

#     embeddings = model.encode(
#         answers,
#         batch_size=32,
#         normalize_embeddings=True,
#         convert_to_numpy=True,
#     )

#     print("Embeddings shape:", embeddings.shape)

#     out_path = Path("evaluation/results/answers_embeddings.npy")
#     np.save(out_path, embeddings)
#     print("Saved:", out_path)


# if __name__ == "__main__":
#     main()
import json
import numpy as np
from pathlib import Path

RESULTS_FILES = [
    Path("evaluation/results/run_required_20251222_194723.jsonl"),
    Path("evaluation/results/run_custom_20251222_201857.jsonl"),
]

EMB_PATH = Path("evaluation/results/answers_embeddings.npy")


def load_runs(paths):
    """
    Charge les lignes JSONL et reconstruit un champ 'config' = '{chunking}/{representation}'
    """
    rows = []
    for p in paths:
        if not p.exists():
            print("Missing:", p)
            continue

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)

                ans = r.get("answer")
                if not ans:
                    continue

                # ✅ reconstruire config à partir des vrais champs du JSON
                chunking = r.get("chunking")
                repr_ = r.get("representation")
                if chunking is None or repr_ is None:
                    # on skippe proprement si format inattendu
                    continue

                r["config"] = f"{chunking}/{repr_}"
                r["_source_file"] = p.name
                rows.append(r)

    return rows


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    # embeddings normalisés -> cosinus = produit scalaire
    return emb @ emb.T


def group_key(r):
    # même query + même config (fixed/bm25, child/emb, etc.)
    return (r.get("query_id"), r.get("config"))


def main():
    rows = load_runs(RESULTS_FILES)
    emb = np.load(EMB_PATH)

    print("Rows:", len(rows))
    print("Emb:", emb.shape)

    if len(rows) != emb.shape[0]:
        raise RuntimeError(
            f"Mismatch rows vs embeddings: rows={len(rows)} emb={emb.shape[0]}.\n"
            "➡️ Assure-toi d'avoir généré answers_embeddings.npy à partir des mêmes fichiers JSONL "
            "et dans le même ordre."
        )

    cos = cosine_matrix(emb)

    # ------------------------------------------------------------------
    # A) Stabilité selon K (même query_id + config)
    # ------------------------------------------------------------------
    by_qc = {}
    for i, r in enumerate(rows):
        key = group_key(r)
        by_qc.setdefault(key, []).append(i)

    stabilities = []
    for key, idxs in by_qc.items():
        if len(idxs) < 2:
            continue

        sims = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                sims.append(float(cos[idxs[a], idxs[b]]))

        stabilities.append((key, float(np.mean(sims)), len(idxs)))

    stabilities.sort(key=lambda x: x[1])  # plus instable en premier

    print("\n=== Top 10 configs les plus INSTABLES (selon K) ===")
    for (qid, cfg), s, n in stabilities[:10]:
        print(f"{qid} | {cfg} | n={n} | mean_cos={s:.3f}")

    print("\n=== Top 10 configs les plus STABLES (selon K) ===")
    for (qid, cfg), s, n in stabilities[-10:]:
        print(f"{qid} | {cfg} | n={n} | mean_cos={s:.3f}")

    # ------------------------------------------------------------------
    # B) Comparaison BM25 vs EMB (même query_id + même chunking, même k)
    # ------------------------------------------------------------------
    index = {}
    for i, r in enumerate(rows):
        qid = r.get("query_id")
        cfg = r.get("config")  # ex: "fixed/bm25" ou "child/emb"
        k = r.get("k")
        if qid is None or cfg is None or k is None:
            continue
        index[(qid, cfg, k)] = i

    pairs = []
    for (qid, cfg, k), i_bm25 in index.items():
        if cfg.endswith("/bm25"):
            cfg_emb = cfg.replace("/bm25", "/emb")
            j = index.get((qid, cfg_emb, k))
            if j is not None:
                pairs.append((qid, cfg, cfg_emb, k, float(cos[i_bm25, j])))

    if not pairs:
        print("\n⚠️ Aucune paire BM25 vs EMB trouvée. Vérifie que tu as bien des configs bm25 ET emb.")
        return

    pairs.sort(key=lambda x: x[4])  # plus différent d'abord

    print("\n=== Top 10 BM25 vs EMB les plus DIFFÉRENTS (cos faible) ===")
    for qid, c1, c2, k, s in pairs[:10]:
        print(f"{qid} | {c1} vs {c2} | k={k} | cos={s:.3f}")

    print("\n=== Top 10 BM25 vs EMB les plus SIMILAIRES (cos élevé) ===")
    for qid, c1, c2, k, s in pairs[-10:]:
        print(f"{qid} | {c1} vs {c2} | k={k} | cos={s:.3f}")


if __name__ == "__main__":
    main()

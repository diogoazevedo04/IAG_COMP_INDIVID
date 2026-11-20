"""Indexacao (Chroma ou NumPy, cosseno), pesquisa densa, BM25 e fusao RRF.
config: vector_store: chroma | numpy   (numpy = alternativa sem dependencias pesadas)"""
import json, re, unicodedata
from pathlib import Path


def embed(texts, model, batch=16):
    import ollama
    out = []
    for i in range(0, len(texts), batch):
        out.extend(ollama.embed(model=model, input=texts[i:i + batch]).embeddings)
    return out


def bm25_tokens(text, strip_accents=False):
    text = text.lower()
    if strip_accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", text)


def rrf(rankings, k=60):
    """rankings: listas de dicts com chunk_id, por ordem. Soma 1/(k+posicao) por lista."""
    score, keep = {}, {}
    for lst in rankings:
        for pos, c in enumerate(lst, 1):
            score[c["chunk_id"]] = score.get(c["chunk_id"], 0.0) + 1.0 / (k + pos)
            keep.setdefault(c["chunk_id"], c)
    ordered = sorted(score, key=lambda cid: -score[cid])
    return [{**keep[cid], "score": score[cid]} for cid in ordered]


class Index:
    def __init__(self, root: Path, cfg):
        self.cfg = cfg
        self.store = cfg.get("vector_store", "chroma")
        self.dir = root / "index" / f"c{cfg['chunk_tokens']}_o{cfg['overlap_tokens']}"
        self.chunks, self.by_id, self._bm25, self._col, self._mat = [], {}, None, None, None

    def build(self, chunks):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "chunks.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        vecs = []
        for i in range(0, len(chunks), 64):
            vecs.extend(embed([c["texto"] for c in chunks[i:i + 64]], self.cfg["embedding_model"]))  # SEM instrucao
            print(f"  embeddings {min(i + 64, len(chunks))}/{len(chunks)}", end="\r")
        print()
        if self.store == "numpy":
            import numpy as np
            a = np.array(vecs, dtype="float32")
            a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
            np.save(self.dir / "vectors.npy", a)
            return
        import chromadb
        client = chromadb.PersistentClient(path=str(self.dir / "chroma"))
        try:
            client.delete_collection("chunks")
        except Exception:
            pass
        col = client.create_collection("chunks", metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), 64):
            part = chunks[i:i + 64]
            col.add(ids=[c["chunk_id"] for c in part], embeddings=vecs[i:i + 64],
                    documents=[c["texto"] for c in part],
                    metadatas=[{"doc_id": c["doc_id"], "inicio": c["inicio"], "fim": c["fim"]} for c in part])

    def load(self):
        self.chunks = [json.loads(l) for l in (self.dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        if self.store == "numpy":
            import numpy as np
            self._mat = np.load(self.dir / "vectors.npy")
        else:
            import chromadb
            self._col = chromadb.PersistentClient(path=str(self.dir / "chroma")).get_collection("chunks")
        if self.cfg["search"] == "hybrid":
            from rank_bm25 import BM25Okapi
            b = self.cfg["bm25"]
            self._sa = b.get("strip_accents", False)
            self._bm25 = BM25Okapi([bm25_tokens(c["texto"], self._sa) for c in self.chunks], k1=b["k1"], b=b["b"])
        return self

    def dense(self, query, n):
        qv = embed([self.cfg["query_instruction"] + query], self.cfg["embedding_model"])[0]
        if self.store == "numpy":
            import numpy as np
            q = np.array(qv, dtype="float32"); q /= np.linalg.norm(q) + 1e-12
            sims = self._mat @ q
            top = np.argsort(-sims)[:n]
            return [{**self.chunks[i], "score": float(sims[i])} for i in top]
        r = self._col.query(query_embeddings=[qv], n_results=n)
        return [{**self.by_id[cid], "score": 1.0 - d} for cid, d in zip(r["ids"][0], r["distances"][0])]

    def lexical(self, query, n):
        sc = self._bm25.get_scores(bm25_tokens(query, self._sa))
        top = sorted(range(len(sc)), key=lambda i: -sc[i])[:n]
        return [{**self.chunks[i], "score": float(sc[i])} for i in top if sc[i] > 0]

    def retrieve(self, query):
        n = self.cfg["top_k_metric"]
        if self.cfg["search"] == "dense":
            return self.dense(query, n)
        f = self.cfg["fetch"]
        return rrf([self.dense(query, f), self.lexical(query, f)], k=self.cfg["rrf_k"])[:n]

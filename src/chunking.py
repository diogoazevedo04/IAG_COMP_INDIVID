"""Segmentacao em chunks por tokens (tokenizer do modelo de embeddings), com offsets em caracteres."""
import json
from pathlib import Path

_TOK = {}


def get_tokenizer(name):
    if name not in _TOK:
        from transformers import AutoTokenizer
        _TOK[name] = AutoTokenizer.from_pretrained(name)
    return _TOK[name]


def chunk_text(text, tok, size, overlap):
    """Janela deslizante de `size` tokens com passo size-overlap.
    Devolve [{inicio, fim, n_tokens}] com offsets em caracteres (fim excluido)."""
    assert 0 <= overlap < size
    offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    n, step, i, out = len(offs), size - overlap, 0, []
    while i < n:
        j = min(i + size, n)
        a, b = offs[i][0], offs[j - 1][1]
        a += len(text[a:b]) - len(text[a:b].lstrip())  # sem espacos no inicio
        out.append({"inicio": a, "fim": b, "n_tokens": j - i})
        if j == n:
            break
        i += step
    return out


def load_meta(root: Path):
    p = root / "dados" / "metadados.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_chunks(root: Path, cfg):
    tok = get_tokenizer(cfg["tokenizer"])
    chunks = []
    for r in load_meta(root):
        with open(root / "dados" / r["ficheiro_limpo"], encoding="utf-8", newline="") as f:
            text = f.read()
        for k, c in enumerate(chunk_text(text, tok, cfg["chunk_tokens"], cfg["overlap_tokens"])):
            chunks.append({
                "chunk_id": f"{r['doc_id']}#{k}", "doc_id": r["doc_id"], "titulo": r["titulo"],
                "entidade": r["entidade"], "url": r["url"], "area": r["area"],
                "inicio": c["inicio"], "fim": c["fim"], "n_tokens": c["n_tokens"],
                "texto": text[c["inicio"]:c["fim"]],
            })
    return chunks

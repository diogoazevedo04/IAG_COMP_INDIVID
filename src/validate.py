"""Valida metadados.jsonl e perguntas.jsonl e ajuda a anotar trechos.

Uso:
  python src/validate.py check                 # valida tudo
  python src/validate.py find a2_doc "texto"   # devolve inicio/fim e o JSON do trecho
"""
import hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "dados"
AREA = 2
META_FIELDS = ["doc_id", "titulo", "entidade", "url", "data_acesso", "area"]
Q_FIELDS = ["id", "area", "tipo", "pergunta", "resposta_referencia", "trechos"]
EXPECTED = {"factual": 8, "multi": 4, "sem_resposta": 3}


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_clean(rel):
    # newline="" para nao converter fins de linha: os offsets sao sobre o ficheiro tal como esta
    with open(DADOS / rel, encoding="utf-8", newline="") as f:
        return f.read()


def check():
    errs, warns = [], []
    meta = {}
    mp = DADOS / "metadados.jsonl"
    if not mp.exists():
        sys.exit("Falta dados/metadados.jsonl")
    for r in read_jsonl(mp):
        for f in META_FIELDS:
            if not r.get(f):
                errs.append(f"[meta] {r.get('doc_id')}: campo em falta '{f}'")
        d = r["doc_id"]
        if not d.startswith(f"a{AREA}_"):
            errs.append(f"[meta] {d}: prefixo deve ser a{AREA}_")
        if d in meta:
            errs.append(f"[meta] doc_id duplicado {d}")
        meta[d] = r
        fp = DADOS / r.get("ficheiro_limpo", f"limpo/{d}.txt")
        if not fp.exists():
            errs.append(f"[meta] {d}: ficheiro limpo em falta"); continue
        raw = fp.read_bytes()
        if hashlib.sha256(raw).hexdigest() != r.get("sha256"):
            errs.append(f"[meta] {d}: sha256 nao coincide (texto limpo mudou depois de registado!)")
        if b"\r" in raw:
            errs.append(f"[meta] {d}: contem \\r (fins de linha tem de ser \\n)")
    total = sum(r.get("n_palavras", 0) for r in meta.values())
    print(f"Corpus: {len(meta)} documentos, {total} palavras (meta 20000)")
    if total < 20000:
        warns.append(f"[corpus] faltam {20000-total} palavras")

    qp = DADOS / "perguntas.jsonl"
    if qp.exists():
        qs = read_jsonl(qp)
        ids = Counter(q["id"] for q in qs)
        for i, n in ids.items():
            if n > 1:
                errs.append(f"[q] id duplicado {i}")
        for q in qs:
            qid = q.get("id")
            for f in Q_FIELDS:
                if f not in q:
                    errs.append(f"[q] {qid}: campo em falta '{f}'")
            if q.get("tipo") not in EXPECTED:
                errs.append(f"[q] {qid}: tipo invalido {q.get('tipo')}")
            if q.get("area") != AREA:
                errs.append(f"[q] {qid}: area deve ser {AREA}")
            ts = q.get("trechos", [])
            if q.get("tipo") == "sem_resposta" and ts:
                errs.append(f"[q] {qid}: sem_resposta deve ter trechos vazios")
            if q.get("tipo") in ("factual", "multi") and not ts:
                errs.append(f"[q] {qid}: sem trechos")
            if q.get("tipo") == "multi" and len({t["doc_id"] for t in ts}) < 2:
                errs.append(f"[q] {qid}: multi deve usar trechos de >=2 documentos")
            if q.get("tipo") == "factual" and len(ts) != 1:
                warns.append(f"[q] {qid}: factual com {len(ts)} trechos (esperado 1)")
            for t in ts:
                d = t["doc_id"]
                if d not in meta:
                    errs.append(f"[q] {qid}: doc_id desconhecido {d}"); continue
                text = read_clean(meta[d]["ficheiro_limpo"])
                a, b = t["inicio"], t["fim"]
                if not (0 <= a < b <= len(text)):
                    errs.append(f"[q] {qid}: offsets fora do texto ({a},{b}) len={len(text)}")
                elif text[a:b] != t.get("texto"):
                    errs.append(f"[q] {qid}: texto != ficheiro[{a}:{b}] em {d} (corpus mudou?)")
        counts = Counter(q.get("tipo") for q in qs)
        print(f"Perguntas: {len(qs)} -> {dict(counts)}")
        for k, v in EXPECTED.items():
            if counts.get(k, 0) != v:
                warns.append(f"[q] tipo {k}: {counts.get(k,0)} (esperado {v})")
    else:
        warns.append("[q] dados/perguntas.jsonl ainda nao existe")

    for w in warns:
        print("AVISO", w)
    for e in errs:
        print("ERRO ", e)
    print("\nRESULTADO:", "OK" if not errs else f"{len(errs)} erro(s)")
    sys.exit(1 if errs else 0)


def find(doc_id, snippet):
    text = read_clean(f"limpo/{doc_id}.txt")
    pos, hits = 0, []
    while (i := text.find(snippet, pos)) != -1:
        hits.append(i); pos = i + 1
    if not hits:
        sys.exit("Nao encontrado. Copia o texto exatamente do ficheiro limpo (python src/ingest.py show).")
    if len(hits) > 1:
        print(f"AVISO: {len(hits)} ocorrencias, a usar a primeira. Torna o trecho mais especifico.")
    i = hits[0]
    print(json.dumps({"doc_id": doc_id, "inicio": i, "fim": i + len(snippet), "texto": snippet},
                     ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        check()
    elif len(sys.argv) == 4 and sys.argv[1] == "find":
        find(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)

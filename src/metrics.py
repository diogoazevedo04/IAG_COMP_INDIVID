"""Recall@5, MRR@10 e agregacao (Decisoes comuns, secoes 4 e 5)."""
import json
from pathlib import Path


def hit(c, t):
    """Acerto: mesmo documento e sobreposicao >= 50% do mais curto (caracteres)."""
    if c["doc_id"] != t["doc_id"]:
        return False
    ov = max(0, min(c["fim"], t["fim"]) - max(c["inicio"], t["inicio"]))
    return ov >= 0.5 * min(c["fim"] - c["inicio"], t["fim"] - t["inicio"])


def recall_at_k(chunks, trechos, k=5):
    if not trechos:
        return None
    top = chunks[:k]
    return sum(any(hit(c, t) for c in top) for t in trechos) / len(trechos)


def mrr_at_k(chunks, trechos, k=10):
    if not trechos:
        return None
    for pos, c in enumerate(chunks[:k], 1):
        if any(hit(c, t) for t in trechos):
            return 1.0 / pos
    return 0.0


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()] if Path(p).exists() else []


def aggregate(perguntas, respostas, avaliacoes, chunk_stats=None):
    pq = {q["id"]: q for q in perguntas}
    rs = {r["id"]: r for r in respostas}
    av = {a["id"]: a for a in avaliacoes}

    def block(tipos):
        ids = [i for i, q in pq.items() if q["tipo"] in tipos and i in rs]
        ans = [i for i in ids if pq[i]["tipo"] != "sem_resposta"]
        rec = [recall_at_k(rs[i]["chunks"], pq[i]["trechos"]) for i in ans]
        mrr = [mrr_at_k(rs[i]["chunks"], pq[i]["trechos"]) for i in ans]
        ev = [av[i]["juiz"] for i in ids if i in av]
        ev_ans = [av[i]["juiz"] for i in ans if i in av]
        sem = [i for i in ids if pq[i]["tipo"] == "sem_resposta"]
        return {
            "n_perguntas": len(ids),
            "recall_at_5": mean(rec), "mrr_at_10": mean(mrr),
            "correcao_media": mean([e["correcao"] for e in ev_ans]),
            "fidelidade_media": mean([e["fidelidade"] for e in ev]),
            "citacoes_media": mean([e["citacoes"] for e in ev]),
            "abstencao_correta": (sum(av[i]["juiz"]["abstencao"] == "correta" for i in sem if i in av) / len(sem)) if sem else None,
            "abstencao_indevida": (sum(av[i]["juiz"]["abstencao"] == "indevida" for i in ans if i in av) / len(ans)) if ans else None,
        }

    out = block({"factual", "multi", "sem_resposta"})
    out["por_tipo"] = {t: block({t}) for t in ("factual", "multi", "sem_resposta")}
    if chunk_stats:
        out["chunks"] = chunk_stats
    return out


def choose(mA, mB):
    """Regra da secao 5: Recall@5; se a diferenca < 0.05, MRR@10; depois fidelidade."""
    if abs(mA["recall_at_5"] - mB["recall_at_5"]) >= 0.05:
        return "A" if mA["recall_at_5"] > mB["recall_at_5"] else "B"
    if mA["mrr_at_10"] != mB["mrr_at_10"]:
        return "A" if mA["mrr_at_10"] > mB["mrr_at_10"] else "B"
    return "A" if (mA["fidelidade_media"] or 0) >= (mB["fidelidade_media"] or 0) else "B"

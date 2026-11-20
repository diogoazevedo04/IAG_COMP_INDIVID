"""Testes sem modelos nem rede: python tests/test_offline.py"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import metrics as M, judge as J
from chunking import chunk_text
from retrieval import rrf, bm25_tokens


class FakeTok:  # 1 token = 1 palavra (so para testar a logica de offsets)
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


def test_chunker():
    text = " ".join(f"w{i}" for i in range(100))
    ch = chunk_text(text, FakeTok(), 20, 5)
    assert ch[0]["inicio"] == 0 and ch[-1]["fim"] == len(text)
    assert all(c["n_tokens"] <= 20 for c in ch)
    # sobreposicao de 5 tokens entre chunks consecutivos
    a, b = text[ch[0]["inicio"]:ch[0]["fim"]].split(), text[ch[1]["inicio"]:ch[1]["fim"]].split()
    assert a[-5:] == b[:5]
    assert ch[-1]["n_tokens"] > 5  # sem chunk final so de sobreposicao


def test_hit_rule():
    t = {"doc_id": "d", "inicio": 100, "fim": 200}
    assert M.hit({"doc_id": "d", "inicio": 150, "fim": 400}, t)       # 50/100 = 50% do mais curto
    assert not M.hit({"doc_id": "d", "inicio": 160, "fim": 400}, t)   # 40%
    assert not M.hit({"doc_id": "x", "inicio": 100, "fim": 200}, t)   # outro doc
    big = {"doc_id": "d", "inicio": 0, "fim": 1000}                   # chunk grande contem o trecho
    assert M.hit(big, t)


def test_recall_mrr():
    ts = [{"doc_id": "d", "inicio": 0, "fim": 10}, {"doc_id": "e", "inicio": 0, "fim": 10}]
    chunks = [{"doc_id": "z", "inicio": 0, "fim": 10}, {"doc_id": "d", "inicio": 0, "fim": 10},
              {"doc_id": "q", "inicio": 0, "fim": 10}, {"doc_id": "q", "inicio": 0, "fim": 10},
              {"doc_id": "q", "inicio": 0, "fim": 10}, {"doc_id": "e", "inicio": 0, "fim": 10}]
    assert M.recall_at_k(chunks, ts, 5) == 0.5      # so o d esta nos 5 primeiros
    assert M.mrr_at_k(chunks, ts, 10) == 0.5        # primeiro acerto na posicao 2
    assert M.recall_at_k(chunks, [], 5) is None


def test_rrf_and_bm25():
    a = [{"chunk_id": "1"}, {"chunk_id": "2"}]
    b = [{"chunk_id": "2"}, {"chunk_id": "3"}]
    r = rrf([a, b], k=60)
    assert [c["chunk_id"] for c in r] == ["2", "1", "3"]
    assert abs(r[0]["score"] - (1/62 + 1/61)) < 1e-12
    assert bm25_tokens("Autocarro Nº 5, Guimarães") == ["autocarro", "nº", "5", "guimarães"]
    from rank_bm25 import BM25Okapi
    docs = [bm25_tokens(x) for x in ["passe mensal custa 30 euros", "bicicleta partilhada", "horario do comboio"]]
    sc = BM25Okapi(docs, k1=1.5, b=0.75).get_scores(bm25_tokens("passe mensal"))
    assert sc[0] > sc[1] and sc[0] > sc[2]


def test_judge_logic():
    assert J.abstencao_label("sem_resposta", True) == "correta"
    assert J.abstencao_label("sem_resposta", False) == "em_falta"
    assert J.abstencao_label("factual", True) == "indevida"
    assert J.abstencao_label("multi", False) == "nao_aplicavel"
    raw = J.parse('texto {"correcao": 2, "frases": [{"apoiada": true}, {"apoiada": false}], "citacoes": 1, "lingua": "pt-PT"} fim')
    f = J.finalize(raw, "factual", False)
    assert f["fidelidade"] == 0.5 and f["correcao"] == 2
    assert J.finalize({}, "factual", True)["correcao"] == 0
    c = J.finalize({}, "sem_resposta", True)
    assert c["correcao"] is None and c["fidelidade"] is None and c["citacoes"] is None
    assert J.finalize({"correcao": 2, "frases": []}, "sem_resposta", False)["correcao"] == 0


def test_choose():
    mk = lambda r, m, f: {"recall_at_5": r, "mrr_at_10": m, "fidelidade_media": f}
    assert M.choose(mk(0.8, 0.5, 0.9), mk(0.6, 0.9, 0.9)) == "A"   # diferenca >= 0.05: recall
    assert M.choose(mk(0.70, 0.5, 0.9), mk(0.72, 0.6, 0.9)) == "B"  # < 0.05: MRR
    assert M.choose(mk(0.70, 0.5, 0.8), mk(0.70, 0.5, 0.9)) == "B"  # empate: fidelidade


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)

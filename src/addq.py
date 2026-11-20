"""Acrescenta uma pergunta a dados/perguntas.jsonl, calculando os offsets sozinho.

  python src/addq.py --tipo factual \\
     --pergunta "Qual e o horario da linha de apoio ao cliente da Guimabus?" \\
     --resposta "De segunda a sexta, das 09:00 as 18:00." \\
     --trecho "a2_amt_cm_guimaraes|Linha de Apoio ao Cliente~~09:00 às 18.00"

--trecho "DOC_ID|TEXTO"           texto exato do ficheiro limpo (usa \\n para quebras de linha)
--trecho "DOC_ID|INICIO~~FIM"     do inicio de INICIO ate ao fim de FIM (primeira ocorrencia depois do inicio):
                                  basta copiar as primeiras e as ultimas palavras do trecho
O inicio tem de ser unico no documento; se nao for, alonga-o.
  python src/addq.py --listar     mostra as perguntas e a contagem por tipo
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "dados"
Q = DADOS / "perguntas.jsonl"
AREA = 2
ESPERADO = {"factual": 8, "multi": 4, "sem_resposta": 3}


def read_clean(doc_id):
    p = DADOS / "limpo" / f"{doc_id}.txt"
    if not p.exists():
        sys.exit(f"ERRO: nao existe {p}")
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def locate(spec):
    doc_id, sep, frag = spec.partition("|")
    if not sep or not frag:
        sys.exit(f'ERRO: formato "DOC_ID|TEXTO": {spec!r}')
    text = read_clean(doc_id.strip())
    frag = frag.replace("\\n", "\n")
    start_s, sep2, end_s = frag.partition("~~")
    n = text.count(start_s)
    if n == 0:
        sys.exit(f"ERRO: inicio nao encontrado em {doc_id}: {start_s!r}\n(copia do ficheiro limpo: python src/ingest.py show {doc_id})")
    if n > 1:
        sys.exit(f"ERRO: o inicio ocorre {n} vezes em {doc_id}; alonga-o ate ficar unico: {start_s!r}")
    i = text.index(start_s)
    if sep2:
        j = text.find(end_s, i + len(start_s))
        if j == -1:
            sys.exit(f"ERRO: fim nao encontrado depois do inicio: {end_s!r}")
        k = j + len(end_s)
    else:
        k = i + len(start_s)
    if k - i > 1500:
        print(f"AVISO: trecho com {k-i} caracteres; trechos curtos e especificos avaliam melhor a recuperacao.")
    return {"doc_id": doc_id.strip(), "inicio": i, "fim": k, "texto": text[i:k]}


def load():
    if not Q.exists():
        return []
    return [json.loads(l) for l in Q.read_text(encoding="utf-8").splitlines() if l.strip()]


def listar(qs):
    for q in qs:
        docs = ",".join(sorted({t["doc_id"] for t in q["trechos"]})) or "-"
        print(f"{q['id']}  {q['tipo']:<13} {q['pergunta'][:70]}  [{docs}]")
    c = Counter(q["tipo"] for q in qs)
    print("\n" + "  ".join(f"{k}: {c.get(k,0)}/{v}" for k, v in ESPERADO.items()) + f"   total {len(qs)}/15")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tipo", choices=list(ESPERADO))
    ap.add_argument("--pergunta")
    ap.add_argument("--resposta")
    ap.add_argument("--trecho", action="append", default=[])
    ap.add_argument("--id")
    ap.add_argument("--listar", action="store_true")
    a = ap.parse_args()
    qs = load()
    if a.listar:
        return listar(qs)
    if not (a.tipo and a.pergunta and a.resposta):
        ap.error("--tipo, --pergunta e --resposta sao obrigatorios")
    trechos = [locate(s) for s in a.trecho]
    if a.tipo == "sem_resposta" and trechos:
        sys.exit("ERRO: sem_resposta nao tem trechos")
    if a.tipo == "factual" and len(trechos) != 1:
        sys.exit("ERRO: factual precisa de exatamente 1 trecho")
    if a.tipo == "multi" and len({t["doc_id"] for t in trechos}) < 2:
        sys.exit("ERRO: multi precisa de trechos de pelo menos 2 documentos")
    nums = [int(q["id"].rsplit("q", 1)[1]) for q in qs if "q" in q["id"]]
    qid = a.id or f"a{AREA}_q{(max(nums) if nums else 0) + 1:02d}"
    if any(q["id"] == qid for q in qs):
        sys.exit(f"ERRO: id ja existe: {qid}")
    rec = {"id": qid, "area": AREA, "tipo": a.tipo, "pergunta": a.pergunta,
           "resposta_referencia": a.resposta, "trechos": trechos}
    with open(Q, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"OK {qid} ({a.tipo})")
    for t in trechos:
        print(f"   {t['doc_id']} [{t['inicio']}:{t['fim']}] {t['texto'][:100]!r}")
    listar(qs + [rec])


if __name__ == "__main__":
    main()

"""Orquestracao. Exemplos:
  python src/run.py build  --config config/config_A.yaml
  python src/run.py answer --config config/config_A.yaml [--only a2_q01]
  python src/run.py judge  --config config/config_A.yaml
  python src/run.py metrics --config config/config_A.yaml
  python src/run.py retrieval --config config/config_A.yaml   # so recuperacao (sem LLM): Recall@5/MRR@10 -> resultados/A/recuperacao.json
  python src/run.py inspect --config config/config_A.yaml --id a2_q11   # pergunta, ouro, resposta, juiz e top-5 chunks
  python src/run.py tabela                 # tabela comparativa A/B/C em Markdown (para o relatorio)
  python src/run.py valida [--interativo [--so-citacoes]|--calcular]   # amostra de 15 respostas para pontuar a mao / concordancia com o juiz
  python src/run.py choose                                    # vencedora entre A e B
Todos os passos retomam de onde ficaram (ids ja feitos sao saltados)."""
import argparse, json, sys, time
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402


def load_cfg(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def res_dir(cfg):
    d = ROOT / "resultados" / cfg["nome"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def append(p, obj):
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def cmd_build(a):
    from chunking import build_chunks
    from retrieval import Index
    cfg = load_cfg(a.config)
    chunks = build_chunks(ROOT, cfg)
    toks = [c["n_tokens"] for c in chunks]
    stats = {"n_chunks": len(chunks), "tokens_medio": sum(toks) / len(toks), "tokens_max": max(toks), "tokens_min": min(toks)}
    print(stats)
    idx = Index(ROOT, cfg)
    idx.build(chunks)
    (idx.dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("Indice em", idx.dir)


def cmd_answer(a):
    from retrieval import Index
    from generate import answer, is_abstention
    cfg = load_cfg(a.config)
    idx = Index(ROOT, cfg).load()
    out = res_dir(cfg) / "respostas.jsonl"
    done = {r["id"] for r in M.jl(out)}
    for q in M.jl(ROOT / "dados" / "perguntas.jsonl"):
        if q["id"] in done or (a.only and q["id"] != a.only):
            continue
        t0 = time.time()
        chunks = idx.retrieve(q["pergunta"])
        llm_chunks = chunks[:cfg["top_k_llm"]]
        resp = answer(q["pergunta"], llm_chunks, cfg)
        rec = {"id": q["id"], "config": cfg["nome"],
               "chunks": [{k: c[k] for k in ("chunk_id", "doc_id", "inicio", "fim", "score")} for c in chunks],
               "resposta": resp, "abstencao": is_abstention(resp), "segundos": round(time.time() - t0, 1)}
        append(out, rec)
        print(q["id"], "abstencao" if rec["abstencao"] else "ok", f"{rec['segundos']}s")


def cmd_judge(a):
    import judge as J
    cfg = load_cfg(a.config)
    d = res_dir(cfg)
    idx = Index_chunks(cfg)
    qs = {q["id"]: q for q in M.jl(ROOT / "dados" / "perguntas.jsonl")}
    out = d / "avaliacoes.jsonl"
    done = {r["id"] for r in M.jl(out)}
    pending = [r for r in M.jl(d / "respostas.jsonl") if r["id"] not in done]
    for n, r in enumerate(pending):
        q = qs[r["id"]]
        ab = J.abstencao_label(q["tipo"], r["abstencao"])
        if ab in ("correta", "indevida"):  # nao gasta tokens do juiz
            raw = {}
        else:
            prompt = J.build_prompt(q, r["resposta"], [idx[c["chunk_id"]] for c in r["chunks"]], cfg["top_k_llm"])
            raw = J.parse(J.call_judge(prompt, cfg))
            if n < len(pending) - 1:
                time.sleep(cfg["judge"]["pause_s"])
        append(out, {"id": r["id"], "config": cfg["nome"], "tipo": q["tipo"], "juiz": J.finalize(raw, q["tipo"], r["abstencao"])})
        print(r["id"], ab)


def Index_chunks(cfg):
    from retrieval import Index
    i = Index(ROOT, cfg)
    return {c["chunk_id"]: c for c in map(json.loads, (i.dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines())}


def cmd_metrics(a, retrieval_only=False):
    cfg = load_cfg(a.config)
    d = res_dir(cfg)
    from retrieval import Index
    sp = Index(ROOT, cfg).dir / "stats.json"
    stats = json.loads(sp.read_text()) if sp.exists() else None
    m = M.aggregate(M.jl(ROOT / "dados" / "perguntas.jsonl"), M.jl(d / "respostas.jsonl"),
                    [] if retrieval_only else M.jl(d / "avaliacoes.jsonl"), stats)
    if not retrieval_only:
        (d / "metricas.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(m, indent=2, ensure_ascii=False))


def cmd_retrieval(a):
    """So recuperacao (sem LLM nem juiz): corre o retriever em todas as perguntas e calcula Recall@5/MRR@10."""
    from retrieval import Index
    cfg = load_cfg(a.config)
    idx = Index(ROOT, cfg).load()
    qs = M.jl(ROOT / "dados" / "perguntas.jsonl")
    recs = []
    for q in qs:
        chunks = idx.retrieve(q["pergunta"])
        recs.append({"id": q["id"], "config": cfg["nome"],
                     "chunks": [{k: c[k] for k in ("chunk_id", "doc_id", "inicio", "fim", "score")} for c in chunks]})
    d = res_dir(cfg)
    with open(d / "recuperacao.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sp = idx.dir / "stats.json"
    m = M.aggregate(qs, recs, [], json.loads(sp.read_text()) if sp.exists() else None)
    (d / "recuperacao.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Config {cfg['nome']}  (chunks {cfg['chunk_tokens']}/{cfg['overlap_tokens']}, {cfg['search']})")
    print(f"{'':14}{'n':>3}  {'Recall@5':>9}  {'MRR@10':>7}")
    f4 = lambda x: "  -  " if x is None else f"{x:.3f}"
    print(f"{'respondiveis':14}{sum(t['n_perguntas'] for k, t in m['por_tipo'].items() if k != 'sem_resposta'):>3}  {f4(m['recall_at_5']):>9}  {f4(m['mrr_at_10']):>7}")
    for k in ("factual", "multi"):
        t = m["por_tipo"][k]
        print(f"{k:14}{t['n_perguntas']:>3}  {f4(t['recall_at_5']):>9}  {f4(t['mrr_at_10']):>7}")
    print("\nPor pergunta (posicao do 1.o chunk que acerta, nos 10 primeiros):")
    pq = {q["id"]: q for q in qs}
    for r in recs:
        q = pq[r["id"]]
        if q["tipo"] == "sem_resposta":
            continue
        mrr = M.mrr_at_k(r["chunks"], q["trechos"], 10)
        pos = f"#{round(1 / mrr)}" if mrr else "falhou"
        print(f"  {r['id']}  {q['tipo']:<8} recall@5={M.recall_at_k(r['chunks'], q['trechos'], 5):.2f}  1.o acerto: {pos}")


def cmd_inspect(a):
    """Analise de erros: tudo sobre uma pergunta numa configuracao."""
    cfg = load_cfg(a.config)
    d = res_dir(cfg)
    q = next(q for q in M.jl(ROOT / "dados" / "perguntas.jsonl") if q["id"] == a.id)
    r = next(r for r in M.jl(d / "respostas.jsonl") if r["id"] == a.id)
    ev = next((e for e in M.jl(d / "avaliacoes.jsonl") if e["id"] == a.id), None)
    full = Index_chunks(cfg)
    print(f"{q['id']} [{q['tipo']}]  config {cfg['nome']}\nPERGUNTA: {q['pergunta']}\nREFERENCIA: {q['resposta_referencia']}")
    for t in q["trechos"]:
        print(f"OURO  {t['doc_id']}[{t['inicio']}:{t['fim']}]: {t['texto'][:200]!r}")
    print(f"\nRESPOSTA (abstencao={r['abstencao']}, {r.get('segundos')}s):\n{r['resposta']}")
    if ev:
        j = ev["juiz"]
        print(f"\nJUIZ: correcao={j['correcao']} fidelidade={j['fidelidade']} citacoes={j['citacoes']} "
              f"abstencao={j['abstencao']} lingua={j['lingua']} | {j['justificacao']}")
        for f in j["frases"]:
            print(f"   {'apoiada    ' if f.get('apoiada') else 'NAO apoiada'} [{f.get('excerto')}] {f.get('frase')}")
    print(f"\nTOP-{cfg['top_k_llm']} ENVIADOS AO LLM:")
    for i, c in enumerate(r["chunks"][:cfg["top_k_llm"]], 1):
        hits = [f"{t['doc_id']}[{t['inicio']}:{t['fim']}]" for t in q["trechos"] if M.hit(c, t)]
        print(f"[{i}] {c['chunk_id']} chars {c['inicio']}-{c['fim']} score={c['score']:.3f}" + (f"   <-- ACERTA {', '.join(hits)}" if hits else ""))
        print("     " + full[c["chunk_id"]]["texto"][:a.chars].replace("\n", " ") + "...")
    outros = [c["chunk_id"] for c in r["chunks"][cfg["top_k_llm"]:]]
    if outros:
        print(f"\n(posicoes {cfg['top_k_llm'] + 1}-{len(r['chunks'])}: {', '.join(outros)})")


def cmd_valida(a):
    """Amostra fixa (seed 42) de 15 respostas avaliadas, para pontuar a mao; --calcular mede a concordancia."""
    import csv, random
    out = ROOT / "resultados" / "validacao_manual.csv"
    cols = ["config", "id", "tipo", "pergunta", "resposta_referencia", "resposta_sistema", "juiz_correcao",
            "juiz_fidelidade", "juiz_citacoes", "manual_correcao", "manual_fidelidade", "manual_citacoes", "notas"]
    if a.interativo:
        rows = list(csv.DictReader(open(out, encoding="utf-8-sig", newline="")))
        cfgs = {c: load_cfg(ROOT / "config" / f"config_{c}.yaml") for c in "ABC"}
        chunks_by = {c: Index_chunks(cfgs[c]) for c in "ABC"}
        resp_by = {c: {r["id"]: r for r in M.jl(ROOT / "resultados" / c / "respostas.jsonl")} for c in "ABC"}
        pend = rows if a.so_citacoes else [r for r in rows if r["manual_correcao"].strip() == ""]

        def pergunta(txt, validos):
            while True:
                v = input(txt).strip()
                if v in validos:
                    return v
                print(f"  valor invalido (usa um de: {', '.join(sorted(validos))})")

        def gravar():
            with open(out, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

        if a.so_citacoes:
            print(f"A repontuar SO as citacoes de {len(pend)} respostas (o valor inicial fica guardado em notas). "
                  "Escala: 2 = todas as frases factuais citadas E cada [n] aponta para um excerto que as apoia; "
                  "1 = faltam citacoes ou alguma aponta para o excerto errado; 0 = sem citacoes ou quase todas erradas.")
        else:
            print(f"{len(pend)} por pontuar ({len(rows) - len(pend)} ja feitas). Grava a cada resposta; Ctrl+C sai e podes retomar.")
        try:
            for n, r in enumerate(pend, 1):
                rec = resp_by[r["config"]][r["id"]]
                print("=" * 100)
                print(f"[{n}/{len(pend)}]  config {r['config']}  {r['id']} ({r['tipo']})")
                print(f"PERGUNTA: {r['pergunta']}\nREFERENCIA: {r['resposta_referencia']}\n\nRESPOSTA DO SISTEMA:\n{r['resposta_sistema']}\n")
                for i, c in enumerate(rec["chunks"][:5], 1):
                    print(f"--- excerto [{i}] ({c['chunk_id']})\n{chunks_by[r['config']][c['chunk_id']]['texto']}\n")
                print("Pontua so com a resposta, a referencia e os 5 excertos acima (conhecimento externo nao conta).")
                if a.so_citacoes:
                    ant = r["manual_citacoes"]
                    r["manual_citacoes"] = pergunta(f"citacoes (0 / 1 / 2) [anterior: {ant}]: ", {"0", "1", "2"})
                    if "citacoes_inicial=" not in r["notas"]:
                        r["notas"] = (r["notas"] + " | " if r["notas"] else "") + f"citacoes_inicial={ant}"
                    gravar()
                    continue
                r["manual_correcao"] = pergunta("correcao (0 errada / 1 parcial / 2 certa): ", {"0", "1", "2"})
                tot = int(pergunta("n.o de frases factuais da resposta: ", {str(i) for i in range(1, 30)}))
                sup = int(pergunta("quantas estao apoiadas pelos excertos: ", {str(i) for i in range(0, tot + 1)}))
                r["manual_fidelidade"] = round(sup / tot, 3)
                r["manual_citacoes"] = pergunta("citacoes (0 / 1 / 2): ", {"0", "1", "2"})
                r["notas"] = input("notas (opcional): ").strip()
                gravar()
        except (KeyboardInterrupt, EOFError):
            print("\nInterrompido; o que ja fizeste ficou gravado.")
        return
    if not a.calcular:
        qs = {q["id"]: q for q in M.jl(ROOT / "dados" / "perguntas.jsonl")}
        cands = []
        for c in "ABC":
            d = ROOT / "resultados" / c
            ev, rs = M.jl(d / "avaliacoes.jsonl"), {r["id"]: r for r in M.jl(d / "respostas.jsonl")}
            if not ev:
                sys.exit(f"Falta avaliacoes.jsonl da config {c}: corre o juiz nas tres configuracoes primeiro "
                         f"(senao a amostra mudava).")
            for e in ev:
                if e["juiz"]["correcao"] is not None and not rs[e["id"]]["abstencao"]:
                    q, j = qs[e["id"]], e["juiz"]
                    cands.append({"config": c, "id": e["id"], "tipo": q["tipo"], "pergunta": q["pergunta"],
                                  "resposta_referencia": q["resposta_referencia"], "resposta_sistema": rs[e["id"]]["resposta"],
                                  "juiz_correcao": j["correcao"], "juiz_fidelidade": "" if j["fidelidade"] is None else round(j["fidelidade"], 3),
                                  "juiz_citacoes": "" if j["citacoes"] is None else j["citacoes"],
                                  "manual_correcao": "", "manual_fidelidade": "", "manual_citacoes": "", "notas": ""})
        cands.sort(key=lambda r: (r["config"], r["id"]))
        amostra = random.Random(42).sample(cands, min(15, len(cands)))
        amostra.sort(key=lambda r: (r["config"], r["id"]))
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(amostra)
        print(f"{len(amostra)} respostas (de {len(cands)} candidatas) em {out}")
        print("Preenche manual_correcao (0-2), manual_fidelidade (frases apoiadas / total) e manual_citacoes (0-2), "
              "sem olhar para as colunas juiz_*. Depois: python src/run.py valida --calcular")
        return
    import io
    txt = open(out, encoding="utf-8-sig", newline="").read()
    cab = txt.split("\n", 1)[0]
    delim = ";" if cab.count(";") > cab.count(",") else ","  # Excel em PT guarda CSV com ;
    rows = [r for r in csv.DictReader(io.StringIO(txt), delimiter=delim) if r["manual_correcao"].strip() != ""]
    if not rows:
        sys.exit("Ainda nao ha colunas manual_* preenchidas.")
    num = lambda x: float(str(x).replace(",", "."))
    print(f"{len(rows)} respostas pontuadas a mao")
    for campo, jc, mc in (("correcao", "juiz_correcao", "manual_correcao"), ("citacoes", "juiz_citacoes", "manual_citacoes")):
        par = [(num(r[jc]), num(r[mc])) for r in rows if r[jc] != "" and r[mc].strip() != ""]
        if par:
            print(f"{campo}: concordancia exata {sum(x == y for x, y in par)}/{len(par)} ({100 * sum(x == y for x, y in par) / len(par):.0f}%)")
    par = [(num(r["juiz_fidelidade"]), num(r["manual_fidelidade"]), r) for r in rows if r["juiz_fidelidade"] != "" and r["manual_fidelidade"].strip() != ""]
    if par:
        print(f"fidelidade: diferenca absoluta media {sum(abs(x - y) for x, y, _ in par) / len(par):.3f}; "
              f"dentro de 0.10: {sum(abs(x - y) <= 0.10 for x, y, _ in par)}/{len(par)}")
    import re as _re
    ini = [(num(r["juiz_citacoes"]), num(_re.search(r"citacoes_inicial=(\d)", r["notas"]).group(1)))
           for r in rows if r["juiz_citacoes"] != "" and _re.search(r"citacoes_inicial=(\d)", r["notas"])]
    if ini:
        print(f"citacoes (1.a pontuacao, antes de corrigir a leitura da escala): concordancia exata "
              f"{sum(x == y for x, y in ini)}/{len(ini)} ({100 * sum(x == y for x, y in ini) / len(ini):.0f}%)")
    print("\nDiscordancias:")
    for r in rows:
        dif = [f"{k}: juiz {r['juiz_' + k]} vs manual {r['manual_' + k]}" for k in ("correcao", "fidelidade", "citacoes")
               if r["manual_" + k].strip() != "" and r["juiz_" + k] != "" and abs(num(r["juiz_" + k]) - num(r["manual_" + k])) > (0.10 if k == "fidelidade" else 0)]
        if dif:
            print(f"  {r['config']} {r['id']}: " + "; ".join(dif))


def cmd_tabela(_):
    """Tabela Markdown A/B/C a partir de resultados/*/metricas.json e respostas.jsonl."""
    ms, lat = {}, {}
    for c in "ABC":
        d = ROOT / "resultados" / c
        ms[c] = json.loads((d / "metricas.json").read_text(encoding="utf-8"))
        t = [r["segundos"] for r in M.jl(d / "respostas.jsonl") if "segundos" in r]
        lat[c] = sum(t) / len(t) if t else None
    g = lambda c, *k: (lambda v: v)(ms[c] if not k else __import__("functools").reduce(lambda a, b: a[b], k, ms[c]))
    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    rows = [
        ("Recall@5 (12 respondiveis)", lambda c: f3(g(c, "recall_at_5"))),
        ("MRR@10", lambda c: f3(g(c, "mrr_at_10"))),
        ("Recall@5 factuais (8)", lambda c: f3(g(c, "por_tipo", "factual", "recall_at_5"))),
        ("Recall@5 multidocumento (4)", lambda c: f3(g(c, "por_tipo", "multi", "recall_at_5"))),
        ("Correcao media (0-2)", lambda c: f3(g(c, "correcao_media"))),
        ("Fidelidade media (0-1)", lambda c: f3(g(c, "fidelidade_media"))),
        ("Citacoes medias (0-2)", lambda c: f3(g(c, "citacoes_media"))),
        ("Abstencao correta (3 sem resposta)", lambda c: f"{round(g(c, 'abstencao_correta') * 3)}/3"),
        ("Abstencao indevida (12 respondiveis)", lambda c: f"{round(g(c, 'abstencao_indevida') * 12)}/12"),
        ("Latencia media (s por pergunta)", lambda c: f"{lat[c]:.0f}"),
        ("Chunks / tokens medios", lambda c: f"{g(c, 'chunks', 'n_chunks')} / {g(c, 'chunks', 'tokens_medio'):.0f}"),
    ]
    print("| Metrica | A (500/50, densa) | B (200/20, densa) | C (500/50, hibrida) |\n|---|---|---|---|")
    for nome, fn in rows:
        print(f"| {nome} | " + " | ".join(fn(c) for c in "ABC") + " |")


def cmd_choose(_):
    ms = {}
    for c in "AB":
        d = ROOT / "resultados" / c
        f = d / "metricas.json" if (d / "metricas.json").exists() else d / "recuperacao.json"
        print(f"{c}: a usar {f.name}")
        ms[c] = json.loads(f.read_text())
    print("Vencedora entre A e B:", M.choose(ms["A"], ms["B"]))
    if ms["A"].get("fidelidade_media") is None or ms["B"].get("fidelidade_media") is None:
        print("(sem fidelidade do juiz: so vale se Recall@5 ou MRR@10 desempataram; em empate total falta o juiz)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("build", cmd_build), ("answer", cmd_answer), ("judge", cmd_judge),
                     ("metrics", cmd_metrics), ("retrieval", cmd_retrieval)]:
        s = sub.add_parser(name); s.add_argument("--config", required=True)
        if name == "answer":
            s.add_argument("--only")
        s.set_defaults(f=fn)
    si = sub.add_parser("inspect"); si.add_argument("--config", required=True); si.add_argument("--id", required=True)
    si.add_argument("--chars", type=int, default=350); si.set_defaults(f=cmd_inspect)
    sv = sub.add_parser("valida"); sv.add_argument("--calcular", action="store_true"); sv.add_argument("--interativo", action="store_true"); sv.add_argument("--so-citacoes", action="store_true"); sv.set_defaults(f=cmd_valida)
    sub.add_parser("tabela").set_defaults(f=cmd_tabela)
    sub.add_parser("choose").set_defaults(f=cmd_choose)
    a = p.parse_args(); a.f(a)


if __name__ == "__main__":
    main()

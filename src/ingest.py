"""Ingestao do corpus da area 2 (Mobilidade sustentavel).

Uso:
  python src/ingest.py add --doc-id a2_cmg_pmus --original ~/Downloads/pmus.pdf \
      --titulo "Plano de Mobilidade Urbana Sustentavel" --entidade "Municipio de Guimaraes" \
      --url https://... [--subtema transportes_publicos] [--tipo pdf] \
      [--data-documento 2023-05-01] [--nao-guimaraes]
  (opcional) --cortar 1:140 --cortar 2210:    remove blocos (ver numeros com: python src/ingest.py raw <doc_id>)
  (opcional) --drop-linhas "^Dimensão Indicador" --drop-linhas "^\\d+$"   remove cabecalhos/n.o de pagina
  python src/ingest.py status          # palavras por documento e total (meta: 20000)
  python src/ingest.py show a2_cmg_pmus   # mostra o texto limpo com numero de linha

Regras: texto limpo em UTF-8, NFC, fins de linha \\n. Os offsets das perguntas
sao contados neste ficheiro, por isso NAO o edites depois de anotar.
"""
import argparse, hashlib, json, re, shutil, sys, unicodedata
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "dados" / "originais"
LIMPO = ROOT / "dados" / "limpo"
META = ROOT / "dados" / "metadados.jsonl"
AREA = 2
META_TARGET = 20000


def extract(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF (opcional; melhor qualidade)
            with fitz.open(path) as doc:
                return "\n\n".join(page.get_text("text") for page in doc)
        except ImportError:
            from pypdf import PdfReader  # alternativa pura em Python
            return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
    if ext in (".html", ".htm"):
        import trafilatura
        html = path.read_text(encoding="utf-8", errors="replace")
        text = trafilatura.extract(html, include_comments=False, include_tables=True,
                                   favor_recall=True)
        if not text:  # fallback
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for t in soup(["script", "style", "nav", "footer", "header"]):
                t.decompose()
            text = soup.get_text("\n")
        return text
    return path.read_text(encoding="utf-8", errors="replace")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")                      # hifen suave
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # palavras partidas no fim de linha (PDF)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def n_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def load_meta():
    if not META.exists():
        return []
    return [json.loads(l) for l in META.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_meta(rows):
    rows = sorted(rows, key=lambda r: r["doc_id"])
    with META.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_add(a):
    if not a.doc_id.startswith(f"a{AREA}_"):
        sys.exit(f"doc_id tem de comecar por a{AREA}_")
    src = Path(a.original).expanduser()
    if not src.exists():
        sys.exit(f"Ficheiro nao existe: {src}")
    dest_orig = ORIG / (a.doc_id + src.suffix.lower())
    if src.resolve() != dest_orig.resolve():
        shutil.copy2(src, dest_orig)
    out = LIMPO / f"{a.doc_id}.txt"
    rows = load_meta()
    existing = next((r for r in rows if r["doc_id"] == a.doc_id), None)
    if existing and out.exists() and not a.force:
        sys.exit("Documento ja existe. Se queres mesmo refazer o texto limpo (invalida "
                 "anotacoes!) usa --force.")
    raw_text = extract(dest_orig)
    if a.cortar:  # remove intervalos de linhas do texto EXTRAIDO (ver: ingest.py raw <doc_id>)
        lines = raw_text.split("\n")
        rem = set()
        for spec in a.cortar:
            ini, _, fim = spec.partition(":")
            rem.update(range(int(ini) - 1, int(fim) if fim else len(lines)))
        raw_text = "\n".join(l for i, l in enumerate(lines) if i not in rem)
        print(f"--cortar: removidas {len(rem)} linhas")
    drops = [re.compile(p) for p in (a.drop_linhas or [])]
    if drops:
        kept = [l for l in raw_text.split("\n") if not any(d.search(l) for d in drops)]
        print(f"--drop-linhas: removidas {len(raw_text.split(chr(10))) - len(kept)} linhas")
        raw_text = "\n".join(kept)
    text = clean(raw_text)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    row = {
        "doc_id": a.doc_id, "titulo": a.titulo, "entidade": a.entidade, "url": a.url,
        "data_acesso": a.data_acesso or date.today().isoformat(), "area": AREA,
        "subtema": a.subtema, "tipo": a.tipo or src.suffix.lower().lstrip("."),
        "data_documento": a.data_documento, "especifico_guimaraes": not a.nao_guimaraes,
        "ficheiro_limpo": f"limpo/{a.doc_id}.txt",
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "n_palavras": n_words(text),
        "n_palavras_nao_numericas": len([w for w in re.findall(r"\w+", text) if not w.isdigit()]),
        "limpeza": [f"cortar:{c}" for c in (a.cortar or [])] + [f"drop_linhas:{p}" for p in (a.drop_linhas or [])],
    }
    rows = [r for r in rows if r["doc_id"] != a.doc_id] + [row]
    save_meta(rows)
    print(f"OK {a.doc_id}: {row['n_palavras']} palavras, {len(text)} caracteres")
    print("Confirma o texto limpo antes de anotar (python src/ingest.py show", a.doc_id + ")")


def cmd_status(_):
    rows = load_meta()
    total = 0
    print(f"{'doc_id':40} {'palavras':>9} {'nao_num':>8}  subtema")
    for r in rows:
        total += r["n_palavras"]
        print(f"{r['doc_id']:40} {r['n_palavras']:>9} {r.get('n_palavras_nao_numericas', ''):>8}  {r.get('subtema') or ''}")
    print(f"\nTotal: {total} / {META_TARGET} palavras ({100*total/META_TARGET:.0f}%) em {len(rows)} documentos")
    if total < META_TARGET:
        print(f"Faltam {META_TARGET-total} palavras.")


def cmd_raw(a):
    """Texto extraido do original, com numero de linha (para escolher --cortar)."""
    cands = sorted(ORIG.glob(a.doc_id + ".*"))
    if not cands:
        sys.exit(f"Sem original para {a.doc_id} em {ORIG}")
    for i, line in enumerate(extract(cands[0]).split("\n"), 1):
        print(f"{i:6}| {line}")


def cmd_show(a):
    with open(LIMPO / f"{a.doc_id}.txt", encoding="utf-8", newline="") as f:
        text = f.read()
    for i, line in enumerate(text.split("\n"), 1):
        print(f"{i:5}| {line}")


def main():
    try:
        import signal; signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add")
    s.add_argument("--doc-id", required=True)
    s.add_argument("--original", required=True)
    s.add_argument("--titulo", required=True)
    s.add_argument("--entidade", required=True)
    s.add_argument("--url", required=True)
    s.add_argument("--data-acesso")
    s.add_argument("--subtema")
    s.add_argument("--tipo")
    s.add_argument("--data-documento")
    s.add_argument("--nao-guimaraes", action="store_true")
    s.add_argument("--force", action="store_true")
    s.add_argument("--drop-linhas", action="append", metavar="REGEX",
                   help="remove linhas que contenham esta regex (cabecalhos/rodapes repetidos); repetivel")
    s.set_defaults(f=cmd_add)
    sub.add_parser("status").set_defaults(f=cmd_status)
    s.add_argument("--cortar", action="append", metavar="A:B",
                   help="remove as linhas A..B (1-based, inclusivo; B vazio = ate ao fim) do texto extraido; repetivel")
    rw = sub.add_parser("raw"); rw.add_argument("doc_id"); rw.set_defaults(f=cmd_raw)
    sh = sub.add_parser("show"); sh.add_argument("doc_id"); sh.set_defaults(f=cmd_show)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

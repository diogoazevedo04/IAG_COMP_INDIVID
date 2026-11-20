"""LLM juiz (GPT-OSS 120B no Groq): grelha da secao 6 das Decisoes comuns."""
import json, os, re, time
import requests

URL = "https://api.groq.com/openai/v1/chat/completions"

PROMPT = """Avalias a resposta de um sistema de perguntas e respostas sobre mobilidade sustentável em Guimarães.
Devolve APENAS um objeto JSON, sem texto à volta.

Pergunta: {pergunta}

Resposta de referência: {referencia}

Excertos recuperados (a única evidência que o sistema podia usar):
{excertos}

Resposta do sistema:
{resposta}

Critérios:
- "correcao": 2 se todos os factos essenciais da referência estão certos; 1 se parcialmente certa ou incompleta; 0 se errada ou contradiz a referência.
- "frases": divide a resposta do sistema em frases factuais. Para cada uma: "frase" (texto), "apoiada" (true só se algum dos excertos acima a apoia; conhecimento externo NÃO conta), "excerto" (número do excerto que a apoia, ou null).
- "citacoes": 2 se todas as frases factuais têm citação [n] e as citações apontam para excertos que apoiam a frase; 1 se faltam citações ou alguma aponta para o excerto errado; 0 se não há citações ou quase todas estão erradas.
- "lingua": "pt-PT", "pt-BR" ou "outra".
- "justificacao": uma frase curta.

Formato exato:
{{"correcao": 0, "frases": [{{"frase": "...", "apoiada": true, "excerto": 1}}], "citacoes": 0, "lingua": "pt-PT", "justificacao": "..."}}"""


def build_prompt(q, resposta, chunks, k=5):
    ex = "\n\n".join(f"[{i}] {c['texto']}" for i, c in enumerate(chunks[:k], 1))
    return PROMPT.format(pergunta=q["pergunta"], referencia=q["resposta_referencia"], excertos=ex, resposta=resposta)


def parse(content):
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError("sem JSON na resposta do juiz")
    return json.loads(m.group(0))


def abstencao_label(tipo, abstained):
    if tipo == "sem_resposta":
        return "correta" if abstained else "em_falta"
    return "indevida" if abstained else "nao_aplicavel"


def finalize(raw, tipo, abstained):
    """Aplica as regras da grelha: abstencao calculada no codigo; fidelidade recalculada a partir das frases."""
    ab = abstencao_label(tipo, abstained)
    if ab == "correta":
        return {"correcao": None, "frases": [], "fidelidade": None, "citacoes": None,
                "abstencao": ab, "lingua": None, "justificacao": "Abstenção correta."}
    if ab == "indevida":
        return {"correcao": 0, "frases": [], "fidelidade": None, "citacoes": None,
                "abstencao": ab, "lingua": None, "justificacao": "Abstenção indevida."}
    frases = raw.get("frases", [])
    fid = (sum(1 for f in frases if f.get("apoiada")) / len(frases)) if frases else None
    out = {"correcao": raw.get("correcao"), "frases": frases, "fidelidade": fid, "citacoes": raw.get("citacoes"),
           "abstencao": ab, "lingua": raw.get("lingua"), "justificacao": raw.get("justificacao", "")}
    if ab == "em_falta":  # devia abster-se e respondeu: correcao 0
        out["correcao"] = 0
    return out


def call_judge(prompt, cfg):
    j = cfg["judge"]
    headers = {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]}
    body = {"model": j["model"], "temperature": j["temperature"], "reasoning_effort": "low",
            "max_completion_tokens": 2500, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(j["max_retries"]):
        r = requests.post(URL, headers=headers, json=body, timeout=120)
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 30)) + 5 * attempt
            print(f"  429, a esperar {wait:.0f}s"); time.sleep(wait); continue
        if r.status_code >= 400:
            raise RuntimeError(f"Groq {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError("Groq: demasiados 429")

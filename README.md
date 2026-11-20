# IAG 2026/27 - Area 2: Mobilidade sustentavel (William Mest)

## Instalacao
Ver "Guia de instalacao" do grupo (Ollama + `gemma4:12b` + `qwen3-embedding:0.6b`, chave `GROQ_API_KEY` em variavel de ambiente).
    pip install -r requirements.txt

## 1. Base de conhecimento
    python src/ingest.py add --doc-id a2_<entidade>_<tema> --original <ficheiro> --titulo "..." --entidade "..." --url ... --subtema ...
    python src/ingest.py status          # progresso para as 20 000 palavras
    python src/ingest.py show <doc_id>   # rever o texto limpo ANTES de anotar
Depois de anotar perguntas, o texto limpo nao pode mudar.

## 2. Conjunto de teste
Editar `dados/perguntas.jsonl` (15 perguntas: 8 factual, 4 multi, 3 sem_resposta).
    python src/validate.py find <doc_id> "texto exato"   # gera inicio/fim/texto
    python src/validate.py check

## 3. Experiencias (repetir para A, B e C)
    python src/run.py build  --config config/config_A.yaml
    python src/run.py retrieval --config config/config_A.yaml   # Recall@5 / MRR@10 sem LLM
    python src/run.py answer --config config/config_A.yaml
    python src/run.py judge  --config config/config_A.yaml      # ~45 s por avaliacao (Groq)
    python src/run.py metrics --config config/config_A.yaml
    python src/run.py choose                                     # vencedora A vs B -> copiar chunk_tokens/overlap para config_C.yaml

## Testes sem modelos
    python tests/test_offline.py

# Information Extraction Pipeline

Dieses Projekt führt eine zweistufige Information-Extraction-Pipeline aus:

1. Named Entity Recognition (NER)
2. Relation Extraction

Die Pipeline kann entweder mit einem Mock-Backend für schnelle lokale Tests oder mit einem echten vLLM-Backend für reale Modellläufe ausgeführt werden.

## Projektstruktur

```text
.
├── data/
│   └── sample.json
├── outputs/
│   └── predictions_mock.json
├── prompts/
│   ├── ner/
│   │   ├── v1_baseline.txt
│   │   ├── v2_definitions.txt
│   │   └── v3_fewshot.txt
│   └── relation/
│       ├── v1_baseline.txt
│       ├── v2_definitions.txt
│       ├── v3_fewshot.txt
│       └── v4_candidate_reasoning.txt
├── src/
│   ├── data_loader.py
│   ├── entities.py
│   ├── evaluation.py
│   ├── llm.py
│   ├── prompts.py
│   ├── relations.py
│   ├── schemas.py
│   └── main.py
└── README.md
```

## Installation

Das Projekt verwendet `uv`.

```bash
uv sync
```

Danach sollte die virtuelle Umgebung aktiviert sein oder über `uv run` automatisch verwendet werden.

## Ausführung

Das Projekt wird als Python-Modul über `src.main` gestartet.

```bash
python -m src.main [OPTIONEN]
```

## Mock-Durchlauf

Der Mock-Durchlauf eignet sich zum Testen der Pipeline ohne echtes LLM.

```bash
python -m src.main \
  --input data/sample.json \
  --output outputs/predictions_mock.json \
  --backend mock \
  --limit 5 \
  --ner-prompt prompts/ner/v1_baseline.txt \
  --relation-prompt prompts/relation/v1_baseline.txt
```



Beispielausgabe:

```text
Building predictions: 100%|██████████████████████████████████| 1/1 [00:00<00:00, 24105.20it/s]
```

Die erzeugten Predictions werden anschließend in folgender Datei gespeichert:

```text
outputs/predictions_mock.json
```

## Realer Durchlauf mit vLLM

Für einen echten Modelllauf wird das Backend `vllm` verwendet. Zusätzlich muss ein Modell angegeben werden.

```bash
python -m src.main \
  --input data/sample.json \
  --output outputs/predictions_vllm.json \
  --backend vllm \
  --model <MODEL_NAME_OR_PATH> \
  --ner-prompt prompts/ner/v3_fewshot.txt \
  --relation-prompt prompts/relation/v4_candidate_reasoning.txt
```

Beispiel:

```bash
python -m src.main \
  --input data/sample.json \
  --output outputs/predictions_vllm.json \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --ner-prompt prompts/ner/v3_fewshot.txt \
  --relation-prompt prompts/relation/v4_candidate_reasoning.txt
```

## Realer Durchlauf mit Tensor Parallelism

Wenn mehrere GPUs genutzt werden sollen, kann `--tp` gesetzt werden.

```bash
python -m src.main \
  --input data/sample.json \
  --output outputs/predictions_vllm_tp2.json \
  --backend vllm \
  --model <MODEL_NAME_OR_PATH> \
  --tp 2 \
  --ner-prompt prompts/ner/v3_fewshot.txt \
  --relation-prompt prompts/relation/v4_candidate_reasoning.txt
```

## Gold-Entities verwenden

Mit `--gold-entities` wird die NER-Stufe übersprungen. Stattdessen werden die Entities aus der Gold-Datei verwendet. Danach wird nur noch Relation Extraction ausgeführt.

```bash
python -m src.main \
  --input data/sample.json \
  --output outputs/predictions_gold_entities.json \
  --backend vllm \
  --model <MODEL_NAME_OR_PATH> \
  --gold data/gold.json \
  --gold-entities \
  --relation-prompt prompts/relation/v4_candidate_reasoning.txt
```

Wichtig: `--gold-entities` benötigt immer zusätzlich den Parameter `--gold`.

## Evaluation

Wenn eine Gold-Datei übergeben wird, wird nach dem Durchlauf automatisch eine Evaluation der Relationen berechnet.

```bash
python -m src.main \
  --input data/sample.json \
  --output outputs/predictions_eval.json \
  --backend mock \
  --gold data/gold.json \
  --limit 5 \
  --ner-prompt prompts/ner/v1_baseline.txt \
  --relation-prompt prompts/relation/v1_baseline.txt
```

Beispielausgabe:

```text
Evaluation:
{
  ...
}
```

## Prompt-Versionen

Die verwendeten Prompts können über `--ner-prompt` und `--relation-prompt` ausgewählt werden.

### NER-Prompts

```text
prompts/ner/v1_baseline.txt
prompts/ner/v2_definitions.txt
prompts/ner/v3_fewshot.txt
```

### Relation-Prompts

```text
prompts/relation/v1_baseline.txt
prompts/relation/v2_definitions.txt
prompts/relation/v3_fewshot.txt
prompts/relation/v4_candidate_reasoning.txt
```

## Beispiele für Prompt-Vergleiche

### Baseline

```bash
python -m src.main \
  --input data/sample.json \
  --output outputs/predictions_baseline.json \
  --backend vllm \
  --model <MODEL_NAME_OR_PATH> \
  --ner-prompt prompts/ner/v1_baseline.txt \
  --relation-prompt prompts/relation/v1_baseline.txt
```

### Definitions-Prompts

```bash
python -m src.main \
  --input data/sample.json \
  --output outputs/predictions_definitions.json \
  --backend vllm \
  --model <MODEL_NAME_OR_PATH> \
  --ner-prompt prompts/ner/v2_definitions.txt \
  --relation-prompt prompts/relation/v2_definitions.txt
```

### Few-shot NER und Candidate Reasoning

```bash
python -m src.main \
  --input data/sample.json \
  --output outputs/predictions_fewshot_reasoning.json \
  --backend vllm \
  --model <MODEL_NAME_OR_PATH> \
  --ner-prompt prompts/ner/v3_fewshot.txt \
  --relation-prompt prompts/relation/v4_candidate_reasoning.txt
```

## Parameterübersicht

| Parameter | Beschreibung | Default |
|---|---|---|
| `--input` | Pfad zur Input-Datei | `data/sample.json` |
| `--output` | Pfad zur Output-Datei | `outputs/predictions.json` |
| `--backend` | Backend für die Generierung: `mock` oder `vllm` | `mock` |
| `--model` | Modellname oder Modellpfad für vLLM | `None` |
| `--gold` | Pfad zur Gold-Datei für Evaluation oder Gold-Entities | `None` |
| `--limit` | Optionales Limit für die Anzahl der Dokumente | `None` |
| `--ner-prompt` | Pfad zum NER-Prompt | `prompts/ner_prompt.txt` |
| `--relation-prompt` | Pfad zum Relation-Prompt | `prompts/relation_prompt.txt` |
| `--tp` | Tensor Parallel Size für vLLM | `1` |
| `--gold-entities` | Nutzt Gold-Entities statt vorhergesagter Entities | deaktiviert |

## Output-Format

Die Predictions werden als JSON gespeichert.

Beispiel:

```json
[
  {
    "document_id": "doc_1",
    "text": "...",
    "entities": [
      {
        "id": "E1",
        "text": "...",
        "type": "..."
      }
    ],
    "relations": [
      {
        "head": "E1",
        "tail": "E2",
        "type": "..."
      }
    ]
  }
]
```

## Hinweise

- Für schnelle Tests sollte `--backend mock` verwendet werden.
- Für echte Läufe muss `--backend vllm` zusammen mit `--model` verwendet werden.
- Die Prompt-Pfade sollten explizit angegeben werden, da die Defaults in `main.py` möglicherweise nicht zur aktuellen Ordnerstruktur passen.
- Für reproduzierbare Experimente sollten Output-Dateien nach Backend, Modell und Prompt-Version benannt werden.
- Bei größeren Modellen kann `--tp` erhöht werden, wenn mehrere GPUs verfügbar sind.
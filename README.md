# Relation Extraction mit vLLM

Dieses Projekt implementiert eine zweistufige Pipeline für Relation Extraction:

1. **Named Entity Recognition (NER)**
2. **Relation Prediction** zwischen erkannten oder vorgegebenen Entities

Die Pipeline kann entweder mit einem Mock-Backend ohne GPU getestet oder mit vLLM auf einer GPU ausgeführt werden.

---

## Setup mit uv

Virtuelle Umgebung erstellen:

```bash
uv venv
```

Virtuelle Umgebung aktivieren:

```bash
source .venv/bin/activate
```

Abhängigkeiten installieren:

```bash
uv pip install -r requirements.txt
```

Falls das Projekt als Package installiert werden soll:

```bash
uv pip install -e .
```

---

## CoNLL04 exportieren

Vor der Ausführung kann der CoNLL04-Datensatz exportiert werden:

```bash
uv run python -m src.export_conll04 --split test
```

Dabei werden typischerweise Dateien wie diese erzeugt:

```text
data/conll04_test.json
data/conll04_test_gold.json
```

---

## Pipeline ausführen

Die Hauptpipeline wird über `src.main` gestartet.

### Lokaler Test ohne GPU

Für einen schnellen Test ohne GPU kann das Mock-Backend verwendet werden:

```bash
uv run python -m src.main \
  --backend mock \
  --input data/conll04_test.json \
  --output outputs/mock_conll04_test_limit5.json \
  --gold data/conll04_test_gold.json \
  --limit 5
```

---

## vLLM-Test auf GPU

Für die Ausführung mit vLLM wird ein Modell angegeben:

```bash
uv run python -m src.main \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --input data/conll04_test.json \
  --output outputs/vllm_conll04_test.json \
  --gold data/conll04_test_gold.json \
  --limit 20
```

---

## Tensor Parallelism

Für größere Modelle kann Tensor Parallelism über `--tp` gesetzt werden:

```bash
uv run python -m src.main \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --input data/conll04_test.json \
  --output outputs/vllm_conll04_test.json \
  --gold data/conll04_test_gold.json \
  --limit 20 \
  --tp 2
```

Standardwert:

```text
--tp 1
```

---

## Gold-Entities verwenden

Standardmäßig führt die Pipeline zuerst NER aus und verwendet anschließend die vorhergesagten Entities für die Relation Extraction.

Alternativ können die Entities direkt aus der Gold-Datei verwendet werden. Das ist nützlich, wenn nur die Relation Extraction evaluiert werden soll:

```bash
uv run python -m src.main \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --input data/conll04_test.json \
  --output outputs/vllm_conll04_test_gold_entities.json \
  --gold data/conll04_test_gold.json \
  --limit 20 \
  --gold-entities
```

Wichtig: `--gold-entities` benötigt immer zusätzlich `--gold`.

---

## Eigene Prompts verwenden

Die NER- und Relation-Prompts können über eigene Prompt-Dateien gesetzt werden:

```bash
uv run python -m src.main \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --input data/conll04_test.json \
  --output outputs/vllm_custom_prompts.json \
  --gold data/conll04_test_gold.json \
  --ner-prompt prompts/ner_prompt.txt \
  --relation-prompt prompts/relation_prompt.txt
```

Standardwerte:

```text
--ner-prompt prompts/ner_prompt.txt
--relation-prompt prompts/relation_prompt.txt
```

---

## Argumente

| Argument            | Beschreibung                                          | Standard                      |
| ------------------- | ----------------------------------------------------- | ----------------------------- |
| `--input`           | Pfad zur Eingabe-Datei                                | `data/sample.json`            |
| `--output`          | Pfad zur Ausgabe-Datei                                | `outputs/predictions.json`    |
| `--backend`         | Backend für die Generierung: `mock` oder `vllm`       | `mock`                        |
| `--model`           | Modellname für vLLM                                   | `None`                        |
| `--gold`            | Pfad zur Gold-Datei für Evaluation oder Gold-Entities | `None`                        |
| `--limit`           | Begrenzung der Anzahl der Dokumente                   | `None`                        |
| `--ner-prompt`      | Pfad zur NER-Prompt-Datei                             | `prompts/ner_prompt.txt`      |
| `--relation-prompt` | Pfad zur Relation-Prompt-Datei                        | `prompts/relation_prompt.txt` |
| `--tp`              | Tensor Parallel Size für vLLM                         | `1`                           |
| `--gold-entities`   | Verwendet Gold-Entities statt vorhergesagter Entities | deaktiviert                   |

---

## Output

Die Pipeline erzeugt eine JSON-Datei mit Predictions.

Jede Prediction enthält:

```json
{
  "document_id": "...",
  "text": "...",
  "entities": [
    {
      "id": "...",
      "text": "...",
      "type": "..."
    }
  ],
  "relations": [
    {
      "head": "...",
      "tail": "...",
      "type": "..."
    }
  ]
}
```

Der genaue Aufbau der Relationen hängt von den verwendeten Schemas und Parsern in `src.schemas` und `src.relations` ab.

---

## Evaluation

Wenn `--gold` gesetzt ist, wird nach der Prediction automatisch eine Evaluation ausgeführt.

Dabei werden die vorhergesagten Relationen und die Gold-Relationen verglichen.

Ausgegeben werden:

* Precision
* Recall
* F1

Beispiel:

```bash
uv run python -m src.main \
  --backend mock \
  --input data/conll04_test.json \
  --output outputs/mock_eval.json \
  --gold data/conll04_test_gold.json \
  --limit 5
```

Am Ende erscheint eine Ausgabe wie:

```text
Evaluation:
{
  "precision": ...,
  "recall": ...,
  "f1": ...
}
```

---

## Typischer Workflow

```bash
# 1. Umgebung erstellen und Abhängigkeiten installieren
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Daten exportieren
uv run python -m src.export_conll04 --split test

# 3. Mock-Test ausführen
uv run python -m src.main \
  --backend mock \
  --input data/conll04_test.json \
  --output outputs/mock_conll04_test_limit5.json \
  --gold data/conll04_test_gold.json \
  --limit 5

# 4. vLLM-Test ausführen
uv run python -m src.main \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --input data/conll04_test.json \
  --output outputs/vllm_conll04_test.json \
  --gold data/conll04_test_gold.json \
  --limit 20

# 5. Relation Extraction mit Gold-Entities testen
uv run python -m src.main \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --input data/conll04_test.json \
  --output outputs/vllm_conll04_test_gold_entities.json \
  --gold data/conll04_test_gold.json \
  --limit 20 \
  --gold-entities
```

---

## Hinweise

Wenn `--gold` angegeben wird, aber die Datei nicht existiert, gibt das Programm eine Warnung aus.

Wenn `--gold-entities` verwendet wird, muss `--gold` gesetzt sein, da die Entities aus der Gold-Datei geladen werden.

Die Pipeline verarbeitet NER- und Relation-Prompts batchweise über `llm.generate_batch`.

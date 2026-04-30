Jetzt machst du dein Projekt **abgabefähig lauffähig**: README + klare Befehle.

Erstelle/ersetze `README.md`:

````bash
cat > README.md << 'EOF'
# Relation Extraction mit vLLM

Dieses Projekt implementiert eine zweistufige Relation-Extraction-Pipeline:

1. Named Entity Recognition
2. Relation Prediction zwischen Entity-Paaren

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

## CoNLL04 exportieren

```bash
python -m src.export_conll04 --split test
```

## Lokaler Test ohne GPU

```bash
python -m src.main \
  --backend mock \
  --input data/conll04_test.json \
  --output outputs/mock_conll04_test_limit5.json \
  --gold data/conll04_test_gold.json \
  --limit 5
```

## vLLM-Test auf GPU

```bash
python -m src.main \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --input data/conll04_test.json \
  --output outputs/vllm_conll04_test.json \
  --gold data/conll04_test_gold.json \
  --limit 20
```

## Output

Die Pipeline erzeugt JSON mit:

* Dokument-ID
* Text
* erkannte Entities
* vorhergesagte Relationen

## Evaluation

Wenn `--gold` gesetzt ist, berechnet die Pipeline:

* Precision
* Recall
* F1
  EOF



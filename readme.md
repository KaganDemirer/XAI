# Post-Hoc Analyse eines Large Language Model (LLM) mit Logit Lens

Ziel dieser Arbeit ist es, ein Large Language Model (LLM) zu analysieren. Hierfür wird ein Logit Lens verwendet, um die Entscheidungen des Modells zu visualisieren und zu interpretieren. Hierfür werden zwei Large Language Models verwendet: OpenAI's GPT-2 und Phi 1_5 von Microsoft.

Beide Modelle werden mit verschiedensten Prompts getestet um Schlüsse über die Traningdaten und die Funktionsweise des Modells zu ziehen.

## 1. Installation

### 1.1. Clonen des Repositories

Zuerst muss das Repository geklont werden. Hierfür kann folgender Befehl verwendet werden:

```bash
git clone https://github.com/KaganDemirer/XAI.git
```

### 1.2. Installation der benötigten Bibliotheken

Um den geschriebenen Code auszuführen muss Python installiert sein.
Getestet wurde mit 3.9.6
Höhere Python Versionen wurden nicht getestet und könnten zu Fehlern führen.

Zuerst müssen die benötigten Python Bibliotheken installiert werden. Hierfür kann die `requirements.txt` Datei verwendet werden.

```bash
pip install -r requirements.txt
```


## 2. Anwendung

Anschließend können die Jupyter Notebooks ausgeführt werden. Vor der Ausführung kann im Jupyter Notebook über die Variable 'prompt' der Text, der analysiert werden soll, geändert werden.

## 3. Bekannte Probleme

Es gab Schwierigkeiten bei der Implementierung von Modellen. In einigen Fällen war eine Berechtigung erforderlich. Andere Modelle waren aufgrund ihres Speicherplatzbedarfs zu groß. Einige Modelle lieferten am Ende keine zufriedenstellenden Ergebnisse und waren daher nicht auswertbar.

## 4. Quellen

https://medium.com/@TaaniyaArora/visualizing-gpt2-word-embeddings-on-tensorboard-ea5c8fef9efa
https://nnsight.net/notebooks/tutorials/logit_lens/
https://huggingface.co/microsoft/phi-1_5
https://huggingface.co/openai-community/gpt2
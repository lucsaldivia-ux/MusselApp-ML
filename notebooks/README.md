# Notebooks

Pensados para **Google Colab gratis** (o Kaggle, ~30 h de GPU a la semana). Guardan checkpoints en
Google Drive para poder reanudar si se corta la sesión.

1. `01_preparar_dataset.ipynb`: une las etiquetas de Label Studio y separa por bandeja (`src/split_sin_fuga.py`).
2. `02_entrenar.ipynb`: entrena el detector (MediaPipe Model Maker; alternativa RF-DETR Nano).
3. `03_evaluar_exportar.ipynb`: métricas del CONTRATO.md y exportación a `.tflite` int8.

# Notebooks

**`entrenar_rfdetr.ipynb`**: para Google Colab gratis (GPU T4). Guarda los checkpoints en Google Drive
(`MusselApp-ML/checkpoints/`) y se reanuda solo si se corta la sesión.

1. Une las etiquetas de Label Studio (`dataset_v1/etiquetas/*.zip`) y separa por bandeja.
2. (Opcional) Pre-entrena con un dataset público de `DATOS_EXTERNOS.md`.
3. Entrena RF-DETR Nano con nuestras fotos.
4. Mide contra las metas del go/no-go (mAP, recall de chorito, confusión de especie, error del conteo).
5. Exporta a `.tflite` (FP32, FP16 e INT8) con `labels.txt` y `modelo.json` para la app.

Se eligió RF-DETR Nano en vez de MediaPipe Model Maker porque Model Maker no instala en el Python 3.12 de
Colab, y RF-DETR detecta mejor objetos chicos y amontonados.

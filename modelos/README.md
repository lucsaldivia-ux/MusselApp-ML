# Modelos entrenados

Los archivos `.tflite` se publican en las **Releases** de este repositorio (no en git).

| Versión | Fecha | Datos | Arquitectura | mAP@0,5 (prueba) | Tamaño INT8 | Release | Notas |
|---|---|---|---|---|---|---|---|
| model_v0_sintetico | 2026-11-25 | 80 fotos dibujadas (`src/dataset_sintetico.py`) | RF-DETR Nano, rfdetr 1.12.0 | 1,00 (sintético) | 30,1 MB | [model-v0-sintetico](https://github.com/lucsaldivia-ux/MusselApp-ML/releases/tag/model-v0-sintetico) | **Solo para integrar la app.** No sirve con fotos reales |
| model_v1 | — | Fotos del equipo | RF-DETR Nano | — | — | — | Pendiente (paso 9) |

## Lo que se aprendió con model_v0 (2026-11-25)

- Entrenar, reanudar, evaluar, predecir y exportar funcionan con `rfdetr==1.12.0` (probado en CPU con Windows).
- En Windows, los scripts necesitan `if __name__ == "__main__":` (multiproceso). En Colab (Linux) no hace falta.
- Columnas de puntajes del `.tflite`: 0 = chorito, 1 = cholga, 2 = referencia, 3 sin uso. `predict()` usa los mismos índices.
- La exportación INT8 avisa un error al escribir su `.rfdetr.json`: el `.tflite` se genera igual.

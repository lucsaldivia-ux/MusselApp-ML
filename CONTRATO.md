# Contrato entre el modelo y la app

Este es el acuerdo fijo entre este repositorio (entrenamiento) y la app (rama `feature/ml-conteo`).
Si algo cambia, se actualiza aquí y en `docs/plan-fase-final.md` de la app, y se avisa a Ramón.

## Archivos que se entregan a la app

En `app/src/main/assets/ml/`:

| Archivo | Contenido |
|---|---|
| `model_vN.tflite` | Modelo cuantizado (int8; fp16 si int8 pierde más de 3 puntos de mAP) |
| `labels.txt` | Una etiqueta por línea, en este orden: `chorito`, `cholga`, `referencia` |
| `modelo.json` | Descripción de entrada y salida (abajo) |

## modelo.json

Modelo: **RF-DETR Nano** (Apache 2.0), exportado con `rfdetr==1.12.0` (ver `notebooks/entrenar_rfdetr.ipynb`).

```json
{
  "version": "model_v1",
  "arquitectura": "RF-DETR Nano (Apache 2.0)",
  "archivo": "model_v1.tflite",
  "entrada": { "formato": "NHWC float32", "normalizacion": "ImageNet (media 0.485,0.456,0.406; desv. 0.229,0.224,0.225)" },
  "etiquetas": ["chorito", "cholga", "referencia"],
  "umbral_confianza": 0.40,
  "referencia_mm": 40.0
}
```

Lo que hay que saber en la app (de la documentación de RF-DETR, 6-nov-2026):

- **Entrada:** imagen RGB redimensionada al tamaño del tensor de entrada (384×384 en Nano), en **float32** y
  normalizada con media/desviación de ImageNet. No es uint8 de 0 a 255.
- **Salidas sin nombre** (`output_0`, `output_1`): las **cajas** son el tensor de rango 3 con última dimensión 4
  (centro x, centro y, ancho, alto, normalizados 0-1); los **puntajes** son el otro tensor de rango 3, a los que se
  aplica sigmoide. ⚠ Con 3 clases, el tensor de puntajes también puede terminar en 4: identificar las salidas por
  su **orden** con el archivo `.rfdetr.json` que acompaña a cada `.tflite`. Se confirma en el paso 9 con el modelo real.
- **INT8 es de rango dinámico** (pesos int8, cálculos en float). Si pierde más de 3 puntos de mAP, se usa el FP16.
- La exportación a TFLite de RF-DETR es **experimental**: si falla, la alternativa es ONNX Runtime para Android (MIT).

## Formato en Firestore (lo que sube la app y lee el portal)

```js
ml: { version: "model_v1", total: 64, choritos: 58, cholgas: 6, mmPorPx: 0.21,
      tallaChoritoMm: 54.2, tallaCholgaMm: 71.0, editado: true },
detecciones: [ { e: "C", p: 0.93, x: 120, y: 340, w: 60, h: 28, mm: 55, o: "M" } ]
// e: "C" chorito, "K" cholga · p: confianza 0-1 · x,y,w,h en escala 0-1000 · mm: largo · o: "M" modelo, "O" operario
```

## Metas mínimas (go/no-go)

| Métrica | Meta |
|---|---|
| mAP@0,5 por especie | ≥ 0,75 |
| Recall de chorito | ≥ 0,85 |
| Confusión chorito/cholga | ≤ 10 % |
| Error medio del conteo | ≤ 10 % del total |
| Error del calibre contra pie de metro | ≤ 3 mm |
| Inferencia en un teléfono de gama media | ≤ 1,5 s |

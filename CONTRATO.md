# Contrato entre el modelo y la app

Este es el acuerdo fijo entre este repositorio (entrenamiento) y la app (rama `feature/ml-conteo`).
Si algo cambia, se actualiza aquí y en `docs/plan-fase-final.md` de la app, y se avisa a Ramón.

## Archivos que se entregan a la app

En `app/src/main/assets/ml/`:

| Archivo | Contenido |
|---|---|
| `model_vN.tflite` | Modelo cuantizado (int8; fp16 si int8 pierde más de 3 puntos de mAP) |
| `labels.txt` | Una etiqueta por línea, en este orden: `chorito`, `cholga`, `referencia` (columnas 0, 1 y 2 de los puntajes) |
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

Lo que hay que saber en la app (**verificado el 2026-11-25** exportando RF-DETR Nano con `rfdetr==1.12.0`):

| Tensor | Forma | Contenido |
|---|---|---|
| Entrada 0 | `[1, 384, 384, 3]` float32 | Imagen RGB redimensionada a 384×384, normalizada con media/desviación de ImageNet (no uint8) |
| Salida 0 | `[1, 300, 4]` float32 | **Puntajes** sin sigmoide. Columna 0 = chorito, 1 = cholga, 2 = referencia; la 3 no se usa |
| Salida 1 | `[1, 300, 4]` float32 | **Cajas**: centro x, centro y, ancho y alto, normalizados 0-1 |

- ⚠ Con 3 clases **las dos salidas miden `[1, 300, 4]`** y en INT8 pierden el nombre: se toman **por posición**
  (0 = puntajes, 1 = cajas). En FP16/FP32 se llaman `labels` y `dets`.
- La columna c es la línea c de `labels.txt` (0 = chorito). La última columna sobra y se ignora.
- Verificado con un modelo entrenado y exportado: 28 de 28 individuos de una foto de prueba con la clase correcta.
- Post-proceso en la app: `SalidaRfDetr.decodificar(...)` (rama `feature/ml-reglas` de la app), con pruebas.
- Tamaños: INT8 de rango dinámico ~30 MB (el que va en la app), FP16 ~55 MB, FP32 ~109 MB.
- Si INT8 pierde más de 3 puntos frente a FP32, se usa FP16. Se mide con  (procesa como la app).
- Con XNNPACK (activo por defecto en LiteRT), en un PC: INT8 ~0,22 s, FP16 y FP32 ~0,21 s por foto ().
- La exportación a TFLite de RF-DETR es **experimental**; la alternativa es ONNX Runtime para Android (MIT).

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

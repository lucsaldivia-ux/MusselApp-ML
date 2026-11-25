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

```json
{
  "version": "model_v1",
  "archivo": "model_v1.tflite",
  "entrada": { "ancho": 384, "alto": 384, "canales": "RGB", "tipo": "uint8", "normalizacion": "0-255" },
  "etiquetas": ["chorito", "cholga", "referencia"],
  "salida": "cajas [N,4] (ymin,xmin,ymax,xmax normalizadas 0-1), clases [N], confianzas [N], N <= 300",
  "umbral_confianza": 0.40,
  "nms_iou": 0.50,
  "referencia_mm": 40.0
}
```

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

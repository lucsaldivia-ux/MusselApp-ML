# MusselApp-ML

Entrenamiento del modelo de **conteo automático de mitílidos** (chorito / cholga) para la app
[MusselApp](https://github.com/jesuscardenas-jacv1/MusselApp-Nube). Fase final del ramo DSY1105 (Duoc UC).

- **El modelo corre en el teléfono** (LiteRT / TensorFlow Lite): sin servicios de pago ni internet.
- **Todo es gratis:** Google Colab o Kaggle para entrenar, Label Studio para etiquetar, Google Drive para el dataset.
- **Licencias aptas para uso comercial:** Apache 2.0 / MIT. No se usa Ultralytics YOLO (AGPL).

## Qué hay aquí y qué no

| Aquí (git) | Fuera de git |
|---|---|
| Notebooks, scripts, contrato, protocolo | Fotos y etiquetas → Google Drive `MusselApp-ML/dataset_v1/` |
| `modelos/README.md` con métricas de cada versión | Archivos `.tflite` → GitHub Releases de este repo |

## Estructura

```
notebooks/   entrenar_rfdetr.ipynb (Colab gratis: prepara, entrena, mide y exporta)
src/         split_sin_fuga.py (separa entrenamiento/validación/prueba por bandeja)
             validar_calibre.py (mide el error del calibre contra la regla: meta del go/no-go)
             evaluar_tflite.py (mide el .tflite de la app contra las metas del go/no-go, como el teléfono)
             revisar_etiquetas.py (revisa cada lote etiquetado contra la planilla antes de entrenar)
             dataset_sintetico.py (fotos dibujadas para probar el pipeline sin fotos reales)
modelos/     README.md con la versión, métricas y fecha de cada modelo
CONTRATO.md  formato del modelo que espera la app (no cambiar sin avisar a Ramón)
```

## Datos

1. **Fotos propias** (lo principal): protocolo en el documento de equipo. Nombre `M<bandeja>_<iniciales>_<n>.jpg`.
2. **Datasets públicos** (solo para pre-entrenar la detección de "individuo"): únicamente con licencia
   **CC BY 4.0 o CC0**, anotando el autor y el enlace en `DATOS_EXTERNOS.md`. No sirven para distinguir
   chorito de cholga (son de otras especies y lugares).

Etiquetas: `chorito`, `cholga`, `referencia` (cuadrado negro de 40 mm) y `duda` (se revisa y se excluye).

## Revisar un lote etiquetado (antes de entregarlo)

Compara los recuadros de cada foto con el conteo a mano de `planilla_fotos` (descargada como CSV) y avisa
individuos sin etiquetar, fotos sin referencia, duplicados, exceso de `duda`, nombres mal puestos y fotos `_cal`
incompletas. Termina con código 1 si hay errores.

```
python src/revisar_etiquetas.py --coco result.json --planilla planilla_fotos.csv [--autor JC]
python src/revisar_etiquetas.py --prueba
```

## Separar el dataset sin fuga

Las fotos de una misma bandeja nunca deben quedar en conjuntos distintos:

```
python src/split_sin_fuga.py etiquetas/*.json --salida dataset_v1/split --semilla 42
```

## Medir el error del calibre (meta: ≤ 3 mm)

En cada bandeja del set de calibre se toma una foto extra terminada en `_cal` (`M002_JC_cal.jpg`) con **solo los
5 individuos medidos, en fila de izquierda a derecha en el orden 1..5 de la planilla**, más la referencia.
Se etiqueta como las demás. Luego, con la planilla exportada a CSV:

```
python src/validar_calibre.py --coco etiquetas.json --imagenes dataset_v1/crudas --planilla planilla_fotos.csv
python src/validar_calibre.py --prueba   # autoprueba con fotos dibujadas de largo conocido
```

Usa el mismo cálculo que `Calibre.kt` en la app (referencia → mm por píxel, Otsu, mancha más grande y eje mayor
por PCA). Con las etiquetas mide solo el calibre; con las predicciones del modelo en formato COCO mide el conjunto.

## Evaluar el `.tflite` que va en la app

El notebook mide el modelo de PyTorch; en el teléfono corre el `.tflite` INT8. Este script mide ese archivo con el
mismo preproceso y post-proceso que la app (sigmoide, umbral 0,40 y duplicados) y lo compara con FP16/FP32:

```
python src/evaluar_tflite.py --dataset dataset_v1 --modelo model_v1_int8.tflite --modelo model_v1_fp32.tflite
python src/evaluar_tflite.py --dataset dataset_v1 --modelo model_v1_int8.tflite --predicciones pred.json
python src/evaluar_tflite.py --prueba
```

`--predicciones` guarda lo detectado en formato COCO y sirve como `--coco` de `validar_calibre.py` (detector + calibre
juntos). Necesita `tensorflow` (usa XNNPACK como Android). **Ojo:** en Windows, el intérprete de `ai-edge-litert` 2.1
corre sin XNNPACK: tarda ~12 veces más y el FP16 da resultados erróneos (28,8 % de precisión). No sirve para medir.

Con `model-v0-sintetico` (12 fotos sintéticas de prueba): INT8 error de conteo 1,4 %, recall de chorito 98,3 %,
confusión 0 %, referencia en el 100 % de las fotos, 0,22 s por foto en el PC.

## Equipo

Lucas Saldivia (QA & Data, responsable) · Ramón Osorio (UI/UX & Mobile) · Jesús Cárdenas (Front-end)

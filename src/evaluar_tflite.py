"""Evalúa el `.tflite` que va en la app contra las metas del go/no-go, procesando como lo hará el teléfono.

El notebook mide el modelo de PyTorch (FP32). En el teléfono corre el `.tflite` INT8, que puede perder precisión al
cuantizarse. Este script mide ese archivo, con el mismo preproceso y el mismo post-proceso que la app:

- foto enderezada según su EXIF, redimensionada a 384×384, RGB / 255 y normalización de ImageNet;
- salida 0 = logits [1, 300, 4] (0 chorito, 1 cholga, 2 referencia, 3 sin uso), salida 1 = cajas cx, cy, w, h (0-1);
- sigmoide, umbral 0,40 y duplicados con IoU > 0,5 sin importar la clase (igual que `SalidaRfDetr.kt`).

Uso:
    python src/evaluar_tflite.py --dataset dataset_v1 --modelo model_v1_int8.tflite [--modelo model_v1_fp32.tflite]
    python src/evaluar_tflite.py --dataset dataset_v1 --modelo m.tflite --predicciones pred.json
    python src/evaluar_tflite.py --prueba          # autoprueba del post-proceso y de la comparación (sin modelo)

`--dataset` es la carpeta con `test/_annotations.coco.json` y sus fotos (la que arma el notebook). Con varios
`--modelo` compara cada uno contra el primero: si el INT8 pierde más de 3 puntos de recall frente al FP32, conviene el
FP16. `--predicciones` guarda lo detectado en formato COCO; sirve como `--coco` de `validar_calibre.py` para medir
detector + calibre juntos, como en el teléfono.

Los tiempos son del computador: sirven para comparar modelos, no reemplazan la medición en un teléfono (≤ 1,5 s).
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
from PIL import Image, ImageOps

LADO = 384
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESV = np.array([0.229, 0.224, 0.225], np.float32)
ETIQUETAS = ['chorito', 'cholga', 'referencia']   # labels.txt de la app
UMBRAL = 0.40
IOU_DUPLICADO = 0.5
IOU_ACIERTO = 0.5
PERDIDA_MAXIMA = 3.0   # puntos de recall que puede perder el INT8 frente al primero
MOTOR = ''

METAS = [
    # (clave, texto, meta, cumple)
    ('error_conteo', 'Error medio del conteo', '≤ 10 %', lambda v: v <= 10),
    ('recall_chorito', 'Recall de chorito', '≥ 85 %', lambda v: v >= 85),
    ('confusion', 'Confusión de especie', '≤ 10 %', lambda v: v <= 10),
]


# ── Mismo post-proceso que SalidaRfDetr.kt ──

def iou(a, b):
    """a y b como (x0, y0, x1, y1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decodificar(cajas, logits, clases=3, umbral=UMBRAL, iou_duplicado=IOU_DUPLICADO):
    """cajas [N, 4] (cx, cy, w, h) y logits [N, C] → lista de (confianza, clase, x0, y0, x1, y1) normalizados."""
    candidatas = []
    for i in range(cajas.shape[0]):
        c = int(np.argmax(logits[i, :clases]))
        conf = float(1.0 / (1.0 + np.exp(-float(logits[i, c]))))
        if conf < umbral:
            continue
        cx, cy, w, h = (float(v) for v in cajas[i])
        x0, y0 = min(max(cx - w / 2, 0.0), 1.0), min(max(cy - h / 2, 0.0), 1.0)
        x1, y1 = min(max(cx + w / 2, 0.0), 1.0), min(max(cy + h / 2, 0.0), 1.0)
        if x1 <= x0 or y1 <= y0:
            continue
        candidatas.append((conf, c, x0, y0, x1, y1))
    candidatas.sort(key=lambda d: -d[0])
    if iou_duplicado >= 1.0:
        return candidatas
    elegidas = []
    for d in candidatas:
        if all(iou(d[2:], e[2:]) <= iou_duplicado for e in elegidas):
            elegidas.append(d)
    return elegidas


def preparar(img):
    x = np.asarray(img.convert('RGB').resize((LADO, LADO), Image.BILINEAR), dtype=np.float32) / 255.0
    return ((x - MEDIA) / DESV)[None]


def abrir(ruta, ancho, alto):
    """Abre la foto con la orientación del etiquetado (el notebook ya la guarda derecha; esto cubre otras carpetas)."""
    cruda = Image.open(ruta)
    girada = ImageOps.exif_transpose(cruda)
    for img in (girada, cruda):
        if img.size == (ancho, alto):
            return img.convert('RGB')
    raise ValueError(f'{os.path.basename(ruta)}: mide {girada.size} y las etiquetas dicen {(ancho, alto)}')


# ── Comparación con las etiquetas ──

def comparar(reales, detectadas):
    """reales y detectadas: listas de (especie, (x0, y0, x1, y1)) en la misma escala. Devuelve contadores."""
    r = {'reales': 0, 'detectadas': 0, 'choritos': 0, 'choritos_ok': 0, 'cholgas': 0, 'cholgas_ok': 0,
         'emparejados': 0, 'confundidos': 0, 'ref_real': 0, 'ref_ok': 0}
    conchas_r = [x for x in reales if x[0] in ('chorito', 'cholga')]
    conchas_d = [x for x in detectadas if x[0] in ('chorito', 'cholga')]
    r['reales'], r['detectadas'] = len(conchas_r), len(conchas_d)
    usadas = set()
    for especie, caja in conchas_r:
        r['choritos' if especie == 'chorito' else 'cholgas'] += 1
        mejor, mejor_iou = None, IOU_ACIERTO
        for j, (_, cd) in enumerate(conchas_d):
            v = iou(caja, cd)
            if j not in usadas and v >= mejor_iou:
                mejor, mejor_iou = j, v
        if mejor is None:
            continue
        usadas.add(mejor)
        r['emparejados'] += 1
        if conchas_d[mejor][0] != especie:
            r['confundidos'] += 1
        else:
            r['choritos_ok' if especie == 'chorito' else 'cholgas_ok'] += 1
    refs_r = [c for e, c in reales if e == 'referencia']
    refs_d = [c for e, c in detectadas if e == 'referencia']
    if refs_r:
        r['ref_real'] = 1
        r['ref_ok'] = int(any(iou(a, b) >= IOU_ACIERTO for a in refs_r for b in refs_d))
    return r


def resumir(por_foto):
    total = {k: sum(f[k] for f in por_foto) for k in por_foto[0]} if por_foto else {}
    errores = [abs(f['detectadas'] - f['reales']) / f['reales'] * 100 for f in por_foto if f['reales']]
    pct = lambda a, b: a / b * 100 if b else float('nan')
    return {
        'fotos': len(por_foto),
        'error_conteo': statistics.mean(errores) if errores else float('nan'),
        'recall_chorito': pct(total.get('choritos_ok', 0), total.get('choritos', 0)),
        'recall_cholga': pct(total.get('cholgas_ok', 0), total.get('cholgas', 0)),
        'confusion': pct(total.get('confundidos', 0), total.get('emparejados', 0)),
        'precision': pct(total.get('emparejados', 0), total.get('detectadas', 0)),
        'referencia': pct(total.get('ref_ok', 0), total.get('ref_real', 0)),
    }


# ── Evaluación de un modelo ──

def interprete(ruta, hilos):
    """Intérprete con XNNPACK, el acelerador de CPU que LiteRT usa por defecto en Android.

    Ojo: en Windows, el Interpreter de ai-edge-litert 2.1 corre SIN XNNPACK y es ~12 veces más lento (2,5 s en vez
    de 0,2 s con el FP32); por eso se prefiere el de TensorFlow, que sí lo activa.
    """
    global MOTOR
    try:
        import warnings
        warnings.filterwarnings('ignore', message='.*tf.lite.Interpreter is deprecated.*')
        import tensorflow as tf
        it = tf.lite.Interpreter(model_path=ruta, num_threads=hilos,
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO)
        MOTOR = f'TensorFlow Lite {tf.__version__} con XNNPACK, {hilos} hilos'
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=ruta, num_threads=hilos)
        MOTOR = f'ai-edge-litert, {hilos} hilos (en Windows puede ir sin XNNPACK: tiempos inflados)'
    it.allocate_tensors()
    ent = it.get_input_details()[0]
    if list(ent['shape']) != [1, LADO, LADO, 3] or ent['dtype'] != np.float32:
        raise ValueError(f'{os.path.basename(ruta)}: la entrada es {list(ent["shape"])} {ent["dtype"]}, '
                         f'se esperaba [1, {LADO}, {LADO}, 3] float32 (contrato en CONTRATO.md)')
    return it


def evaluar(ruta_modelo, coco, carpeta, hilos=4, umbral=UMBRAL):
    it = interprete(ruta_modelo, hilos)
    ent = it.get_input_details()[0]
    sal = it.get_output_details()
    cats = {c['id']: c['name'] for c in coco['categories']}
    por_imagen = {}
    for a in coco['annotations']:
        por_imagen.setdefault(a['image_id'], []).append(a)
    por_foto, tiempos, predicciones = [], [], []
    for im in coco['images']:
        img = abrir(os.path.join(carpeta, im['file_name']), im['width'], im['height'])
        W, H = img.size
        it.set_tensor(ent['index'], preparar(img))
        t = time.perf_counter()
        it.invoke()
        tiempos.append((time.perf_counter() - t) * 1000)
        # Por posición: en INT8 los tensores pierden el nombre (CONTRATO.md)
        logits = it.get_tensor(sal[0]['index'])[0]
        cajas = it.get_tensor(sal[1]['index'])[0]
        dets = decodificar(cajas, logits, umbral=umbral)
        detectadas = [(ETIQUETAS[c], (x0 * W, y0 * H, x1 * W, y1 * H)) for _, c, x0, y0, x1, y1 in dets]
        reales = []
        for a in por_imagen.get(im['id'], []):
            x, y, w, h = a['bbox']
            reales.append((cats[a['category_id']], (x, y, x + w, y + h)))
        por_foto.append(comparar(reales, detectadas))
        for (conf, c, *_), (_, (x0, y0, x1, y1)) in zip(dets, detectadas):
            predicciones.append({'image_id': im['id'], 'category_id': c, 'score': round(conf, 4),
                                 'bbox': [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]})
    res = resumir(por_foto)
    res['ms_mediana'] = statistics.median(tiempos[1:] or tiempos)   # la primera inferencia calienta el intérprete
    res['mb'] = os.path.getsize(ruta_modelo) / 1e6
    return res, predicciones


def coco_predicciones(coco, predicciones):
    """Mismo formato que las etiquetas (sirve como --coco de validar_calibre.py)."""
    cats = [{'id': i, 'name': n} for i, n in enumerate(ETIQUETAS)]
    anots = [dict(p, id=i + 1, area=p['bbox'][2] * p['bbox'][3], iscrowd=0) for i, p in enumerate(predicciones)]
    return {'images': coco['images'], 'categories': cats, 'annotations': anots}


def informe(nombres, resultados):
    f = lambda v: '—' if v != v else f'{v:.1f}'
    ancho = max(14, *(len(n) for n in nombres))
    print(f'\n{"Métrica":<26}{"Meta":<9}' + ''.join(f'{n:>{ancho + 2}}' for n in nombres))
    for clave, texto, meta, cumple in METAS:
        celdas = []
        for r in resultados:
            v = r[clave]
            marca = '' if v != v else (' ✓' if cumple(v) else ' ✗')
            celdas.append(f'{f(v) + " %" + marca:>{ancho + 2}}')
        print(f'{texto:<26}{meta:<9}' + ''.join(celdas))
    for clave, texto, unidad in [('recall_cholga', 'Recall de cholga', ' %'), ('precision', 'Precisión', ' %'),
                                 ('referencia', 'Referencia encontrada', ' %'), ('ms_mediana', 'Tiempo en PC', ' ms'),
                                 ('mb', 'Tamaño', ' MB')]:
        print(f'{texto:<26}{"":<9}' + ''.join(f'{f(r[clave]) + unidad:>{ancho + 2}}' for r in resultados))
    print(f'\nFotos de prueba: {resultados[0]["fotos"]}. Tiempo medido en este computador ({MOTOR}): en un '
          f'teléfono de gama media la meta es ≤ 1,5 s y hay que medirla ahí.')
    principal = resultados[0]
    cumple_todo = all(cumple(principal[c]) for c, _, _, cumple in METAS if principal[c] == principal[c])
    print(f'{nombres[0]}: {"CUMPLE" if cumple_todo else "NO CUMPLE"} las metas de detección '
          f'(el calibre se mide aparte con validar_calibre.py).')
    if principal['referencia'] == principal['referencia'] and principal['referencia'] < 95:
        print(f'OJO: encuentra la referencia en {principal["referencia"]:.0f} % de las fotos; sin ella no hay calibre.')
    for n, r in zip(nombres[1:], resultados[1:]):
        diferencia = r['recall_chorito'] - principal['recall_chorito']
        if diferencia > PERDIDA_MAXIMA:
            print(f'OJO: {nombres[0]} tiene {diferencia:.1f} puntos menos de recall de chorito que {n}; '
                  f'conviene usar el FP16 en la app.')


# ── Autoprueba sin modelo ──

def autoprueba():
    errores = []

    def revisar(cond, texto):
        print(('  ok   ' if cond else '  FALLA ') + texto)
        if not cond:
            errores.append(texto)

    logit = lambda p: float(np.log(p / (1 - p)))
    cajas = np.array([[0.30, 0.30, 0.10, 0.20],    # chorito 0,9
                      [0.31, 0.30, 0.10, 0.20],    # duplicado del anterior, cholga 0,6 → se descarta
                      [0.70, 0.50, 0.12, 0.24],    # cholga 0,8
                      [0.10, 0.10, 0.05, 0.05],    # referencia 0,95
                      [0.50, 0.90, 0.10, 0.10],    # bajo el umbral (0,3)
                      [0.90, 0.90, 0.00, 0.10]],   # sin ancho → se descarta
                     np.float32)
    logits = np.full((6, 4), -9.0, np.float32)
    logits[0, 0] = logit(0.9); logits[1, 1] = logit(0.6); logits[2, 1] = logit(0.8)
    logits[3, 2] = logit(0.95); logits[4, 0] = logit(0.3); logits[5, 0] = logit(0.9)
    logits[:, 3] = 5.0   # la última columna no se usa aunque sea la más alta
    d = decodificar(cajas, logits)
    revisar([x[1] for x in d] == [2, 0, 1], f'post-proceso: clases {[x[1] for x in d]} (esperado [2, 0, 1])')
    revisar(abs(d[1][2] - 0.25) < 1e-6 and abs(d[1][5] - 0.40) < 1e-6, 'post-proceso: centro → esquina')

    reales = [('chorito', (0, 0, 10, 10)), ('chorito', (20, 0, 30, 10)), ('cholga', (40, 0, 50, 10)),
              ('chorito', (60, 0, 70, 10)), ('referencia', (80, 0, 90, 10))]
    detectadas = [('chorito', (0, 0, 10, 10)), ('cholga', (20, 0, 30, 10)), ('cholga', (40, 0, 50, 10)),
                  ('chorito', (100, 0, 110, 10)), ('referencia', (80, 0, 90, 10))]
    r = resumir([comparar(reales, detectadas)])
    revisar(abs(r['recall_chorito'] - 100 / 3) < 0.01, f'recall de chorito {r["recall_chorito"]:.1f} (esperado 33,3)')
    revisar(abs(r['confusion'] - 100 / 3) < 0.01, f'confusión {r["confusion"]:.1f} (esperado 33,3)')
    revisar(r['error_conteo'] == 0 and r['precision'] == 75, f'conteo {r["error_conteo"]} y precisión {r["precision"]}')
    revisar(r['referencia'] == 100, 'referencia encontrada')
    print('\nAutoprueba ' + ('OK' if not errores else f'con {len(errores)} fallas'))
    return 0 if not errores else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', help='carpeta con test/_annotations.coco.json y sus fotos')
    ap.add_argument('--split', default='test')
    ap.add_argument('--modelo', action='append', help='archivo .tflite; repetir para comparar (el primero es el de la app)')
    ap.add_argument('--umbral', type=float, default=UMBRAL)
    ap.add_argument('--hilos', type=int, default=4)
    ap.add_argument('--predicciones', help='guarda lo que detecta el primer modelo en formato COCO')
    ap.add_argument('--prueba', action='store_true', help='autoprueba sin modelo')
    a = ap.parse_args()
    if a.prueba:
        return autoprueba()
    if not a.dataset or not a.modelo:
        ap.error('faltan --dataset y --modelo (o usa --prueba)')
    carpeta = os.path.join(a.dataset, a.split)
    coco = json.load(open(os.path.join(carpeta, '_annotations.coco.json'), encoding='utf-8'))
    nombres, resultados = [], []
    for i, ruta in enumerate(a.modelo):
        print(f'Evaluando {os.path.basename(ruta)} en {len(coco["images"])} fotos…', flush=True)
        res, pred = evaluar(ruta, coco, carpeta, a.hilos, a.umbral)
        nombres.append(os.path.splitext(os.path.basename(ruta))[0][:28])
        resultados.append(res)
        if i == 0 and a.predicciones:
            json.dump(coco_predicciones(coco, pred), open(a.predicciones, 'w', encoding='utf-8'))
            print(f'Predicciones guardadas en {a.predicciones}')
    informe(nombres, resultados)
    return 0


if __name__ == '__main__':
    sys.exit(main())

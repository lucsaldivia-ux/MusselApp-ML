"""Pre-etiqueta fotos para Label Studio: dibuja solo los recuadros de cada concha y el de la referencia.

Usa el modelo pre-entrenado (encuentra "un mejillón", no la especie) y un detector clásico del cuadrado negro de 40 mm.
Así, en Label Studio solo queda **revisar**: corregir recuadros que falten o sobren y, en bandejas mezcladas, elegir
la especie de cada concha.

La especie de cada recuadro sale de la planilla (`choritos_a_mano` / `cholgas_a_mano` de la bandeja):
- bandeja solo de choritos → todos `chorito`;
- bandeja solo de cholgas → todos `cholga`;
- bandeja mezclada o sin planilla → todos `duda` (hay que elegir uno por uno; `revisar_etiquetas.py` avisa si quedan).
`--especie` fuerza una especie para todas las fotos.

Uso (en Colab, sección 3b del notebook, o en un PC con rfdetr instalado):
    python src/preetiquetar.py --modelo checkpoint_best_total.pth --fotos fotos/ --planilla planilla_fotos.csv \\
        --salida tareas.json --vista vista/
    python src/preetiquetar.py --prueba      # autoprueba sin modelo

`tareas.json` se importa en Label Studio (ver README, "Pre-etiquetado"). Las imágenes se referencian como
`/data/local-files/?d=<carpeta>/<archivo>`, para Label Studio con almacenamiento de archivos locales.
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

UMBRAL = 0.20   # bajo a propósito: borrar un recuadro sobrante es más rápido que dibujar uno que falta
IOU_DUPLICADO = 0.5
EXTENSIONES = ('.jpg', '.jpeg', '.png')
COLORES = {'chorito': '#0F766E', 'cholga': '#C76C05', 'referencia': '#1E40AF', 'duda': '#9CA3AF'}


# ── Referencia de 40 mm: cuadrado negro, sólido, en fondo claro ──

def otsu(valores):
    hist = np.bincount(valores.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    suma = np.dot(np.arange(256), hist)
    suma_b = peso_b = 0.0
    mejor, umbral = -1.0, 0
    for t in range(256):
        peso_b += hist[t]
        if peso_b == 0:
            continue
        peso_f = total - peso_b
        if peso_f == 0:
            break
        suma_b += t * hist[t]
        m_b, m_f = suma_b / peso_b, (suma - suma_b) / peso_f
        var = peso_b * peso_f * (m_b - m_f) ** 2
        if var > mejor:
            mejor, umbral = var, t
    return umbral


def buscar_referencia(img, lado_max=1000):
    """Devuelve (x0, y0, x1, y1) del cuadrado negro más cuadrado y más lleno, o None.

    Una concha oscura ocupa ~78 % de su recuadro (es una elipse); el cuadrado impreso, ~100 %. Por eso se exige
    lados parecidos (0,85–1,18) y relleno ≥ 0,90.
    """
    from scipy import ndimage
    escala = min(1.0, lado_max / max(img.size))
    chica = img.convert('L').resize((max(1, int(img.width * escala)), max(1, int(img.height * escala))))
    lum = np.asarray(chica, dtype=np.uint8)
    oscuro = lum <= min(otsu(lum), 90)          # el cuadrado es casi negro
    etiquetas, n = ndimage.label(oscuro)
    if n == 0:
        return None
    area_min = 0.0005 * lum.size
    mejor, mejor_puntaje = None, 0.0
    for i, sl in enumerate(ndimage.find_objects(etiquetas), start=1):
        if sl is None:
            continue
        alto, ancho = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if alto * ancho < area_min or min(alto, ancho) < 8:
            continue
        proporcion = ancho / alto
        relleno = (etiquetas[sl] == i).sum() / (alto * ancho)
        if not (0.85 <= proporcion <= 1.18) or relleno < 0.90:
            continue
        puntaje = relleno - abs(1 - proporcion)
        if puntaje > mejor_puntaje:
            mejor, mejor_puntaje = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop), puntaje
    if mejor is None:
        return None
    return tuple(v / escala for v in mejor)


# ── Detecciones ──

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def limpiar(cajas, referencia):
    """Quita duplicados y las cajas que en realidad son la referencia. cajas: [(x0, y0, x1, y1, puntaje)]."""
    elegidas = []
    for c in sorted(cajas, key=lambda c: -c[4]):
        if referencia and iou(c[:4], referencia) > 0.3:
            continue
        if all(iou(c[:4], e[:4]) <= IOU_DUPLICADO for e in elegidas):
            elegidas.append(c)
    return elegidas


def resultado_ls(caja, etiqueta, ancho, alto, puntaje=None, n=0):
    x0, y0, x1, y1 = caja
    r = {'id': f'p{n}', 'from_name': 'label', 'to_name': 'image', 'type': 'rectanglelabels',
         'original_width': ancho, 'original_height': alto, 'image_rotation': 0,
         'value': {'x': x0 / ancho * 100, 'y': y0 / alto * 100, 'width': (x1 - x0) / ancho * 100,
                   'height': (y1 - y0) / alto * 100, 'rotation': 0, 'rectanglelabels': [etiqueta]}}
    if puntaje is not None:
        r['score'] = round(float(puntaje), 3)
    return r


# ── Planilla ──

def especies_planilla(ruta):
    especies = {}
    if not ruta:
        return especies
    with open(ruta, encoding='utf-8-sig', newline='') as f:
        for fila in csv.DictReader(f):
            bandeja = (fila.get('bandeja') or '').strip().upper()
            try:
                ch = int(float((fila.get('choritos_a_mano') or '').replace(',', '.')))
                ko = int(float((fila.get('cholgas_a_mano') or '').replace(',', '.')))
            except ValueError:
                continue
            especies[bandeja] = ('chorito' if ko == 0 and ch > 0 else 'cholga' if ch == 0 and ko > 0 else 'duda', ch + ko)
    return especies


def bandeja_de(nombre):
    m = re.match(r'(M\d{3})_', os.path.basename(nombre), re.IGNORECASE)
    return m.group(1).upper() if m else None


# ── Proceso ──

def preetiquetar(fotos, detectar, salida, carpeta_ls='fotos', planilla=None, especie=None, vista=None):
    """detectar(img) → [(x0, y0, x1, y1, puntaje)] en píxeles. Devuelve las tareas y un resumen por foto."""
    especies = especies_planilla(planilla)
    tareas, resumen = [], []
    if vista:
        os.makedirs(vista, exist_ok=True)
    for ruta in fotos:
        nombre = os.path.basename(ruta)
        img = ImageOps.exif_transpose(Image.open(ruta)).convert('RGB')   # Label Studio la muestra así, derecha
        ancho, alto = img.size
        bandeja = bandeja_de(nombre)
        etiqueta, esperados = especies.get(bandeja, (especie or 'duda', None))
        if especie:
            etiqueta = especie
        referencia = buscar_referencia(img)
        conchas = limpiar(detectar(img), referencia)
        resultados = [resultado_ls(c[:4], etiqueta, ancho, alto, c[4], i) for i, c in enumerate(conchas)]
        if referencia:
            resultados.append(resultado_ls(referencia, 'referencia', ancho, alto, None, len(resultados)))
        tareas.append({'data': {'image': f'/data/local-files/?d={carpeta_ls}/{nombre}'},
                       'predictions': [{'model_version': 'preentreno_externo', 'result': resultados}]})
        resumen.append((nombre, etiqueta, len(conchas), esperados, referencia is not None))
        if vista:
            dib = img.copy()
            d = ImageDraw.Draw(dib)
            grosor = max(3, ancho // 400)
            for c in conchas:
                d.rectangle(c[:4], outline=COLORES[etiqueta], width=grosor)
            if referencia:
                d.rectangle(referencia, outline=COLORES['referencia'], width=grosor * 2)
            dib.thumbnail((1600, 1600))
            dib.save(os.path.join(vista, os.path.splitext(nombre)[0] + '_vista.jpg'), quality=85)
    json.dump(tareas, open(salida, 'w', encoding='utf-8'), ensure_ascii=False)
    for nombre, etiqueta, n, esperados, ref in resumen:
        extra = f' (planilla: {esperados})' if esperados is not None else ''
        print(f'{nombre}: {n} {etiqueta}{extra}' + ('' if ref else '  · SIN referencia detectada: dibújala a mano'))
    print(f'\n{len(tareas)} fotos pre-etiquetadas → {salida}')
    return tareas, resumen


def detector_rfdetr(ruta_modelo, umbral):
    from rfdetr import RFDETR
    modelo = RFDETR.from_checkpoint(ruta_modelo)

    def detectar(img):
        det = modelo.predict(img, threshold=umbral)
        return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(p)) for b, p in zip(det.xyxy, det.confidence)]
    return detectar


# ── Autoprueba sin modelo ──

def autoprueba():
    fallas = 0

    def revisar(cond, texto):
        nonlocal fallas
        fallas += not cond
        print(('  ok    ' if cond else '  FALLA ') + texto)

    with tempfile.TemporaryDirectory() as d:
        # Foto dibujada: fondo claro, 3 conchas (elipses oscuras) y la referencia (cuadrado negro de 200 px)
        img = Image.new('RGB', (1600, 1200), (236, 232, 222))
        dib = ImageDraw.Draw(img)
        conchas = [(300, 400, 560, 520), (700, 600, 900, 900), (1100, 300, 1400, 450)]
        for c in conchas:
            dib.ellipse(c, fill=(30, 35, 50))
        dib.rectangle((1300, 900, 1500, 1100), fill=(5, 5, 5))
        dib.ellipse((100, 900, 260, 1060), fill=(25, 25, 25))   # concha redonda: no debe confundirse con la referencia
        foto = os.path.join(d, 'M201_LS_01.jpg')
        img.save(foto, quality=95)
        ref = buscar_referencia(Image.open(foto))
        revisar(ref is not None and abs(ref[0] - 1300) < 8 and abs(ref[3] - 1100) < 8,
                f'encuentra la referencia y no la concha redonda: {ref}')
        planilla = os.path.join(d, 'planilla.csv')
        with open(planilla, 'w', encoding='utf-8', newline='') as f:
            f.write('bandeja,choritos_a_mano,cholgas_a_mano\nM201,0,4\n')

        def falso(img):   # "modelo" que devuelve las conchas, un duplicado y una caja sobre la referencia
            return [c + (0.9,) for c in conchas] + [(302, 402, 558, 518, 0.6), (1295, 895, 1505, 1105, 0.8),
                                                    (100, 900, 260, 1060, 0.7)]
        tareas, resumen = preetiquetar([foto], falso, os.path.join(d, 't.json'), planilla=planilla,
                                       vista=os.path.join(d, 'vista'))
        res = tareas[0]['predictions'][0]['result']
        etiquetas = [r['value']['rectanglelabels'][0] for r in res]
        revisar(etiquetas.count('cholga') == 4 and etiquetas.count('referencia') == 1,
                f'bandeja solo de cholgas → 4 cholga + 1 referencia, sin duplicado ni la caja de la referencia: {etiquetas}')
        v = res[0]['value']
        revisar(0 <= v['x'] <= 100 and 0 < v['width'] <= 100 and res[0]['original_width'] == 1600,
                'coordenadas en porcentaje, como las pide Label Studio')
        revisar(tareas[0]['data']['image'] == '/data/local-files/?d=fotos/M201_LS_01.jpg', 'ruta de la imagen')
        revisar(os.path.isfile(os.path.join(d, 'vista', 'M201_LS_01_vista.jpg')), 'genera la vista previa')
        revisar(especies_planilla(planilla)['M201'][0] == 'cholga', 'especie desde la planilla')
    print('\nAutoprueba ' + ('OK' if not fallas else f'con {fallas} fallas'))
    return 1 if fallas else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modelo', help='checkpoint_best_total.pth del pre-entrenamiento (o de un modelo posterior)')
    ap.add_argument('--fotos', help='carpeta con las fotos')
    ap.add_argument('--planilla', help='planilla_fotos en CSV (para saber la especie de cada bandeja)')
    ap.add_argument('--especie', choices=['chorito', 'cholga', 'duda'], help='forzar una especie para todas')
    ap.add_argument('--umbral', type=float, default=UMBRAL)
    ap.add_argument('--autor', help='solo las fotos de estas iniciales (M200_LS_01.jpg → LS)')
    ap.add_argument('--carpeta-ls', default='fotos', help='subcarpeta de las fotos dentro de la raíz de Label Studio')
    ap.add_argument('--salida', default='tareas.json')
    ap.add_argument('--vista', help='carpeta para vistas previas con los recuadros dibujados')
    ap.add_argument('--prueba', action='store_true')
    a = ap.parse_args()
    if a.prueba:
        return autoprueba()
    if not a.modelo or not a.fotos:
        ap.error('faltan --modelo y --fotos (o usa --prueba)')
    fotos = sorted(os.path.join(a.fotos, f) for f in os.listdir(a.fotos) if f.lower().endswith(EXTENSIONES)
                   and (not a.autor or f'_{a.autor.upper()}_' in f.upper()))
    if not fotos:
        ap.error(f'no hay fotos en {a.fotos}' + (f' con las iniciales {a.autor}' if a.autor else ''))
    if a.vista:
        os.makedirs(a.vista, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.salida)), exist_ok=True)
    preetiquetar(fotos, detector_rfdetr(a.modelo, a.umbral), a.salida, a.carpeta_ls, a.planilla, a.especie, a.vista)
    return 0


if __name__ == '__main__':
    sys.exit(main())

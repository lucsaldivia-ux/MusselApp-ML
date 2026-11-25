"""Mide el error del calibre (meta del go/no-go: error <= 3 mm) con el set de calibre.

Protocolo (docs/instrucciones-ml/equipo.md de la app): en cada bandeja del set de calibre se toma una foto extra,
con nombre terminado en `_cal` (por ejemplo `M002_JC_cal.jpg`), que tiene SOLO los 5 individuos medidos con regla,
en una fila de izquierda a derecha en el orden 1..5 de la planilla, más la referencia de 40 mm.

El cálculo es el mismo de la app (`Calibre.kt`): mm por píxel con el recuadro de la referencia, luminancia del
recorte de cada concha, umbral de Otsu y largo del eje mayor por PCA. Así se valida lo que hará el teléfono.

Uso:
    python src/validar_calibre.py --coco etiquetas.json --imagenes carpeta/ --planilla planilla_fotos.csv
    python src/validar_calibre.py --prueba        # autoprueba con fotos dibujadas de largo conocido

`--coco` acepta las etiquetas de Label Studio (recuadros dibujados a mano: valida solo el calibre) o las
predicciones del modelo en el mismo formato (valida detector + calibre, como en el teléfono).
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile

import numpy as np
from PIL import Image, ImageOps

REFERENCIA_MM = 40.0
PROPORCION_MIN, PROPORCION_MAX = 0.9, 1.1
PIXELES_MINIMOS = 20
META_MM = 3.0
N_MEDIDOS = 5


# ── Mismo algoritmo que Calibre.kt ──

def mm_por_pixel(ancho_px, alto_px, lado_mm=REFERENCIA_MM):
    if ancho_px <= 0 or alto_px <= 0:
        return None
    if not PROPORCION_MIN <= ancho_px / alto_px <= PROPORCION_MAX:
        return None
    return lado_mm / ((ancho_px + alto_px) / 2)


def umbral_otsu(lum):
    hist = np.bincount(np.clip(lum.ravel(), 0, 255), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    suma_total = float((np.arange(256) * hist).sum())
    suma_fondo = peso_fondo = 0.0
    mejor, umbral = -1.0, 128
    for t in range(256):
        peso_fondo += hist[t]
        if peso_fondo == 0:
            continue
        peso_frente = total - peso_fondo
        if peso_frente == 0:
            break
        suma_fondo += t * hist[t]
        mf = suma_fondo / peso_fondo
        mfr = (suma_total - suma_fondo) / peso_frente
        var = peso_fondo * peso_frente * (mf - mfr) ** 2
        if var > mejor:
            mejor, umbral = var, t
    return umbral


def mancha_mayor(mascara):
    """Se queda con la mancha oscura conectada más grande (4 vecinos): descarta trozos de conchas vecinas."""
    alto, ancho = mascara.shape
    plano = mascara.ravel()
    etiqueta = np.zeros(plano.size, dtype=np.int32)
    mejor, mejor_n, actual = 0, 0, 0
    for inicio in np.flatnonzero(plano):
        if etiqueta[inicio]:
            continue
        actual += 1
        etiqueta[inicio] = actual
        pila, n = [inicio], 0
        while pila:
            i = pila.pop()
            n += 1
            x = i % ancho
            for j in (i - ancho, i + ancho, i - 1 if x > 0 else -1, i + 1 if x < ancho - 1 else -1):
                if 0 <= j < plano.size and plano[j] and not etiqueta[j]:
                    etiqueta[j] = actual
                    pila.append(j)
        if n > mejor_n:
            mejor, mejor_n = actual, n
    return (etiqueta == mejor).reshape(alto, ancho)


def largo_eje_mayor_px(lum):
    ys, xs = np.nonzero(mancha_mayor(lum <= umbral_otsu(lum)))
    if len(xs) < PIXELES_MINIMOS:
        return None
    dx, dy = xs - xs.mean(), ys - ys.mean()
    cxx, cyy, cxy = (dx * dx).sum(), (dy * dy).sum(), (dx * dy).sum()
    traza, det = cxx + cyy, cxx * cyy - cxy * cxy
    lam = traza / 2 + math.sqrt(max(0.0, traza * traza / 4 - det))
    vx, vy = lam - cyy, cxy
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        vx, vy = 1.0, 0.0
    n = math.hypot(vx, vy)
    p = dx * (vx / n) + dy * (vy / n)
    return float(p.max() - p.min() + 1)


def luminancia(img_rgb):
    a = np.asarray(img_rgb, dtype=np.float64)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.int64)


# ── Lectura de datos ──

def abrir_imagen(ruta, ancho_coco, alto_coco):
    """Abre la foto con la misma orientación que usó el etiquetado (el navegador aplica la rotación EXIF)."""
    cruda = Image.open(ruta)
    girada = ImageOps.exif_transpose(cruda)
    for img in (girada, cruda):
        if img.size == (ancho_coco, alto_coco):
            return img.convert('RGB')
    raise ValueError(f'{os.path.basename(ruta)}: mide {girada.size} y las etiquetas dicen {(ancho_coco, alto_coco)}')


def nombre_real(nombre):
    """Label Studio antepone un código al nombre del archivo ("1a2b3c4d-M002_JC_cal.jpg")."""
    base = os.path.basename(nombre)
    m = re.search(r'(M\d{3}_.*)$', base)
    return m.group(1) if m else base


def leer_planilla(ruta):
    medidas = {}
    with open(ruta, encoding='utf-8-sig', newline='') as f:
        for fila in csv.DictReader(f):
            if (fila.get('set_calibre') or '').strip().lower() not in ('si', 'sí'):
                continue
            largos = []
            for i in range(1, N_MEDIDOS + 1):
                valor = (fila.get(f'largo_{i}_mm') or '').strip().replace(',', '.')
                largos.append((fila.get(f'especie_{i}', '').strip().lower(), float(valor) if valor else None))
            medidas[fila['bandeja'].strip().upper()] = largos
    return medidas


def validar(coco, carpeta, medidas):
    cats = {c['id']: c['name'] for c in coco['categories']}
    cajas = {}
    for a in coco['annotations']:
        cajas.setdefault(a['image_id'], []).append(a)
    filas, avisos = [], []
    for im in coco['images']:
        nombre = nombre_real(im['file_name'])
        if not re.search(r'_cal\.(jpe?g|png)$', nombre, re.I):
            continue
        bandeja = nombre.split('_')[0].upper()
        if bandeja not in medidas:
            avisos.append(f'{nombre}: la bandeja {bandeja} no está marcada como set de calibre en la planilla')
            continue
        anots = cajas.get(im['id'], [])
        refs = [a for a in anots if cats.get(a['category_id']) == 'referencia']
        conchas = sorted((a for a in anots if cats.get(a['category_id']) in ('chorito', 'cholga')),
                         key=lambda a: a['bbox'][0] + a['bbox'][2] / 2)
        if len(refs) != 1 or len(conchas) != N_MEDIDOS:
            avisos.append(f'{nombre}: se esperaban 1 referencia y {N_MEDIDOS} conchas; hay {len(refs)} y {len(conchas)}')
            continue
        mmpx = mm_por_pixel(refs[0]['bbox'][2], refs[0]['bbox'][3])
        if mmpx is None:
            avisos.append(f'{nombre}: la referencia no es cuadrada (foto inclinada)')
            continue
        img = abrir_imagen(os.path.join(carpeta, nombre), im['width'], im['height'])
        for i, a in enumerate(conchas):
            especie_real, largo_real = medidas[bandeja][i]
            if largo_real is None:
                continue
            x, y, w, h = (int(round(v)) for v in a['bbox'])
            px = largo_eje_mayor_px(luminancia(img.crop((x, y, x + w, y + h))))
            medido = px * mmpx if px else None
            filas.append({
                'foto': nombre, 'n': i + 1, 'especie': cats[a['category_id']], 'especie_planilla': especie_real,
                'real_mm': largo_real, 'medido_mm': None if medido is None else round(medido, 1),
                'error_mm': None if medido is None else round(medido - largo_real, 1),
            })
    return filas, avisos


def resumen(filas, avisos):
    for a in avisos:
        print('AVISO', a)
    print(f"\n{'foto':<22}{'n':>2}  {'especie':<9}{'real':>7}{'medido':>8}{'error':>7}")
    for f in filas:
        medido = '-' if f['medido_mm'] is None else f"{f['medido_mm']:.1f}"
        error = '-' if f['error_mm'] is None else f"{f['error_mm']:+.1f}"
        marca = '' if f['especie'] == f['especie_planilla'] or not f['especie_planilla'] else '  (especie distinta a la planilla)'
        print(f"{f['foto']:<22}{f['n']:>2}  {f['especie']:<9}{f['real_mm']:>7.1f}{medido:>8}{error:>7}{marca}")
    errores = [abs(f['error_mm']) for f in filas if f['error_mm'] is not None]
    sin_medir = sum(1 for f in filas if f['error_mm'] is None)
    if not errores:
        print('\nNo hubo individuos medibles.')
        return {'n': 0, 'cumple': False}
    r = {
        'n': len(errores), 'sin_medir': sin_medir,
        'error_medio_mm': round(float(np.mean(errores)), 2),
        'error_maximo_mm': round(float(np.max(errores)), 2),
        'dentro_de_3mm': round(sum(e <= META_MM for e in errores) / len(errores), 3),
    }
    r['cumple'] = r['error_medio_mm'] <= META_MM and sin_medir == 0
    print(f"\nIndividuos: {r['n']} (sin medir: {sin_medir}) · error medio {r['error_medio_mm']} mm · "
          f"máximo {r['error_maximo_mm']} mm · dentro de 3 mm: {r['dentro_de_3mm']:.0%}")
    print('META DE CALIBRE (error medio <= 3 mm):', 'CUMPLE' if r['cumple'] else 'NO CUMPLE')
    return r


# ── Autoprueba: fotos dibujadas con largos conocidos ──

def autoprueba():
    from PIL import ImageDraw
    rng = np.random.default_rng(7)
    carpeta = tempfile.mkdtemp(prefix='calibre_prueba_')
    imagenes, anots, planilla = [], [], []
    for k in range(3):
        mmpx = 0.12 + 0.04 * k                        # distinta distancia de la cámara en cada foto
        W, H = 3000, 2000
        img = Image.new('RGB', (W, H), (238, 236, 230))
        d = ImageDraw.Draw(img)
        lado = REFERENCIA_MM / mmpx
        d.rectangle([60, 60, 60 + lado, 60 + lado], fill=(10, 10, 10))
        iid = k + 1
        anots.append({'id': len(anots) + 1, 'image_id': iid, 'category_id': 3, 'bbox': [60, 60, lado, lado]})
        fila = {'bandeja': f'M9{k:02d}', 'set_calibre': 'si'}
        for i in range(N_MEDIDOS):
            especie = 'cholga' if i % 3 == 2 else 'chorito'
            largo_mm = float(rng.uniform(40, 85))
            L, A = largo_mm / mmpx, largo_mm / mmpx * 0.45
            cx, cy = 500 + i * 520, 1200   # apretadas: la vecina entra al recuadro y la mancha mayor la descarta
            ang = math.radians(float(rng.uniform(0, 180)))
            pts = [(cx + L / 2 * math.cos(t) * math.cos(ang) - A / 2 * math.sin(t) * math.sin(ang),
                    cy + L / 2 * math.cos(t) * math.sin(ang) + A / 2 * math.sin(t) * math.cos(ang))
                   for t in np.linspace(0, 2 * math.pi, 200)]
            d.polygon(pts, fill=(30, 35, 55))
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            anots.append({'id': len(anots) + 1, 'image_id': iid, 'category_id': 1 if especie == 'chorito' else 2,
                          'bbox': [min(xs) - 2, min(ys) - 2, max(xs) - min(xs) + 4, max(ys) - min(ys) + 4]})
            fila[f'especie_{i + 1}'], fila[f'largo_{i + 1}_mm'] = especie, f'{largo_mm:.1f}'
        nombre = f"{fila['bandeja']}_XX_cal.jpg"
        img.save(os.path.join(carpeta, nombre), quality=90)
        imagenes.append({'id': iid, 'file_name': 'abcd1234-' + nombre, 'width': W, 'height': H})
        planilla.append(fila)
    coco = {'images': imagenes, 'annotations': anots,
            'categories': [{'id': 1, 'name': 'chorito'}, {'id': 2, 'name': 'cholga'}, {'id': 3, 'name': 'referencia'}]}
    ruta_csv = os.path.join(carpeta, 'planilla.csv')
    campos = ['bandeja', 'set_calibre'] + [c for i in range(1, 6) for c in (f'especie_{i}', f'largo_{i}_mm')]
    with open(ruta_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(planilla)
    filas, avisos = validar(coco, carpeta, leer_planilla(ruta_csv))
    r = resumen(filas, avisos)
    assert r['n'] == 15 and not avisos, 'la autoprueba debía medir 15 individuos sin avisos'
    assert r['error_maximo_mm'] < 1.0, 'con fotos dibujadas el error debe ser menor a 1 mm'
    print('\nAUTOPRUEBA OK')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--coco', help='etiquetas o predicciones en formato COCO')
    p.add_argument('--imagenes', help='carpeta con las fotos _cal')
    p.add_argument('--planilla', help='planilla_fotos exportada como CSV')
    p.add_argument('--salida', help='guardar el detalle y el resumen en este JSON')
    p.add_argument('--prueba', action='store_true', help='autoprueba con fotos dibujadas')
    a = p.parse_args()
    if a.prueba:
        autoprueba()
        return
    if not (a.coco and a.imagenes and a.planilla):
        p.error('faltan --coco, --imagenes y --planilla (o usa --prueba)')
    with open(a.coco, encoding='utf-8') as f:
        coco = json.load(f)
    filas, avisos = validar(coco, a.imagenes, leer_planilla(a.planilla))
    r = resumen(filas, avisos)
    if a.salida:
        with open(a.salida, 'w', encoding='utf-8') as f:
            json.dump({'resumen': r, 'detalle': filas, 'avisos': avisos}, f, ensure_ascii=False, indent=2)
    sys.exit(0 if r['cumple'] else 1)


if __name__ == '__main__':
    main()

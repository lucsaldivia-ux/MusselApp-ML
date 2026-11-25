"""Revisa un lote etiquetado (export COCO de Label Studio) contra la planilla `planilla_fotos` antes de entrenar.

Encuentra los errores que más dañan el entrenamiento cuando todavía hay tiempo de corregirlos:

- individuos sin etiquetar (o de más): compara los recuadros de cada foto con `choritos_a_mano` y `cholgas_a_mano`;
- fotos sin la `referencia` (sin ella no hay calibre) o con más de una;
- recuadros duplicados (doble clic), diminutos o con una etiqueta que no existe;
- demasiadas `duda` en una foto;
- nombres de archivo fuera del formato `M<bandeja>_<iniciales>_<n>.jpg` o que no están en la planilla;
- fotos `_cal` que no traen exactamente 5 conchas + la referencia, o cuya bandeja no tiene los 5 largos.

Uso:
    python src/revisar_etiquetas.py --coco result.json --planilla planilla_fotos.csv
    python src/revisar_etiquetas.py --coco lote1.json lote2.json --planilla planilla_fotos.csv --autor JC
    python src/revisar_etiquetas.py --prueba

La planilla se descarga de Drive como CSV (Archivo → Descargar → CSV). Termina con código 1 si hay ERRORES
(lo que hay que corregir sí o sí); los AVISOS conviene mirarlos pero no bloquean.
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile

ETIQUETAS = {'chorito', 'cholga', 'referencia', 'duda'}
CONCHAS = ('chorito', 'cholga')
N_CAL = 5
DUDA_MAXIMA = 0.15        # sobre esta fracción de dudas en una foto, avisa
IOU_DUPLICADO = 0.85
LADO_MINIMO_PX = 8        # recuadros más chicos suelen ser clics por error
META_POR_ESPECIE = 500    # meta del equipo: 500 choritos y 500 cholgas
PATRON = re.compile(r'^M(\d{3})_([A-Z]{2,3})_(\d{1,2}|cal)\.(jpe?g|png)$', re.IGNORECASE)


def nombre_real(nombre):
    """Label Studio antepone un código al nombre del archivo ("1a2b3c4d-M002_JC_cal.jpg")."""
    base = os.path.basename(nombre)
    m = re.search(r'(M\d{3}_.*)$', base)
    return m.group(1) if m else base


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def numero(valor):
    valor = (valor or '').strip().replace(',', '.')
    try:
        return float(valor) if valor else None
    except ValueError:
        return 'malo'


def leer_planilla(ruta):
    bandejas = {}
    with open(ruta, encoding='utf-8-sig', newline='') as f:
        for i, fila in enumerate(csv.DictReader(f), start=2):
            bandeja = (fila.get('bandeja') or '').strip().upper()
            if not bandeja or 'borrar' in (fila.get('notas') or '').lower():
                continue   # filas vacías o de ejemplo
            fila['_linea'] = i
            bandejas[bandeja] = fila
    return bandejas


def leer_coco(rutas):
    """Une varios exports; devuelve {nombre_real: {'ancho', 'alto', 'cajas': [(etiqueta, bbox)]}}."""
    fotos = {}
    for ruta in rutas:
        coco = json.load(open(ruta, encoding='utf-8'))
        cats = {c['id']: c['name'].strip().lower() for c in coco['categories']}
        ids = {}
        for im in coco['images']:
            nombre = nombre_real(im['file_name'])
            ids[im['id']] = nombre
            fotos.setdefault(nombre, {'ancho': im.get('width'), 'alto': im.get('height'), 'cajas': [],
                                      'origen': os.path.basename(ruta)})
        for a in coco['annotations']:
            if a['image_id'] in ids:
                fotos[ids[a['image_id']]]['cajas'].append((cats.get(a['category_id'], '?'), a['bbox']))
    return fotos


def revisar(fotos, planilla, autor=None):
    errores, avisos = [], []
    totales = {e: 0 for e in ETIQUETAS}
    vistas_por_bandeja = {}

    for nombre in sorted(fotos):
        foto = fotos[nombre]
        m = PATRON.match(nombre)
        if not m:
            errores.append(f'{nombre}: el nombre no sigue el formato M<bandeja>_<iniciales>_<n>.jpg')
            continue
        bandeja, iniciales, n = f'M{m.group(1)}', m.group(2).upper(), m.group(3).lower()
        if autor and iniciales != autor.upper():
            continue
        vistas_por_bandeja.setdefault(bandeja, []).append(nombre)
        cajas = foto['cajas']
        cuenta = {e: 0 for e in ETIQUETAS}
        for etiqueta, bbox in cajas:
            if etiqueta not in ETIQUETAS:
                errores.append(f'{nombre}: etiqueta desconocida "{etiqueta}" (solo chorito, cholga, referencia, duda)')
                continue
            cuenta[etiqueta] += 1
            if min(bbox[2], bbox[3]) < LADO_MINIMO_PX:
                avisos.append(f'{nombre}: recuadro de {etiqueta} de {bbox[2]:.0f}×{bbox[3]:.0f} px (¿clic por error?)')
        for e in ETIQUETAS:
            totales[e] += cuenta[e]
        for i in range(len(cajas)):
            for j in range(i + 1, len(cajas)):
                if iou(cajas[i][1], cajas[j][1]) > IOU_DUPLICADO:
                    avisos.append(f'{nombre}: dos recuadros casi iguales ({cajas[i][0]} y {cajas[j][0]}): ¿duplicado?')

        if cuenta['referencia'] == 0:
            errores.append(f'{nombre}: sin recuadro de referencia (sin ella no hay calibre)')
        elif cuenta['referencia'] > 1:
            avisos.append(f'{nombre}: {cuenta["referencia"]} referencias; debería haber una')
        individuos = cuenta['chorito'] + cuenta['cholga'] + cuenta['duda']
        if individuos and cuenta['duda'] / individuos > DUDA_MAXIMA:
            avisos.append(f'{nombre}: {cuenta["duda"]} de {individuos} son duda ({cuenta["duda"] / individuos:.0%}); '
                          f'revisa la luz (de costado) o si se pueden decidir')

        fila = planilla.get(bandeja)
        if fila is None:
            errores.append(f'{nombre}: la bandeja {bandeja} no está en la planilla')
            continue
        if n == 'cal':
            revisar_cal(nombre, cajas, cuenta, fila, errores, avisos)
            continue
        ch, ko = numero(fila.get('choritos_a_mano')), numero(fila.get('cholgas_a_mano'))
        if ch in (None, 'malo') or ko in (None, 'malo'):
            errores.append(f'{nombre}: la planilla (línea {fila["_linea"]}) no tiene choritos_a_mano / cholgas_a_mano')
            continue
        esperado = int(ch) + int(ko)
        if individuos != esperado:
            faltan = esperado - individuos
            texto = f'faltan {faltan}' if faltan > 0 else f'sobran {-faltan}'
            errores.append(f'{nombre}: {individuos} individuos etiquetados y la planilla dice {esperado} ({texto})')
        elif cuenta['duda'] == 0 and (cuenta['chorito'] != int(ch) or cuenta['cholga'] != int(ko)):
            avisos.append(f'{nombre}: total correcto, pero {cuenta["chorito"]} choritos / {cuenta["cholga"]} cholgas '
                          f'y la planilla dice {int(ch)} / {int(ko)}: revisa la especie de algunos')

    # Fotos que la planilla nombra pero que no vienen en el export
    for bandeja, fila in sorted(planilla.items()):
        if autor and (fila.get('autor') or '').strip().upper() != autor.upper():
            continue
        if bandeja not in vistas_por_bandeja:
            continue   # bandeja de otro lote: no se exige
        for archivo in (fila.get('archivos') or '').split():
            if archivo not in fotos:
                avisos.append(f'{archivo}: está en la planilla (bandeja {bandeja}) pero no en el export')
        if (fila.get('set_calibre') or '').strip().lower() in ('si', 'sí'):
            if not any(PATRON.match(f) and f.lower().startswith(bandeja.lower() + '_') and '_cal.' in f.lower()
                       for f in fotos):
                avisos.append(f'{bandeja}: marcada como set de calibre, pero no hay foto _cal en el export')
    return errores, avisos, totales


def revisar_cal(nombre, cajas, cuenta, fila, errores, avisos):
    if (fila.get('set_calibre') or '').strip().lower() not in ('si', 'sí'):
        errores.append(f'{nombre}: es foto _cal, pero la bandeja no tiene set_calibre = si en la planilla')
    conchas = cuenta['chorito'] + cuenta['cholga'] + cuenta['duda']
    if conchas != N_CAL:
        errores.append(f'{nombre}: la foto _cal debe tener exactamente {N_CAL} conchas y tiene {conchas}')
    largos = [numero(fila.get(f'largo_{i}_mm')) for i in range(1, N_CAL + 1)]
    if any(v in (None, 'malo') for v in largos):
        errores.append(f'{nombre}: faltan largos en la planilla (largo_1_mm … largo_{N_CAL}_mm)')
    elif any(not 10 <= v <= 200 for v in largos):
        avisos.append(f'{nombre}: algún largo de la planilla está fuera de 10–200 mm (¿cm en vez de mm?)')
    # Especie en el orden de la fila (izquierda → derecha), como lo usa validar_calibre.py
    en_fila = sorted([(b[0] + b[2] / 2, e) for e, b in cajas if e in CONCHAS + ('duda',)])
    if len(en_fila) == N_CAL:
        for i, (_, e) in enumerate(en_fila, start=1):
            esperada = (fila.get(f'especie_{i}') or '').strip().lower()
            if esperada and e != 'duda' and e != esperada:
                avisos.append(f'{nombre}: el individuo {i} (de izquierda a derecha) está etiquetado {e} '
                              f'y la planilla dice {esperada}: ¿la fila quedó en otro orden?')


def informe(errores, avisos, totales, fotos):
    for e in errores:
        print(f'ERROR  {e}')
    for a in avisos:
        print(f'AVISO  {a}')
    print(f'\n{len(fotos)} fotos · {totales["chorito"]} choritos · {totales["cholga"]} cholgas · '
          f'{totales["duda"]} duda · {totales["referencia"]} referencias')
    print(f'Meta del equipo: {META_POR_ESPECIE} por especie (este lote aporta '
          f'{totales["chorito"] / META_POR_ESPECIE:.0%} de choritos y {totales["cholga"] / META_POR_ESPECIE:.0%} de cholgas).')
    print('Resultado: ' + ('LISTO para entregar' if not errores else f'{len(errores)} errores por corregir'))


# ── Autoprueba ──

def autoprueba():
    def caja(x, e):
        return (e, [x, 100, 60, 30])
    fotos = {
        'M001_JC_01.jpg': {'cajas': [caja(10 + 70 * i, 'chorito') for i in range(5)] + [caja(400, 'cholga'), caja(800, 'referencia')]},
        'M001_JC_02.jpg': {'cajas': [caja(10 + 70 * i, 'chorito') for i in range(4)] + [caja(400, 'cholga')]},  # falta 1 y la ref
        'M002_JC_cal.jpg': {'cajas': [caja(10, 'chorito'), caja(100, 'cholga'), caja(200, 'chorito'), caja(300, 'chorito'),
                                      caja(400, 'cholga'), caja(800, 'referencia')]},
        'M003_JC_01.jpg': {'cajas': [caja(10, 'chorito'), ('chorito', [11, 100, 60, 30]), caja(800, 'referencia')]},  # duplicado
        'foto_rara.jpg': {'cajas': []},
    }
    for f in fotos.values():
        f.update(ancho=1000, alto=800, origen='prueba')
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, 'planilla.csv')
        campos = ['bandeja', 'autor', 'archivos', 'choritos_a_mano', 'cholgas_a_mano', 'set_calibre'] + \
            [f'{c}_{i}{s}' for i in range(1, 6) for c, s in (('especie', ''), ('largo', '_mm'))] + ['notas']
        with open(ruta, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerow({'bandeja': 'M000', 'autor': 'JC', 'notas': 'ejemplo: borrar esta fila'})
            w.writerow({'bandeja': 'M001', 'autor': 'JC', 'archivos': 'M001_JC_01.jpg M001_JC_02.jpg M001_JC_03.jpg',
                        'choritos_a_mano': '5', 'cholgas_a_mano': '1', 'set_calibre': 'no'})
            fila2 = {'bandeja': 'M002', 'autor': 'JC', 'choritos_a_mano': '30', 'cholgas_a_mano': '4', 'set_calibre': 'si'}
            for i, (e, l) in enumerate([('chorito', 55), ('chorito', 61), ('chorito', 48), ('chorito', 52), ('cholga', 69)], 1):
                fila2[f'especie_{i}'], fila2[f'largo_{i}_mm'] = e, str(l)
            w.writerow(fila2)
            w.writerow({'bandeja': 'M003', 'autor': 'JC', 'choritos_a_mano': '2', 'cholgas_a_mano': '0', 'set_calibre': 'no'})
        errores, avisos, totales = revisar(fotos, leer_planilla(ruta))
    texto = '\n'.join(errores + avisos)
    casos = [
        ('detecta el individuo que falta', 'M001_JC_02.jpg: 5 individuos etiquetados y la planilla dice 6 (faltan 1)'),
        ('detecta la foto sin referencia', 'M001_JC_02.jpg: sin recuadro de referencia'),
        ('detecta la foto de la planilla que no vino', 'M001_JC_03.jpg: está en la planilla'),
        ('detecta el orden de la fila _cal', 'M002_JC_cal.jpg: el individuo 2'),
        ('detecta el duplicado', 'M003_JC_01.jpg: dos recuadros casi iguales'),
        ('detecta el nombre fuera de formato', 'foto_rara.jpg: el nombre no sigue el formato'),
        ('ignora la fila de ejemplo', None),
    ]
    fallas = 0
    for texto_caso, buscado in casos:
        ok = (buscado in texto) if buscado else ('M000' not in texto)
        fallas += not ok
        print(('  ok    ' if ok else '  FALLA ') + texto_caso)
    ok = 'M001_JC_01.jpg' not in texto
    fallas += not ok
    print(('  ok    ' if ok else '  FALLA ') + 'no se queja de la foto correcta')
    print('\nAutoprueba ' + ('OK' if not fallas else f'con {fallas} fallas'))
    return 1 if fallas else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coco', nargs='+', help='uno o más exports COCO de Label Studio (result.json)')
    ap.add_argument('--planilla', help='planilla_fotos descargada como CSV')
    ap.add_argument('--autor', help='revisar solo las fotos de estas iniciales (por ejemplo JC)')
    ap.add_argument('--prueba', action='store_true', help='autoprueba sin datos reales')
    a = ap.parse_args()
    if a.prueba:
        return autoprueba()
    if not a.coco or not a.planilla:
        ap.error('faltan --coco y --planilla (o usa --prueba)')
    fotos = leer_coco(a.coco)
    errores, avisos, totales = revisar(fotos, leer_planilla(a.planilla), a.autor)
    informe(errores, avisos, totales, fotos)
    return 1 if errores else 0


if __name__ == '__main__':
    sys.exit(main())

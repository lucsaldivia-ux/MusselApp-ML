"""Prepara un dataset público (export COCO de Roboflow) para el pre-entrenamiento: deja una sola clase, "mejillon".

El dataset "mussel" de Stephen (DATOS_EXTERNOS.md) trae mejillones, percebes y ostras. Para pre-entrenar solo
interesa que el modelo aprenda a encontrar "un mejillón", así que:

- se quedan solo las anotaciones de la clase de mejillón (nombre que contenga "mussel" o "mejill");
- percebes y ostras se quitan (las fotos se conservan: enseñan qué NO es un mejillón);
- se descarta la categoría "padre" que agrega Roboflow (supercategory "none");
- si el export trae solo `train/` (así viene el de Stephen), se separa en 80 % / 10 % / 10 % con semilla fija;
  si trae `train/` y `valid/` pero no `test/`, se copia `valid/` (RF-DETR lo usa para la evaluación final).

Después, el entrenamiento con nuestras fotos parte de estos pesos con 3 clases (chorito, cholga, referencia): RF-DETR
reemplaza la capa de clases y conserva lo aprendido para encontrar individuos.

Uso:
    python src/preparar_externo.py --entrada mussel.v1.coco.zip --salida /content/externo
    python src/preparar_externo.py --entrada carpeta_descomprimida/ --salida externo/
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import unicodedata
import zipfile

CLAVES = ('mussel', 'mejill')
NOMBRE = 'mejillon'


def normal(texto):
    return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode().lower()


def es_mejillon(cat):
    return any(k in normal(cat['name']) for k in CLAVES) and (cat.get('supercategory') or '').lower() != 'none'


def filtrar(coco):
    ids = {c['id'] for c in coco['categories'] if es_mejillon(c)}
    if not ids:
        nombres = [c['name'] for c in coco['categories']]
        raise SystemExit(f'No encuentro la clase de mejillón entre {nombres}; revisa CLAVES en este script')
    anots = []
    for a in coco['annotations']:
        if a['category_id'] in ids:
            anots.append(dict(a, id=len(anots) + 1, category_id=1))
    return {'images': coco['images'], 'annotations': anots,
            'categories': [{'id': 1, 'name': NOMBRE, 'supercategory': 'mitilido'}]}, ids


def buscar_raiz(carpeta):
    """Roboflow a veces mete todo dentro de una subcarpeta: busca la que tiene train/_annotations.coco.json."""
    for raiz, _, archivos in os.walk(carpeta):
        if os.path.basename(raiz) == 'train' and '_annotations.coco.json' in archivos:
            return os.path.dirname(raiz)
    raise SystemExit(f'No encuentro train/_annotations.coco.json dentro de {carpeta} (¿se exportó en formato COCO?)')


def separar_train(salida, semilla=42, proporciones=(0.8, 0.1)):
    """Reparte train/ en train, valid y test cuando el export no trae validación."""
    carpeta = os.path.join(salida, 'train')
    coco = json.load(open(os.path.join(carpeta, '_annotations.coco.json'), encoding='utf-8'))
    imagenes = sorted(coco['images'], key=lambda im: im['id'])
    random.Random(semilla).shuffle(imagenes)
    n = len(imagenes)
    corte1, corte2 = int(n * proporciones[0]), int(n * (proporciones[0] + proporciones[1]))
    grupos = {'train': imagenes[:corte1], 'valid': imagenes[corte1:corte2], 'test': imagenes[corte2:]}
    os.rename(carpeta, carpeta + '_todo')
    origen = carpeta + '_todo'
    resumen = {}
    for split, ims in grupos.items():
        destino = os.path.join(salida, split)
        os.makedirs(destino)
        ids = {im['id'] for im in ims}
        anots = [a for a in coco['annotations'] if a['image_id'] in ids]
        for im in ims:
            shutil.move(os.path.join(origen, im['file_name']), os.path.join(destino, im['file_name']))
        json.dump({'images': ims, 'annotations': anots, 'categories': coco['categories']},
                  open(os.path.join(destino, '_annotations.coco.json'), 'w', encoding='utf-8'))
        resumen[split] = (len(ims), len(anots), 'separado de train')
    shutil.rmtree(origen)
    return resumen


def preparar(entrada, salida):
    temporal = None
    if zipfile.is_zipfile(entrada):
        temporal = tempfile.mkdtemp()
        zipfile.ZipFile(entrada).extractall(temporal)
        entrada = temporal
    raiz = buscar_raiz(entrada)
    shutil.rmtree(salida, ignore_errors=True)
    resumen = {}
    for split in ('train', 'valid', 'test'):
        origen = os.path.join(raiz, split)
        if not os.path.isfile(os.path.join(origen, '_annotations.coco.json')):
            continue
        coco = json.load(open(os.path.join(origen, '_annotations.coco.json'), encoding='utf-8'))
        nuevo, ids = filtrar(coco)
        destino = os.path.join(salida, split)
        os.makedirs(destino)
        for im in coco['images']:
            archivo = os.path.join(origen, im['file_name'])
            if os.path.isfile(archivo):
                shutil.copy(archivo, os.path.join(destino, os.path.basename(im['file_name'])))
            im['file_name'] = os.path.basename(im['file_name'])
        json.dump(nuevo, open(os.path.join(destino, '_annotations.coco.json'), 'w', encoding='utf-8'))
        quitadas = len(coco['annotations']) - len(nuevo['annotations'])
        resumen[split] = (len(coco['images']), len(nuevo['annotations']), quitadas)
    if 'train' not in resumen:
        raise SystemExit('El export debe traer al menos train/')
    if 'valid' not in resumen:
        resumen = separar_train(salida)
    elif 'test' not in resumen:
        shutil.copytree(os.path.join(salida, 'valid'), os.path.join(salida, 'test'))
        resumen['test'] = resumen['valid'][:2] + ('copia de valid',)
    if temporal:
        shutil.rmtree(temporal, ignore_errors=True)
    for split, datos in resumen.items():
        print(f'{split}: {datos[0]} fotos, {datos[1]} mejillones' + (f', {datos[2]} anotaciones de otras clases quitadas'
              if isinstance(datos[2], int) else f' ({datos[2]})'))
    return resumen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--entrada', required=True, help='zip o carpeta del export COCO de Roboflow')
    ap.add_argument('--salida', required=True, help='carpeta nueva con train/, valid/ y test/ de una sola clase')
    a = ap.parse_args()
    preparar(a.entrada, a.salida)
    return 0


if __name__ == '__main__':
    sys.exit(main())

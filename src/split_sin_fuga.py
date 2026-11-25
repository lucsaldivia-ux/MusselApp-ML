"""Separa un dataset COCO en entrenamiento / validación / prueba SIN fuga de datos.

Las fotos de una misma bandeja (prefijo del nombre, por ejemplo M012 en M012_JC_02.jpg) quedan
siempre en el mismo conjunto. Si quedaran repartidas, el modelo "vería" en el entrenamiento casi la
misma foto que luego se usa para medirlo, y las métricas saldrían infladas.

Uso:
    python src/split_sin_fuga.py etiquetas/*.json --salida dataset_v1/split --semilla 42
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

CATEGORIAS_VALIDAS = ("chorito", "cholga", "referencia")  # "duda" se excluye


def bandeja_de(nombre_archivo: str) -> str:
    """M012_JC_02.jpg -> M012. Label Studio a veces antepone un prefijo con guion: se quita."""
    base = Path(nombre_archivo).name
    base = base.split("-", 1)[1] if "-" in base and not base.startswith("M") else base
    return base.split("_", 1)[0]


def unir(archivos):
    """Une varios exports COCO en uno, renumerando ids y dejando solo las categorías válidas."""
    categorias = [{"id": i + 1, "name": n} for i, n in enumerate(CATEGORIAS_VALIDAS)]
    id_por_nombre = {c["name"]: c["id"] for c in categorias}
    imagenes, anotaciones, vistos = [], [], {}
    for ruta in archivos:
        coco = json.loads(Path(ruta).read_text(encoding="utf-8"))
        nombre_cat = {c["id"]: c["name"] for c in coco["categories"]}
        nuevo_id = {}
        for img in coco["images"]:
            clave = Path(img["file_name"]).name
            if clave in vistos:  # la misma foto exportada dos veces: se usa la primera
                nuevo_id[img["id"]] = None
                continue
            nid = len(imagenes) + 1
            vistos[clave] = nid
            nuevo_id[img["id"]] = nid
            imagenes.append({**img, "id": nid, "file_name": clave})
        for a in coco["annotations"]:
            nombre = nombre_cat.get(a["category_id"])
            iid = nuevo_id.get(a["image_id"])
            if iid is None or nombre not in id_por_nombre:
                continue
            anotaciones.append({**a, "id": len(anotaciones) + 1, "image_id": iid,
                                "category_id": id_por_nombre[nombre]})
    return {"images": imagenes, "annotations": anotaciones, "categories": categorias}


def separar(coco, proporciones=(0.70, 0.15, 0.15), semilla=42):
    """Reparte bandejas completas entre los tres conjuntos."""
    por_bandeja = defaultdict(list)
    for img in coco["images"]:
        por_bandeja[bandeja_de(img["file_name"])].append(img["id"])
    bandejas = sorted(por_bandeja)
    random.Random(semilla).shuffle(bandejas)
    n = len(bandejas)
    n_val = max(1, round(n * proporciones[1])) if n >= 3 else 0
    n_test = max(1, round(n * proporciones[2])) if n >= 3 else 0
    grupos = {
        "test": bandejas[:n_test],
        "val": bandejas[n_test:n_test + n_val],
        "train": bandejas[n_test + n_val:],
    }
    conjuntos = {}
    for nombre, lista in grupos.items():
        ids = {i for b in lista for i in por_bandeja[b]}
        conjuntos[nombre] = {
            "images": [im for im in coco["images"] if im["id"] in ids],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
            "categories": coco["categories"],
        }
    return conjuntos, grupos


def resumen(conjuntos):
    filas = []
    for nombre, c in conjuntos.items():
        cuenta = defaultdict(int)
        for a in c["annotations"]:
            cuenta[a["category_id"]] += 1
        nombres = {cat["id"]: cat["name"] for cat in c["categories"]}
        detalle = ", ".join(f"{nombres[k]}={v}" for k, v in sorted(cuenta.items()))
        filas.append(f"{nombre:5s}: {len(c['images'])} fotos · {detalle or 'sin etiquetas'}")
    return "\n".join(filas)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("archivos", nargs="+", help="exports COCO (.json) de Label Studio")
    p.add_argument("--salida", required=True)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()

    coco = unir(args.archivos)
    conjuntos, grupos = separar(coco, semilla=args.semilla)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, c in conjuntos.items():
        (salida / f"{nombre}.json").write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    (salida / "bandejas.json").write_text(json.dumps(grupos, indent=2), encoding="utf-8")
    print(resumen(conjuntos))


if __name__ == "__main__":
    main()

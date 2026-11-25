"""Genera un dataset SINTÉTICO (dibujado) para probar el pipeline sin fotos reales.

No sirve para entrenar el modelo final: solo para comprobar que el notebook, el entrenamiento, la
evaluación y la exportación funcionan de punta a punta antes de que lleguen las fotos del equipo.

Dibuja sobre fondo claro: "choritos" (elipses oscuras lisas), "cholgas" (elipses más grandes con estrías
radiales) y el cuadrado negro de referencia. Deja la carpeta en el formato que espera RF-DETR:
train/, valid/ y test/, cada una con sus imágenes y _annotations.coco.json.

Uso:
    python src/dataset_sintetico.py --salida /content/sintetico --fotos 60
"""
import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

CLASES = ["chorito", "cholga", "referencia"]


def concha(draw, cx, cy, largo, ancho, angulo, estriada, rng):
    """Dibuja una valva girada y devuelve su caja [x, y, w, h]."""
    def girar(x, y):
        return (cx + x * math.cos(angulo) - y * math.sin(angulo), cy + x * math.sin(angulo) + y * math.cos(angulo))

    puntos, angulos = [], []
    for i in range(48):
        t = 2 * math.pi * i / 48
        # Forma de gota: más angosta en la punta (umbo, en t = 0) que en el borde opuesto
        r = 1 - 0.35 * max(0.0, math.cos(t))
        puntos.append(girar(largo / 2 * math.cos(t), ancho / 2 * math.sin(t) * r))
        angulos.append(t)
    tono = rng.randint(15, 55)
    draw.polygon(puntos, fill=(tono, tono + rng.randint(0, 10), tono + rng.randint(5, 25)))
    if estriada:  # costillas radiales desde la punta hasta el borde: el rasgo que distingue a la cholga
        umbo = puntos[0]
        for p, t in zip(puntos, angulos):
            if math.pi * 0.55 <= t <= math.pi * 1.45:
                fin = (umbo[0] + (p[0] - umbo[0]) * 0.97, umbo[1] + (p[1] - umbo[1]) * 0.97)
                draw.line([umbo, fin], fill=(tono + 45, tono + 45, tono + 50), width=max(1, round(largo / 120)))
    xs, ys = [p[0] for p in puntos], [p[1] for p in puntos]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def foto(indice, ancho, alto, rng):
    img = Image.new("RGB", (ancho, alto), tuple(rng.randint(215, 245) for _ in range(3)))
    draw = ImageDraw.Draw(img)
    anotaciones = []
    escala = ancho / 2000  # tamaños pensados para fotos de 2000 px de ancho
    # Referencia de 40 mm en una esquina (a ~5 px por mm en 2000 px)
    lado = round(rng.randint(150, 210) * escala)
    margen = round(30 * escala)
    rx, ry = rng.choice([(margen, margen), (ancho - lado - margen, alto - lado - margen)])
    draw.rectangle([rx, ry, rx + lado, ry + lado], fill=(10, 10, 10))
    anotaciones.append((2, [rx, ry, lado, lado]))
    ocupadas = [(rx, ry, rx + lado, ry + lado)]
    for _ in range(rng.randint(15, 45)):
        cholga = rng.random() < 0.25
        largo = (rng.uniform(200, 320) if cholga else rng.uniform(150, 260)) * escala
        for _intento in range(30):
            cx, cy = rng.uniform(largo / 2, ancho - largo / 2), rng.uniform(largo / 2, alto - largo / 2)
            caja = (cx - largo / 2, cy - largo / 2, cx + largo / 2, cy + largo / 2)
            if all(caja[2] < o[0] or caja[0] > o[2] or caja[3] < o[1] or caja[1] > o[3] for o in ocupadas):
                ocupadas.append(caja)
                bbox = concha(draw, cx, cy, largo, largo * rng.uniform(0.42, 0.55), rng.uniform(0, math.pi), cholga, rng)
                anotaciones.append((1 if cholga else 0, bbox))
                break
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.2)))
    return img, anotaciones


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--salida", required=True)
    p.add_argument("--fotos", type=int, default=60)
    p.add_argument("--ancho", type=int, default=2000)
    p.add_argument("--alto", type=int, default=1500)
    p.add_argument("--semilla", type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.semilla)
    salida = Path(args.salida)
    reparto = {"train": int(args.fotos * 0.7), "valid": int(args.fotos * 0.15)}
    reparto["test"] = args.fotos - reparto["train"] - reparto["valid"]
    n = 0
    for conjunto, cantidad in reparto.items():
        carpeta = salida / conjunto
        carpeta.mkdir(parents=True, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASES)]}
        for _ in range(cantidad):
            n += 1
            nombre = f"M9{n:02d}_SI_01.jpg"
            img, anots = foto(n, args.ancho, args.alto, rng)
            img.save(carpeta / nombre, quality=90)
            iid = len(coco["images"]) + 1
            coco["images"].append({"id": iid, "file_name": nombre, "width": args.ancho, "height": args.alto})
            for clase, bbox in anots:
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": iid, "category_id": clase + 1,
                                            "bbox": [round(v, 1) for v in bbox], "area": round(bbox[2] * bbox[3], 1), "iscrowd": 0})
        (carpeta / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
        print(f"{conjunto}: {cantidad} fotos, {len(coco['annotations'])} etiquetas")


if __name__ == "__main__":
    main()

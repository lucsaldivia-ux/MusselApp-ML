# Datos externos usados

Solo datasets con licencia **CC BY 4.0** o **CC0**. Se usan para **pre-entrenar** la detección de "individuo"
(contar), no para distinguir chorito de cholga. Descargar desde Roboflow Universe requiere una cuenta gratuita.
Al usar un dataset CC BY 4.0 hay que **nombrar al autor y el enlace** (en este archivo y en la presentación).

## Elegidos

| Dataset | Enlace | Licencia | Autor | Imágenes | Clases | Para qué |
|---|---|---|---|---|---|---|
| mussel (Stephen) | https://universe.roboflow.com/stephen-7b2qu/mussel-l4ogs | CC BY 4.0 | Stephen (Roboflow Universe) | 501 | percebe, mejillón, ostra | Pre-entrenar "individuo": fotos de terreno, varias con muchos mejillones juntos. Usar solo la clase mejillón |

## Revisados y descartados

| Dataset | Enlace | Licencia | Motivo |
|---|---|---|---|
| mussel detection (jaredgreeneperkins) | https://universe.roboflow.com/jaredgreeneperkins-gmail-com/mussel-detection | CC BY 4.0 | Tiene 1 sola imagen |
| mussel detection (MFGC) | https://universe.roboflow.com/mfgc/mussel-detection-8g2q9 | CC BY 4.0 | 1.500 imágenes, pero un solo mejillón por foto, de cerca: no enseña a contar. Opcional como respaldo |

Revisado el 2026-11-25; licencia de "mussel" (Stephen) confirmada otra vez en su página: CC BY 4.0. Volver a revisarla antes de descargar.

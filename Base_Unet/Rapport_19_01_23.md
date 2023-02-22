# Rapport TER Sujet 113
## 19/01/2023

Synstème :<br>
CPU : AMD® Ryzen 9 3900x 12-core processor × 24<br>
GPU : NVIDIA Corporation GP102 [GeForce GTX 1080 Ti]<br><br>
Une époque prend environ 3.5 minutes à calculer.<br>

## Ce qui a été fait
- Entrainement de 2 réseaux sur LW4 5000 et I3 5000
- Lecture des articles sur le UNet et la distillation de réseaux
- Calcul des scores IoU

----

### <ins>Indice IoU (Jaccard)
$$
J(A,B) = \frac{|A \cap B|}{|A\cup B|}
$$
Soit la division du nombre de pixels partagés dans les deux images par le nombre de pixels au total.

| Données      | IoU réseau entraîné sur LW4 | IoU réseau entraîné sur I3 |
| ----------- | ----------- | ------- |
| LW4      | 0.141      | 0.141 |
| I3   | 0.141        | 0.132 |

Il y a peut-être une erreur dans mon code au vu des résultats.

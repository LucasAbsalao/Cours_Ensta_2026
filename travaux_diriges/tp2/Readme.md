# TD n° 2 - 27 Janvier 2026

##  1. Parallélisation ensemble de Mandelbrot

L'ensensemble de Mandebrot est un ensemble fractal inventé par Benoit Mandelbrot permettant d'étudier la convergence ou la rapidité de divergence dans le plan complexe de la suite récursive suivante :

$$
\left\{
\begin{array}{l}
    c\,\,\textrm{valeurs\,\,complexe\,\,donnée}\\
    z_{0} = 0 \\
    z_{n+1} = z_{n}^{2} + c
\end{array}}
\right.
$$
dépendant du paramètre $c$.

Il est facile de montrer que si il existe un $N$ tel que $\mid z_{N} \mid > 2$, alors la suite $z_{n}$ diverge. Cette propriété est très utile pour arrêter le calcul de la suite puisqu'on aura détecter que la suite a divergé. La rapidité de divergence est le plus petit $N$ trouvé pour la suite tel que $\mid z_{N} \mid > 2$.

On fixe un nombre d'itérations maximal $N_{\textrm{max}}$. Si jusqu'à cette itération, aucune valeur de $z_{N}$ ne dépasse en module 2, on considère que la suite converge.

L'ensemble de Mandelbrot sur le plan complexe est l'ensemble des valeurs de $c$ pour lesquels la suite converge.

Pour l'affichage de cette suite, on calcule une image de $W\times H$ pixels telle qu'à chaque pixel $(p_{i},p_{j})$, de l'espace image, on associe une valeur complexe  $c = x_{min} + p_{i}.\frac{x_{\textrm{max}}-x_{\textrm{min}}}{W} + i.\left(y_{\textrm{min}} + p_{j}.\frac{y_{\textrm{max}}-y_{\textrm{min}}}{H}\right)$. Pour chacune des valeurs $c$ associées à chaque pixel, on teste si la suite converge ou diverge.

- Si la suite converge, on affiche le pixel correspondant en noir
- Si la suite diverge, on affiche le pixel avec une couleur correspondant à la rapidité de divergence.

1. À partir du code séquentiel `mandelbrot.py`, faire une partition équitable par bloc suivant les lignes de l'image pour distribuer le calcul sur `nbp` processus  puis rassembler l'image sur le processus zéro pour la sauvegarder. Calculer le temps d'exécution pour différents nombre de tâches et calculer le speedup. Comment interpréter les résultats obtenus ?
2. Réfléchissez à une meilleur répartition statique des lignes au vu de l'ensemble obtenu sur notre exemple et mettez la en œuvre. Calculer le temps d'exécution pour différents nombre de tâches et calculer le speedup et comparez avec l'ancienne répartition. Quel problème pourrait se poser avec une telle stratégie ?
3. Mettre en œuvre une stratégie maître-esclave pour distribuer les différentes lignes de l'image à calculer. Calculer le speedup avec cette approche et comparez  avec les solutions différentes. Qu'en concluez-vous ?

## 2. Produit matrice-vecteur

On considère le produit d'une matrice carrée $A$ de dimension $N$ par un vecteur $u$ de même dimension dans $\mathbb{R}$. La matrice est constituée des cœfficients définis par $A_{ij} = (i+j) \mod N  + 1$. 

Par soucis de simplification, on supposera $N$ divisible par le nombre de tâches `nbp` exécutées.

### a - Produit parallèle matrice-vecteur par colonne

Afin de paralléliser le produit matrice–vecteur, on décide dans un premier temps de partitionner la matrice par un découpage par bloc de colonnes. Chaque tâche contiendra $N_{\textrm{loc}}$ colonnes de la matrice. 

- Calculer en fonction du nombre de tâches la valeur de Nloc
- Paralléliser le code séquentiel `matvec.py` en veillant à ce que chaque tâche n’assemble que la partie de la matrice utile à sa somme partielle du produit matrice-vecteur. On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.
- Calculer le speed-up obtenu avec une telle approche

### b - Produit parallèle matrice-vecteur par ligne

Afin de paralléliser le produit matrice–vecteur, on décide dans un deuxième temps de partitionner la matrice par un découpage par bloc de lignes. Chaque tâche contiendra $N_{\textrm{loc}}$ lignes de la matrice.

- Calculer en fonction du nombre de tâches la valeur de Nloc
- paralléliser le code séquentiel `matvec.py` en veillant à ce que chaque tâche n’assemble que la partie de la matrice utile à son produit matrice-vecteur partiel. On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.
- Calculer le speed-up obtenu avec une telle approche

## 3. Entraînement pour l'examen écrit

Alice a parallélisé en partie un code sur machine à mémoire distribuée. Pour un jeu de données spécifiques, elle remarque que la partie qu’elle exécute en parallèle représente en temps de traitement 90% du temps d’exécution du programme en séquentiel.

En utilisant la loi d’Amdhal, pouvez-vous prédire l’accélération maximale que pourra obtenir Alice avec son code (en considérant n ≫ 1) ?

À votre avis, pour ce jeu de donné spécifique, quel nombre de nœuds de calcul semble-t-il raisonnable de prendre pour ne pas trop gaspiller de ressources CPU ?

En effectuant son cacul sur son calculateur, Alice s’aperçoit qu’elle obtient une accélération maximale de quatre en augmentant le nombre de nœuds de calcul pour son jeu spécifique de données.

En doublant la quantité de donnée à traiter, et en supposant la complexité de l’algorithme parallèle linéaire, quelle accélération maximale peut espérer Alice en utilisant la loi de Gustafson ?

# Réponses

## 1

### 1

En divisant l'image en 6 blocs horizontaux, nous avons observé ce résultat : les processus centraux (2 et 3) ont été les plus rapides, tandis que les processus périphériques (0, 1, 4, 5) ont pris plus de temps.

Ce phénomène s'explique par l'optimisation implémentée dans le code (le test de la cardioïde et du disque principal).

- Les processus centraux (2 et 3) traitent la majeure partie de l'intérieur de l'ensemble (la zone plus lumineuse). Grâce au test géométrique préalable, l'algorithme détecte immédiatement ces points et retourne le résultat sans entrer dans la boucle de calcul itérative. Le coût de calcul devient alors négligeable ($O(1)$).

- Les processus périphériques (0, 1, 4, 5) traitent les zones de frontière fractale. Pour ces points, le test géométrique échoue, forçant l'algorithme à exécuter la boucle de calcul complète pour déterminer la vitesse de divergence.

En conclusion, bien que l'optimisation accélère drastiquement le traitement du centre, elle aggrave le déséquilibre de charge (load imbalance) en rendant les tâches centrales triviales et plus vites par rapport aux tâches périphériques plus complexes.

**Speed-Up = 2.57**

### 2

Dans la nouvelle stratégie (Cyclique), chaque processus calcule une moyenne de lignes faciles et difficiles. Le temps de calcul sera presque identique pour tous les rangs. Par conséquent, le speed-up sera beaucoup plus élévé, car personne n'attend inutilement.

**Speed-Up = 3.27**

le principal problème est la **complexité de reconstruction**: Le processus maître (Rank 0) doit effectuer une étape de post-traitement pour remettre les lignes dans le bon ordre.

### 3

La stratégie Maître-Esclave offre la meilleure robustesse face à l'irrégularité de l'ensemble de Mandelbrot. Elle lisse parfaitement le déséquilibre de charge observé avec le partitionnement par blocs.

**speed up: 3.39**

Cependant, pour un faible nombre de processeurs, la perte d'un cœur qui sera dédié uniquement à la gestion (le Maître) pénalise le speed-up global. Dans ce cas, la stratégie Maître-Esclave a obtenu en speedup similaire au speedup de la stratégie cyclique.


## 2

Le **speed-up** est inférieur à 1 car l'**overhead** de communication lié à la parallélisation est supérieur 
au temps d'exécution séquentiel des opérations. Il est probable qu'avec une matrice de dimensions plus importantes, 
la parallélisation devienne plus efficace e t permette d'obtenir un speed-up supérieur à 1.

Le speed-up du produit matrice-vecteur par partitionnement horizontal (lignes) est généralement supérieur à celui du partitionnement vertical (colonnes). Cela s'explique par le fait que l'opération de **réduction** (Allreduce avec somme arithmétique) est plus coûteuse que l'opération de **collecte** (Gather). Dans cette dernière, chaque tâche contribue simplement à une partie du vecteur final, sans nécessiter de calculs supplémentaires lors de la communication.


## 3

### 1

1/f où f est la partie non parallélisable: 1/0.1 = 10.

### 2

Il semble raisonnable de choisir entre 9 et 10 nœuds de calcul.

Avec 9 nœuds, nous obtenons un speed-up de 5, ce qui représente la moitié de l'accélération maximale théorique ($S_{max}=10$). À ce stade, l'efficacité parallèle ($E = \frac{S(n)}{n}$) est d'environ 55%.

Choisir plus de nœuds ferait chuter l'efficacité en dessous du seuil critique de 50%, ce qui signifierait que plus de la moitié de la puissance de calcul serait gaspillée (rendement décroissant), sans gain significatif de performance.

### 3

La loi de Gustafson, prend en compte le fait qu'en augmentant la puissance de calcul, on tend généralement à augmenter la taille du problème traité (Weak Scaling).

- Temps de la partie séquentielle ($T_{seq}$) : Il représente 0,1 unité de temps (les 10% initiaux).

- Temps de la partie parallèle ($T_{par}$) : Puisque la complexité est linéaire, si on double les données, le travail double. $0,9 \times 2 =$ 1,8 unité de temps.

- Pour l'accélération maximale on considère $n \to \infty$.

$$S_{max\_nouveau} = \frac{\text{Travail Total Nouveau}}{\text{Partie Séquentielle}} = \frac{0,1 + 1,8}{0,1} = \frac{1,9}{0,1} = 19$$

En doublant la quantité de données, la part du code parallélisable devient prépondérante par rapport à la partie séquentielle fixe. Alice peut donc espérer une nouvelle accélération maximale théorique de 19.
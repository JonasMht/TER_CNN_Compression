# TER_Jonas_Mehtali

## Quelques précisions sur les données :

Lors de la réunion Jeudi, nous t'avons présenté les deux types de cellules avec lesquels nous travaillons.
Il y a 2 types de cellules, les cellules "I3" et les cellules "LW4".
Elles se trouvent dans le dossier "data" qui est disposition sur seafile : https://seafile.unistra.fr/d/50335450aeee434ca5d0/

Dans ce dossier "data", il y a quattre fichiers et deux dossiers.


  - Les quattre fichiers sont des stacks d'images associés à des annotation de mitochondries
    en format ".tiff", je te recommande d'installer le logiciel "Fiji" pour visualiser ces images,
    ce qui te donnera un premier apperçu des données.

  - Les deux dossiers contiennent les images pré-traitées en "patchs" de taille 256x256 pixels (réduction de taille
    nécessaire pour l'entrainement pour cause de mémoire), ainsi que les labels correspondants à ces images, qui sont les
    annotations de segmentation de mitochondries. Ces dossiers vont te permettre d'entrainer ton réseau soit
    sur les cellules "I3" soit sur les cellules "LW4".
    
## Code 

Le notebook "U-Net_training" va te permettre d'entraîner un U-Net (avec la librairie Pytorch).
L'entrainement est fonctionnelle, tu n'as qu'à le lancer pour te familiariser avec le code.
Petite précision : Je sauvegarde les poids de mon réseaux toutes les 3 épochs, en format ".pth" 
dans le dossier "network_weight" qui est créer lorsque l'entraînement est lancé. 

## À faire pour commencer

Dans un premier temps, familiarise toi avec les données, l'entraînement, Pytorch, essaye de faire tourner 
sur ton GPU etc etc...
Puis, une fois l'entrainement terminé, essayer de prédire des segmentations sur les images brutes (".tiff")
à l'aide des poids du réseau sauvegardés en ".pth" 




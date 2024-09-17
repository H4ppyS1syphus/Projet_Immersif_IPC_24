# 🌿 Projet Immersif 2024 : Détection des Plantes et Identification des Maladies

Bienvenue dans le dépôt du **Projet Immersif 2024**, où nous exploitons l'apprentissage automatique et profond pour la détection des plantes et le diagnostic des maladies. Ce projet couvre divers aspects du traitement d'images et de l'IA, avec un focus sur l'analyse de la santé des plantes.

---

## 🛠️ Instructions d'installation

### 1. Cloner le dépôt

Pour commencer, clonez ce dépôt en utilisant la commande suivante :

\`\`\`bash
git clone https://github.com/H4ppyS1syphus/Projet_Immersif_IPC_24.git
\`\`\`

### 2. Configurer l'environnement Python

Ce projet utilise Python ainsi que des dépendances spécifiques pour le traitement d'images et l'apprentissage automatique. Suivez les étapes ci-dessous pour configurer votre environnement :

#### a. Créer un environnement virtuel

Dans le répertoire du projet, exécutez la commande suivante pour créer un environnement virtuel :

\`\`\`bash
python -m venv .venv
\`\`\`

#### b. Activer l'environnement virtuel

- Sur **Windows** :

\`\`\`bash
.venv\\Scripts\\activate
\`\`\`

- Sur **macOS/Linux** :

\`\`\`bash
source .venv/bin/activate
\`\`\`

#### c. Installer les dépendances

Une fois l'environnement activé, installez les dépendances nécessaires avec :

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## 🚀 Instructions d'utilisation

### 1. Détection des Plantes

Le script suivant permet de détecter les plantes dans une image. Exécutez la commande suivante :

\`\`\`bash
python plant_detection/detect.py --image <chemin_vers_image>
\`\`\`

Remplacez \`<chemin_vers_image>\` par le chemin vers l'image à analyser.

### 2. Détection des Maladies

Pour détecter les maladies sur les plantes identifiées, utilisez le script suivant :

\`\`\`bash
python disease_detection/detect_disease.py --image <chemin_vers_image>
\`\`\`

Cela analysera l'image et renverra des informations sur les maladies potentielles affectant la plante.

---

## 📂 Structure du projet

- **Bases de données**
  - Feuilles diverses
  - Fraises

- **Divers**
  - scikit-learn
  - Un article sur YOLO
  - Un article sur YOLO v.9
  - [Page GitHub de YOLO v.9](https://github.com/ultralytics/yolov9)
  - Intégration de YOLO v.8 sur Jetson Nano
  - Cours de traitement d'images

---

## 💻 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir des pull requests pour améliorer le code, corriger des bugs ou proposer de nouvelles fonctionnalités.

---

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
"""

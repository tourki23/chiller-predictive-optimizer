Markdown
# ❄️ Industrial Chiller Predictive Optimizer


[![CI/CD Deployment](https://github.com/tourki23/chiller-predictive-optimizer/actions/workflows/main.yml/badge.svg)](https://github.com/tourki23/chiller-predictive-optimizer/actions)
[![Azure Deployed](https://img.shields.io/badge/Azure-App%20Service-blue)](https://chiller-optimizer-tourki-a0eka5dvfahmbbfn.francecentral-01.azurewebsites.net)
## 📌 Présentation du Projet
Ce projet est un **Jumeau Numérique (Digital Twin)** piloté par l'IA, conçu pour optimiser l'efficacité énergétique des systèmes HVAC industriels (refroidisseurs/chillers). 

L'application utilise des modèles de Machine Learning pour prédire la charge thermique et optimiser les points de consigne en temps réel, permettant de réduire la consommation électrique tout en maintenant les contraintes opérationnelles.

---

## 🛠️ Stack Technique

| Secteur | Technologies |
| :--- | :--- |
| **Frontend / Dashboard** | Dash (Plotly), Dash Bootstrap Components |
| **Data Processing** | Pandas, Numpy, Scikit-Learn |
| **Modélisation IA** | TensorFlow (Deep Learning), XGBoost |
| **Backend & Base de données** | Flask-Caching, SQLAlchemy |
| **DevOps & Cloud** | Docker, GitHub Actions (CI/CD), Azure Container Registry (ACR) |
| **Hosting** | Azure App Service (Web App for Containers) |

---

## 🏗️ Architecture MLOps

Le projet suit une approche de déploiement continu (CD) moderne :
1. **Local Dev :** Développement et tests unitaires avec `pytest`.
2. **CI (GitHub Actions) :** À chaque `push`, le code est testé, une image Docker est construite et envoyée vers l'ACR.
3. **CD (Azure) :** L'Azure Web App récupère automatiquement la nouvelle image et redémarre le service sans interruption.

---

## 🚀 Installation Locale

1. **Cloner le repository :**
   ```bash
   git clone [https://github.com/tourki23/chiller-predictive-optimizer.git](https://github.com/tourki23/chiller-predictive-optimizer.git)
   cd chiller-predictive-optimizer
Lancer avec Docker (Recommandé) :

Bash
docker-compose up --build
L'application sera disponible sur http://localhost:8050.

Lancer manuellement :

Bash
pip install -r requirements.txt
python app.py
🧪 Tests
Le projet inclut une suite de tests unitaires pour valider la logique de prédiction et l'intégrité des données :

Bash
pytest tests/
👨‍💻 Auteur
MAhmoud tourki - Data Scientist 



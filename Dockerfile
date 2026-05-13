# Utiliser une image Python officielle légère
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système requises (libgomp1 pour XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du code source
COPY . .

# --- AJOUT CRUCIAL ---
# Créer le dossier des logs et s'assurer qu'il est accessible en écriture
# Cela évite que ton logging.basicConfig ne fasse planter l'app
RUN mkdir -p /app/app_logs && chmod 777 /app/app_logs

# Exposer le port sur lequel Dash va tourner
EXPOSE 9050

# Commande pour démarrer l'application
# On lance le script depuis la racine /app pour que les chemins relatifs vers 
# 'trained_models' et 'axima_poc.db' soient corrects.
CMD ["python", "app/UI.py"]

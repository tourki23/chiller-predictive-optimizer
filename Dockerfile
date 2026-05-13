# Utiliser une image Python officielle légère
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système requises (ex: libgomp1 est souvent requis par XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du code source
COPY . .

# Exposer le port sur lequel Dash va tourner
EXPOSE 9050

# Commande pour démarrer l'application (en pointant vers le dossier app)
CMD ["gunicorn", "--bind", "0.0.0.0:9050", "--timeout", "600", "app.UI:server"]

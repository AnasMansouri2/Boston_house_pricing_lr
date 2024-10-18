# Utiliser une image Python 3.7 comme base
FROM python:3.7

# Copier tous les fichiers dans le répertoire /app
COPY . /app

# Définir le répertoire de travail à /app
WORKDIR /app

# Installer les dépendances à partir de requirements.txt
RUN pip install -r requirements.txt

# Exposer le port 5050
EXPOSE 5050

# Commande pour exécuter l'application Flask
CMD ["python", "appli.py"]

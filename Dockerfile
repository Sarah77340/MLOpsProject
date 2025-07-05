# Utilise une image officielle Python
FROM python:3.10-slim

# Répertoire de travail
WORKDIR /app

# Copie les dépendances et installe-les
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copie tout le code de l'application
COPY backend ./backend
COPY model ./model
COPY data ./data
COPY mlruns ./mlruns

# Expose le port de l’API
EXPOSE 8000

# Commande pour lancer FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

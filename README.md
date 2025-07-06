## Version 3

### Telechargement 

## Dans le terminal VSCode, exécute :
dvc init
git commit -m "Init DVC"

## Dans le terminal, lancer avec la commande :
docker compose up --build

(-> ça fait tout le build avec le reload automatique si modification du code)

# Pour info :
- si création d'une nouvelle version de main.py, par exemple main3.py,
    alors il faut modifier dans le *Dockerfile* le chiffre dans `main:app`  :

    ```CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]```

    par :

    ```CMD ["uvicorn", "backend.main3:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]```



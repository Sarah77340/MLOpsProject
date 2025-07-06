"""
Télécharge le dernier modèle enregistré dans MLflow (DagsHub)
et l’enregistre localement sous model/emotion_model.keras
"""

import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import tempfile

# ── 1) Config MLflow ────────────────────────────────────────────────────────────
TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]          # injecté par le workflow
EXPERIMENT_ID = "0"                                       # default experiment
ARTIFACT_PATH_IN_RUN = "model"                            # où tu logues le modèle
LOCAL_MODEL_PATH = Path("model/emotion_model.keras")      # cible dans le repo

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# ── 2) Récupérer le dernier run (le + récent) ───────────────────────────────────
runs = client.search_runs(
    experiment_ids=[EXPERIMENT_ID],
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

if not runs:
    raise RuntimeError("Aucun run MLflow trouvé dans l'expérience 0")

run = runs[0]
run_id = run.info.run_id
print(f"� latest run_id = {run_id}")

# ── 3) Télécharger l’artifact ‘model’ de ce run ────────────────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    download_uri = f"runs:/{run_id}/{ARTIFACT_PATH_IN_RUN}"
    local_tmp = mlflow.artifacts.download_artifacts(download_uri, tmpdir)
    # local_tmp contient le dossier sauvegardé par mlflow.keras.save_model

    # ─ Copier/écraser dans model/
    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Cas 1 : modèle déjà sous forme .keras
    keras_files = list(Path(local_tmp).rglob("*.keras"))
    if keras_files:
        shutil.copy(keras_files[0], LOCAL_MODEL_PATH)
    else:
        # Cas 2 : SavedModel → on copie tout le dossier
        target_dir = LOCAL_MODEL_PATH.parent / "emotion_model"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(local_tmp, target_dir)

print(f"Modèle téléchargé dans {LOCAL_MODEL_PATH.parent.resolve()}")

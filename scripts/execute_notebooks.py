"""Ejecuta y guarda los seis notebooks en orden."""
from pathlib import Path
import os

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
NAMES = [
    "01_datos_originales_y_eda.ipynb",
    "02_limpieza_de_transacciones.ipynb",
    "03_feature_engineering_y_dataset_modelado.ipynb",
    "04_modelo_con_datos_originales.ipynb",
    "05_modelo_con_datos_procesados.ipynb",
    "06_comparacion_y_entrega_de_resultados.ipynb",
]
OUT = ROOT / "notebooks/executed"
OUT.mkdir(parents=True, exist_ok=True)

for name in NAMES:
    print(f">>> {name}")
    source = ROOT / "notebooks" / name
    notebook = nbformat.read(source, as_version=4)
    NotebookClient(notebook, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute(cwd=str(ROOT))
    nbformat.write(notebook, OUT / name)
    print("OK")

print("Ejecución completa")

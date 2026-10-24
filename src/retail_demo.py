"""Lógica compartida para la demo guiada de Explotación de Datos."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, median_absolute_error,
                             precision_score, r2_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
CUTOFF = pd.Timestamp("2011-09-01")
TARGET_END = pd.Timestamp("2011-10-01")
SNAPSHOTS = [pd.Timestamp("2011-06-01"), pd.Timestamp("2011-07-01"), pd.Timestamp("2011-08-01"), pd.Timestamp("2011-09-01")]
DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
MODEL_PARAMS = {"n_estimators": 250, "min_samples_leaf": 4, "max_features": "sqrt", "n_jobs": -1, "random_state": SEED}


def ensure_folders(root: Path) -> None:
    for relative in ("data/raw", "data/interim", "data/processed", "artifacts", "reports"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def source_paths(root: Path) -> tuple[Path, Path]:
    raw_dir = root / "data/raw"
    return raw_dir / "online_retail.zip", raw_dir / "Online Retail.xlsx"


def load_original(root: Path) -> tuple[pd.DataFrame, bool]:
    """Descarga una vez la fuente oficial y devuelve las transacciones sin alterar."""
    ensure_folders(root)
    zip_path, xlsx_path = source_paths(root)
    was_downloaded = False
    if not xlsx_path.exists():
        if not zip_path.exists():
            response = requests.get(DATA_URL, timeout=180)
            response.raise_for_status()
            zip_path.write_bytes(response.content)
            was_downloaded = True
        with zipfile.ZipFile(zip_path) as archive:
            name = next(item for item in archive.namelist() if item.lower().endswith(".xlsx"))
            xlsx_path.write_bytes(archive.read(name))
    frame = pd.read_excel(xlsx_path, engine="openpyxl")
    for column in ["InvoiceNo", "StockCode", "Description", "Country"]:
        frame[column] = frame[column].astype("string")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    frame.to_parquet(root / "data/interim/transacciones_originales.parquet", index=False)
    return frame, was_downloaded


def original_diagnostic(raw: pd.DataFrame) -> pd.DataFrame:
    counts = {
        "Total de líneas": len(raw),
        "Cliente sin identificador": int(raw["CustomerID"].isna().sum()),
        "Factura cancelada": int(raw["InvoiceNo"].str.startswith("C", na=False).sum()),
        "Cantidad no positiva": int((raw["Quantity"] <= 0).sum()),
        "Precio no positivo": int((raw["UnitPrice"] <= 0).sum()),
        "Duplicado exacto": int(raw.duplicated().sum()),
    }
    result = pd.DataFrame({"regla": counts.keys(), "filas": counts.values()})
    result["porcentaje"] = 100 * result["filas"] / len(raw)
    return result


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = raw.copy()
    for column in ["InvoiceNo", "StockCode", "Description", "Country"]:
        work[column] = work[column].astype("string").str.strip()
    work["Description"] = work["Description"].fillna("SIN_DESCRIPCION")
    work["Country"] = work["Country"].str.upper()
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"])
    work["line_total"] = work["Quantity"] * work["UnitPrice"]
    work["es_producto"] = work["StockCode"].str.fullmatch(r"\d+[A-Z]?").fillna(False)
    cancelled = work["InvoiceNo"].str.startswith("C", na=False)
    has_customer = work["CustomerID"].notna()
    valid_sale = (~cancelled) & has_customer & (work["Quantity"] > 0) & (work["UnitPrice"] > 0)
    duplicate = work.duplicated(subset=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"], keep="first")
    audit = pd.DataFrame([
        ("Archivo original", len(work), "Punto de partida"),
        ("Sin CustomerID", int((~has_customer).sum()), "No se puede construir un perfil por cliente"),
        ("Factura cancelada", int(cancelled.sum()), "No se interpreta como venta"),
        ("Cantidad o precio no positivo", int(((work["Quantity"] <= 0) | (work["UnitPrice"] <= 0)).sum()), "No se interpreta como venta"),
        ("Duplicado exacto entre ventas candidatas", int((valid_sale & duplicate).sum()), "Se descarta sólo si toda la línea coincide"),
        ("Códigos de servicio/no producto", int((valid_sale & ~work["es_producto"]).sum()), "Se conservan en gasto, no se usan como diversidad de productos"),
    ], columns=["regla", "filas_afectadas", "motivo"])
    return work.loc[valid_sale & ~duplicate].copy(), audit


def build_model_tables(raw: pd.DataFrame, clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crea snapshots mensuales comparables y una evaluación realmente temporal."""
    raw, clean = raw.copy(), clean.copy()
    for frame in (raw, clean):
        frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
        frame["InvoiceNo"] = frame["InvoiceNo"].astype("string")
        frame["line_total"] = frame["Quantity"] * frame["UnitPrice"]

    original_parts, processed_parts, targets = [], [], []
    for cutoff in SNAPSHOTS:
        horizon_end = cutoff + pd.Timedelta(days=30)
        raw_history = raw[(raw["InvoiceDate"] < cutoff) & raw["CustomerID"].notna()].copy()
        clean_history = clean[(clean["InvoiceDate"] < cutoff) & clean["CustomerID"].notna()].copy()
        future = clean[(clean["InvoiceDate"] >= cutoff) & (clean["InvoiceDate"] < horizon_end)].copy()
        customer_ids = np.intersect1d(raw_history["CustomerID"].unique(), clean_history["CustomerID"].unique())
        target = pd.DataFrame({"CustomerID": customer_ids})
        target["snapshot_date"] = cutoff
        target = target.merge(future.groupby("CustomerID")["line_total"].sum().rename("gasto_futuro"), on="CustomerID", how="left")
        target["gasto_futuro"] = target["gasto_futuro"].fillna(0.0)
        target["compra_futura"] = (target["gasto_futuro"] > 0).astype(int)
        targets.append(target)

        # El baseline reproduce el atajo: pocos agregados sobre datos sin reglas de negocio.
        baseline = raw_history[raw_history["CustomerID"].isin(customer_ids)].groupby("CustomerID").agg(
            lineas_raw=("InvoiceNo", "size"), facturas_raw=("InvoiceNo", "nunique"),
            cantidad_neta_raw=("Quantity", "sum"), importe_neto_raw=("line_total", "sum"),
            fecha_ultima_raw=("InvoiceDate", "max"),
        ).reset_index()
        baseline["dias_desde_ultima_raw"] = (cutoff - baseline.pop("fecha_ultima_raw")).dt.days
        baseline["snapshot_date"] = cutoff
        original_parts.append(target.merge(baseline, on=["CustomerID", "snapshot_date"]))

        # Perfil procesado: RFM, intensidad reciente, tendencia, diversidad y devoluciones.
        sales = clean_history[clean_history["CustomerID"].isin(customer_ids)].copy()
        invoices = sales.groupby(["CustomerID", "InvoiceNo"])["line_total"].sum().rename("total_factura")
        profile = sales.groupby("CustomerID").agg(
            ultima_fecha=("InvoiceDate", "max"), primera_fecha=("InvoiceDate", "min"),
            frecuencia_facturas=("InvoiceNo", "nunique"), gasto_historico=("line_total", "sum"),
            cantidad_promedio=("Quantity", "mean"),
            dias_activos=("InvoiceDate", lambda x: x.dt.normalize().nunique()), pais=("Country", "last"),
        ).reset_index()
        product_sales = sales[sales["es_producto"]]
        profile = profile.merge(product_sales.groupby("CustomerID")["StockCode"].nunique().rename("productos_distintos"), on="CustomerID", how="left")
        profile["productos_distintos"] = profile["productos_distintos"].fillna(0)
        profile = profile.merge(invoices.groupby("CustomerID").agg(ticket_promedio="mean", ticket_mediano="median"), on="CustomerID", how="left")
        profile["recencia_dias"] = (cutoff - profile.pop("ultima_fecha")).dt.days
        profile["antiguedad_dias"] = (cutoff - profile.pop("primera_fecha")).dt.days

        for days in (30, 90):
            recent = sales[sales["InvoiceDate"] >= cutoff - pd.Timedelta(days=days)]
            recent_features = recent.groupby("CustomerID").agg(**{
                f"gasto_{days}d": ("line_total", "sum"), f"facturas_{days}d": ("InvoiceNo", "nunique"),
                f"productos_{days}d": ("StockCode", "nunique"),
            })
            profile = profile.merge(recent_features, on="CustomerID", how="left")
        previous = sales[(sales["InvoiceDate"] >= cutoff - pd.Timedelta(days=60)) & (sales["InvoiceDate"] < cutoff - pd.Timedelta(days=30))]
        previous_spend = previous.groupby("CustomerID")["line_total"].sum().rename("gasto_30d_previo")
        profile = profile.merge(previous_spend, on="CustomerID", how="left")
        profile["tendencia_gasto_30d"] = profile["gasto_30d"].fillna(0) / (profile["gasto_30d_previo"].fillna(0) + 1)

        returns = raw_history[(raw_history["Quantity"] < 0) & raw_history["CustomerID"].isin(customer_ids)].copy()
        returns["importe_devolucion"] = returns["line_total"].abs()
        return_features = returns.groupby("CustomerID").agg(lineas_devolucion=("InvoiceNo", "size"), importe_devolucion=("importe_devolucion", "sum"))
        profile = profile.merge(return_features, on="CustomerID", how="left")
        for column in ["gasto_30d", "facturas_30d", "productos_30d", "gasto_90d", "facturas_90d", "productos_90d", "gasto_30d_previo", "lineas_devolucion", "importe_devolucion"]:
            profile[column] = profile[column].fillna(0)
        profile["tasa_devolucion"] = profile["importe_devolucion"] / profile["gasto_historico"].clip(lower=1)
        for column in ["gasto_historico", "ticket_promedio", "ticket_mediano", "gasto_30d", "gasto_90d"]:
            profile[f"log_{column}"] = np.log1p(profile[column].clip(lower=0))
        profile["snapshot_date"] = cutoff
        processed_parts.append(target.merge(profile, on=["CustomerID", "snapshot_date"]))

    original = pd.concat(original_parts, ignore_index=True)
    processed = pd.concat(processed_parts, ignore_index=True)
    target_all = pd.concat(targets, ignore_index=True)
    split_by_date = {SNAPSHOTS[0]: "train", SNAPSHOTS[1]: "train", SNAPSHOTS[2]: "validation", SNAPSHOTS[3]: "test"}
    for table in (original, processed):
        table["particion"] = table["snapshot_date"].map(split_by_date)
        table.replace([np.inf, -np.inf], np.nan, inplace=True)
    manifest = processed[["CustomerID", "snapshot_date", "compra_futura", "particion"]].copy()
    assert original[["CustomerID", "snapshot_date", "compra_futura"]].equals(processed[["CustomerID", "snapshot_date", "compra_futura"]])
    return original, processed, manifest, target_all


def make_leaky_table(processed: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Crea a propósito un ejemplo de fuga para enseñarlo y descartarlo.

    Este dataset NO representa un modelo válido. La variable añadida cuenta actividad
    ocurrida después del corte, exactamente en la ventana cuyo comportamiento se
    quiere predecir. Se utiliza sólo en el notebook 04 para mostrar que una métrica
    espectacular puede ser evidencia de un error de preparación, no de un buen modelo.
    """
    transactions = clean.copy()
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
    table = processed.copy()
    counts = []
    for snapshot in SNAPSHOTS:
        end = snapshot + pd.Timedelta(days=30)
        after_cutoff = transactions[(transactions["InvoiceDate"] >= snapshot) & (transactions["InvoiceDate"] < end)]
        count = after_cutoff.groupby("CustomerID").size().rename("lineas_posteriores_al_corte").reset_index()
        count["snapshot_date"] = snapshot
        counts.append(count)
    leaked_activity = pd.concat(counts, ignore_index=True)
    table = table.merge(leaked_activity, on=["CustomerID", "snapshot_date"], how="left")
    table["lineas_posteriores_al_corte"] = table["lineas_posteriores_al_corte"].fillna(0)
    return table


def model_metrics(y: pd.Series, probability: np.ndarray, threshold: float = .5) -> dict:
    pred = (probability >= threshold).astype(int)
    return {
        "prevalencia": float(y.mean()), "umbral": float(threshold), "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probability)), "matriz_confusion": confusion_matrix(y, pred).tolist(),
    }


def regression_metrics(y: pd.Series, predicted: np.ndarray) -> dict:
    return {"mae": float(mean_absolute_error(y, predicted)), "medae": float(median_absolute_error(y, predicted)), "r2": float(r2_score(y, predicted)), "n_compradores_evaluados": int(len(y))}


def no_purchase_baseline(y: pd.Series) -> dict:
    """Referencia honesta: nunca contactar a nadie.

    En un problema con pocos compradores puede lograr una accuracy aparentemente alta,
    pero su recall y F1 para compradores son cero. Se usa en los notebooks como un
    recordatorio de que accuracy no equivale a utilidad de negocio.
    """
    return model_metrics(y.astype(int), np.zeros(len(y)), threshold=.5)


def ranking_metrics(test: pd.DataFrame, score_column: str = "probabilidad_compra", fraction: float = .10) -> dict:
    """Mide la calidad de una lista priorizada, no sólo de una etiqueta binaria."""
    count = max(1, int(np.ceil(len(test) * fraction)))
    top = test.nlargest(count, score_column)
    prevalence = test["compra_futura"].mean()
    precision_at_k = top["compra_futura"].mean()
    total_spend = test["gasto_futuro"].sum()
    return {
        "criterio_ordenamiento": score_column,
        "porcentaje_contactado": fraction,
        "clientes_contactados": int(count),
        "precision_at_k": float(precision_at_k),
        "lift_at_k": float(precision_at_k / prevalence) if prevalence else 0.0,
        "compras_capturadas": int(top["compra_futura"].sum()),
        "porcentaje_gasto_futuro_capturado": float(top["gasto_futuro"].sum() / total_spend) if total_spend else 0.0,
    }


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num = [name for name in X.columns if name not in cat]
    return ColumnTransformer([
        ("numericas", Pipeline([("impute", SimpleImputer(strategy="median"))]), num),
        ("categoricas", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def train_models(table: pd.DataFrame, root: Path, tag: str) -> tuple[pd.DataFrame, dict, Pipeline, Pipeline]:
    features = [col for col in table.columns if col not in {"CustomerID", "snapshot_date", "compra_futura", "gasto_futuro", "particion"}]
    X = table[features].copy()
    y = table["compra_futura"].astype(int)
    train, validation, test_mask = table["particion"].eq("train"), table["particion"].eq("validation"), table["particion"].eq("test")
    classifier = Pipeline([("preprocess", make_preprocessor(X)), ("model", RandomForestClassifier(class_weight="balanced", **MODEL_PARAMS))])
    classifier.fit(X.loc[train], y.loc[train])
    validation_probability = classifier.predict_proba(X.loc[validation])[:, 1]
    thresholds = np.arange(.25, .76, .05)
    selected_threshold = float(max(thresholds, key=lambda threshold: f1_score(y.loc[validation], validation_probability >= threshold)))
    test = table.loc[test_mask, ["CustomerID", "snapshot_date", "compra_futura", "gasto_futuro"]].copy()
    test["probabilidad_compra"] = classifier.predict_proba(X.loc[test_mask])[:, 1]
    buyers_train = train & y.eq(1)
    regressor = Pipeline([("preprocess", make_preprocessor(X)), ("model", RandomForestRegressor(**MODEL_PARAMS))])
    regressor.fit(X.loc[buyers_train], table.loc[buyers_train, "gasto_futuro"])
    test["gasto_condicional_estimado"] = np.maximum(regressor.predict(X.loc[test_mask]), 0)
    test["valor_esperado"] = test["probabilidad_compra"] * test["gasto_condicional_estimado"]
    buyer_test = test["compra_futura"].eq(1)
    metrics = {
        "clasificacion": model_metrics(test["compra_futura"], test["probabilidad_compra"], selected_threshold),
        "ranking_por_probabilidad": ranking_metrics(test),
        "ranking_por_valor_esperado": ranking_metrics(test, score_column="valor_esperado"),
        "umbral_seleccionado_validacion": selected_threshold,
        "regresion_condicional": regression_metrics(test.loc[buyer_test, "gasto_futuro"], test.loc[buyer_test, "gasto_condicional_estimado"]),
        "n_train": int(train.sum()), "n_validation": int(validation.sum()), "n_test": int(len(test)),
    }
    test.to_parquet(root / f"artifacts/predicciones_{tag}.parquet", index=False)
    with open(root / f"reports/metricas_modelo_{tag}.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    joblib.dump(classifier, root / f"artifacts/modelo_{tag}_clasificacion.joblib")
    joblib.dump(regressor, root / f"artifacts/modelo_{tag}_regresion.joblib")
    return test, metrics, classifier, regressor

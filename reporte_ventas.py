"""Limpieza de ventas, estadísticas por categoría y gráficos, parametrizado por JSON."""
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parent
logger = logging.getLogger("reporte_ventas")


def configurar_logging() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(BASE / "reporte_ventas.log", encoding="utf-8"),
        ],
    )


def cargar_config(ruta: Path) -> dict:
    with open(ruta, encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Config cargada: %s", ruta.name)
    return config


def cargar_datos(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta, encoding="utf-8")
    logger.info("Datos cargados: %d filas", len(df))
    return df


def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Duplicados eliminados: %d", n - len(df))
    return df


def normalizar_texto(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    for col in columnas:
        df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    return df


def convertir_tipos(df: pd.DataFrame) -> pd.DataFrame:
    # "$271.570" -> 271570 (el punto es separador de miles)
    df["precio_unitario"] = pd.to_numeric(
        df["precio_unitario"].astype("string").str.replace(r"[^\d]", "", regex=True),
        errors="coerce",
    )
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


def imputar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="number").columns:
        nulos = int(df[col].isna().sum())
        if nulos:
            df[col] = df[col].fillna(df[col].median())
            logger.info("Nulos imputados con mediana en '%s': %d", col, nulos)
    nulos = int(df["fecha"].isna().sum())
    if nulos:
        df["fecha"] = df["fecha"].fillna(df["fecha"].median())
        logger.info("Nulos imputados con mediana en 'fecha': %d", nulos)
    return df


def eliminar_atipicos_iqr(df: pd.DataFrame, columna: str, factor: float) -> pd.DataFrame:
    q1, q3 = np.percentile(df[columna], [25, 75])
    iqr = q3 - q1
    bajo, alto = q1 - factor * iqr, q3 + factor * iqr
    mascara = (df[columna] >= bajo) & (df[columna] <= alto)
    logger.info("Atípicos en '%s' (rango %.2f-%.2f) eliminados: %d", columna, bajo, alto, int((~mascara).sum()))
    return df[mascara].reset_index(drop=True)


def resumen_por_categoria(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for cat, grupo in df.groupby("categoria"):
        monto = grupo["monto"].to_numpy(dtype=float)
        filas.append({
            "categoria": cat,
            "media": np.mean(monto),
            "mediana": np.median(monto),
            "desv_std": np.std(monto, ddof=1) if len(monto) > 1 else np.nan,
        })
    return pd.DataFrame(filas)


def graficar(df: pd.DataFrame, resumen: pd.DataFrame, ruta: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ejes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=resumen, x="categoria", y="media", hue="categoria", ax=ejes[0], legend=False)
    ejes[0].set_title("Monto medio por categoría")
    ejes[0].set_ylabel("Monto medio")
    sns.boxplot(data=df, x="categoria", y="monto", hue="categoria", ax=ejes[1], legend=False)
    ejes[1].set_title("Distribución del monto por categoría")
    ejes[1].set_ylabel("Monto")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info("Gráfico guardado: %s", ruta.name)


def abrir_archivo(ruta: Path) -> None:
    if hasattr(os, "startfile"):
        os.startfile(ruta)
    else:
        logger.warning("Apertura automática solo disponible en Windows")


def main() -> None:
    configurar_logging()
    cfg = cargar_config(BASE / "config_reporte.json")

    df = cargar_datos(BASE / cfg["archivo_entrada"])
    df = normalizar_texto(df, cfg["columnas_texto"])
    df = convertir_tipos(df)
    df = eliminar_duplicados(df)
    df = imputar_nulos(df)
    df = eliminar_atipicos_iqr(df, cfg["columna_atipicos"], cfg["factor_iqr"])
    df["monto"] = df["unidades"] * df["precio_unitario"]

    df.to_csv(BASE / cfg["archivo_limpio"], index=False, encoding="utf-8")
    resumen = resumen_por_categoria(df)
    resumen.to_csv(BASE / cfg["archivo_resumen"], index=False, encoding="utf-8")
    logger.info("Resumen por categoría:\n%s", resumen.round(2).to_string(index=False))

    ruta_grafico = BASE / cfg["archivo_grafico"]
    graficar(df, resumen, ruta_grafico)
    if cfg.get("mostrar_grafico", False):
        abrir_archivo(ruta_grafico)
    logger.info("Proceso finalizado")


if __name__ == "__main__":
    main()

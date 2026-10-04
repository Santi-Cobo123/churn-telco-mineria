"""Fase 4a - Modelo descriptivo: segmentación de clientes con K-Means."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES, METRICS, RANDOM_STATE

CLUSTER_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "NumServicios"]


def elegir_k(df, k_range=range(2, 9)):
    """Método del codo y coeficiente de silueta para elegir el número de grupos."""
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    inercias, siluetas = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(X, km.labels_, sample_size=min(3000, len(X)),
                                         random_state=RANDOM_STATE))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(list(k_range), inercias, "o-", color="#1f77b4", label="Inercia")
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_ylabel("Inercia", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(list(k_range), siluetas, "s--", color="#d62728", label="Silueta")
    ax2.set_ylabel("Coeficiente de silueta", color="#d62728")
    ax1.set_title("Método del codo y silueta")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_kmeans_codo_silueta.png", dpi=150)
    plt.close(fig)
    tabla = pd.DataFrame({"k": list(k_range), "inercia": inercias, "silueta": siluetas})
    tabla.to_csv(METRICS / "kmeans_seleccion_k.csv", index=False)
    return tabla


def segmentar(df, k):
    """Ajusta K-Means con k grupos y devuelve el perfil de cada segmento."""
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    df = df.copy()
    df["Segmento"] = km.labels_

    perfil = df.groupby("Segmento").agg(
        clientes=("Segmento", "size"),
        **{f"{c}_media": (c, "mean") for c in CLUSTER_FEATURES},
        tasa_churn=("Churn", lambda s: (s == "Yes").mean()),
    ).round(3)
    perfil.to_csv(METRICS / "kmeans_perfil_segmentos.csv")

    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=km.labels_, cmap="Set2", s=8, alpha=0.7)
    ax.legend(*sc.legend_elements(), title="Segmento")
    ax.set_title(f"Segmentos de clientes (K-Means, k={k}) proyectados con PCA")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_kmeans_segmentos_pca.png", dpi=150)
    plt.close(fig)
    return df, perfil

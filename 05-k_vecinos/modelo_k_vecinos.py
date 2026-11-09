"""Clasificación de hogares pobres con k vecinos más cercanos (kNN).

Ejecute desde cualquier carpeta con: python modelo_k_vecinos.py
El archivo de datos esperado está en datos/datos.xlsx, hoja hogares.
"""

# Antes de importar las librerías del análisis, instalar solamente las ausentes.
import importlib.util
import subprocess
import sys

PAQUETES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
}
faltantes = [instalacion for modulo, instalacion in PAQUETES.items()
             if importlib.util.find_spec(modulo) is None]
if faltantes:
    print("Instalando librerías faltantes:", ", ".join(faltantes))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *faltantes])

# Importaciones del análisis, después de comprobar/instalar dependencias.
from pathlib import Path
import unicodedata
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, train_test_split,
                                     cross_val_predict)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Rutas relativas al propio archivo para que el script funcione desde cualquier ubicación.
CARPETA = Path(__file__).resolve().parent
ARCHIVO_DATOS = CARPETA / "datos" / "datos.xlsx"
CARPETA_RESULTADOS = CARPETA / "resultados"
CARPETA_FIGURAS = CARPETA / "visualizaciones"
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
ARCHIVO_EXCEL = CARPETA_RESULTADOS / "k_vecinos_resultados.xlsx"
RANDOM_STATE = 42
VARIABLES_X = ["ingreso_pc", "educacion_jefe"]
VARIABLE_Y = "pobre"


def mensaje(texto):
    """Imprime avances breves en español."""
    print(f"\n{texto}")


def guardar_figura(figura, nombre):
    """Guarda PNG numerado a 150 dpi y cierra la figura."""
    ruta = CARPETA_FIGURAS / nombre
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)
    print(f"Figura guardada: {ruta.name}")


def normalizar_texto(valor):
    """Quita tildes y espacios para reconocer variantes Sí/Si/No."""
    texto = str(valor).strip().lower().replace("�", "i")
    return "".join(c for c in unicodedata.normalize("NFKD", texto)
                   if not unicodedata.combining(c))


def codificar_objetivo(serie):
    """Convierte la clase positiva Sí en 1 y No en 0."""
    mapeo = {"si": 1, "no": 0}
    valores = serie.map(normalizar_texto).map(mapeo)
    if valores.isna().any():
        inesperados = sorted(serie[valores.isna()].astype(str).unique().tolist())
        raise ValueError(f"Valores inesperados en '{VARIABLE_Y}': {inesperados}")
    return valores.astype(int)


def metricas(nombre, y_real, probabilidades, umbral=0.5):
    """Calcula las cuatro métricas solicitadas para un umbral dado."""
    y_predicho = (np.asarray(probabilidades) >= umbral).astype(int)
    return {
        "modelo": nombre,
        "accuracy": accuracy_score(y_real, y_predicho),
        "precision": precision_score(y_real, y_predicho, zero_division=0),
        "recall_sensibilidad": recall_score(y_real, y_predicho, zero_division=0),
        "AUC": roc_auc_score(y_real, probabilidades) if len(np.unique(y_real)) == 2 else np.nan,
    }


def crear_pipeline_knn(k, peso):
    """Escala dentro del pipeline: el escalador aprende solo del pliegue de entrenamiento."""
    return Pipeline([
        ("escalador", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=peso)),
    ])


def main():
    # 1. Carga, revisión inicial y descripción del conjunto.
    mensaje("PASO 1 | Cargando y revisando datos")
    if not ARCHIVO_DATOS.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ARCHIVO_DATOS}")
    datos = pd.read_excel(ARCHIVO_DATOS, sheet_name="hogares", engine="openpyxl")
    faltantes_por_columna = datos.isna().sum()
    porcentaje_si = (datos[VARIABLE_Y].map(normalizar_texto) == "si").mean() * 100
    descripcion = pd.DataFrame([
        {"elemento": "archivo", "valor": str(ARCHIVO_DATOS.relative_to(CARPETA))},
        {"elemento": "hoja", "valor": "hogares"},
        {"elemento": "filas", "valor": len(datos)},
        {"elemento": "columnas", "valor": len(datos.columns)},
        {"elemento": "columnas disponibles", "valor": ", ".join(datos.columns)},
        {"elemento": "casos pobres = Sí (%)", "valor": porcentaje_si},
        {"elemento": "filas con algún dato faltante", "valor": int(datos.isna().any(axis=1).sum())},
    ] + [
        {"elemento": f"faltantes en {col}", "valor": int(n)}
        for col, n in faltantes_por_columna.items()
    ])
    print(f"Datos: {datos.shape[0]} filas y {datos.shape[1]} columnas. Sí: {porcentaje_si:.1f}%. ")
    print(f"Filas con algún faltante: {datos.isna().any(axis=1).sum()}.")
    requeridas = VARIABLES_X + [VARIABLE_Y]
    no_disponibles = [col for col in requeridas if col not in datos.columns]
    if no_disponibles:
        raise KeyError(f"Faltan columnas necesarias: {no_disponibles}")

    # 2. Codificación de la respuesta y preparación de variables de entrada.
    mensaje("PASO 2 | Codificando la clase y preparando las variables")
    y_completo = codificar_objetivo(datos[VARIABLE_Y])
    # get_dummies convierte cada categoría de texto en columnas 0/1; aquí las dos X
    # solicitadas ya son numéricas, por lo que se conservan como tales.
    x_crudo = datos[VARIABLES_X].copy()
    columnas_texto = x_crudo.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if columnas_texto:
        x_preparado = pd.get_dummies(x_crudo, columns=columnas_texto, dtype=int)
    else:
        x_preparado = x_crudo.copy()
    # Para mantener los hogares con entradas faltantes, la mediana se estima dentro
    # del pipeline, evitando que información de prueba influya en el entrenamiento.
    mascara = y_completo.notna()
    x = x_preparado.loc[mascara].reset_index(drop=True)
    y = y_completo.loc[mascara].reset_index(drop=True)
    indice_original = datos.index[mascara].to_numpy()
    print(f"Clase: Sí=1 ({int(y.sum())}), No=0 ({int((y == 0).sum())}). X usadas: {list(x.columns)}.")
    if y.nunique() != 2:
        raise ValueError("Se requieren ambas clases (Sí y No) para entrenar y evaluar el clasificador.")

    # 3. Separación estratificada: conserva proporciones de Sí/No en ambos grupos.
    mensaje("PASO 3 | Separando entrenamiento (75 %) y prueba (25 %)")
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_test = x.iloc[idx_train], x.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    print(f"Entrenamiento: {len(idx_train)} hogares ({y_train.mean():.1%} positivos). "
          f"Prueba: {len(idx_test)} ({y_test.mean():.1%} positivos).")

    # 4. Comparación justa de kNN con y sin escalado.
    mensaje("PASO 4 | Comparando kNN con y sin escalado")
    k_base = min(5, len(x_train))
    modelo_sin = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("knn", KNeighborsClassifier(n_neighbors=k_base)),
    ])
    modelo_sin.fit(x_train, y_train)
    modelo_con = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k_base)),
    ])
    modelo_con.fit(x_train, y_train)
    escalado = pd.DataFrame([
        {"modelo": f"kNN k={k_base}, sin escalar", "accuracy_prueba": accuracy_score(y_test, modelo_sin.predict(x_test)),
         "explicacion": "Las distancias pueden quedar dominadas por la variable de mayor magnitud."},
        {"modelo": f"kNN k={k_base}, escalado", "accuracy_prueba": accuracy_score(y_test, modelo_con.predict(x_test)),
         "explicacion": "StandardScaler ajustado solo con entrenamiento pone las variables en escalas comparables."},
    ])
    print(escalado[["modelo", "accuracy_prueba"]].to_string(index=False))
    print("El escalador se ajusta dentro del pipeline usando solo entrenamiento; así prueba no filtra información.")

    # 5. GridSearchCV prueba k impares y dos formas de ponderar a los vecinos.
    mensaje("PASO 5 | Buscando k y el peso óptimos con validación cruzada")
    k_valores = [k for k in range(1, 102, 2) if k <= len(x_train) - 1]
    if not k_valores:
        raise ValueError("Hay muy pocos datos de entrenamiento para probar k hasta 101.")
    pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    busqueda_modelo = GridSearchCV(
        crear_pipeline_knn(1, "uniform"),
        {"knn__n_neighbors": k_valores, "knn__weights": ["uniform", "distance"]},
        scoring="accuracy", cv=pliegues, return_train_score=True, n_jobs=-1,
        refit=True,
    )
    busqueda_modelo.fit(x_train, y_train)
    mejor_k = int(busqueda_modelo.best_params_["knn__n_neighbors"])
    mejor_peso = busqueda_modelo.best_params_["knn__weights"]
    busqueda = pd.DataFrame(busqueda_modelo.cv_results_).sort_values(
        ["param_knn__n_neighbors", "param_knn__weights"]
    ).reset_index(drop=True)
    busqueda = busqueda.rename(columns={
        "param_knn__n_neighbors": "k", "param_knn__weights": "peso",
        "mean_train_score": "accuracy_entrenamiento_promedio",
        "std_train_score": "desviacion_entrenamiento",
        "mean_test_score": "accuracy_validacion_promedio",
        "std_test_score": "desviacion_validacion", "rank_test_score": "puesto",
    })
    columnas_busqueda = ["k", "peso", "accuracy_entrenamiento_promedio",
                         "desviacion_entrenamiento", "accuracy_validacion_promedio",
                         "desviacion_validacion", "puesto"]
    busqueda = busqueda[columnas_busqueda]
    print(f"Mejor combinación: k={mejor_k}, peso={mejor_peso}; accuracy media CV="
          f"{busqueda_modelo.best_score_:.3f}.")
    por_k = busqueda.groupby("k", as_index=False).agg(
        accuracy_entrenamiento=("accuracy_entrenamiento_promedio", "mean"),
        accuracy_validacion=("accuracy_validacion_promedio", "mean"),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(por_k["k"], por_k["accuracy_entrenamiento"], marker="o", ms=3, label="Entrenamiento (promedio de pesos)")
    ax.plot(por_k["k"], por_k["accuracy_validacion"], marker="o", ms=3, label="Validación cruzada (promedio de pesos)")
    ax.axvline(mejor_k, color="black", linestyle="--", alpha=0.65, label=f"k elegido = {mejor_k}")
    ax.set(title="Exactitud según el número de vecinos", xlabel="k (número de vecinos)", ylabel="Exactitud")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.25); ax.legend()
    guardar_figura(fig, "01_exactitud_k.png")
    print("Con k=1, cada observación de entrenamiento suele ser su propio vecino más cercano: ")
    print("por eso la exactitud de entrenamiento se acerca a 100 %, aunque puede ser sobreajuste.")

    # El mejor pipeline ya fue ajustado por GridSearchCV con todo entrenamiento.
    modelo_knn = busqueda_modelo.best_estimator_
    prob_train_knn = modelo_knn.predict_proba(x_train)[:, 1]
    prob_test_knn = modelo_knn.predict_proba(x_test)[:, 1]

    # 6. Mapas de decisión en el plano de las dos variables numéricas.
    mensaje("PASO 6 | Dibujando mapas de decisión")
    def predecir_malla(k, xx, yy):
        datos_malla = pd.DataFrame({VARIABLES_X[0]: xx.ravel(), VARIABLES_X[1]: yy.ravel()})
        m = crear_pipeline_knn(k, mejor_peso)
        m.fit(x_train, y_train)
        return m.predict_proba(datos_malla)[:, 1].reshape(xx.shape)

    min_x, max_x = x[VARIABLES_X[0]].min(), x[VARIABLES_X[0]].max()
    min_y, max_y = x[VARIABLES_X[1]].min(), x[VARIABLES_X[1]].max()
    gx = np.linspace(min_x, max_x, 160); gy = np.linspace(min_y, max_y, 160)
    xx, yy = np.meshgrid(gx, gy)
    ks_mapa = [1, mejor_k, 75]
    fig, ejes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    for ax, k_mapa in zip(ejes, ks_mapa):
        z = predecir_malla(k_mapa, xx, yy)
        ax.contourf(xx, yy, z, levels=[0, 0.5, 1], alpha=0.22, colors=["#4C78A8", "#E45756"])
        ax.contour(xx, yy, z, levels=[0.5], colors="black", linewidths=0.9)
        ax.scatter(x_train[VARIABLES_X[0]], x_train[VARIABLES_X[1]], c=y_train,
                   cmap="coolwarm", edgecolor="white", linewidth=0.25, s=13, alpha=0.7)
        ax.set(title=f"k = {k_mapa}", xlabel="Ingreso per cápita")
        ax.grid(alpha=0.15)
    ejes[0].set_ylabel("Años de educación del jefe")
    fig.suptitle("Regiones de clasificación de pobreza (puntos: hogares de entrenamiento)")
    guardar_figura(fig, "02_mapas_decision.png")
    print("Cada fondo muestra la clase estimada: azul No y rojo Sí; fronteras irregulares indican decisiones locales.")

    # 7. Ejemplo: se muestran los vecinos más cercanos de un hogar reservado para prueba.
    mensaje("PASO 7 | Inspeccionando vecinos de un hogar de prueba")
    hogar_pos = 0
    hogar = x_test.iloc[[hogar_pos]]
    distancia, posiciones = modelo_knn.named_steps["knn"].kneighbors(
        modelo_knn.named_steps["escalador"].transform(hogar), n_neighbors=mejor_k
    )
    vecinos_y = y_train.iloc[posiciones[0]].to_numpy()
    if mejor_peso == "distance":
        pesos = 1 / np.maximum(distancia[0], np.finfo(float).eps)
    else:
        pesos = np.ones(len(vecinos_y))
    voto_si = pesos[vecinos_y == 1].sum()
    voto_no = pesos[vecinos_y == 0].sum()
    prob_votos_si = voto_si / (voto_si + voto_no)
    vecinos = x_train.iloc[posiciones[0]][VARIABLES_X].reset_index(drop=True).copy()
    vecinos.insert(0, "hogar_prueba", "Ejemplo 1")
    vecinos.insert(0, "numero_vecino", np.arange(1, mejor_k + 1))
    vecinos["pobre_real"] = np.where(vecinos_y == 1, "Sí", "No")
    vecinos["distancia_escalada"] = distancia[0]
    vecinos["peso_voto"] = pesos
    vecinos["voto_positivo_ponderado"] = np.where(vecinos_y == 1, pesos, 0)
    vecinos["voto_negativo_ponderado"] = np.where(vecinos_y == 0, pesos, 0)
    vecinos_ejemplo = vecinos.copy()
    resumen_votos = pd.DataFrame([
        {"elemento": "ingreso_pc del hogar", "valor": float(hogar.iloc[0][VARIABLES_X[0]])},
        {"elemento": "educacion_jefe del hogar", "valor": float(hogar.iloc[0][VARIABLES_X[1]])},
        {"elemento": "pobre real del hogar", "valor": "Sí" if y_test.iloc[hogar_pos] else "No"},
        {"elemento": "votos ponderados Sí", "valor": float(voto_si)},
        {"elemento": "votos ponderados No", "valor": float(voto_no)},
        {"elemento": "probabilidad de Sí por proporción de votos", "valor": float(prob_votos_si)},
        {"elemento": "probabilidad Sí reportada por el modelo", "valor": float(prob_test_knn[hogar_pos])},
    ])
    vecinos_ejemplo = pd.concat([
        resumen_votos.assign(numero_vecino=np.nan, ingreso_pc=np.nan, educacion_jefe=np.nan,
                             pobre_real=np.nan, distancia_escalada=np.nan, peso_voto=np.nan,
                             voto_positivo_ponderado=np.nan, voto_negativo_ponderado=np.nan)
            .rename(columns={"elemento": "hogar_prueba", "valor": "detalle"})[
                ["hogar_prueba", "detalle", "numero_vecino", "ingreso_pc", "educacion_jefe", "pobre_real",
                 "distancia_escalada", "peso_voto", "voto_positivo_ponderado", "voto_negativo_ponderado"]],
        vecinos.assign(detalle="")[["hogar_prueba", "detalle", "numero_vecino", "ingreso_pc", "educacion_jefe",
                                     "pobre_real", "distancia_escalada", "peso_voto",
                                     "voto_positivo_ponderado", "voto_negativo_ponderado"]]
    ], ignore_index=True)
    print(f"Hogar de prueba: real={'Sí' if y_test.iloc[hogar_pos] else 'No'}; "
          f"probabilidad estimada de Sí={prob_test_knn[hogar_pos]:.3f}. Se guardan sus {mejor_k} vecinos.")

    # 8. Matriz de confusión para kNN con el umbral solicitado de 0,5.
    mensaje("PASO 8 | Construyendo matriz de confusión con umbral 0,5")
    pred_test_05 = (prob_test_knn >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred_test_05, labels=[1, 0])
    vp, fn, fp, vn = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predice Sí", "Predice No"])
    ax.set_yticks([0, 1], labels=["Real Sí", "Real No"])
    rótulos = [[f"VP\n{vp}\nSí correctamente detectado", f"FN\n{fn}\nSí que se escapó"],
               [f"FP\n{fp}\nFalsa alarma de Sí", f"VN\n{vn}\nNo correctamente detectado"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, rótulos[i][j], ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black")
    ax.set_title("Matriz de confusión de kNN (umbral 0,5)")
    ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
    guardar_figura(fig, "03_matriz_confusion.png")
    print(f"VP={vp}: positivos detectados; FN={fn}: positivos que se escaparon; "
          f"FP={fp}: falsas alarmas; VN={vn}: negativos correctamente reconocidos.")

    # 9. Comparación en entrenamiento y prueba contra modelos de referencia.
    mensaje("PASO 9 | Comparando métricas y revisando sobreajuste")
    preprocesador_num = Pipeline([("imputador", SimpleImputer(strategy="median")), ("escalador", StandardScaler())])
    modelo_log = Pipeline([("preparacion", preprocesador_num),
                           ("clasificador", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    modelo_arbol = Pipeline([("imputador", SimpleImputer(strategy="median")),
                             ("clasificador", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    modelo_base = None
    modelos = [("kNN", modelo_knn), ("Regresión logística", modelo_log), ("Árbol de decisión", modelo_arbol)]
    metricas_registros = []
    for nombre, modelo in modelos:
        if nombre != "kNN":
            modelo.fit(x_train, y_train)
        for grupo, x_eval, y_eval in [("entrenamiento", x_train, y_train), ("prueba", x_test, y_test)]:
            prob = modelo.predict_proba(x_eval)[:, 1]
            fila = metricas(nombre, y_eval, prob, 0.5)
            fila["conjunto"] = grupo
            metricas_registros.append(fila)
    clase_mayoritaria = int(y_train.mean() >= 0.5)
    for grupo, y_eval in [("entrenamiento", y_train), ("prueba", y_test)]:
        pred = np.full(len(y_eval), clase_mayoritaria)
        prob = pred.astype(float)
        metricas_registros.append({
            "modelo": "Línea base (clase mayoritaria)", "conjunto": grupo,
            "accuracy": accuracy_score(y_eval, pred),
            "precision": precision_score(y_eval, pred, zero_division=0),
            "recall_sensibilidad": recall_score(y_eval, pred, zero_division=0),
            "AUC": 0.5,
        })
    metricas_df = pd.DataFrame(metricas_registros)[
        ["modelo", "conjunto", "accuracy", "precision", "recall_sensibilidad", "AUC"]
    ]
    fila_knn_train = metricas_df.query("modelo == 'kNN' and conjunto == 'entrenamiento'").iloc[0]
    fila_knn_test = metricas_df.query("modelo == 'kNN' and conjunto == 'prueba'").iloc[0]
    brecha = float(fila_knn_train["accuracy"] - fila_knn_test["accuracy"])
    sobreajuste = "Posible sobreajuste: la exactitud de entrenamiento supera claramente la de prueba." if brecha > 0.10 else "No se observa una brecha grande de exactitud; revisar también otras métricas y validación."
    metricas_df["brecha_accuracy_train_menos_prueba_knn"] = np.where(metricas_df["modelo"] == "kNN", brecha, np.nan)
    metricas_df["lectura_sobreajuste_knn"] = np.where(metricas_df["modelo"] == "kNN", sobreajuste, "")
    print(metricas_df.to_string(index=False, float_format=lambda n: f"{n:.3f}"))
    print(sobreajuste)

    # 10. Curvas ROC y métricas para varios umbrales.
    mensaje("PASO 10 | Trazando ROC y evaluando umbrales")
    fig, ax = plt.subplots(figsize=(8, 6))
    for nombre, modelo in modelos:
        prob = modelo.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_valor = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC = {auc_valor:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC = 0,500)")
    ax.set(title="Curvas ROC en el conjunto de prueba", xlabel="Tasa de falsos positivos", ylabel="Sensibilidad")
    ax.grid(alpha=0.25); ax.legend(loc="lower right")
    guardar_figura(fig, "04_curva_roc.png")

    # Umbral recomendado se elige con probabilidades fuera de pliegue del entrenamiento,
    # nunca mirando el resultado de prueba. Se maximiza recall con precisión >= 0,25;
    # si ninguno lo consigue, se elige el de mayor recall y, en empate, mayor precisión.
    oof_prob = cross_val_predict(
        clone(modelo_knn), x_train, y_train, cv=pliegues, method="predict_proba", n_jobs=-1
    )[:, 1]
    candidatos = []
    for umbral in np.round(np.arange(0.1, 1.0, 0.1), 1):
        p = (oof_prob >= umbral).astype(int)
        candidatos.append({"umbral": float(umbral),
                           "sensibilidad_validacion": recall_score(y_train, p, zero_division=0),
                           "precision_validacion": precision_score(y_train, p, zero_division=0)})
    elegibles = [f for f in candidatos if f["precision_validacion"] >= 0.25]
    elegido = max(elegibles or candidatos,
                  key=lambda f: (f["sensibilidad_validacion"], f["precision_validacion"], -f["umbral"]))
    umbral_recomendado = elegido["umbral"]
    umbral_rows = []
    for umbral in np.round(np.arange(0.1, 1.0, 0.1), 1):
        fila = metricas("kNN", y_test, prob_test_knn, float(umbral))
        fila["umbral"] = float(umbral)
        fila["recall_validacion_entrenamiento"] = next(f["sensibilidad_validacion"] for f in candidatos if f["umbral"] == float(umbral))
        fila["precision_validacion_entrenamiento"] = next(f["precision_validacion"] for f in candidatos if f["umbral"] == float(umbral))
        fila["umbral_recomendado"] = "Sí" if float(umbral) == umbral_recomendado else ""
        umbral_rows.append(fila)
    umbrales = pd.DataFrame(umbral_rows)[
        ["umbral", "accuracy", "precision", "recall_sensibilidad", "AUC",
         "recall_validacion_entrenamiento", "precision_validacion_entrenamiento", "umbral_recomendado"]
    ]
    print(f"Umbral recomendado para dar prioridad a sensibilidad: {umbral_recomendado:.1f} "
          f"(seleccionado con predicciones de validación del entrenamiento; precisión mínima objetivo 0,25).")
    print("Al bajar el umbral se marcan más hogares como positivos: normalmente sube sensibilidad y también pueden subir FP.")

    # 11. Predicciones del conjunto de prueba, junto con clase real y probabilidad.
    mensaje("PASO 11 | Guardando las predicciones del conjunto de prueba")
    predicciones = x_test[VARIABLES_X].reset_index(drop=True).copy()
    predicciones.insert(0, "fila_origen_excel", indice_original[idx_test] + 2)
    predicciones["pobre_real"] = np.where(y_test.to_numpy() == 1, "Sí", "No")
    predicciones["probabilidad_Si"] = prob_test_knn
    predicciones["clase_predicha_umbral_0_5"] = np.where(prob_test_knn >= 0.5, "Sí", "No")
    predicciones["clase_predicha_umbral_recomendado"] = np.where(prob_test_knn >= umbral_recomendado, "Sí", "No")
    print(f"Se generaron {len(predicciones)} predicciones (una por hogar de prueba).")

    # Resumen y libro Excel: cada DataFrame se guarda en su propia pestaña.
    auc_knn = float(roc_auc_score(y_test, prob_test_knn))
    resumen = pd.DataFrame([
        {"cifra_clave": "filas del archivo", "valor": len(datos)},
        {"cifra_clave": "casos pobres Sí (%)", "valor": porcentaje_si},
        {"cifra_clave": "hogares de entrenamiento", "valor": len(x_train)},
        {"cifra_clave": "hogares de prueba", "valor": len(x_test)},
        {"cifra_clave": "k elegido", "valor": mejor_k},
        {"cifra_clave": "peso elegido", "valor": mejor_peso},
        {"cifra_clave": "accuracy kNN prueba (umbral 0,5)", "valor": float(fila_knn_test["accuracy"])},
        {"cifra_clave": "precision kNN prueba (umbral 0,5)", "valor": float(fila_knn_test["precision"])},
        {"cifra_clave": "sensibilidad kNN prueba (umbral 0,5)", "valor": float(fila_knn_test["recall_sensibilidad"])},
        {"cifra_clave": "AUC kNN prueba", "valor": auc_knn},
        {"cifra_clave": "umbral recomendado por sensibilidad", "valor": umbral_recomendado},
        {"cifra_clave": "brecha accuracy entrenamiento - prueba", "valor": brecha},
        {"cifra_clave": "lectura de sobreajuste", "valor": sobreajuste},
        {"cifra_clave": "limitación", "valor": "Asociación predictiva no demuestra causalidad; solo se usaron ingreso_pc y educacion_jefe."},
    ])
    hojas = {
        "resumen": resumen,
        "descripcion": descripcion,
        "escalado": escalado,
        "busqueda": busqueda,
        "vecinos_ejemplo": vecinos_ejemplo,
        "matriz_confusion": pd.DataFrame([
            {"real": "Sí", "predicho": "Sí", "celda": "VP", "casos": vp},
            {"real": "Sí", "predicho": "No", "celda": "FN", "casos": fn},
            {"real": "No", "predicho": "Sí", "celda": "FP", "casos": fp},
            {"real": "No", "predicho": "No", "celda": "VN", "casos": vn},
        ]),
        "metricas": metricas_df,
        "umbrales": umbrales,
        "predicciones": predicciones,
    }
    with pd.ExcelWriter(ARCHIVO_EXCEL, engine="openpyxl") as escritor:
        for hoja, tabla in hojas.items():
            tabla.to_excel(escritor, sheet_name=hoja, index=False)
    # Formato simple para que las hojas sean más fáciles de leer en Excel.
    libro = load_workbook(ARCHIVO_EXCEL)
    for hoja in libro.worksheets:
        hoja.freeze_panes = "A2"
        hoja.auto_filter.ref = hoja.dimensions
        for celda in hoja[1]:
            celda.font = Font(bold=True, color="FFFFFF")
            celda.fill = PatternFill("solid", fgColor="234F6F")
            celda.alignment = Alignment(wrap_text=True, vertical="center")
        hoja.row_dimensions[1].height = 30
        for col in hoja.columns:
            letra = get_column_letter(col[0].column)
            ancho = min(max(max(len(str(c.value or "")) for c in col[:100]) + 2, 12), 48)
            hoja.column_dimensions[letra].width = ancho
    libro.save(ARCHIVO_EXCEL)
    print(f"Excel guardado: {ARCHIVO_EXCEL}")

    # Informe breve generado con cifras de esta ejecución y guía para interpretar entregables.
    mejor_modelo_test = metricas_df[metricas_df["conjunto"] == "prueba"].sort_values("AUC", ascending=False).iloc[0]
    informe = f"""# Clasificación de pobreza de hogares con kNN

## Resultado principal

Se clasificaron hogares como pobres (Sí) usando únicamente `ingreso_pc` y `educacion_jefe`. La partición estratificada dejó {len(x_train)} hogares para entrenamiento y {len(x_test)} para prueba. La validación cruzada eligió **k = {mejor_k}** y ponderación **{mejor_peso}**. En prueba, kNN obtuvo accuracy **{fila_knn_test['accuracy']:.3f}**, precisión **{fila_knn_test['precision']:.3f}**, sensibilidad **{fila_knn_test['recall_sensibilidad']:.3f}** y AUC **{auc_knn:.3f}** con umbral 0,5. La línea base predice siempre la clase más común; compárela en la hoja `metricas`. El modelo con mayor AUC de prueba fue **{mejor_modelo_test['modelo']}** ({mejor_modelo_test['AUC']:.3f}).

## Lectura de los archivos

- `modelo_k_vecinos.py`: análisis reproducible desde la línea de comandos; crea las carpetas de resultados y visualizaciones si faltan.
- `modelo_k_vecinos.ipynb`: explicación didáctica paso a paso; ejecute las celdas en orden para reproducir el análisis.
- `datos/datos.xlsx`: copia del conjunto de hogares usado, hoja `hogares`.
- `resultados/k_vecinos_resultados.xlsx`: `resumen` reúne cifras clave; `descripcion` documenta filas, columnas y faltantes; `escalado` compara exactitud sin/con escala; `busqueda` contiene los resultados de k y pesos; `vecinos_ejemplo` muestra los vecinos y votos; `matriz_confusion` guarda VP, FN, FP y VN; `metricas` compara kNN, regresión logística, árbol y línea base en entrenamiento y prueba; `umbrales` compara umbrales de 0,1 a 0,9 y marca el recomendado; `predicciones` guarda clase real, probabilidad y predicción de prueba.
- `visualizaciones/01_exactitud_k.png`: exactitud de entrenamiento y validación según k; una brecha amplia sugiere sobreajuste y k=1 puede memorizar.
- `visualizaciones/02_mapas_decision.png`: regiones estimadas para k=1, el k elegido y k=75.
- `visualizaciones/03_matriz_confusion.png`: VP detecta un Sí, FN es un Sí escapado, FP es una falsa alarma y VN identifica correctamente un No.
- `visualizaciones/04_curva_roc.png`: sensibilidad frente a falsos positivos; cuanto más cerca de la esquina superior izquierda, mejor separación; AUC resume esa capacidad.

## Umbral y sobreajuste

Como es más costoso dejar escapar un caso positivo, se recomienda probar el umbral **{umbral_recomendado:.1f}**, escogido para favorecer la sensibilidad usando predicciones de validación del conjunto de entrenamiento y una precisión mínima objetivo de 0,25. Confirme su conveniencia con el costo operativo de las falsas alarmas. {sobreajuste} La brecha de accuracy entre entrenamiento y prueba fue {brecha:.3f}.

## Párrafo para informe de investigación

Se estimó un clasificador kNN para identificar hogares clasificados como pobres a partir del ingreso per cápita y los años de educación del jefe del hogar. Los hiperparámetros se seleccionaron mediante validación cruzada estratificada de cinco pliegues y el desempeño se evaluó en una muestra de prueba separada, comparándolo con una línea base, regresión logística y árbol de decisión. Las métricas observadas describen capacidad predictiva en estos datos y no deben interpretarse como evidencia causal: una asociación entre las variables y la pobreza no prueba que una variable cause la otra. Además, el modelo usa solo dos predictores, depende de la calidad y representatividad de la base, y sus resultados pueden cambiar en otras poblaciones o periodos; el umbral recomendado también implica un equilibrio entre positivos detectados y falsas alarmas.
"""
    (CARPETA / "informe_k_vecinos.md").write_text(informe, encoding="utf-8")
    print(f"Informe guardado: {CARPETA / 'informe_k_vecinos.md'}")
    mensaje("ANÁLISIS TERMINADO | Archivos actualizados en resultados/ y visualizaciones/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()

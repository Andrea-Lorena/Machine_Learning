"""Regresión logística múltiple para explicar y predecir la deserción estudiantil."""

# Se comprueba e instala cada dependencia antes de importar las librerías del análisis.
import importlib.util
import subprocess
import sys

DEPENDENCIAS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
}
faltantes = [paquete for modulo, paquete in DEPENDENCIAS.items() if importlib.util.find_spec(modulo) is None]
if faltantes:
    print("Instalando dependencias faltantes:", ", ".join(faltantes))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *faltantes])

# Importaciones del análisis, después de comprobar las dependencias.
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def normalizar_si_no(serie):
    """Homogeneiza Sí/Si/No, incluidos textos con problemas de codificación comunes."""
    texto = serie.astype("string").str.strip().str.lower()
    texto = texto.str.replace("í", "i", regex=False).str.replace("�", "i", regex=False)
    return texto.map({"si": 1, "no": 0})


def guardar_figura(figura, ruta):
    """Guarda una figura PNG y la cierra para liberar memoria."""
    figura.tight_layout()
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def ejecutar_analisis():
    # Las rutas se resuelven desde la carpeta donde está este script.
    carpeta = Path(__file__).resolve().parent
    carpeta_datos = carpeta / "datos"
    carpeta_resultados = carpeta / "resultados"
    carpeta_figuras = carpeta / "visualizaciones"
    carpeta_resultados.mkdir(exist_ok=True)
    carpeta_figuras.mkdir(exist_ok=True)
    archivo_datos = carpeta_datos / "datos.xlsx"
    archivo_salida = carpeta_resultados / "regresion_logistica_resultados.xlsx"

    # Paso 1: carga, revisión inicial y descripción de la clase positiva.
    datos = pd.read_excel(archivo_datos, sheet_name="estudiantes")
    faltantes_por_columna = datos.isna().sum()
    total_faltantes = int(faltantes_por_columna.sum())
    positivo = normalizar_si_no(datos["desercion"])
    if positivo.isna().any():
        raise ValueError("La columna desercion debe contener únicamente Sí y No.")
    porcentaje_si = float(positivo.mean() * 100)
    print("PASO 1 | Datos cargados desde", archivo_datos)
    print(f"Filas: {len(datos)} | Columnas: {datos.shape[1]} | Celdas faltantes: {total_faltantes}")
    print(f"Casos de deserción = Sí: {int(positivo.sum())} ({porcentaje_si:.1f} %)")

    descripcion = pd.DataFrame({
        "indicador": ["filas", "columnas", "celdas_faltantes", "porcentaje_desercion_si", "casos_desercion_si", "casos_desercion_no"],
        "valor": [len(datos), datos.shape[1], total_faltantes, porcentaje_si, int(positivo.sum()), int((positivo == 0).sum())],
    })
    faltantes = faltantes_por_columna.rename_axis("columna").reset_index(name="datos_faltantes")
    descripcion = pd.concat([descripcion, pd.DataFrame({"indicador": ["faltantes_por_columna"], "valor": ["Ver detalle_faltantes"]})], ignore_index=True)

    # Paso 2: se construyen las variables del modelo: Sí=1, No=0.
    predictores = ["promedio_semestre", "distancia_km", "trabaja", "beca"]
    modelado = pd.DataFrame(index=datos.index)
    modelado["promedio_semestre"] = pd.to_numeric(datos["promedio_semestre"], errors="coerce")
    modelado["distancia_km"] = pd.to_numeric(datos["distancia_km"], errors="coerce")
    modelado["trabaja"] = normalizar_si_no(datos["trabaja"])
    modelado["beca"] = normalizar_si_no(datos["beca"])
    modelado["desercion"] = positivo
    if modelado.isna().any().any():
        raise ValueError("Hay valores faltantes o categorías distintas de Sí/No en las columnas del modelo.")
    X = modelado[predictores].astype(float)
    y = modelado["desercion"].astype(int)
    print("PASO 2 | Codificación: desercion Sí=1 / No=0; trabaja Sí=1 / No=0 (referencia: No); beca Sí=1 / No=0 (referencia: No).")

    # Paso 3: se separan 75 % para entrenar y 25 % para evaluar, conservando la proporción de clases.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"PASO 3 | Entrenamiento: {len(y_train)} casos ({y_train.mean():.1%} Sí); prueba: {len(y_test)} ({y_test.mean():.1%} Sí).")

    # Paso 4: el modelo de scikit-learn estima probabilidades; statsmodels aporta p e intervalos.
    modelo = LogisticRegression(random_state=42, max_iter=1000)
    modelo.fit(X_train, y_train)
    X_train_sm = sm.add_constant(X_train, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo_sm = sm.Logit(y_train, X_train_sm).fit(disp=False, maxiter=200)
    parametros = modelo_sm.params
    intervalos = modelo_sm.conf_int()
    pvalores = modelo_sm.pvalues
    etiquetas = {
        "const": "Intercepto (todas las X=0)",
        "promedio_semestre": "promedio_semestre (por +1 punto)",
        "distancia_km": "distancia_km (por +1 km)",
        "trabaja": "trabaja: Sí frente a No (referencia No)",
        "beca": "beca: Sí frente a No (referencia No)",
    }
    interpretaciones = []
    for nombre, beta in parametros.items():
        if nombre == "const":
            texto = "Intercepto: log-chances cuando promedio=0, distancia=0, trabaja=No y beca=No; puede no ser un perfil realista."
        else:
            direccion = "se asocia con mayores" if beta > 0 else "se asocia con menores" if beta < 0 else "no cambia"
            texto = f"El signo {beta:+.3f} indica que, manteniendo las demás variables constantes, {etiquetas[nombre]} {direccion} chances estimadas de deserción (asociación, no causalidad)."
        interpretaciones.append(texto)
    coeficientes = pd.DataFrame({
        "variable": [etiquetas.get(n, n) for n in parametros.index],
        "beta": parametros.values,
        "IC95_inferior_beta": intervalos[0].values,
        "IC95_superior_beta": intervalos[1].values,
        "valor_p": pvalores.values,
        "odds_ratio_exp_beta": np.exp(parametros.values),
        "exp_IC95_inferior": np.exp(intervalos[0].values),
        "exp_IC95_superior": np.exp(intervalos[1].values),
        "interpretacion_del_signo": interpretaciones,
    })
    ecuacion = "p = 1 / (1 + e^-z), donde z = " + " + ".join(f"({parametros[n]:.6f} * {n})" for n in parametros.index if n != "const") + f" + ({parametros['const']:.6f})"
    print("PASO 4 | Modelos ajustados en entrenamiento (scikit-learn y statsmodels).")
    print("Ecuación estimada:", ecuacion)

    # La curva sigmoide muestra el efecto de promedio_semestre y deja las otras X fijas.
    rango_promedio = np.linspace(X["promedio_semestre"].min(), X["promedio_semestre"].max(), 250)
    perfil_sigmoide = pd.DataFrame({
        "promedio_semestre": rango_promedio,
        "distancia_km": X["distancia_km"].mean(),
        "trabaja": 0,  # Categoría de referencia: No.
        "beca": 0,     # Categoría de referencia: No.
    })
    prob_sigmoide = modelo.predict_proba(perfil_sigmoide[predictores])[:, 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rango_promedio, prob_sigmoide, color="#2878B5", lw=2.5)
    ax.set(
        title="Curva sigmoide: probabilidad estimada de deserción",
        xlabel="Promedio del semestre",
        ylabel="Probabilidad estimada de deserción (Sí)",
        ylim=(0, 1),
    )
    ax.text(
        0.02, 0.03,
        "Distancia fijada en su media; trabaja=No y beca=No",
        transform=ax.transAxes, fontsize=9, color="#444444",
    )
    ax.grid(alpha=0.25)
    guardar_figura(fig, carpeta_figuras / "04_curva_sigmoide.png")
    print("PASO 4 | Curva sigmoide guardada: visualizaciones/04_curva_sigmoide.png (otras variables fijas).")

    # Paso 5: distribución de probabilidades de prueba, separada por la clase real.
    prob_test = modelo.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for clase, nombre, color in [(0, "No desertó (No)", "#2878B5"), (1, "Desertó (Sí)", "#D9534F")]:
        ax.hist(prob_test[y_test.to_numpy() == clase], bins=20, alpha=0.60, label=nombre, color=color, edgecolor="white")
    ax.set(title="Probabilidad predicha de deserción en prueba", xlabel="Probabilidad estimada de deserción (Sí)", ylabel="Número de estudiantes")
    ax.legend()
    guardar_figura(fig, carpeta_figuras / "01_probabilidades.png")
    print("PASO 5 | Histograma guardado: visualizaciones/01_probabilidades.png")

    # Paso 6: la matriz usa filas reales y columnas predichas con umbral 0,5.
    umbral_base = 0.5
    pred_test = (prob_test >= umbral_base).astype(int)
    matriz = confusion_matrix(y_test, pred_test, labels=[1, 0])
    vp, fn, fp, vn = int(matriz[0, 0]), int(matriz[0, 1]), int(matriz[1, 0]), int(matriz[1, 1])
    tabla_confusion = pd.DataFrame([
        {"celda": "VP", "real": "Sí", "predicho": "Sí", "conteo": vp, "significado": "Positivo verdadero: desertó y el modelo lo detectó."},
        {"celda": "FN", "real": "Sí", "predicho": "No", "conteo": fn, "significado": "Falso negativo: desertó y el modelo no lo detectó (caso que preocupa)."},
        {"celda": "FP", "real": "No", "predicho": "Sí", "conteo": fp, "significado": "Falso positivo: no desertó, pero el modelo alertó."},
        {"celda": "VN", "real": "No", "predicho": "No", "conteo": vn, "significado": "Verdadero negativo: no desertó y el modelo acertó."},
    ])
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    imagen = ax.imshow([[vp, fn], [fp, vn]], cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predicho: Sí", "Predicho: No"])
    ax.set_yticks([0, 1], labels=["Real: Sí", "Real: No"])
    ax.set_title("Matriz de confusión (umbral = 0,5)")
    for (i, j), (sigla, valor) in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [("VP", vp), ("FN", fn), ("FP", fp), ("VN", vn)]):
        ax.text(j, i, f"{sigla}\n{valor}", ha="center", va="center", fontsize=14, color="black")
    fig.colorbar(imagen, ax=ax, label="Conteo")
    guardar_figura(fig, carpeta_figuras / "02_matriz_confusion.png")
    print(f"PASO 6 | Matriz guardada. VP={vp}, FN={fn}, FP={fp}, VN={vn}.")

    # Paso 7: se comparan las métricas de train y test con una predicción base constante.
    clase_mayoritaria = int(y_train.value_counts().idxmax())
    prob_base = np.full(len(y_test), float(clase_mayoritaria))
    def calcular_metricas(nombre, reales, probabilidades, predicciones):
        return {
            "conjunto": nombre,
            "accuracy": accuracy_score(reales, predicciones),
            "precision": precision_score(reales, predicciones, zero_division=0),
            "recall_sensibilidad": recall_score(reales, predicciones, zero_division=0),
            "AUC_ROC": roc_auc_score(reales, probabilidades),
        }
    prob_train = modelo.predict_proba(X_train)[:, 1]
    pred_train = (prob_train >= umbral_base).astype(int)
    pred_base_train = np.full(len(y_train), clase_mayoritaria)
    pred_base_test = np.full(len(y_test), clase_mayoritaria)
    metricas = pd.DataFrame([
        calcular_metricas("Modelo - entrenamiento", y_train, prob_train, pred_train),
        calcular_metricas("Modelo - prueba", y_test, prob_test, pred_test),
        calcular_metricas("Línea base - entrenamiento (siempre clase mayoritaria)", y_train, np.full(len(y_train), float(clase_mayoritaria)), pred_base_train),
        calcular_metricas("Línea base - prueba (siempre clase mayoritaria)", y_test, prob_base, pred_base_test),
    ])
    diferencia_auc = float(metricas.loc[0, "AUC_ROC"] - metricas.loc[1, "AUC_ROC"])
    diferencia_accuracy = float(metricas.loc[0, "accuracy"] - metricas.loc[1, "accuracy"])
    sobreajuste = "Posible sobreajuste: revisar la brecha entre entrenamiento y prueba." if diferencia_auc > 0.10 or diferencia_accuracy > 0.10 else "No se observa una brecha grande entre entrenamiento y prueba; no hay señal clara de sobreajuste con esta comparación."
    explicacion_metricas = pd.DataFrame([
        {"metrica": "accuracy", "lectura_sencilla": "De cada 10 estudiantes, cuántos clasifica correctamente, aproximadamente."},
        {"metrica": "precision", "lectura_sencilla": "De cada 10 alertas de deserción, cuántas corresponden a estudiantes que sí desertan."},
        {"metrica": "recall_sensibilidad", "lectura_sencilla": "De cada 10 estudiantes que sí desertan, a cuántos detecta el modelo."},
        {"metrica": "AUC_ROC", "lectura_sencilla": "Probabilidad de ordenar a un desertor por encima de un no desertor; 0,5 es azar y 1 es separación perfecta."},
    ])
    print("PASO 7 | Métricas (umbral 0,5) y línea base calculadas.")
    print(metricas.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Comparación train-prueba:", sobreajuste)

    # Paso 8: curva ROC y medidas de umbral se calculan sobre prueba para mostrar el comportamiento observado.
    fpr, tpr, roc_thresholds = roc_curve(y_test, prob_test)
    auc_prueba = roc_auc_score(y_test, prob_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2878B5", lw=2, label=f"Modelo (AUC = {auc_prueba:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    for t in (0.2, 0.5, 0.8):
        pred_t = (prob_test >= t).astype(int)
        cm_t = confusion_matrix(y_test, pred_t, labels=[0, 1])
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        sensibilidad_t = tp_t / (tp_t + fn_t) if tp_t + fn_t else 0
        falsas_alarmas_t = fp_t / (fp_t + tn_t) if fp_t + tn_t else 0
        ax.scatter(falsas_alarmas_t, sensibilidad_t, s=55, label=f"Umbral {t:.1f}")
        ax.annotate(f"{t:.1f}", (falsas_alarmas_t, sensibilidad_t), xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="Tasa de falsas alarmas (FPR)", ylabel="Sensibilidad (TPR)", title="Curva ROC en el conjunto de prueba", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    guardar_figura(fig, carpeta_figuras / "03_curva_roc.png")

    filas_umbral = []
    for t in np.round(np.arange(0.1, 1.0, 0.1), 1):
        p = (prob_test >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, p, labels=[0, 1]).ravel()
        filas_umbral.append({
            "umbral": t,
            "sensibilidad_recall": tp_t / (tp_t + fn_t) if tp_t + fn_t else 0,
            "precision": tp_t / (tp_t + fp_t) if tp_t + fp_t else 0,
            "falsas_alarmas_FPR": fp_t / (fp_t + tn_t) if fp_t + tn_t else 0,
            "VP": tp_t, "FN": fn_t, "FP": fp_t, "VN": tn_t,
        })
    tabla_umbrales = pd.DataFrame(filas_umbral)
    # Se visualiza cómo cambian precisión y sensibilidad cuando se mueve el umbral.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tabla_umbrales["umbral"], tabla_umbrales["precision"], marker="o", lw=2, label="Precisión")
    ax.plot(tabla_umbrales["umbral"], tabla_umbrales["sensibilidad_recall"], marker="o", lw=2, label="Sensibilidad (recall)")
    ax.set(
        title="Precisión y sensibilidad según el umbral",
        xlabel="Umbral de clasificación",
        ylabel="Proporción",
        xticks=tabla_umbrales["umbral"],
        ylim=(0, 1.05),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    guardar_figura(fig, carpeta_figuras / "05_precision_sensibilidad_umbral.png")
    print("PASO 8 | Gráfico precisión-sensibilidad guardado: visualizaciones/05_precision_sensibilidad_umbral.png")
    # Se prioriza sensibilidad; en empates se elige el umbral con mayor precisión.
    recomendado = tabla_umbrales.sort_values(["sensibilidad_recall", "precision", "umbral"], ascending=[False, False, False]).iloc[0]
    recomendacion = f"Umbral sugerido: {recomendado['umbral']:.1f} (sensibilidad {recomendado['sensibilidad_recall']:.1%}, precisión {recomendado['precision']:.1%}, falsas alarmas {recomendado['falsas_alarmas_FPR']:.1%} en prueba). Se priorizó sensibilidad; confirme el costo operativo de las alertas y seleccione el umbral final con datos de validación antes de usarlo en producción."
    print("PASO 8 | Curva ROC guardada. La tabla de umbrales usa el conjunto de prueba.")
    print(recomendacion)

    # Paso 9: archivo de predicciones para revisar cada caso de prueba.
    predicciones = pd.DataFrame({
        "indice_fila_origen": X_test.index,
        "valor_real": np.where(y_test.to_numpy() == 1, "Sí", "No"),
        "probabilidad_desercion_si": prob_test,
        "clase_predicha_umbral_0_5": np.where(pred_test == 1, "Sí", "No"),
    }).sort_values("indice_fila_origen")
    print("PASO 9 | Predicciones de prueba preparadas:", len(predicciones), "casos.")

    # Se agrega una hoja de resumen y se guardan todas las tablas en un solo Excel.
    resumen = pd.DataFrame([
        {"cifra_clave": "Filas analizadas", "valor": len(datos), "lectura": "Registros disponibles en la hoja estudiantes."},
        {"cifra_clave": "Porcentaje desercion Sí", "valor": porcentaje_si, "lectura": "Proporción de clase positiva en los datos completos (%)."},
        {"cifra_clave": "Accuracy modelo prueba", "valor": float(metricas.loc[metricas.conjunto == "Modelo - prueba", "accuracy"].iloc[0]), "lectura": "Aciertos totales en prueba con umbral 0,5."},
        {"cifra_clave": "Precision modelo prueba", "valor": float(metricas.loc[metricas.conjunto == "Modelo - prueba", "precision"].iloc[0]), "lectura": "Proporción de alertas correctas en prueba."},
        {"cifra_clave": "Sensibilidad modelo prueba", "valor": float(metricas.loc[metricas.conjunto == "Modelo - prueba", "recall_sensibilidad"].iloc[0]), "lectura": "Proporción de desertores detectados en prueba."},
        {"cifra_clave": "AUC modelo prueba", "valor": auc_prueba, "lectura": "Capacidad de ordenar positivos antes que negativos."},
        {"cifra_clave": "AUC diferencia train-prueba", "valor": diferencia_auc, "lectura": "Brecha usada junto con accuracy para comentar sobreajuste."},
        {"cifra_clave": "Revisión sobreajuste", "valor": sobreajuste, "lectura": "Comparación descriptiva; no es una prueba definitiva."},
        {"cifra_clave": "Umbral sugerido sensibilidad", "valor": float(recomendado["umbral"]), "lectura": recomendacion},
        {"cifra_clave": "Ecuación logística estimada", "valor": ecuacion, "lectura": "z contiene intercepto y coeficientes; p transforma z en probabilidad."},
        {"cifra_clave": "Positivo verdadero (VP)", "valor": vp, "lectura": "Desertó y el modelo detectó la deserción."},
        {"cifra_clave": "Falso negativo (FN)", "valor": fn, "lectura": "Desertó, pero el modelo no lo detectó."},
        {"cifra_clave": "Falso positivo (FP)", "valor": fp, "lectura": "No desertó, pero recibió una alerta."},
        {"cifra_clave": "Verdadero negativo (VN)", "valor": vn, "lectura": "No desertó y el modelo tampoco alertó."},
    ])
    with pd.ExcelWriter(archivo_salida, engine="openpyxl") as escritor:
        descripcion.to_excel(escritor, sheet_name="descripcion", index=False)
        faltantes.to_excel(escritor, sheet_name="detalle_faltantes", index=False)
        coeficientes.to_excel(escritor, sheet_name="coeficientes", index=False)
        tabla_confusion.to_excel(escritor, sheet_name="matriz_confusion", index=False)
        metricas.to_excel(escritor, sheet_name="metricas", index=False)
        explicacion_metricas.to_excel(escritor, sheet_name="lectura_metricas", index=False)
        tabla_umbrales.to_excel(escritor, sheet_name="umbrales", index=False)
        predicciones.to_excel(escritor, sheet_name="predicciones", index=False)
        resumen.to_excel(escritor, sheet_name="resumen", index=False)
    print("RESULTADO | Excel guardado en", archivo_salida)
    print("RESULTADO | Figuras PNG guardadas en", carpeta_figuras)


if __name__ == "__main__":
    ejecutar_analisis()

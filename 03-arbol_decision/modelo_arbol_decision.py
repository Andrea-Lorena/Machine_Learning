"""Clasificación de hogares pobres mediante árbol de decisión."""

# PASO: dependencias (antes de cualquier import adicional)
import importlib.util
import subprocess
import sys

PAQUETES = {"pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
            "matplotlib": "matplotlib", "openpyxl": "openpyxl"}
faltantes = [pip for modulo, pip in PAQUETES.items()
             if importlib.util.find_spec(modulo) is None]
if faltantes:
    print("Instalando paquetes faltantes:", faltantes)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *faltantes])

# PASO: imports y configuración
from pathlib import Path
import unicodedata
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # En el cuaderno se ejecuta desde la carpeta del proyecto.
    BASE_DIR = Path.cwd().resolve()
DATA_DIR, RESULTS_DIR, PLOTS_DIR = (BASE_DIR / "datos", BASE_DIR / "resultados",
                                    BASE_DIR / "visualizaciones")
for carpeta in (DATA_DIR, RESULTS_DIR, PLOTS_DIR):
    carpeta.mkdir(parents=True, exist_ok=True)
RUTA_DATOS = DATA_DIR / "datos.xlsx"
RUTA_EXCEL = RESULTS_DIR / "arbol_decision_resultados.xlsx"
VARS = ["ingreso_pc", "educacion_jefe", "hacinamiento", "acueducto",
        "piso_tierra", "internet", "zona"]

def normalizar(valor):
    texto = unicodedata.normalize("NFKD", str(valor))
    return "".join(c for c in texto if not unicodedata.combining(c)).strip().lower()

def guardar(figura, nombre):
    figura.savefig(PLOTS_DIR / nombre, dpi=150, bbox_inches="tight")
    plt.close(figura)

def calcular_metricas(y_real, prob, pred):
    return {"accuracy": accuracy_score(y_real, pred),
            "precision": precision_score(y_real, pred, zero_division=0),
            "recall": recall_score(y_real, pred, zero_division=0),
            "AUC": roc_auc_score(y_real, prob) if len(np.unique(y_real)) == 2 else np.nan}

# PASO: carga y descripción de los datos
print("PASO 1/11 | Cargando datos y revisando su calidad...")
if not RUTA_DATOS.exists():
    raise FileNotFoundError(f"No se encontró {RUTA_DATOS}. Coloque datos.xlsx dentro de datos/.")
datos = pd.read_excel(RUTA_DATOS, sheet_name="hogares", engine="openpyxl")
faltan = [c for c in VARS + ["pobre"] if c not in datos.columns]
if faltan:
    raise ValueError(f"Faltan columnas requeridas: {faltan}")
print(f"  Filas: {len(datos):,}; columnas: {len(datos.columns)}; faltantes: {datos.isna().sum().sum()}")

# PASO: convertir la respuesta a 0/1 y separar X numéricas y categóricas
print("PASO 2/11 | Codificando pobreza y variables de texto...")
y_txt = datos["pobre"].map(normalizar)
if not set(y_txt.dropna().unique()).issubset({"si", "no"}):
    raise ValueError(f"Se esperaba Sí/No en pobre; valores: {sorted(y_txt.unique())}")
y = y_txt.map({"no": 0, "si": 1}).astype(int)
X = datos[VARS].copy()
cat = [c for c in VARS if not pd.api.types.is_numeric_dtype(X[c])]
num = [c for c in VARS if c not in cat]
porcentaje_si = y.mean() * 100
descripcion = pd.DataFrame({"indicador": ["filas", "columnas", "celdas_faltantes",
    "casos_pobre_Si", "porcentaje_pobre_Si", "casos_pobre_No", "predictoras"],
    "valor": [len(datos), len(datos.columns), int(datos.isna().sum().sum()), int(y.sum()),
              porcentaje_si, int((y == 0).sum()), ", ".join(VARS)]})
print(f"  Sí: {int(y.sum())} de {len(y)} hogares ({porcentaje_si:.1f} %). "
      f"Categóricas codificadas con indicadores 0/1: {cat}")

# PASO: separar entrenamiento y prueba de forma estratificada
print("PASO 3/11 | Dividiendo 75 % para entrenamiento y 25 % para prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
    random_state=RANDOM_STATE, stratify=y)
print(f"  Entrenamiento: {len(y_train)}; prueba: {len(y_test)}.")

# PASO: ajustar preprocesamiento y seleccionar parámetros con validación cruzada
print("PASO 4/11 | Buscando parámetros con 5 pliegues y AUC...")
pre = ColumnTransformer([("texto", OneHotEncoder(handle_unknown="ignore"), cat),
                         ("numeros", "passthrough", num)],
                        verbose_feature_names_out=False)
pipe = Pipeline([("pre", pre), ("arbol", DecisionTreeClassifier(random_state=RANDOM_STATE))])
grid = {"arbol__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "arbol__min_samples_leaf": [5, 10, 20, 30],
        "arbol__ccp_alpha": [0, .001, .005, .01]}
busqueda = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1,
                        return_train_score=True, refit=True)
busqueda.fit(X_train, y_train)
modelo = busqueda.best_estimator_
busqueda_df = pd.DataFrame(busqueda.cv_results_).sort_values("rank_test_score")
arbol = modelo.named_steps["arbol"]
features = modelo.named_steps["pre"].get_feature_names_out()
print(f"  Mejor configuración: {busqueda.best_params_}; AUC CV={busqueda.best_score_:.3f}.")

# PASO: dibujar el árbol final
fig, ax = plt.subplots(figsize=(24, 14))
plot_tree(arbol, feature_names=features, class_names=["No", "Sí"], filled=True,
          rounded=True, proportion=True, precision=2, fontsize=8, ax=ax)
ax.set_title("Árbol de decisión: predicción de pobreza (Sí)")
guardar(fig, "01_arbol.png")
print("  Árbol guardado en visualizaciones/01_arbol.png.")

# PASO: reglas y tres hojas con más positivos
print("PASO 5/11 | Extrayendo reglas y explicando las hojas principales...")
regla_texto = export_text(arbol, feature_names=list(features), decimals=2, show_weights=True)
t = arbol.tree_
hojas = np.flatnonzero(t.children_left == t.children_right)
lista_hojas = []
for h in hojas:
    n = int(round(t.n_node_samples[h]))
    valores = t.value[h][0]
    # En scikit-learn reciente, tree_.value guarda proporciones de clase.
    positivos = int(round(valores[1] * n)) if len(valores) > 1 else 0
    condiciones, nodo = [], int(h)
    while nodo != 0:
        padre = int(np.flatnonzero((t.children_left == nodo) | (t.children_right == nodo))[0])
        nombre, corte = features[t.feature[padre]], t.threshold[padre]
        condiciones.append(f"{nombre} {'≤' if t.children_left[padre] == nodo else '>'} {corte:.2f}")
        nodo = padre
    lista_hojas.append({"nodo_hoja": int(h), "casos_entrenamiento": n,
        "casos_Si_entrenamiento": positivos,
        "porcentaje_Si_en_hoja": 100 * positivos / n if n else 0,
        "regla": " Y ".join(reversed(condiciones)) or "Todos los hogares"})
hojas_df = pd.DataFrame(lista_hojas).sort_values(
    ["casos_Si_entrenamiento", "casos_entrenamiento"], ascending=False)
top3 = hojas_df.head(3)
explicacion_reglas = "\n".join(f"{i}. Si se cumple {r.regla}, la hoja reunió "
    f"{r.casos_Si_entrenamiento} casos Sí de {r.casos_entrenamiento} hogares "
    f"({r.porcentaje_Si_en_hoja:.1f} %)." for i, r in enumerate(top3.itertuples(), 1))
reglas_df = pd.DataFrame({"seccion": ["Reglas completas (export_text)", "Tres hojas con más casos Sí"],
                          "contenido": [regla_texto, explicacion_reglas]})
print(explicacion_reglas)

# PASO: curva de exactitud por profundidad
print("PASO 6/11 | Comparando exactitud para profundidades de 1 a 15...")
xtr, xte = pre.fit_transform(X_train), pre.transform(X_test)
prof_reg = []
for d in range(1, 16):
    m = DecisionTreeClassifier(max_depth=d,
        min_samples_leaf=busqueda.best_params_["arbol__min_samples_leaf"],
        ccp_alpha=busqueda.best_params_["arbol__ccp_alpha"], random_state=RANDOM_STATE)
    m.fit(xtr, y_train)
    prof_reg.append({"max_depth": d, "accuracy_entrenamiento": accuracy_score(y_train, m.predict(xtr)),
                     "accuracy_prueba": accuracy_score(y_test, m.predict(xte))})
profundidad_df = pd.DataFrame(prof_reg)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(profundidad_df.max_depth, profundidad_df.accuracy_entrenamiento, marker="o", label="Entrenamiento")
ax.plot(profundidad_df.max_depth, profundidad_df.accuracy_prueba, marker="o", label="Prueba")
ax.set(xlabel="Profundidad máxima", ylabel="Exactitud", title="Exactitud por profundidad")
ax.set_xticks(range(1, 16)); ax.grid(alpha=.25); ax.legend()
guardar(fig, "02_profundidad.png")
brecha = profundidad_df.iloc[-1].accuracy_entrenamiento - profundidad_df.iloc[-1].accuracy_prueba
texto_sobreajuste = ("A profundidad alta se observa una brecha entre entrenamiento y prueba, señal posible de sobreajuste: "
    "el árbol aprende detalles que no se generalizan." if brecha > .08 else
    "No se observa una gran brecha entre entrenamiento y prueba a profundidad 15; la selección final se basa en CV.")
print("  " + texto_sobreajuste)

# PASO: matriz de confusión con umbral 0,5
print("PASO 7/11 | Calculando matriz de confusión de prueba con umbral 0,5...")
prob_arbol_test = modelo.predict_proba(X_test)[:, 1]
pred_arbol_test = (prob_arbol_test >= .5).astype(int)
tn, fp, fn, vp = confusion_matrix(y_test, pred_arbol_test, labels=[0, 1]).ravel()
matriz_df = pd.DataFrame([[vp, fn], [fp, tn]], index=["Real Sí", "Real No"],
                         columns=["Predicho Sí", "Predicho No"])
fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow([[vp, fn], [fp, tn]], cmap="Blues")
ax.set_xticks([0, 1], ["Predicho Sí", "Predicho No"]); ax.set_yticks([0, 1], ["Real Sí", "Real No"])
for (i, j), valor, rotulo in [((0,0), vp, "VP"), ((0,1), fn, "FN"), ((1,0), fp, "FP"), ((1,1), tn, "VN")]:
    ax.text(j, i, f"{rotulo}\n{valor}", ha="center", va="center", fontsize=13)
ax.set_title("Matriz de confusión en prueba (umbral 0,5)"); guardar(fig, "03_matriz_confusion.png")
print(f"  VP={vp}: detectados; FN={fn}: positivos que se escaparon; FP={fp}: falsas alarmas; VN={tn}: negativos correctos.")

# PASO: medir árbol, regresión logística y línea base
print("PASO 8/11 | Comparando métricas de árbol, logística y clase más común...")
logistica = Pipeline([("pre", pre), ("escala", StandardScaler(with_mean=False)),
                      ("logistica", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
logistica.fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE).fit(X_train, y_train)
modelos = {"Árbol": modelo, "Logística": logistica, "Línea base (clase más común)": base}
reg_metricas = []
for nombre, m in modelos.items():
    for conjunto, xx, yy in [("Entrenamiento", X_train, y_train), ("Prueba", X_test, y_test)]:
        p = m.predict_proba(xx)[:, 1]; clase = (p >= .5).astype(int)
        reg_metricas.append({"modelo": nombre, "conjunto": conjunto, **calcular_metricas(yy, p, clase)})
metricas_df = pd.DataFrame(reg_metricas)
print(metricas_df[metricas_df.conjunto == "Prueba"].to_string(index=False, float_format=lambda z: f"{z:.3f}"))

# PASO: ROC y evaluación de umbrales
print("PASO 9/11 | Trazando curvas ROC y buscando umbral sensible...")
fig, ax = plt.subplots(figsize=(7, 6))
for nombre, m in [("Árbol", modelo), ("Logística", logistica)]:
    p = m.predict_proba(X_test)[:, 1]; fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_test, p):.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar (AUC=0,500)")
ax.set(xlabel="Tasa de falsos positivos", ylabel="Sensibilidad", title="Curvas ROC en prueba")
ax.legend(); ax.grid(alpha=.25); guardar(fig, "04_curva_roc.png")
umbrales_reg = []
for u in np.arange(.1, 1., .1):
    pred = (prob_arbol_test >= u).astype(int)
    umbrales_reg.append({"umbral": round(float(u), 1), "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "sensibilidad_recall": recall_score(y_test, pred, zero_division=0),
        "casos_Si_detectados": int(((pred == 1) & (y_test.to_numpy() == 1)).sum()),
        "casos_Si_escapados": int(((pred == 0) & (y_test.to_numpy() == 1)).sum())})
umbrales_df = pd.DataFrame(umbrales_reg)
elegibles = umbrales_df[umbrales_df.sensibilidad_recall >= .90]
if len(elegibles):
    recomendado = elegibles.sort_values(["precision", "umbral"], ascending=False).iloc[0]
    recomendacion = (f"Umbral recomendado: {recomendado.umbral:.1f}; sensibilidad={recomendado.sensibilidad_recall:.1%}, "
        f"precisión={recomendado.precision:.1%}. Busca detectar al menos 90 % de positivos y, entre ellos, "
        "elige el de mayor precisión. Puede aumentar falsas alarmas; valide los costos antes de usarlo.")
else:
    recomendado = umbrales_df.sort_values(["sensibilidad_recall", "precision"], ascending=False).iloc[0]
    recomendacion = (f"Ningún umbral llega a sensibilidad de 90 %. El umbral {recomendado.umbral:.1f} tiene la mayor "
        f"sensibilidad disponible ({recomendado.sensibilidad_recall:.1%}; precisión {recomendado.precision:.1%}); "
        "detecta más positivos a costa de posibles falsas alarmas.")
print("  " + recomendacion)

# PASO: importancia agregada por variable original
print("PASO 10/11 | Calculando y graficando importancia de variables...")
imp = pd.DataFrame({"variable_codificada": features, "importancia": arbol.feature_importances_})
def variable_original(nombre):
    """Relaciona cada indicador codificado con su variable fuente completa."""
    for columna in cat:
        if nombre.startswith(columna + "_"):
            return columna
    return nombre
imp["variable"] = imp.variable_codificada.map(variable_original)
importancia_df = imp.groupby("variable", as_index=False).importancia.sum().sort_values("importancia", ascending=False)
nota_importancia = ("La importancia resume cuánto se utilizó cada variable para separar los datos en el árbol. "
                    "No dice si la relación es positiva o negativa y no demuestra causalidad.")
fig, ax = plt.subplots(figsize=(8, 5)); orden = importancia_df.sort_values("importancia")
ax.barh(orden.variable, orden.importancia, color="steelblue")
ax.set(xlabel="Importancia (reducción de impureza)", title="Importancia de variables")
guardar(fig, "05_importancia.png"); print("  " + nota_importancia)

# PASO: exportar predicciones y todas las tablas a un único Excel
print("PASO 11/11 | Guardando predicciones, tablas y resumen...")
predicciones_df = pd.DataFrame({"real": y_test.to_numpy(), "probabilidad_Si": prob_arbol_test,
    "clase_predicha_umbral_0_5": pred_arbol_test}, index=X_test.index).sort_index()
predicciones_df.index.name = "indice_fila_original"
mt = metricas_df[(metricas_df.modelo == "Árbol") & (metricas_df.conjunto == "Prueba")].iloc[0]
resumen_df = pd.DataFrame({"concepto": ["hogares", "porcentaje real pobre=Sí", "casos Sí en prueba",
    "casos No en prueba", "parámetros elegidos", "AUC CV árbol", "accuracy árbol prueba",
    "precision árbol prueba (0,5)", "sensibilidad árbol prueba (0,5)", "AUC árbol prueba",
    "umbral recomendado", "recomendación umbral", "sobreajuste", "lectura importancia", "datos", "figuras"],
    "resultado": [len(datos), f"{porcentaje_si:.2f} %", int(y_test.sum()), int((y_test == 0).sum()),
    str(busqueda.best_params_), busqueda.best_score_, mt.accuracy, mt.precision, mt.recall, mt.AUC,
    recomendado.umbral, recomendacion, texto_sobreajuste, nota_importancia,
    "datos/datos.xlsx, hoja hogares", "01_arbol.png a 05_importancia.png"]})
tablas = {"resumen": resumen_df, "descripcion": descripcion, "busqueda": busqueda_df,
    "reglas": reglas_df, "profundidad": profundidad_df, "matriz_confusion": matriz_df.reset_index(),
    "metricas": metricas_df, "umbrales": umbrales_df, "importancia": importancia_df,
    "predicciones": predicciones_df.reset_index()}
with pd.ExcelWriter(RUTA_EXCEL, engine="openpyxl") as writer:
    for nombre, tabla in tablas.items():
        tabla.to_excel(writer, sheet_name=nombre, index=False)
        ws = writer.book[nombre]; ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
        from openpyxl.styles import PatternFill, Font
        for celda in ws[1]:
            celda.fill = PatternFill("solid", fgColor="1F4E78")
            celda.font = Font(bold=True, color="FFFFFF")
        for columna in ws.columns:
            ancho = min(max(max(len(str(c.value or "")) for c in columna) + 2, 12), 60)
            ws.column_dimensions[columna[0].column_letter].width = ancho
print(f"  Excel: {RUTA_EXCEL}")
print(f"  Figuras PNG (dpi=150): {PLOTS_DIR}")
print("  Listo. El cuaderno explica cómo interpretar tablas, métricas y gráficos.")

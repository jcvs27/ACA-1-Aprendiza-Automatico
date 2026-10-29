# ==========================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS Y DEPENDENCIAS
# ==========================================================================
import streamlit as st            # Framework para la creación de interfaces web interactivas
import pandas as pd               # Biblioteca para la manipulación y análisis de estructuras de datos (DataFrames)
import numpy as np                # Operaciones matemáticas optimizadas y transformaciones logarítmicas
import matplotlib.pyplot as plt    # Biblioteca base para el diseño de gráficos estáticos
import seaborn as sns             # Biblioteca de visualización de datos de alto nivel basada en matplotlib
from sklearn.linear_model import LinearRegression, LogisticRegression # Algoritmos de predicción numérica continua y clasificación binaria
from sklearn.tree import DecisionTreeClassifier                      # Clasificador lógico basado en un mapa de decisiones en forma de árbol
from sklearn.neighbors import KNeighborsClassifier                    # Algoritmo de clasificación por vecindad y distancia geométrica
from sklearn.cluster import KMeans                                   # Algoritmo de aprendizaje no supervisado para agrupamiento (clustering)
from sklearn.preprocessing import StandardScaler                      # Normalizador para escalar datos con media=0 y varianza=1
from sklearn.model_selection import train_test_split                 # Utilidad para la partición de datos en entrenamiento y prueba
from sklearn.metrics import classification_report, silhouette_score  # Métricas clave de rendimiento técnico y de clustering

# Configuración de estilos globales estéticos de la página de Streamlit
st.set_page_config(page_title="Dashboard de Analítica y ML", layout="wide")

# ==========================================================================
# 2. CARGA DE DATOS Y PROCESAMIENTO INICIAL CON CACHÉ
# ==========================================================================
@st.cache_data
def cargar_datos():
    """
    Carga el dataset limpio, realiza la conversión del formato de fechas
    y extrae columnas temporales necesarias para los análisis históricos.
    """
    df = pd.read_csv('data_cleaned.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    if df['Order Date'].isna().sum() > 100:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Año'] = df['Order Date'].dt.year
    df['Mes_Año'] = df['Order Date'].dt.to_period('M')
    return df

try:
    df_clean = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar 'data_cleaned.csv': {e}")
    st.stop()

# ==========================================================================
# 3. CONSTRUCCIÓN DE LA INTERFAZ PRINCIPAL
# ==========================================================================
st.title("📊 Plataforma de Analítica de Negocios e Inteligencia Artificial")
st.markdown("**Proyecto ACA 1** | Aplicación interactiva de modelos predictivos y segmentación de clientes.")

# --- MENÚ LATERAL (SIDEBAR) CONTROLES GENERALES Y AUTORES ---
st.sidebar.header("⚙️ Parámetros de Control")
if st.sidebar.checkbox("Ver Información del Dataset", value=True):
    st.sidebar.write(f"📊 Total registros: **{len(df_clean)}**")
    st.sidebar.write(f"👥 Clientes únicos: **{df_clean['Customer ID'].nunique()}**")

# Mostrar el equipo de trabajo en el sidebar de manera limpia y elegante
st.sidebar.markdown("---")
st.sidebar.header("👥 Integrantes de Equipo")
st.sidebar.markdown("""
- **HERNAN MAURICIO GÓMEZ SOTO**
- **JULIO ANDRES RODRIGUEZ CALVO**
- **JUAN CARLOS VARGAS SIERRA**
- **CLARA JAZIEL VIECO CORREA**
- **OSCAR DAVID HERNANDEZ BOHORQUEZ**
""")

# Guía educativa embebida en la barra lateral para usuarios finales
with st.sidebar.expander("📖 Guía del Classification Report"):
    st.markdown("""
    **Columnas:**
    * **Precision (Precisión):** ¿Qué tan seguro es el modelo? Si predice 'Clase A', ¿cuántas veces acierta realmente?
    * **Recall (Exhaustividad):** ¿Qué tan bueno es encontrando casos? De todos los casos reales de 'Clase A', ¿cuántos detectó?
    * **F1-Score:** El promedio armonizado entre precisión y recall. Es la métrica de balance general.
    * **Support (Soporte):** Cantidad de datos reales evaluados para esa clase específica.

    **Filas / Clases:**
    * **Clase 0 / No Recurrente / Baja Venta:** El comportamiento predictivo para la categoría base negativa.
    * **Clase 1 / Recurrente / Alta Venta:** El comportamiento predictivo para la categoría de éxito o alto volumen.
    * **Accuracy (Exactitud):** Porcentaje total de aciertos sobre el total de datos.
    * **Macro Avg:** Promedio simple de las clases (útil si hay desbalance).
    * **Weighted Avg:** Promedio ponderado de las clases según su soporte.
    """)

# Organización de la aplicación en 4 Pestañas principales de navegación
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Exploración e Información (Análisis Histórico)",
    "🔮 Modelos de Ventas (Regresión/Clasificación)",
    "👥 Fidelidad del Cliente (Recompra con Árbol/KNN)",
    "🎯 Segmentación Dinámica (K-Means)"
])

# ------------------------------------------------------------------
# TAB 1: EXPLORACIÓN E INFORMACIÓN (ANÁLISIS DE NEGOCIOS)
# ------------------------------------------------------------------
with tab1:
    st.subheader("📂 Resumen General del Dataset de Ventas")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Primeros registros")
        st.dataframe(df_clean.head(5))
    with col2:
        st.markdown("### Estadísticas descriptivas de Ventas")
        st.dataframe(df_clean[['Sales']].describe())

    st.markdown("### Distribución General de Ventas")
    fig_dist, ax_dist = plt.subplots(1, 2, figsize=(10, 3.2))
    # Histograma de escala original sesgada
    sns.histplot(df_clean['Sales'], bins=30, kde=True, ax=ax_dist[0], color='blue')
    ax_dist[0].set_title("Distribución Original")
    # Histograma con transformación logarítmica para estabilizar la varianza
    sns.histplot(np.log1p(df_clean['Sales']), bins=30, kde=True, ax=ax_dist[1], color='teal')
    ax_dist[1].set_title("Distribución Logarítmica")
    st.pyplot(fig_dist)

    st.markdown("---")
    st.markdown("## 📊 Análisis Exploratorio Adicional de Negocios")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### Evolución Mensual de las Ventas Totales")
        ventas_mensuales = df_clean.groupby('Mes_Año')['Sales'].sum().reset_index()
        ventas_mensuales['Mes_Año'] = ventas_mensuales['Mes_Año'].astype(str)
        fig_ev, ax_ev = plt.subplots(figsize=(8, 3.5))
        sns.lineplot(x='Mes_Año', y='Sales', data=ventas_mensuales, marker='o', color='purple', linewidth=2, ax=ax_ev)
        ax_ev.set_xlabel('Mes y Año')
        ax_ev.set_ylabel('Ventas ($)')
        plt.xticks(rotation=90, fontsize=8)
        st.pyplot(fig_ev)

    with col_g2:
        st.markdown("#### Top Subcategorías que más Generan Ventas")
        top_subcat = df_clean.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_sub, ax_sub = plt.subplots(figsize=(8, 3.5))
        sns.barplot(x='Sales', y='Sub-Category', data=top_subcat, palette='viridis', ax=ax_sub, hue='Sub-Category', legend=False)
        ax_sub.set_xlabel('Ventas Totales ($)')
        ax_sub.set_ylabel('Subcategoría')
        st.pyplot(fig_sub)

# ------------------------------------------------------------------
# TAB 2: MODELOS DE VENTAS (REGRESIÓN Y CLASIFICACIÓN)
# ------------------------------------------------------------------
with tab2:
    st.subheader("🔮 Predicción del Comportamiento de Venta del Producto")
    mediana = df_clean['Sales'].median()
    df_clean['Alta_Venta'] = (df_clean['Sales'] > mediana).astype(int)

    st.info(f"💡 **Guía de Clases en Clasificación de Ventas:**\n* 🔴 **Clase 0 (Baja Venta):** Ventas menores o iguales a la mediana (${mediana:.2f} USD).\n* 🟢 **Clase 1 (Alta Venta):** Ventas que superan la mediana de transacciones (${mediana:.2f} USD).")

    # Extracción de variables categóricas explicativas
    X_raw = df_clean[['Category', 'Sub-Category', 'Region']]
    y_lin = np.log1p(df_clean['Sales'])
    y_log = df_clean['Alta_Venta']

    # Aplicación de One-Hot Encoding omitiendo la primera columna redundante
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    columnas_ref = X_encoded.columns

    # Separación en entrenamiento y validación técnica
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_encoded, y_log, test_size=0.2, random_state=42)

    # Entrenamiento de modelos simultáneos
    mod_multiple = LinearRegression().fit(X_encoded, y_lin)
    mod_logistico = LogisticRegression(max_iter=1000).fit(X_train_l, y_train_l)

    st.markdown("### 🛠️ Ingresar Parámetros del Producto")
    c1, c2, c3 = st.columns(3)
    with c1:
        cat_sel = st.selectbox("Categoría", df_clean['Category'].unique())
    with c2:
        sub_sel = st.selectbox("Subcategoría", df_clean[df_clean['Category'] == cat_sel]['Sub-Category'].unique())
    with c3:
        reg_sel = st.selectbox("Región", df_clean['Region'].unique())

    if st.button("Calcular Predicciones de Venta"):
        # Formatear el diccionario con las mismas columnas usadas en el ajuste del modelo
        input_dict = {col: [False] for col in columnas_ref}
        nuevo_df = pd.DataFrame(input_dict)

        # Activación de variables según el formulario
        if f"Category_{cat_sel}" in nuevo_df.columns: nuevo_df[f"Category_{cat_sel}"] = [True]
        if f"Sub-Category_{sub_sel}" in nuevo_df.columns: nuevo_df[f"Sub-Category_{sub_sel}"] = [True]
        if f"Region_{reg_sel}" in nuevo_df.columns: nuevo_df[f"Region_{reg_sel}"] = [True]

        # Predicciones invirtiendo la escala logarítmica mediante expm1
        pred_lin = np.expm1(mod_multiple.predict(nuevo_df)[0])
        prob_alta = mod_logistico.predict_proba(nuevo_df)[0][1] * 100

        st.markdown("---")
        st.markdown("### 🎯 Resultados Obtenidos")
        r1, r2 = st.columns(2)
        r1.metric("Venta Estimada (Regresión Múltiple)", f"${pred_lin:.2f} USD")
        clase_pred_txt = "🟢 ALTA VENTA (Clase 1)" if prob_alta >= 50 else "🔴 BAJA VENTA (Clase 0)"
        r2.metric("Diagnóstico (Regresión Logística)", clase_pred_txt, delta=f"{prob_alta:.2f}% Probabilidad de Éxito")

        st.markdown("### 📊 Gráfico de Comparación del Producto")
        promedio_categoria = df_clean[df_clean['Category'] == cat_sel]['Sales'].mean()
        promedio_general = df_clean['Sales'].mean()

        fig_bar, ax_bar = plt.subplots(figsize=(8, 3.0))
        valores = [promedio_general, promedio_categoria, pred_lin]
        etiquetas = ['Promedio General', f'Promedio {cat_sel}', 'Tu Producto Estimado']
        colores = ['#A0A0A0', '#4682B4', '#FF4B4B']

        sns.barplot(x=valores, y=etiquetas, palette=colores, ax=ax_bar, hue=etiquetas, legend=False)
        ax_bar.set_title("Comparación del Valor de Venta Estimado ($)", fontsize=10, fontweight="bold")
        ax_bar.set_xlabel("Ventas en USD")
        for i, v in enumerate(valores):
            ax_bar.text(v + (max(valores)*0.01), i, f"${v:.2f}", va="center", fontweight="bold", fontsize=8)
        st.pyplot(fig_bar)

    st.markdown("---")
    st.markdown("### 📈 Reportes de Rendimiento Técnico")
    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        st.markdown("#### 🔹 Coeficiente de Regresión Múltiple")
        r2_score = mod_multiple.score(X_encoded, y_lin)
        st.write(f"**Precisión Global del Modelo (R² Score):** `{r2_score*100:.2f}%`")
    with col_rep2:
        st.markdown("#### 🔹 Reporte de Clasificación (Regresión Logística)")
        y_pred_log = mod_logistico.predict(X_test_l)
        rep_log_dict = classification_report(y_test_l, y_pred_log, output_dict=True)
        st.dataframe(pd.DataFrame(rep_log_dict).transpose().iloc[:-1])

# ------------------------------------------------------------------
# TAB 3: FIDELIDAD DEL CLIENTE (PREDICCIÓN DE RECOMPRA CON ÁRBOL/KNN)
# ------------------------------------------------------------------
with tab3:
    st.subheader("👥 Clasificación de Clientes y Tendencia de Recompra")

    # Agrupación de datos históricos a nivel de ID de cliente
    clientes_pedidos = df_clean.groupby("Customer ID").agg(
        pedidos_unicos=("Order ID", "nunique"),
        ciudad_principal=("City", lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),
        total_gastado=("Sales", "sum"),
        articulos_comprados=("Sales", "count")
    ).reset_index()
    clientes_pedidos["Recompra"] = (clientes_pedidos["pedidos_unicos"] >= 2).astype(int)

    # Filtrado de ciudades para mitigar alta cardinalidad
    ciudades_frec = clientes_pedidos["ciudad_principal"].value_counts().head(20).index
    clientes_filtrados = clientes_pedidos.copy()
    clientes_filtrados["ciudad_principal"] = clientes_filtrados["ciudad_principal"].apply(lambda x: x if x in ciudades_frec else "Otra")

    # Preparación y Ajuste del Árbol de Decisión
    X_recompra = pd.get_dummies(clientes_filtrados[["ciudad_principal", "total_gastado"]], drop_first=True)
    y_recompra = clientes_filtrados["Recompra"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_recompra, y_recompra, test_size=0.25, random_state=42)
    mod_recompra_arbol = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train_r, y_train_r)

    # Preparación, Escalado y Entrenamiento de KNN
    X_knn = clientes_pedidos[["total_gastado", "articulos_comprados"]]
    y_knn = clientes_pedidos["Recompra"]
    escalador_knn = StandardScaler()
    X_knn_esc = escalador_knn.fit_transform(X_knn)
    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_knn_esc, y_knn, test_size=0.25, random_state=42)
    mod_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_k, y_train_k)

    # Controles de simulación de un cliente nuevo
    col_cli1, col_cli2 = st.columns(2)
    with col_cli1:
        st.markdown("### Datos del Cliente")
        ciudad_input = st.selectbox("Ciudad Principal del Cliente", clientes_pedidos["ciudad_principal"].unique())
        gasto_input = st.number_input("Total Gastado acumulado ($)", min_value=1.0, max_value=50000.0, value=1500.0)
        articulos_input = st.slider("Cantidad de Artículos Comprados", min_value=1, max_value=150, value=10)

    with col_cli2:
        if st.button("Evaluar Tendencia de Recompra"):
            # Predicción 1: Árbol de Decisión
            input_arbol = {col: [False] for col in X_recompra.columns if col != "total_gastado"}
            input_arbol["total_gastado"] = [gasto_input]
            df_arbol = pd.DataFrame(input_arbol)
            if f"ciudad_principal_{ciudad_input}" in df_arbol.columns:
                df_arbol[f"ciudad_principal_{ciudad_input}"] = [True]
            df_arbol = df_arbol[X_recompra.columns]
            prob_recompra_arbol = mod_recompra_arbol.predict_proba(df_arbol)[0][1] * 100

            # Predicción 2: KNN (con escalador)
            nuevo_knn = pd.DataFrame({"total_gastado": [gasto_input], "articulos_comprados": [articulos_input]})
            nuevo_knn_esc = escalador_knn.transform(nuevo_knn)
            prob_knn = mod_knn.predict_proba(nuevo_knn_esc)[0][1] * 100

            st.markdown("---")
            st.markdown("#### 🔍 Resultados de Predicción de Recompra")
            st.write(f"🌲 **Probabilidad del Árbol de Decisión:** `{prob_recompra_arbol:.2f}%`")
            st.write(f"📐 **Probabilidad de KNN (Similitud):** `{prob_knn:.2f}%`")

            # Visualización de los clientes parecidos más cercanos (vecinos de KNN)
            distancias, indices = mod_knn.kneighbors(nuevo_knn_esc, n_neighbors=3)
            vecinos_df = clientes_pedidos.iloc[indices[0]]
            st.markdown("**Vecinos más cercanos en la base de datos:**")
            st.dataframe(vecinos_df[["Customer ID", "total_gastado", "articulos_comprados", "Recompra"]])

# ------------------------------------------------------------------
# TAB 4: SEGMENTACIÓN DINÁMICA (K-MEANS) - COMPLETADA
# ------------------------------------------------------------------
with tab4:
    st.subheader("🎯 Segmentación de Clientes Mediante K-Means")
    st.markdown("""
    Este módulo utiliza el algoritmo de aprendizaje no supervisado **K-Means** para agrupar dinámicamente
    a los clientes basándose en su patrón comercial histórico: **gasto acumulado ($)** y **número de artículos comprados**.
    """)

    # Preparación técnica y escalamiento de los datos de clustering
    clientes_pedidos = df_clean.groupby("Customer ID").agg(
        total_gastado=("Sales", "sum"),
        articulos_comprados=("Sales", "count")
    ).reset_index()

    X_clustering = clientes_pedidos[["total_gastado", "articulos_comprados"]]
    escalador_kmeans = StandardScaler()
    X_escalado = escalador_kmeans.fit_transform(X_clustering)

    # Ajustar el modelo de K-Means estableciendo 3 clústeres representativos
    kmeans_modelo = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_escalado)
    clientes_pedidos["Cluster"] = kmeans_modelo.labels_

    # Calcular métrica de calidad del clustering (Coeficiente de Silueta)
    sil_score = silhouette_score(X_escalado, kmeans_modelo.labels_
)

    # Mostrar métricas de evaluación del modelo de clustering
    col_met1, col_met2 = st.columns(2)
    with col_met1:
        st.metric(label="Métrica de Calidad: Coeficiente de Silueta", value=f"{sil_score:.3f}", help="Mide qué tan cercanos están los puntos a su propio clúster en comparación con otros. Un valor cercano a 1 indica excelente agrupación.")
    with col_met2:
        st.info("💡 **Nota Técnica:** Un coeficiente de silueta mayor a 0.50 refleja una estructura de clústeres bien definida y cohesiva en el espacio geométrico escalado.")

    # --- FORMULARIO PARA AGREGAR NUEVO CLIENTE Y CLASIFICARLO ---
    st.markdown("### 👥 Agregar y Segmentar un Nuevo Cliente")
    col_seg1, col_seg2 = st.columns(2)

    # Conservar el estado del cliente simulado usando session_state
    if 'nuevo_cliente_registrado' not in st.session_state:
        st.session_state['nuevo_cliente_registrado'] = False
        st.session_state['nuevo_gasto'] = 2500.0
        st.session_state['nuevos_articulos'] = 30
        st.session_state['nuevo_cluster'] = None

    with col_seg1:
        nuevo_gasto = st.number_input("Monto Total Gastado del Cliente ($)", min_value=1.0, max_value=100000.0, value=float(st.session_state['nuevo_gasto']), step=50.0)
        nuevos_articulos = st.number_input("Número de Artículos Comprados por el Cliente", min_value=1, max_value=500, value=int(st.session_state['nuevos_articulos']), step=1)

    with col_seg2:
        if st.button("Asignar Segmento (Clúster) al Cliente"):
            # Guardar valores de entrada en session_state
            st.session_state['nuevo_gasto'] = nuevo_gasto
            st.session_state['nuevos_articulos'] = nuevos_articulos
            st.session_state['nuevo_cliente_registrado'] = True

            # 1. Crear un DataFrame con el formato correcto para el cliente nuevo
            nuevo_registro_df = pd.DataFrame({
                'total_gastado': [nuevo_gasto],
                'articulos_comprados': [nuevos_articulos]
            })

            # 2. Escalar el nuevo dato usando el StandardScaler ajustado previamente
            nuevo_registro_esc = escalador_kmeans.transform(nuevo_registro_df)

            # 3. Predecir a qué clúster pertenece (centroide más cercano)
            nuevo_cluster = kmeans_modelo.predict(nuevo_registro_esc)[0]
            st.session_state['nuevo_cluster'] = nuevo_cluster

            st.markdown("---")
            st.markdown(f"### 🎯 Segmento Asignado: **Clúster {nuevo_cluster}**")

            # Descripción interpretativa personalizada de los grupos de clientes
            if nuevo_cluster == 0:
                st.info("💡 **Interpretación del Segmento 0 (Clientes Moderados):** Clientes con un volumen de compra intermedio y gasto balanceado. Representan la base de ingresos estables del negocio.")
            elif nuevo_cluster == 1:
                st.warning("💡 **Interpretación del Segmento 1 (Clientes de Baja Frecuencia):** Compradores ocasionales con bajo gasto acumulado y pocos artículos. Potencial objetivo para campañas de activación o cupones de descuento.")
            else:
                st.success("💡 **Interpretación del Segmento 2 (Clientes VIP / Alto Valor):** Clientes corporativos o VIP de alta gama que realizan compras masivas y generan el mayor margen de rentabilidad.")

    st.markdown("---")
    st.markdown("### 📊 Mapa de Distribución de Segmentación Actual")
    # Gráfico que muestra las fronteras de los clústeres y destaca el nuevo cliente ingresado
    fig_cluster, ax_cluster = plt.subplots(figsize=(8, 4.2))
    sns.scatterplot(x='total_gastado', y='articulos_comprados', hue='Cluster', palette='Set1', data=clientes_pedidos, ax=ax_cluster, s=60, alpha=0.8)

    # Si hay un nuevo cliente ingresado, lo pintamos en el gráfico
    if st.session_state['nuevo_cliente_registrado']:
        ax_cluster.scatter(
            st.session_state['nuevo_gasto'],
            st.session_state['nuevos_articulos'],
            color='black',
            edgecolor='yellow',
            s=350,
            marker='*',
            label='Nuevo Cliente Simulando',
            zorder=10
        )
        ax_cluster.legend(title='Segmento / Clúster')

    ax_cluster.set_title("Agrupamiento de Clientes por Comportamiento de Compra", fontsize=11, fontweight='bold')
    ax_cluster.set_xlabel("Total Gastado en USD ($)")
    ax_cluster.set_ylabel("Cantidad de Artículos Comprados")
    st.pyplot(fig_cluster)

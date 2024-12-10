import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Función para generar el árbol binomial del subyacente
def arbol_subyacente(S0, u, d, N):

    # Hacemos un array para nuestro arbol
    arbol = np.zeros((N+1, N+1))

    # Le decimos que el precio inicial es s0
    arbol[0, 0] = S0

    # Rellenamos nuestro árbol con los valores correspondientes
    for i in range(1, N+1):
        for j in range(i+1):
            arbol[j, i] = S0 * (u ** (i-j)) * (d ** j)

    return arbol

# Función para generar el árbol binomial de la opción
def arbol_derivado(arbol_subyacente, K, r, T, N, u, d, tipo="call"):

    # Calculamos delta
    delta = T / N

    # Calculamos B
    B = np.exp(-r*delta)

    # Hacemos un arbol de ceros con las mismas dimensiones del arbol subyacente
    arbol = np.zeros_like(arbol_subyacente)
    
    # Calculamos el payoff para cada caso
    if tipo == "call":
        for i in range(N+1):
            arbol[i, N] = max(arbol_subyacente[i, N] - K, 0) # Payoff de una call para cada nodo final
    elif tipo =="put":
        for i in range(N+1):
            arbol[i, N] = max(K - arbol_subyacente[i, N], 0) # Payoff de una put para cada nodo final
    
    # Valuamos de manera regresiva el árbol

    # Notemos que el indice es N-1, puesto que queremos valuar un tiempo antes del ultimo periodo usando los payoffs del ultimo periodo
    for i in range(N-1, -1, -1): # El índice empieza en el numero de periodos menos 1 y termina en 0, disminuyendo de uno en uno
        for j in range(i+1): # Empieza en cero y se detiene hasta que  llega al último nodo que valua
            X_u = arbol[j, i+1] # Valor del derivado cuando sube
            X_d = arbol[j+1, i+1] # Valor del derivado cuando baja
            q = (B**(-1) - d) / (u - d) # Probabilidad de subida 
            arbol[j, i] = np.exp(-r * delta) * (q * X_u + (1 - q) * X_d) # Calculamos cada nodo del árbol con el descuento
    
    return arbol # Regresamos el árbol con todas las valuaciones

def arbol_alpha(arbol_subyacente, arbol_derivado, K, r, T, N, tipo="call"):

    # Calculamos delta
    delta = T / N

    # Calculamos B
    B = np.exp(-r*delta)

    # Hacemos un arbol de ceros con las mismas dimensiones del arbol subyacente
    arbol = np.zeros_like(arbol_subyacente)
    
    # Calculamos el payoff para cada caso
    if tipo == "call":
        for i in range(N+1):
            arbol[i, N] = max(arbol_subyacente[i, N] - K, 0) # Payoff de una call para cada nodo final
    elif tipo =="put":
        for i in range(N+1):
            arbol[i, N] = max(K - arbol_subyacente[i, N], 0) # Payoff de una put para cada nodo final
    
    # Valuamos de manera regresiva el árbol
    
    for i in range(N): # El índice empieza en el numero de periodos menos 1 y termina en 0, disminuyendo de uno en uno
        X_u = arbol_derivado[i,N]
        X_d = arbol_derivado[i+1,N]
        S_u = arbol_subyacente[i,N]
        S_d = arbol_subyacente[i+1,N]
        arbol[i,N] = round((X_u - X_d)/(S_u - S_d),4)

    # Notemos que el indice es N-1, puesto que queremos valuar un tiempo antes del ultimo periodo usando los payoffs del ultimo periodo
    for i in range(N-1, -1, -1): # El índice empieza en el numero de periodos menos 1 y termina en 0, disminuyendo de uno en uno
        for j in range(i+1): # Empieza en cero y se detiene hasta que  llega al último nodo que valua
            X_u = arbol_derivado[j, i+1] # Valor del derivado cuando sube
            X_d = arbol_derivado[j+1, i+1] # Valor del derivado cuando baja
            S_u = arbol_subyacente[j, i+1]
            S_d = arbol_subyacente[j+1, i+1]
            alpha = (X_u - X_d)/(S_u - S_d)
            arbol[j, i] = round(alpha,4) # Calculo de alpha
    
    return arbol # Regresamos el árbol con todas las valuaciones

def arbol_beta(arbol_subyacente, arbol_derivado, K, r, T, N, tipo="call"):

    # Calculamos delta
    delta = T / N

    # Calculamos B
    B = np.exp(-r*delta)

    # Hacemos un arbol de ceros con las mismas dimensiones del arbol subyacente
    arbol = np.zeros_like(arbol_subyacente)
    
    # Calculamos el payoff para cada caso
    if tipo == "call":
        for i in range(N+1):
            arbol[i, N] = max(arbol_subyacente[i, N] - K, 0) # Payoff de una call para cada nodo final
    elif tipo =="put":
        for i in range(N+1):
            arbol[i, N] = max(K - arbol_subyacente[i, N], 0) # Payoff de una put para cada nodo final
    
    # Valuamos de manera regresiva el árbol

    for i in range(N): # El índice empieza en el numero de periodos menos 1 y termina en 0, disminuyendo de uno en uno
        X_u = arbol_derivado[i,N]
        X_d = arbol_derivado[i+1,N]
        S_u = arbol_subyacente[i,N]
        S_d = arbol_subyacente[i+1,N]
        arbol[i,N] = round((X_u - ((X_u - X_d)*S_u)/(S_u - S_d)),4)
  
    # Notemos que el indice es N-1, puesto que queremos valuar un tiempo antes del ultimo periodo usando los payoffs del ultimo periodo
    for i in range(N-1, -1, -1): # El índice empieza en el numero de periodos menos 1 y termina en 0, disminuyendo de uno en uno
        for j in range(i+1): # Empieza en cero y se detiene hasta que  llega al último nodo que valua
            X_u = arbol_derivado[j, i+1] # Valor del derivado cuando sube
            X_d = arbol_derivado[j+1, i+1] # Valor del derivado cuando baja
            S_u = arbol_subyacente[j, i+1]
            S_d = arbol_subyacente[j+1, i+1]
            beta = B*(X_u - ((X_u - X_d)*S_u)/(S_u - S_d))
            arbol[j, i] = round(beta,4) # Calculo de beta
    
    return arbol # Regresamos el árbol con todas las valuaciones

def combinar_alpha_beta(alpha, beta):
    alpha_beta = [[(alpha[i][j], beta[i][j]) for j in range(len(alpha[0]))] for i in range(len(alpha))]
    matriz = [[f"({x[0]}, {x[1]})" for x in fila] for fila in alpha_beta]
    return matriz

def opcion_americana(arbol_subyacente, K, r, T, N, u, d):

    # Calcular parámetros básicos
    delta = T / N  # Tiempo por paso
    B = np.exp(-r * delta)  # Factor de descuento
    q = (B**(-1) - d) / (u - d)  # Probabilidad neutral al riesgo
    
    # Inicializar la matriz para la opción
    matriz_opcion = np.zeros_like(arbol_subyacente)
    
    # Calcular valores terminales (opción en el último paso)
    for i in range(N + 1):
        matriz_opcion[i, N] = max(0, arbol_subyacente[i, N] - K)
        
    
    # Calcular los valores retrocediendo en el árbol
    for j in range(N - 1, -1, -1):  # Columnas
        for i in range(j + 1):  # Filas
            # Valor esperado descontado
            valor_continuacion = B * (q * matriz_opcion[i, j + 1] + (1 - q) * matriz_opcion[i + 1, j + 1])
            
            # Valor intrínseco
            
            valor_intrinseco = max(0, arbol_subyacente[i, j] - K)
            
            # Opción americana: Tomar el máximo entre continuación e intrínseco
            matriz_opcion[i, j] = max(valor_intrinseco, valor_continuacion)
    
    return matriz_opcion

def grafica_subyacente(N, u, d, S0):

    # Se calculan los nodos del arbol
    arbol = []
    for i in range(N + 1):
        periodo = [S0 * (u**j) * (d**(i - j)) for j in range(i + 1)]
        arbol.append(periodo)

    # Hacemos las lineas que unen a los nodos
    x_conec = []
    y_conec = []
    for i in range(1, len(arbol)):
        for j in range(len(arbol[i])):
            if j < len(arbol[i - 1]):
                # Conectar el nodo actual con el nodo directamente por encima de él.
                x_conec.extend([i - 1, i, None])
                y_conec.extend([arbol[i - 1][j], arbol[i][j], None])
            if j > 0:
                # Conectar el nodo actual con el nodo en la parte superior izquierda.
                x_conec.extend([i - 1, i, None])
                y_conec.extend([arbol[i - 1][j - 1], arbol[i][j], None])

    # Inicializamos la figura
    fig = go.Figure()

    # Graficamos todooo
    fig.add_trace(go.Scatter(
        x=x_conec, 
        y=y_conec,
        mode='lines',
        line=dict(color='#FECC13', width=3),
        name='Conexiones'
    ))

    # Agregamos los nodos y sus valores a la gráfica
    for i, periodo in enumerate(arbol):
        # Posiciones de los nodos
        x_posi = [i] * len(periodo)
        y_posi = periodo
        
        # Agregamos estilo
        fig.add_trace(go.Scatter(
            x=x_posi,
            y=y_posi,
            mode='markers+text',
            marker=dict(size=8, color='#DC4B2C'),
            text=[f"${precio:,.2f}" for precio in periodo],
            textposition='top center',
            name=f'Periodo {i}'
        ))

    # Agreagamos titulo y demas a las graficas
    fig.update_layout(
        title="Árbol binomial del subyacente",
        xaxis=dict(title="Periodo"),
        yaxis=dict(title="Precio"),
        showlegend=False,
        template="plotly_white",
        height = 600
    )

    return fig

def grafica_derivado(matriz_derivado, N):

    derivado = []
    for i in range(0,N+1):
        derivado.append([matriz_derivado[j,i] for j in range(i+1)])

    # Hacemos las lineas que unen a los nodos
    x_conec = []
    y_conec = []
    for i in range(1, len(derivado)):
        for j in range(len(derivado[i])):
            if j < len(derivado[i - 1]):
                # Conectar el nodo actual con el nodo directamente por encima de él.
                x_conec.extend([i - 1, i, None])
                y_conec.extend([derivado[i - 1][j], derivado[i][j], None])
            if j > 0:
                # Conectar el nodo actual con el nodo en la parte superior izquierda.
                x_conec.extend([i - 1, i, None])
                y_conec.extend([derivado[i - 1][j - 1], derivado[i][j], None])

    # Inicializamos la figura
    fig = go.Figure()

    # Graficamos todooo
    fig.add_trace(go.Scatter(
        x=x_conec, 
        y=y_conec,
        mode='lines',
        line=dict(color='#FECC13', width=3),
        name='Conexiones'
    ))

    # Agregamos los nodos y sus valores a la gráfica
    for i, periodo in enumerate(derivado):
        # Posiciones de los nodos
        x_posi = [i] * len(periodo)
        y_posi = periodo
        
        # Agregamos estilo
        fig.add_trace(go.Scatter(
            x=x_posi,
            y=y_posi,
            mode='markers+text',
            marker=dict(size=8, color='#DC4B2C'),
            text=[f"${precio:,.2f}" for precio in periodo],
            textposition='top center',
            name=f'Periodo {i}'
        ))

    # Agreagamos titulo y demas a las graficas
    fig.update_layout(
        title="Árbol del derivado",
        xaxis=dict(title="Periodo"),
        yaxis=dict(title="Precio"),
        showlegend=False,
        template="plotly_white",
        height = 600
    )

    return fig

st.set_page_config("Calculadora de opciones: Put/Call Europea y Americana", layout="wide", menu_items={'About' : " # Bienvenido/a a la calculadora de opciones :D"})

st.markdown('<p style="text-align: center; font-size: 35px; color: #DC4B2C">Calculadora de opciones :D</p>', unsafe_allow_html=True)
st.markdown("""<div style="background-color: var(--primary-background-color); height: 50px;"></div>""", unsafe_allow_html=True)
coluu1, coluu2 = st.columns(2)
with coluu1:
    st.markdown('<p style="text-align: center; font-size: 25px; color: #FECC13">Elige la opción que quieres valuar</p>', unsafe_allow_html=True)
with coluu2:
    c1, c2 = st.columns(2)
    with c2:
            opc1 = st.selectbox("Ver gráfica o tabla",options=["Gráfica Subyacente", "Gráfica Derivado", "Tabla (alpha,beta)"])
    with c1:
        if opc1 == "Gráfica Subyacente":
            st.markdown('<p style="text-align: center; font-size: 25px; color: #FECC13">Árbol del subyacente</p>', unsafe_allow_html=True)
        elif opc1 == "Tabla (alpha,beta)":
            st.markdown('<p style="text-align: center; font-size: 25px; color: #FECC13">Tabla (alpha,beta)</p>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; font-size: 25px; color: #FECC13">Árbol del derivado</div>', unsafe_allow_html=True)

colu1, colu2 = st.columns(2)
with colu1:
    opcion = st.selectbox("Elige el tipo de opcion", options=["Call Europea", "Put Europea", "Call Americana"])
    st.markdown("""<div style="background-color: var(--primary-background-color); height: 50px;"></div>""", unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 25px; color: #FECC13">Elige los parámetros</p>', unsafe_allow_html=True)
    st.markdown("""<div style="background-color: var(--primary-background-color); height: 50px;"></div>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            s0 = st.number_input("Precio del subyacente (s0)", min_value=0.00, max_value=1000000000.00, value=100.00, step=0.10)
    with col2:
        with st.container(border=True):
            u = st.number_input("Selecciona u", min_value=1.000001, value=1.2, step=0.01)
    with col3:
        with st.container(border=True):
            d = st.number_input("Selecciona d", min_value=0.000001, max_value=0.9, step=0.01, value=0.9)
    with col4:
        with st.container(border=True):
            T = st.number_input("Maduración (T, años)", min_value=1, step=1,value=1)
    st.markdown("""<div style="background-color: var(--primary-background-color); height: 50px;"></div>""", unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        with st.container(border=True):
            N = st.number_input("Pon el número de periodos (N)", min_value=1, step=1, value=4)
    with col6:
        with st.container(border=True):
            r = st.number_input("Pon la tasa libre de riesgo (r)", min_value=0.00, step=0.01, value=0.05)
    with col7:
        with st.container(border=True):
            K = st.number_input("Elige el precio strike (K)", min_value=0.00, step=0.01, value=100.00)
    if opcion == "Call Europea":
        columnaa1, columnaa2 = st.columns(2)
        with columnaa1:
            st.markdown("""<div style="background-color: var(--primary-background-color); height: 25px;"></div>""", unsafe_allow_html=True)
            st.markdown('<p style="text-align: center; font-size: 25px; color: #FECC13">Precio hoy de la opción call europea:</p>', unsafe_allow_html=True)
        with columnaa2:
            arbol_sub = arbol_subyacente(s0,u,d,N)
            arbol_deriv = arbol_derivado(arbol_sub, K, r, T, N, u, d, tipo="call")
            st.markdown("""<div style="background-color: var(--primary-background-color); height: 25px;"></div>""", unsafe_allow_html=True)
            st.markdown(f'<p style="text-align: center; font-size: 25px; color: red">${arbol_deriv[0,0]:,.2f}</p>', unsafe_allow_html=True)
    elif opcion == "Put Europea":
        columnaa1, columnaa2 = st.columns(2)
        with columnaa1:
            st.markdown("""<div style="background-color: var(--primary-background-color); height: 25px;"></div>""", unsafe_allow_html=True)
            st.markdown('<p style="text-align: center; font-size: 25px; color: #FECC13">Precio hoy de la opción put europea:</p>', unsafe_allow_html=True)
        with columnaa2:
            arbol_sub = arbol_subyacente(s0,u,d,N)
            arbol_deriv = arbol_derivado(arbol_sub, K, r, T, N, u, d, tipo="put")
            st.markdown("""<div style="background-color: var(--primary-background-color); height: 25px;"></div>""", unsafe_allow_html=True)
            st.markdown(f'<p style="text-align: center; font-size: 25px; color: red">${arbol_deriv[0,0]:,.2f}</p>', unsafe_allow_html=True)
with colu2:
    arbol_sub = arbol_subyacente(s0,u,d,N)
    grafica_sub = grafica_subyacente(N,u,d,s0)
    if opcion == "Call Europea":
        arbol_deriv = arbol_derivado(arbol_sub, K, r, T, N, u, d, tipo="call")
        grafica_deriv = grafica_derivado(arbol_deriv, N)
        if opc1 == "Gráfica Derivado":
         st.plotly_chart(grafica_deriv)
        elif opc1 == "Tabla (alpha,beta)":
            alpha = arbol_alpha(arbol_sub, arbol_deriv, K, r, T, N, tipo="call")
            beta = arbol_beta(arbol_sub, arbol_deriv, K, r, T, N, tipo="call")
            alpha_beta = combinar_alpha_beta(alpha, beta)
            st.dataframe(alpha_beta,use_container_width=True, hide_index=False, height=598)
        else:
         st.plotly_chart(grafica_sub)
    elif opcion == "Put Europea":
        arbol_deriv = arbol_derivado(arbol_sub, K, r, T, N, u, d, tipo="put")
        grafica_deriv = grafica_derivado(arbol_deriv, N)
        if opc1 == "Gráfica Derivado":
         st.plotly_chart(grafica_deriv)
        elif opc1 == "Tabla (alpha,beta)":
            alpha = arbol_alpha(arbol_sub, arbol_deriv, K, r, T, N, tipo="call")
            beta = arbol_beta(arbol_sub, arbol_deriv, K, r, T, N, tipo="call")
            alpha_beta = combinar_alpha_beta(alpha, beta)
            st.dataframe(alpha_beta,use_container_width=True, hide_index=False, height=598)
        else:
         st.plotly_chart(grafica_sub)
    elif opcion == "Call Americana":
        arbol_deriv = opcion_americana(arbol_sub, K, r, T, N, u, d)
        grafica_deriv = grafica_derivado(arbol_deriv, N)
        if opc1 == "Gráfica Derivado":
         st.plotly_chart(grafica_deriv)
        elif opc1 == "Tabla (alpha,beta)":
            alpha = arbol_alpha(arbol_sub, arbol_deriv, K, r, T, N, tipo="call")
            beta = arbol_beta(arbol_sub, arbol_deriv, K, r, T, N, tipo="call")
            alpha_beta = combinar_alpha_beta(alpha, beta)
            st.dataframe(alpha_beta,use_container_width=True, hide_index=False, height=598)
        else:
         st.plotly_chart(grafica_sub)
    
 
 
    



"""
Sistema de Gemelos Digitales para Gestión de Mantenimiento de Equipos de Carguío en Minas a Tajo Abierto
Aplicación principal desarrollada con Streamlit
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import time

# Agregar ruta del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos
from utils.database import init_database, insert_default_data
from modules import auth
from modules import dashboard as dash_mod
from modules import gemelo_digital as gd_mod
from modules import mantenimiento as mant_mod
from modules import reportes as rep_mod
from modules import predictivo as pred_mod
from modules.motor_ia import MotorPredictivo, inicializar_motor_ia, MODELS_DIR

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Gemelo Digital - Minería",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar base de datos
@st.cache_resource
def inicializar_sistema():
    init_database()
    insert_default_data()
    return True

inicializar_sistema()

# ============================================================
# ESTADO DE SESIÓN
# ============================================================
if 'usuario' not in st.session_state:
    st.session_state.usuario = None
if 'token' not in st.session_state:
    st.session_state.token = None
if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = 'Login'
if 'motor_ia' not in st.session_state:
    st.session_state.motor_ia = None

# ============================================================
# FUNCIÓN DE LOGIN
# ============================================================
def mostrar_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #2c3e50;'>⛏️ SISTEMA DE GEMELOS DIGITALES</h1>
            <h3 style='color: #7f8c8d;'>Gestión de Mantenimiento - Equipos de Carguío</h3>
            <p style='color: #95a5a6;'>Minería a Tajo Abierto</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("### 🔐 Iniciar Sesión")
            username = st.text_input("Usuario", placeholder="Ingrese su usuario")
            password = st.text_input("Contraseña", type="password", placeholder="Ingrese su contraseña")
            submit = st.form_submit_button("Ingresar", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Por favor ingrese usuario y contraseña")
                else:
                    resultado = auth.login(username, password)
                    if resultado:
                        st.session_state.usuario = resultado
                        st.session_state.token = resultado['token']
                        st.success("✅ Inicio de sesión exitoso!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Usuario o contraseña incorrectos")
        
        st.info("""
        **Usuarios de demostración:**
        - 👤 Administrador: `admin` / `admin123`
        - 👷 Ingeniero: `ingeniero` / `inge123`
        - 📋 Supervisor: `supervisor` / `super123`
        - 🔧 Técnico: `tecnico` / `tec123`
        """)

# ============================================================
# BARRA LATERAL DE NAVEGACIÓN
# ============================================================
def mostrar_sidebar():
    usuario = st.session_state.usuario
    
    with st.sidebar:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #2c3e50, #34495e); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
            <h4 style='color: white; margin: 0;'>👤 {usuario['nombre']} {usuario['apellido']}</h4>
            <p style='color: #bdc3c7; margin: 5px 0;'>{usuario['rol']}</p>
            <p style='color: #95a5a6; font-size: 12px; margin: 0;'>{usuario['email']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Menú Principal")
        
        # Opciones de menú según rol
        menu_options = []
        
        menu_options.append(('📊 Dashboard', 'Dashboard'))
        
        if auth.has_permission(usuario['rol'], auth.PERMISOS['gemelo_digital']):
            menu_options.append(('🔧 Gemelo Digital', 'Gemelo Digital'))
        
        if auth.has_permission(usuario['rol'], auth.PERMISOS['mantenimiento']):
            menu_options.append(('📋 Mantenimiento', 'Mantenimiento'))
        
        if auth.has_permission(usuario['rol'], auth.PERMISOS['predictivo']):
            menu_options.append(('🤖 Análisis Predictivo', 'Predictivo'))
            menu_options.append(('⚙️ Motor IA', 'Motor IA'))
        
        if auth.has_permission(usuario['rol'], auth.PERMISOS['reportes']):
            menu_options.append(('📑 Reportes', 'Reportes'))
        
        if auth.has_permission(usuario['rol'], auth.PERMISOS['repuestos']):
            menu_options.append(('🔩 Repuestos', 'Repuestos'))
        
        if auth.has_permission(usuario['rol'], auth.PERMISOS['usuarios']):
            menu_options.append(('👥 Gestión Usuarios', 'Usuarios'))
            menu_options.append(('📝 Bitácora', 'Bitacora'))
        
        menu_options.append(('📖 Documentación Scrum', 'Scrum'))
        
        # Mostrar menú
        for label, key in menu_options:
            if st.button(label, use_container_width=True, key=f"btn_{key}"):
                st.session_state.pagina_actual = key
        
        st.markdown("---")
        
        if st.button("🚪 Cerrar Sesión", use_container_width=True, type="secondary"):
            st.session_state.usuario = None
            st.session_state.token = None
            st.session_state.pagina_actual = 'Login'
            st.rerun()
        
        st.markdown(f"""
        <div style='position: absolute; bottom: 10px; left: 10px; right: 10px; text-align: center; color: #95a5a6; font-size: 11px;'>
            <p>Versión 1.0.0</p>
            <p>{datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PÁGINA: DASHBOARD
# ============================================================
def mostrar_dashboard():
    st.title("📊 Dashboard Principal")
    st.markdown("---")
    
    # KPIs
    kpis = dash_mod.get_kpis()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Disponibilidad", f"{kpis['disponibilidad']}%", delta="Meta: 90%")
    with col2:
        st.metric("OEE", f"{kpis['oee']}%", delta="Meta: 75%")
    with col3:
        st.metric("MTBF", f"{kpis['mtbf']:.0f} h", delta="Meta: 500 h")
    with col4:
        st.metric("MTTR", f"{kpis['mttr']:.1f} h", delta="Meta: ≤8 h")
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Total Equipos", kpis['total_equipos'])
    with col6:
        st.metric("Operativos", kpis['operativos'])
    with col7:
        st.metric("Equipos Críticos", kpis['equipos_criticos'], delta_color="inverse")
    with col8:
        st.metric("Costos Mantenimiento", f"${kpis['costos_mantenimiento']:,.0f}")
    
    st.markdown("---")
    
    # Gráficos
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.plotly_chart(dash_mod.chart_estado_equipos(), use_container_width=True)
    
    with col_g2:
        st.plotly_chart(dash_mod.chart_tipo_equipos(), use_container_width=True)
    
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        st.plotly_chart(dash_mod.chart_consumo_combustible(), use_container_width=True)
    
    with col_g4:
        st.plotly_chart(dash_mod.chart_ordenes_por_tipo(), use_container_width=True)
    
    col_g5, col_g6 = st.columns(2)
    
    with col_g5:
        st.plotly_chart(dash_mod.chart_costos_mensuales(), use_container_width=True)
    
    with col_g6:
        st.plotly_chart(dash_mod.chart_horas_operacion(), use_container_width=True)
    
    st.markdown("---")
    
    # Alertas Críticas
    st.subheader("⚠️ Alertas Críticas")
    alertas = dash_mod.get_alertas_criticas()
    
    if not alertas.empty:
        st.dataframe(alertas, use_container_width=True)
    else:
        st.success("✅ No hay alertas críticas en este momento")
    
    # Auto-refresco
    if st.button("🔄 Actualizar Datos"):
        st.rerun()

# ============================================================
# PÁGINA: GEMELO DIGITAL
# ============================================================
def mostrar_gemelo_digital():
    st.title("🔧 Gemelo Digital")
    st.markdown("---")
    
    # Seleccionar equipo
    equipos_df = dash_mod.get_equipos_data()
    
    col_sel1, col_sel2 = st.columns([2, 1])
    
    with col_sel1:
        equipo_seleccionado = st.selectbox(
            "Seleccionar Equipo",
            options=equipos_df['id'].tolist(),
            format_func=lambda x: f"{equipos_df[equipos_df['id']==x]['codigo'].iloc[0]} - {equipos_df[equipos_df['id']==x]['nombre'].iloc[0]}"
        )
    
    with col_sel2:
        if st.button("🔄 Generar Datos en Tiempo Real"):
            gd_mod.generar_datos_en_tiempo_real(equipo_seleccionado)
            st.success("✅ Datos actualizados")
            st.rerun()
    
    # Obtener datos del equipo
    equipo = gd_mod.get_equipo_detalle(equipo_seleccionado)
    datos_actuales = gd_mod.get_ultimos_datos(equipo_seleccionado)
    
    if equipo:
        # Información general del equipo
        st.markdown(f"""
        <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3>{equipo['nombre']}</h3>
            <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;'>
                <div><strong>Código:</strong> {equipo['codigo']}</div>
                <div><strong>Tipo:</strong> {equipo['tipo']}</div>
                <div><strong>Marca:</strong> {equipo['marca']}</div>
                <div><strong>Modelo:</strong> {equipo['modelo']}</div>
                <div><strong>Año:</strong> {equipo['año_fabricacion']}</div>
                <div><strong>Horas:</strong> {equipo['horas_operacion']:.0f}</div>
                <div><strong>Ubicación:</strong> {equipo['ubicacion']}</div>
                <div><strong>Estado:</strong> <span style='color: {'#2ecc71' if equipo['estado']=='Operativo' else '#e74c3c'}'>{equipo['estado']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualización del gemelo y análisis de salud
        col_v1, col_v2 = st.columns([1.5, 1])
        
        with col_v1:
            st.subheader("🖥️ Representación Visual")
            fig = gd_mod.visualizar_gemelo_2d(equipo, datos_actuales)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_v2:
            st.subheader("📊 Análisis de Salud")
            if datos_actuales:
                analisis = gd_mod.analizar_salud_equipo(datos_actuales)
                for a in analisis:
                    with st.expander(f"{a['componente']} - {a['estado']}", expanded=True):
                        st.markdown(f"""
                        <div style='border-left: 4px solid {a['color']}; padding-left: 10px;'>
                            <p><strong>Estado:</strong> <span style='color: {a['color']}'>{a['estado']}</span></p>
                            <p>{a['parametro']}</p>
                            <p style='font-size: 12px; color: #666;'>{a['detalle']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No hay datos de sensores disponibles")
        
        st.markdown("---")
        
        # Tendencias históricas
        st.subheader("📈 Tendencias Históricas")
        fig_temp = dash_mod.chart_tendencias_temperatura(equipo_seleccionado)
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Simulación de fallas
        st.markdown("---")
        st.subheader("⚠️ Simulación de Escenarios de Falla")
        
        col_sim1, col_sim2 = st.columns([1, 1])
        
        with col_sim1:
            tipo_falla = st.selectbox(
                "Tipo de Falla a Simular",
                options=[
                    ('sobrecalentamiento_motor', '🔥 Sobrecalentamiento de Motor'),
                    ('perdida_presion_hidraulica', '💧 Pérdida de Presión Hidráulica'),
                    ('desgaste_neumaticos', '🛞 Desgaste de Neumáticos'),
                    ('falla_frenos', '🛑 Falla en Sistema de Frenos'),
                    ('consumo_excesivo', '⛽ Consumo Excesivo de Combustible')
                ],
                format_func=lambda x: x[1]
            )
            
            intensidad = st.slider("Intensidad de la Falla", 0.1, 1.0, 0.7, 0.1)
        
        with col_sim2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎯 Ejecutar Simulación", type="primary"):
                resultado = gd_mod.simular_falla(equipo_seleccionado, tipo_falla[0], intensidad)
                st.warning(f"**Simulación completada:** {tipo_falla[1]}")
                for alerta in resultado['alertas']:
                    st.error(alerta)
                st.rerun()

# ============================================================
# PÁGINA: MANTENIMIENTO
# ============================================================
def mostrar_mantenimiento():
    st.title("📋 Gestión de Mantenimiento")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Órdenes de Trabajo", "➕ Nueva Orden", "📜 Historial", "🔧 Generación Automática"])
    
    with tab1:
        # Filtros
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filtro_estado = st.selectbox("Filtrar por Estado", 
                options=['Todos', 'Pendiente', 'En Proceso', 'Completada', 'Cancelada'])
        with col_f2:
            filtro_tipo = st.selectbox("Filtrar por Tipo",
                options=['Todos', 'Preventivo', 'Correctivo', 'Predictivo'])
        
        fe = None if filtro_estado == 'Todos' else filtro_estado
        ft = None if filtro_tipo == 'Todos' else filtro_tipo
        
        ordenes = mant_mod.get_ordenes_trabajo(fe, ft)
        
        if not ordenes.empty:
            st.dataframe(ordenes, use_container_width=True)
            
            # Actualizar estado de orden
            st.subheader("Actualizar Estado de Orden")
            col_act1, col_act2, col_act3 = st.columns(3)
            
            with col_act1:
                orden_id = st.selectbox("Seleccionar Orden",
                    options=ordenes['id'].tolist(),
                    format_func=lambda x: ordenes[ordenes['id']==x]['numero_orden'].iloc[0])
            with col_act2:
                nuevo_estado = st.selectbox("Nuevo Estado",
                    options=['Pendiente', 'En Proceso', 'Completada', 'Cancelada'])
            with col_act3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Actualizar"):
                    datos = {'estado': nuevo_estado}
                    if nuevo_estado == 'En Proceso':
                        datos['fecha_inicio'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    elif nuevo_estado == 'Completada':
                        datos['fecha_fin'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    mant_mod.actualizar_orden_trabajo(orden_id, datos)
                    st.success("✅ Orden actualizada")
                    st.rerun()
        else:
            st.info("No hay órdenes de trabajo que mostrar")
    
    with tab2:
        st.subheader("Crear Nueva Orden de Trabajo")
        
        equipos_df = mant_mod.get_equipos_para_mantenimiento()
        tecnicos = mant_mod.get_tecnicos_disponibles()
        
        with st.form("nueva_orden"):
            col_n1, col_n2 = st.columns(2)
            
            with col_n1:
                eq_id = st.selectbox("Equipo",
                    options=equipos_df['id'].tolist(),
                    format_func=lambda x: f"{equipos_df[equipos_df['id']==x]['codigo'].iloc[0]} - {equipos_df[equipos_df['id']==x]['nombre'].iloc[0]}")
                tipo = st.selectbox("Tipo de Mantenimiento", ['Preventivo', 'Correctivo', 'Predictivo'])
                prioridad = st.selectbox("Prioridad", ['Baja', 'Media', 'Alta', 'Critica'])
            
            with col_n2:
                tecnico_id = st.selectbox("Técnico Asignado",
                    options=[None] + [t['id'] for t in tecnicos],
                    format_func=lambda x: "Sin asignar" if x is None else f"{next(t['nombre'] + ' ' + t['apellido'] for t in tecnicos if t['id']==x)}")
                fecha_programada = st.date_input("Fecha Programada")
                costo_estimado = st.number_input("Costo Estimado ($)", min_value=0.0, value=0.0)
            
            titulo = st.text_input("Título de la Orden")
            descripcion = st.text_area("Descripción Detallada")
            
            submit = st.form_submit_button("📝 Crear Orden", type="primary")
            
            if submit:
                if not titulo:
                    st.error("El título es obligatorio")
                else:
                    supervisor_id = st.session_state.usuario['user_id'] if st.session_state.usuario['rol'] in ['Supervisor', 'Administrador'] else None
                    exito, mensaje, orden_id = mant_mod.crear_orden_trabajo(
                        eq_id, tipo, prioridad, titulo, descripcion,
                        fecha_programada.strftime('%Y-%m-%d') if fecha_programada else None,
                        tecnico_id, supervisor_id, costo_estimado
                    )
                    if exito:
                        st.success(f"✅ Orden {mensaje} creada exitosamente")
                    else:
                        st.error(f"❌ Error: {mensaje}")
    
    with tab3:
        st.subheader("Historial de Mantenimiento")
        
        equipos_h = dash_mod.get_equipos_data()
        eq_filtro = st.selectbox("Filtrar por Equipo",
            options=['Todos'] + equipos_h['id'].tolist(),
            format_func=lambda x: "Todos los equipos" if x == 'Todos' else f"{equipos_h[equipos_h['id']==x]['codigo'].iloc[0]}")
        
        eq_id = None if eq_filtro == 'Todos' else eq_filtro
        historial = mant_mod.get_historial_mantenimiento(eq_id)
        
        if not historial.empty:
            st.dataframe(historial, use_container_width=True)
        else:
            st.info("No hay registros en el historial")
    
    with tab4:
        st.subheader("Generación Automática de Órdenes Preventivas")
        st.info("El sistema genera órdenes de mantenimiento preventivo basadas en las horas de operación de los equipos (cada 500 horas).")
        
        if st.button("🚀 Ejecutar Generación Automática", type="primary"):
            ordenes = mant_mod.generar_ordenes_automaticas()
            if ordenes:
                st.success(f"✅ Se generaron {len(ordenes)} órdenes automáticas:")
                for ot in ordenes:
                    st.write(f"  • {ot}")
            else:
                st.info("No se requirieron nuevas órdenes en este momento")

# ============================================================
# PÁGINA: ANÁLISIS PREDICTIVO
# ============================================================
def mostrar_predictivo():
    st.title("🤖 Análisis Predictivo")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Predicción por Equipo", "📋 Historial de Predicciones", "📊 Resumen General"])
    
    with tab1:
        equipos_df = dash_mod.get_equipos_data()
        
        equipo_pred = st.selectbox("Seleccionar Equipo para Análisis",
            options=equipos_df['id'].tolist(),
            format_func=lambda x: f"{equipos_df[equipos_df['id']==x]['codigo'].iloc[0]} - {equipos_df[equipos_df['id']==x]['nombre'].iloc[0]}")
        
        if st.button("🔍 Ejecutar Análisis Predictivo", type="primary"):
            with st.spinner("Entrenando modelos y realizando predicciones..."):
                resultado = pred_mod.predecir_falla_equipo(equipo_pred)
                
                if resultado:
                    st.success("✅ Análisis completado")
                    
                    # Mostrar resultados
                    col_r1, col_r2, col_r3 = st.columns(3)
                    
                    with col_r1:
                        st.metric("Probabilidad de Falla", 
                                 f"{resultado['probabilidad_falla']}%",
                                 delta=f"Precisión: {resultado['precision_modelo']}%")
                    
                    with col_r2:
                        st.metric("Horas Restantes Estimadas",
                                 f"{resultado['horas_restantes_estimadas']:.0f} h",
                                 delta=f"{resultado['dias_restantes_estimados']} días")
                    
                    with col_r3:
                        severidad_color = {
                            'Critica': '#e74c3c',
                            'Alta': '#e67e22',
                            'Media': '#f39c12',
                            'Baja': '#2ecc71'
                        }
                        st.markdown(f"""
                        <div style='text-align: center; padding: 20px; background: {severidad_color[resultado['severidad']]}; border-radius: 10px;'>
                            <h3 style='color: white; margin: 0;'>SEVERIDAD</h3>
                            <h2 style='color: white; margin: 10px 0;'>{resultado['severidad']}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Recomendación
                    st.subheader("💡 Recomendación")
                    st.text_area("", resultado['recomendacion'], height=250)
                    
                    # Factores críticos
                    st.subheader("📊 Factores Críticos")
                    factores_df = pd.DataFrame(resultado['factores_criticos'], 
                                              columns=['Parámetro', 'Importancia'])
                    factores_df['Importancia %'] = factores_df['Importancia'] * 100
                    
                    fig_fact = px.bar(factores_df, x='Parámetro', y='Importancia %',
                                     title='Importancia de Factores en la Predicción',
                                     color='Importancia %',
                                     color_continuous_scale='Reds')
                    st.plotly_chart(fig_fact, use_container_width=True)
                    
                    # Tendencias
                    st.subheader("📈 Análisis de Tendencias")
                    tendencias = pred_mod.analizar_tendencias_falla(equipo_pred)
                    if tendencias:
                        tend_df = pd.DataFrame([
                            {'Parámetro': k, 'Tendencia': v['tendencia'], 
                             'Variación': v['variacion'], 'Valor Actual': v['valor_actual']}
                            for k, v in tendencias.items()
                        ])
                        st.dataframe(tend_df, use_container_width=True)
    
    with tab2:
        st.subheader("Historial de Predicciones")
        predicciones = pred_mod.obtener_predicciones_guardadas()
        
        if not predicciones.empty:
            st.dataframe(predicciones, use_container_width=True)
        else:
            st.info("No hay predicciones almacenadas. Ejecute un análisis primero.")
    
    with tab3:
        st.subheader("Resumen General de Predicciones")
        resumen = pred_mod.obtener_resumen_predicciones()
        
        if resumen:
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.metric("Críticas", resumen.get('Critica', 0))
            with col_s2:
                st.metric("Altas", resumen.get('Alta', 0))
            with col_s3:
                st.metric("Medias", resumen.get('Media', 0))
            with col_s4:
                st.metric("Bajas", resumen.get('Baja', 0))
            
            # Gráfico de resumen
            resumen_df = pd.DataFrame([
                {'Severidad': k, 'Cantidad': v} for k, v in resumen.items()
            ])
            fig_res = px.pie(resumen_df, values='Cantidad', names='Severidad',
                           title='Distribución de Predicciones por Severidad',
                           color_discrete_map={'Critica': '#e74c3c', 'Alta': '#e67e22', 
                                              'Media': '#f39c12', 'Baja': '#2ecc71'})
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.info("No hay datos de predicciones disponibles")

# ============================================================
# PÁGINA: REPORTES
# ============================================================
def mostrar_reportes():
    st.title("📑 Generación de Reportes")
    st.markdown("---")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        st.markdown("""
        <div style='background: #e74c3c; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white;'>📄 Reporte PDF</h3>
            <p style='color: #fadbd8;'>Reporte ejecutivo con KPIs, gráficos y recomendaciones</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Generar PDF", type="primary", use_container_width=True):
            with st.spinner("Generando reporte PDF..."):
                pdf_bytes = rep_mod.generar_reporte_pdf()
                st.download_button(
                    label="⬇️ Descargar Reporte PDF",
                    data=pdf_bytes,
                    file_name=f"Reporte_Ejecutivo_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    with col_r2:
        st.markdown("""
        <div style='background: #3498db; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white;'>📝 Reporte Word</h3>
            <p style='color: #d6eaf8;'>Informe detallado con análisis y conclusiones</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Generar Word", type="primary", use_container_width=True):
            with st.spinner("Generando reporte Word..."):
                word_bytes = rep_mod.generar_reporte_word()
                st.download_button(
                    label="⬇️ Descargar Reporte Word",
                    data=word_bytes,
                    file_name=f"Informe_Detallado_{datetime.now().strftime('%Y%m%d_%H%M')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
    
    with col_r3:
        st.markdown("""
        <div style='background: #27ae60; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white;'>📊 Reporte Excel</h3>
            <p style='color: #d5f5e3;'>Datos completos con múltiples hojas y gráficos</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Generar Excel", type="primary", use_container_width=True):
            with st.spinner("Generando reporte Excel..."):
                excel_bytes = rep_mod.generar_reporte_excel()
                st.download_button(
                    label="⬇️ Descargar Reporte Excel",
                    data=excel_bytes,
                    file_name=f"Datos_Mantenimiento_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    st.markdown("---")
    
    # Vista previa de datos
    st.subheader("📋 Vista Previa de Datos")
    
    tab_pre1, tab_pre2, tab_pre3 = st.tabs(["KPIs Actuales", "Órdenes Recientes", "Resumen de Equipos"])
    
    with tab_pre1:
        kpis = dash_mod.get_kpis()
        kpis_df = pd.DataFrame([
            {'Indicador': 'Disponibilidad (%)', 'Valor': kpis['disponibilidad']},
            {'Indicador': 'OEE (%)', 'Valor': kpis['oee']},
            {'Indicador': 'MTBF (horas)', 'Valor': kpis['mtbf']},
            {'Indicador': 'MTTR (horas)', 'Valor': kpis['mttr']},
            {'Indicador': 'Costos ($)', 'Valor': kpis['costos_mantenimiento']},
            {'Indicador': 'Equipos Críticos', 'Valor': kpis['equipos_criticos']},
        ])
        st.dataframe(kpis_df, use_container_width=True)
    
    with tab_pre2:
        ordenes = dash_mod.get_ordenes_trabajo_data().head(10)
        st.dataframe(ordenes, use_container_width=True)
    
    with tab_pre3:
        equipos = dash_mod.get_equipos_data()
        st.dataframe(equipos, use_container_width=True)

# ============================================================
# PÁGINA: REPUESTOS
# ============================================================
def mostrar_repuestos():
    st.title("🔩 Gestión de Repuestos")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📦 Catálogo de Repuestos", "⚠️ Stock Bajo"])
    
    with tab1:
        repuestos = mant_mod.get_repuestos()
        if not repuestos.empty:
            st.dataframe(repuestos, use_container_width=True)
        else:
            st.info("No hay repuestos registrados")
    
    with tab2:
        bajo_stock = mant_mod.get_repuestos_bajo_stock()
        if not bajo_stock.empty:
            st.warning(f"⚠️ Hay {len(bajo_stock)} repuestos con stock bajo el mínimo")
            st.dataframe(bajo_stock, use_container_width=True)
        else:
            st.success("✅ Todos los repuestos tienen stock adecuado")

# ============================================================
# PÁGINA: GESTIÓN DE USUARIOS
# ============================================================
def mostrar_usuarios():
    st.title("👥 Gestión de Usuarios")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📋 Lista de Usuarios", "➕ Nuevo Usuario"])
    
    with tab1:
        usuarios = auth.get_all_users()
        if usuarios:
            df_usuarios = pd.DataFrame(usuarios)
            st.dataframe(df_usuarios, use_container_width=True)
        else:
            st.info("No hay usuarios registrados")
    
    with tab2:
        with st.form("nuevo_usuario"):
            col_u1, col_u2 = st.columns(2)
            
            with col_u1:
                username = st.text_input("Usuario*")
                password = st.text_input("Contraseña*", type="password")
                nombre = st.text_input("Nombre*")
                apellido = st.text_input("Apellido*")
            
            with col_u2:
                email = st.text_input("Email*")
                rol = st.selectbox("Rol*", ['Administrador', 'Ingeniero', 'Supervisor', 'Tecnico'])
                area = st.text_input("Área")
            
            submit = st.form_submit_button("📝 Crear Usuario", type="primary")
            
            if submit:
                if not all([username, password, nombre, apellido, email]):
                    st.error("Por favor complete todos los campos obligatorios (*)")
                else:
                    exito, mensaje = auth.register_user(username, password, nombre, apellido, email, rol, area)
                    if exito:
                        st.success(f"✅ {mensaje}")
                    else:
                        st.error(f"❌ {mensaje}")

# ============================================================
# PÁGINA: BITÁCORA
# ============================================================
def mostrar_bitacora():
    st.title("📝 Bitácora de Accesos")
    st.markdown("---")
    
    registros = auth.get_bitacora_accesos(200)
    if registros:
        df_bitacora = pd.DataFrame(registros)
        st.dataframe(df_bitacora, use_container_width=True)
    else:
        st.info("No hay registros en la bitácora")

# ============================================================
# PÁGINA: DOCUMENTACIÓN SCRUM
# ============================================================
def mostrar_scrum():
    st.title("📖 Documentación Scrum")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Visión del Proyecto", "📋 Product Backlog", "🚀 Planificación de Sprints", "📊 Artefactos"])
    
    with tab1:
        st.header("🎯 Visión del Proyecto")
        st.markdown("""
        ## Nombre del Proyecto
        **Sistema de Gemelos Digitales para Gestión de Mantenimiento de Equipos de Carguío en Minas a Tajo Abierto**
        
        ## Descripción
        Desarrollar una aplicación web basada en Python y Streamlit que implemente gemelos digitales 
        para monitorear, predecir y gestionar el mantenimiento de equipos de carguío en operaciones 
        mineras a tajo abierto.
        
        ## Objetivos Principales
        - ✅ Reducir tiempos de inactividad no programada de equipos
        - ✅ Implementar mantenimiento predictivo basado en datos
        - ✅ Mejorar la disponibilidad de la flota por encima del 90%
        - ✅ Optimizar costos de mantenimiento en un 15%
        - ✅ Proporcionar visibilidad en tiempo real del estado de los equipos
        
        ## Stakeholders
        - **Product Owner:** Gerencia de Mantenimiento
        - **Scrum Master:** Líder Técnico
        - **Equipo de Desarrollo:** 3 desarrolladores + 1 analista de datos
        - **Usuarios Finales:** Ingenieros, Supervisores, Técnicos de Mantenimiento
        
        ## Criterios de Éxito
        - Disponibilidad del sistema ≥ 99.5%
        - Precisión de predicción de fallas ≥ 80%
        - Reducción de MTTR en un 20%
        - Usuarios activos ≥ 80% del personal objetivo
        - Satisfacción del usuario ≥ 4.2/5
        """)
    
    with tab2:
        st.header("📋 Product Backlog")
        
        backlog_items = [
            {'ID': 'US-001', 'Historia': 'Como administrador, quiero gestionar usuarios y roles para controlar el acceso al sistema', 'Prioridad': 'Alta', 'Puntos': 8, 'Sprint': 1},
            {'ID': 'US-002', 'Historia': 'Como usuario, quiero iniciar sesión de forma segura para acceder a mis funcionalidades', 'Prioridad': 'Alta', 'Puntos': 5, 'Sprint': 1},
            {'ID': 'US-003', 'Historia': 'Como ingeniero, quiero ver un dashboard con KPIs para evaluar el estado de la flota', 'Prioridad': 'Alta', 'Puntos': 13, 'Sprint': 1},
            {'ID': 'US-004', 'Historia': 'Como supervisor, quiero visualizar el gemelo digital de un equipo para monitorear su estado en tiempo real', 'Prioridad': 'Alta', 'Puntos': 13, 'Sprint': 2},
            {'ID': 'US-005', 'Historia': 'Como técnico, quiero crear y gestionar órdenes de trabajo para organizar el mantenimiento', 'Prioridad': 'Alta', 'Puntos': 8, 'Sprint': 2},
            {'ID': 'US-006', 'Historia': 'Como ingeniero, quiero simular escenarios de falla para evaluar respuestas del sistema', 'Prioridad': 'Media', 'Puntos': 8, 'Sprint': 2},
            {'ID': 'US-007', 'Historia': 'Como administrador, quiero generar reportes en PDF para presentar a la gerencia', 'Prioridad': 'Alta', 'Puntos': 8, 'Sprint': 3},
            {'ID': 'US-008', 'Historia': 'Como ingeniero, quiero generar reportes en Word y Excel para análisis detallados', 'Prioridad': 'Media', 'Puntos': 8, 'Sprint': 3},
            {'ID': 'US-009', 'Historia': 'Como ingeniero, quiero ver predicciones de fallas para anticipar mantenimiento', 'Prioridad': 'Alta', 'Puntos': 13, 'Sprint': 3},
            {'ID': 'US-010', 'Historia': 'Como supervisor, quiero ver el historial de mantenimiento para analizar tendencias', 'Prioridad': 'Media', 'Puntos': 5, 'Sprint': 2},
            {'ID': 'US-011', 'Historia': 'Como administrador, quiero gestionar el inventario de repuestos para controlar stock', 'Prioridad': 'Media', 'Puntos': 8, 'Sprint': 3},
            {'ID': 'US-012', 'Historia': 'Como usuario, quiero ver alertas de equipos críticos para actuar rápidamente', 'Prioridad': 'Alta', 'Puntos': 5, 'Sprint': 1},
            {'ID': 'US-013', 'Historia': 'Como administrador, quiero revisar la bitácora de accesos para auditoría', 'Prioridad': 'Baja', 'Puntos': 3, 'Sprint': 3},
            {'ID': 'US-014', 'Historia': 'Como sistema, quiero generar órdenes automáticas para mantenimiento preventivo', 'Prioridad': 'Media', 'Puntos': 8, 'Sprint': 3},
            {'ID': 'US-015', 'Historia': 'Como usuario, quiero exportar datos para análisis externos', 'Prioridad': 'Baja', 'Puntos': 5, 'Sprint': 3},
        ]
        
        df_backlog = pd.DataFrame(backlog_items)
        st.dataframe(df_backlog, use_container_width=True)
        
        st.subheader("📊 Resumen por Sprint")
        sprint_summary = df_backlog.groupby('Sprint').agg(
            Cantidad=('ID', 'count'),
            Puntos_Totales=('Puntos', 'sum')
        ).reset_index()
        st.dataframe(sprint_summary, use_container_width=True)
    
    with tab3:
        st.header("🚀 Planificación de Sprints")
        
        # Sprint 1
        st.subheader("✅ Sprint 1 - Base del Sistema")
        st.markdown("""
        **Duración:** 2 semanas (10 días hábiles)
        
        **Objetivo del Sprint:** Establecer la base del sistema con autenticación, gestión de usuarios y dashboard principal.
        
        **Historias de Usuario:**
        - US-001: Gestión de usuarios y roles (8 pts)
        - US-002: Inicio de sesión seguro (5 pts)
        - US-003: Dashboard con KPIs (13 pts)
        - US-012: Alertas de equipos críticos (5 pts)
        
        **Total de Puntos:** 31 Story Points
        
        **Definición de Terminado:**
        - ✅ Código revisado por pares
        - ✅ Pruebas unitarias aprobadas
        - ✅ Pruebas de integración exitosas
        - ✅ Documentación actualizada
        - ✅ Aprobación del Product Owner
        
        **Capacidad del Equipo:** 3 desarrolladores × 7 horas/día × 10 días = 210 horas
        """)
        
        st.markdown("---")
        
        # Sprint 2
        st.subheader("⚙️ Sprint 2 - Gemelo Digital y Mantenimiento")
        st.markdown("""
        **Duración:** 2 semanas (10 días hábiles)
        
        **Objetivo del Sprint:** Implementar el gemelo digital, gestión de órdenes de trabajo y simulación de fallas.
        
        **Historias de Usuario:**
        - US-004: Visualización gemelo digital (13 pts)
        - US-005: Gestión de órdenes de trabajo (8 pts)
        - US-006: Simulación de escenarios de falla (8 pts)
        - US-010: Historial de mantenimiento (5 pts)
        
        **Total de Puntos:** 34 Story Points
        
        **Definición de Terminado:**
        - ✅ Todas las funcionalidades del Sprint 1 operativas
        - ✅ Gemelo digital visualiza datos en tiempo real
        - ✅ CRUD completo de órdenes de trabajo
        - ✅ Simulaciones generan alertas apropiadas
        - ✅ Documentación técnica actualizada
        """)
        
        st.markdown("---")
        
        # Sprint 3
        st.subheader("🤖 Sprint 3 - Reportes y Análisis Predictivo")
        st.markdown("""
        **Duración:** 2 semanas (10 días hábiles)
        
        **Objetivo del Sprint:** Implementar generación de reportes multi-formato, análisis predictivo y funcionalidades administrativas.
        
        **Historias de Usuario:**
        - US-007: Reportes en PDF (8 pts)
        - US-008: Reportes en Word y Excel (8 pts)
        - US-009: Predicciones de falla (13 pts)
        - US-011: Gestión de repuestos (8 pts)
        - US-013: Bitácora de accesos (3 pts)
        - US-014: Órdenes automáticas (8 pts)
        - US-015: Exportación de datos (5 pts)
        
        **Total de Puntos:** 53 Story Points
        
        **Definición de Terminado:**
        - ✅ Sistema completo y funcional
        - ✅ Todos los reportes se generan correctamente
        - ✅ Modelos predictivos entrenados y evaluados
        - ✅ Pruebas de aceptación de usuario exitosas
        - ✅ Manual de usuario completo
        - ✅ Despliegue en ambiente de producción
        """)
    
    with tab4:
        st.header("📊 Artefactos Scrum")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.subheader("📉 Burndown Chart - Sprint 1")
            # Datos simulados de burndown
            dias = list(range(1, 11))
            ideal = [31, 27.9, 24.8, 21.7, 18.6, 15.5, 12.4, 9.3, 6.2, 3.1, 0]
            real = [31, 30, 26, 22, 19, 16, 13, 10, 7, 4, 0]
            
            burndown_data = pd.DataFrame({
                'Día': dias + [10],
                'Ideal': ideal[:10],
                'Real': real[:10]
            })
            
            fig_burn = px.line(burndown_data, x='Día', y=['Ideal', 'Real'],
                              title='Burndown Chart - Sprint 1',
                              labels={'value': 'Story Points', 'Día': 'Día del Sprint'})
            fig_burn.update_traces(mode='lines+markers')
            st.plotly_chart(fig_burn, use_container_width=True)
        
        with col_a2:
            st.subheader("📊 Velocity Chart")
            velocity_data = pd.DataFrame({
                'Sprint': ['Sprint 1', 'Sprint 2', 'Sprint 3'],
                'Planificado': [31, 34, 53],
                'Completado': [31, 32, 50]
            })
            
            fig_vel = px.bar(velocity_data, x='Sprint', y=['Planificado', 'Completado'],
                           title='Velocity Chart',
                           barmode='group')
            st.plotly_chart(fig_vel, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("📝 Retrospectivas de Sprint")
        
        with st.expander("🔍 Retrospectiva Sprint 1"):
            st.markdown("""
            **Lo que salió bien:**
            - ✅ La base de datos se diseñó e implementó rápidamente
            - ✅ El módulo de autenticación funcionó desde el primer día
            - ✅ Buena comunicación en el equipo
            
            **A mejorar:**
            - ⚠️ Subestimamos la complejidad de los gráficos del dashboard
            - ⚠️ Falta de documentación inicial
            
            **Acciones de mejora:**
            - Dedicar tiempo a la planificación técnica al inicio de cada historia
            - Documentar a medida que se desarrolla
            """)
        
        with st.expander("🔍 Retrospectiva Sprint 2"):
            st.markdown("""
            **Lo que salió bien:**
            - ✅ El gemelo digital superó las expectativas
            - ✅ Integración con módulo de mantenimiento fluida
            - ✅ Revisiones de código más eficientes
            
            **A mejorar:**
            - ⚠️ Pruebas de simulación requirieron más tiempo del esperado
            
            **Acciones de mejora:**
            - Crear datos de prueba más realistas desde el inicio
            """)
        
        with st.expander("🔍 Retrospectiva Sprint 3"):
            st.markdown("""
            **Lo que salió bien:**
            - ✅ Los modelos predictivos lograron buena precisión
            - ✅ Reportes generados en todos los formatos solicitados
            - ✅ Entrega completa y funcional
            
            **A mejorar:**
            - ⚠️ Últimos días con mucha presión por tiempo
            
            **Acciones de mejora:**
            - Mejorar la estimación de historias complejas
            - Considerar buffers de tiempo en la planificación
            """)

# ============================================================
# PÁGINA: MOTOR DE IA
# ============================================================
def mostrar_motor_ia():
    st.title("⚙️ Motor de Inteligencia Artificial")
    st.markdown("---")
    
    # Inicializar motor si no existe
    if st.session_state.motor_ia is None:
        with st.spinner("🔄 Inicializando motor de IA..."):
            try:
                motor = MotorPredictivo()
                # Intentar cargar modelo guardado
                if os.path.exists(MODELS_DIR) and len(os.listdir(MODELS_DIR)) > 0:
                    if motor.cargar():
                        st.success("✅ Modelo cargado desde disco")
                    else:
                        st.info("Entrenando nuevo modelo...")
                        motor.cargar_datos()
                        motor.preparar_datos()
                        motor.entrenar_todos()
                        motor.evaluar_todos()
                        motor.comparar_algoritmos()
                        motor.guardar()
                        st.success("✅ Modelo entrenado y guardado")
                else:
                    st.info("No hay modelos guardados. Entrenando nuevo motor...")
                    motor.cargar_datos()
                    motor.preparar_datos()
                    motor.entrenar_todos()
                    motor.evaluar_todos()
                    motor.comparar_algoritmos()
                    motor.guardar()
                    st.success("✅ Motor de IA listo")
                
                st.session_state.motor_ia = motor
            except Exception as e:
                st.error(f"Error al inicializar motor: {str(e)}")
                return
    
    motor = st.session_state.motor_ia
    
    # Pestañas del motor de IA
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Comparativa Algoritmos", 
        "🔮 Predicción por Equipo", 
        "🏆 Mejor Algoritmo",
        "🔄 Reentrenar",
        "📋 Logs"
    ])
    
    with tab1:
        st.subheader("📊 Comparativa de los 5 Algoritmos")
        
        if motor.resultados_evaluacion:
            df_comparativa = motor.obtener_tabla_comparativa()
            st.dataframe(df_comparativa, use_container_width=True)
            
            # Gráfico de barras comparativo
            st.subheader("📈 Rendimiento por Algoritmo")
            metricas_df = pd.DataFrame([
                {'Algoritmo': r['algoritmo'].upper(), 
                 'F1-Score': r['f1_score'],
                 'AUC-ROC': r['auc_roc'],
                 'Precisión': r['accuracy']}
                for r in motor.resultados_evaluacion.values()
            ])
            
            metricas_melted = metricas_df.melt(id_vars='Algoritmo', var_name='Métrica', value_name='Valor')
            fig_comp = px.bar(metricas_melted, x='Algoritmo', y='Valor', color='Métrica',
                            barmode='group', title='Comparativa de Métricas por Algoritmo',
                            color_discrete_map={'F1-Score': '#3498db', 'AUC-ROC': '#2ecc71', 'Precisión': '#e74c3c'})
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Gráfico de tiempos
            st.subheader("⏱️ Tiempos de Inferencia")
            tiempos_df = pd.DataFrame([
                {'Algoritmo': r['algoritmo'].upper(), 'Tiempo (ms)': r['tiempo_inferencia_ms']}
                for r in motor.resultados_evaluacion.values()
            ])
            fig_tiempos = px.bar(tiempos_df, x='Algoritmo', y='Tiempo (ms)',
                               title='Tiempo de Inferencia por Predicción (ms)',
                               color='Tiempo (ms)', color_continuous_scale='Reds')
            st.plotly_chart(fig_tiempos, use_container_width=True)
            
            # Criterios de selección
            st.subheader("⚖️ Criterios de Selección Ponderados")
            criterios = motor.config['criterios_seleccion']
            crit_df = pd.DataFrame([
                {'Criterio': k.replace('_', ' ').title(), 'Peso (%)': round(v * 100, 1)}
                for k, v in criterios.items()
            ])
            fig_crit = px.pie(crit_df, values='Peso (%)', names='Criterio',
                             title='Ponderación de Criterios para Selección')
            st.plotly_chart(fig_crit, use_container_width=True)
        else:
            st.info("No hay resultados de evaluación. Ejecute reentrenamiento.")
    
    with tab2:
        st.subheader("🔮 Predicción de Falla por Equipo")
        
        equipos_df = dash_mod.get_equipos_data()
        equipo_sel = st.selectbox("Seleccionar Equipo",
            options=equipos_df['id'].tolist(),
            format_func=lambda x: f"{equipos_df[equipos_df['id']==x]['codigo'].iloc[0]} - {equipos_df[equipos_df['id']==x]['nombre'].iloc[0]}")
        
        # Seleccionar algoritmo
        alg_disponibles = list(motor.modelos.keys())
        alg_sel = st.selectbox("Algoritmo a usar (predeterminado: mejor)",
            options=['Mejor Automático'] + alg_disponibles)
        
        if st.button("🔍 Ejecutar Predicción", type="primary"):
            with st.spinner("Analizando datos del equipo..."):
                # Obtener datos del equipo
                datos_eq = gd_mod.get_ultimos_datos(equipo_sel)
                
                if datos_eq:
                    # Preparar vector de características
                    if hasattr(motor, 'caracteristicas'):
                        vector = []
                        for caract in motor.caracteristicas:
                            if caract in datos_eq:
                                vector.append(datos_eq[caract])
                            else:
                                # Para características derivadas, usar valor base
                                if 'temp_motor' in datos_eq:
                                    vector.append(datos_eq['temp_motor'])
                                else:
                                    vector.append(0)
                        
                        algoritmo = None if alg_sel == 'Mejor Automático' else alg_sel
                        resultado = motor.predecir(vector, algoritmo=algoritmo)
                        
                        # Mostrar resultados
                        col_r1, col_r2, col_r3 = st.columns(3)
                        
                        with col_r1:
                            st.metric("Probabilidad de Falla", 
                                     f"{resultado['probabilidad_falla']}%",
                                     delta=f"Algoritmo: {resultado['algoritmo_usado']}")
                        
                        with col_r2:
                            severidad_colores = {
                                'Critica': '#e74c3c', 'Alta': '#e67e22',
                                'Media': '#f39c12', 'Baja': '#2ecc71'
                            }
                            st.markdown(f"""
                            <div style='text-align: center; padding: 15px; background: {severidad_colores[resultado['severidad']]}; border-radius: 10px;'>
                                <h4 style='color: white; margin: 0;'>SEVERIDAD</h4>
                                <h3 style='color: white; margin: 5px 0;'>{resultado['severidad']}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_r3:
                            st.metric("Horas Restantes Estimadas", 
                                     f"{resultado['horas_restantes_estimadas']} h",
                                     delta=f"Tiempo inferencia: {resultado['tiempo_inferencia_ms']} ms")
                        
                        st.markdown("---")
                        
                        # Recomendación
                        st.subheader("💡 Recomendación")
                        st.text_area("", resultado['recomendacion'], height=200)
                        
                        # Factores influyentes
                        if resultado['factores_influyentes']:
                            st.subheader("📊 Factores Más Influyentes")
                            fact_df = pd.DataFrame(resultado['factores_influyentes'], columns=['Factor', 'Importancia'])
                            fact_df['Factor'] = fact_df['Factor'].astype(str).str.replace('_', ' ').str.title()
                            fig_fact = px.bar(fact_df.head(5), x='Factor', y='Importancia',
                                            title='Top 5 Factores de Influencia',
                                            color='Importancia', color_continuous_scale='Reds')
                            st.plotly_chart(fig_fact, use_container_width=True)
                    else:
                        st.warning("El motor no tiene características definidas. Reentrene el modelo.")
                else:
                    st.error("No hay datos de sensores para este equipo")
    
    with tab3:
        st.subheader("🏆 Mejor Algoritmo Seleccionado")
        
        if motor.mejor_algoritmo:
            mejor = motor.mejor_algoritmo
            puntuacion = motor.puntuaciones.get(mejor, {})
            info = motor.modelos.get(mejor, {})
            metricas = motor.resultados_evaluacion.get(mejor, {})
            
            col_m1, col_m2 = st.columns([1, 2])
            
            with col_m1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #2c3e50, #3498db); padding: 25px; border-radius: 15px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>🏆</h2>
                    <h3 style='color: white; margin: 10px 0;'>{mejor.upper()}</h3>
                    <p style='color: #ecf0f1; font-size: 18px;'>Puntuación: <strong>{puntuacion.get('puntuacion_general', 0):.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("F1-Score", metricas.get('f1_score', 0))
                st.metric("AUC-ROC", metricas.get('auc_roc', 0))
                st.metric("Precisión", metricas.get('accuracy', 0))
            
            with col_m2:
                st.write("**Desglose de Puntuación:**")
                desglose = pd.DataFrame([
                    {'Criterio': 'Rendimiento Predictivo', 
                     'Puntuación': puntuacion.get('rendimiento', 0),
                     'Peso': motor.config['criterios_seleccion']['rendimiento_predictivo']},
                    {'Criterio': 'Tiempo de Inferencia', 
                     'Puntuación': puntuacion.get('tiempo', 0),
                     'Peso': motor.config['criterios_seleccion']['tiempo_inferencia']},
                    {'Criterio': 'Interpretabilidad', 
                     'Puntuación': puntuacion.get('interpretabilidad', 0),
                     'Peso': motor.config['criterios_seleccion']['interpretabilidad']},
                    {'Criterio': 'Facilidad de Mantenimiento', 
                     'Puntuación': puntuacion.get('mantenibilidad', 0),
                     'Peso': motor.config['criterios_seleccion']['facilidad_mantenimiento']},
                ])
                st.dataframe(desglose, use_container_width=True)
                
                st.write("**Características del modelo:**")
                st.info(f"""
                - **Tipo:** Clasificación binaria (falla/no falla)
                - **Tiempo de entrenamiento:** {metricas.get('tiempo_entrenamiento_s', 0):.2f} segundos
                - **Tiempo de inferencia:** {metricas.get('tiempo_inferencia_ms', 0):.2f} ms por predicción
                - **Interpretabilidad:** {metricas.get('interpretabilidad', 0)}/10
                - **Mantenibilidad:** {metricas.get('mantenibilidad', 0)}/10
                """)
                
                # Importancia de características
                if info.get('importancias'):
                    st.subheader("📊 Importancia de Características")
                    imp_df = pd.DataFrame(info['importancias'][:10], columns=['Característica', 'Importancia'])
                    imp_df['Característica'] = imp_df['Característica'].astype(str).str.replace('_', ' ').str.title()
                    fig_imp = px.bar(imp_df, x='Importancia', y='Característica', orientation='h',
                                    title='Top 10 Características Más Importantes')
                    st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("No hay algoritmo seleccionado. Ejecute reentrenamiento.")
    
    with tab4:
        st.subheader("🔄 Reentrenar Motor de IA")
        
        st.warning("El reentrenamiento puede tardar varios minutos, especialmente para los modelos de deep learning.")
        
        col_op1, col_op2 = st.columns(2)
        
        with col_op1:
            algoritmos_a_entrenar = st.multiselect(
                "Algoritmos a entrenar",
                options=['random_forest', 'xgboost', 'svm', 'cnn_lstm', 'lstm_ae_rf'],
                default=['random_forest', 'xgboost', 'svm']
            )
        
        with col_op2:
            guardar_despues = st.checkbox("Guardar modelo después de entrenar", value=True)
        
        if st.button("🚀 Iniciar Reentrenamiento", type="primary"):
            with st.spinner("Reentrenando motor de IA..."):
                progress_bar = st.progress(0)
                
                # Cargar y preparar datos frescos
                motor.cargar_datos()
                progress_bar.progress(10)
                
                motor.preparar_datos()
                progress_bar.progress(20)
                
                # Limpiar modelos anteriores
                motor.modelos = {}
                motor.resultados_evaluacion = {}
                
                # Entrenar algoritmos seleccionados
                total = len(algoritmos_a_entrenar)
                for i, alg in enumerate(algoritmos_a_entrenar):
                    st.write(f"Entrenando {alg}...")
                    if alg == 'random_forest':
                        motor.entrenar_random_forest()
                    elif alg == 'xgboost':
                        motor.entrenar_xgboost()
                    elif alg == 'svm':
                        motor.entrenar_svm()
                    elif alg == 'cnn_lstm':
                        motor.entrenar_cnn_lstm()
                    elif alg == 'lstm_ae_rf':
                        motor.entrenar_lstm_ae_rf()
                    
                    progress_bar.progress(20 + int((i + 1) / total * 60))
                
                # Evaluar y comparar
                st.write("Evaluando algoritmos...")
                motor.evaluar_todos()
                progress_bar.progress(85)
                
                st.write("Seleccionando mejor algoritmo...")
                motor.comparar_algoritmos()
                progress_bar.progress(95)
                
                if guardar_despues:
                    motor.guardar()
                
                progress_bar.progress(100)
                
                st.success(f"""
                ✅ Reentrenamiento completado!
                - Algoritmos entrenados: {len(motor.modelos)}
                - Mejor algoritmo: {motor.mejor_algoritmo}
                - Puntuación: {motor.puntuaciones.get(motor.mejor_algoritmo, {}).get('puntuacion_general', 0):.4f}
                """)
                
                st.session_state.motor_ia = motor
    
    with tab5:
        st.subheader("📋 Logs del Motor de IA")
        
        logs = motor.obtener_logs(100)
        if logs:
            log_texto = ''.join(logs)
            st.text_area("", log_texto, height=400)
        else:
            st.info("No hay logs disponibles")
        
        if st.button("🗑️ Limpiar Logs"):
            import os
            log_file = os.path.join(os.path.dirname(MODELS_DIR), 'logs', 'motor_ia.log')
            if os.path.exists(log_file):
                os.remove(log_file)
                st.success("Logs limpiados")
                st.rerun()


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================
def main():
    if st.session_state.usuario is None:
        mostrar_login()
    else:
        mostrar_sidebar()
        
        pagina = st.session_state.pagina_actual
        
        if pagina == 'Dashboard':
            mostrar_dashboard()
        elif pagina == 'Gemelo Digital':
            mostrar_gemelo_digital()
        elif pagina == 'Mantenimiento':
            mostrar_mantenimiento()
        elif pagina == 'Predictivo':
            mostrar_predictivo()
        elif pagina == 'Motor IA':
            mostrar_motor_ia()
        elif pagina == 'Reportes':
            mostrar_reportes()
        elif pagina == 'Repuestos':
            mostrar_repuestos()
        elif pagina == 'Usuarios':
            mostrar_usuarios()
        elif pagina == 'Bitacora':
            mostrar_bitacora()
        elif pagina == 'Scrum':
            mostrar_scrum()
        else:
            mostrar_dashboard()

if __name__ == '__main__':
    main()

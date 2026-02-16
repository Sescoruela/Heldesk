"""
frontend.py — Interfaz Streamlit para el Helpdesk interno con RAG.

Ejecutar con:
    streamlit run frontend.py
"""

import streamlit as st
from backend import KnowledgeBase, rag_pipeline, EXTRACTORS

# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="🛠️ Helpdesk Interno",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Estado de sesión
# ──────────────────────────────────────────────

if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()

if "history" not in st.session_state:
    st.session_state.history = []  # lista de {"query", "response"}

kb: KnowledgeBase = st.session_state.kb

# ──────────────────────────────────────────────
# Sidebar — Configuración y carga de documentos
# ──────────────────────────────────────────────

with st.sidebar:
    # ── Bloque 1: API Key Embeddings ──
    st.header("🔗 Embeddings (xAI)")
    api_key_embeddings = st.text_input(
        "🔑 API Key para Embeddings",
        type="password",
        help="API key de xAI usada para generar embeddings de documentos y consultas.",
    )

    st.divider()

    # ── Bloque 2: API Key LLM ──
    st.header("🤖 LLM — Grok (xAI)")
    api_key_llm = st.text_input(
        "🔑 API Key para Grok (LLM)",
        type="password",
        help="API key de xAI usada para generar respuestas con Grok.",
    )

    # Modelo
    model = st.selectbox(
        "Modelo Grok",
        options=["grok-3-mini", "grok-3"],
        index=0,
        help="Selecciona el modelo de Grok a utilizar.",
    )

    # Top-K
    top_k = st.slider(
        "🔎 Fragmentos a recuperar (top-k)",
        min_value=1,
        max_value=15,
        value=5,
        help="Número de fragmentos del contexto a enviar al LLM.",
    )

    st.divider()

    # ── Carga de documentos ──
    st.header("📁 Base de conocimiento")

    uploaded_files = st.file_uploader(
        "Sube documentos de soporte",
        type=["pdf", "docx", "txt", "md", "log"],
        accept_multiple_files=True,
        help="Formatos soportados: PDF, DOCX, TXT, MD, LOG",
    )

    if uploaded_files:
        if not api_key_embeddings:
            st.warning("⚠️ Introduce la API Key de Embeddings antes de subir documentos.")
        else:
            for f in uploaded_files:
                with st.spinner(f"Procesando *{f.name}*…"):
                    try:
                        n = kb.add_document(f.name, f.read(), api_key=api_key_embeddings)
                        if n > 0:
                            st.success(f"✅ **{f.name}** — {n} fragmentos indexados")
                        else:
                            st.info(f"ℹ️ **{f.name}** ya estaba indexado o vacío")
                    except ValueError as e:
                        st.error(f"❌ {e}")
                    except Exception as e:
                        st.error(f"❌ Error al generar embeddings: {e}")

    # Estadísticas de la KB
    if kb.total_chunks > 0:
        st.divider()
        st.metric("Documentos indexados", kb.total_documents)
        st.metric("Fragmentos totales", kb.total_chunks)
        with st.expander("📄 Fuentes cargadas"):
            for src in kb.sources:
                st.write(f"- {src}")

        if st.button("🗑️ Resetear base de conocimiento", type="secondary"):
            kb.reset()
            st.session_state.kb = KnowledgeBase()
            st.rerun()
    else:
        st.caption("Aún no hay documentos cargados.")

# ──────────────────────────────────────────────
# Área principal — Consulta de incidencias
# ──────────────────────────────────────────────

st.title("🛠️ Helpdesk Interno — Asistente RAG")
st.caption("Describe tu incidencia técnica y recibe una respuesta estructurada basada en la documentación interna.")

# Formulario de consulta
with st.form("incident_form", clear_on_submit=True):
    query = st.text_area(
        "📝 Describe tu incidencia",
        height=120,
        placeholder=(
            "Ejemplo: El usuario no puede conectarse a la VPN corporativa "
            "desde su portátil con Windows 11. Aparece el error «TLS handshake failed»."
        ),
    )
    submitted = st.form_submit_button(
        "🚀 Buscar solución",
        type="primary",
        use_container_width=True,
    )

# Procesar consulta
if submitted:
    # Validaciones
    if not api_key_embeddings or not api_key_llm:
        st.error("⚠️ Introduce ambas API Keys (Embeddings y LLM) en la barra lateral.")
    elif not query.strip():
        st.warning("⚠️ Escribe una descripción de la incidencia.")
    elif kb.total_chunks == 0:
        st.warning("⚠️ Sube al menos un documento a la base de conocimiento antes de consultar.")
    else:
        with st.spinner("🔍 Buscando en la base de conocimiento y generando respuesta…"):
            try:
                result = rag_pipeline(
                    query=query.strip(),
                    kb=kb,
                    api_key_embeddings=api_key_embeddings,
                    api_key_llm=api_key_llm,
                    top_k=top_k,
                    model=model,
                )
                # Guardar en historial
                st.session_state.history.insert(0, {
                    "query":    query.strip(),
                    "answer":   result["answer"],
                    "sources":  result["sources"],
                    "context":  result["context"],
                })
            except Exception as e:
                st.error(f"❌ Error al generar la respuesta: {e}")

# ──────────────────────────────────────────────
# Mostrar historial de respuestas
# ──────────────────────────────────────────────

for i, entry in enumerate(st.session_state.history):
    with st.container(border=True):
        st.markdown(f"**🗨️ Incidencia:** {entry['query']}")
        st.divider()
        st.markdown(entry["answer"])

        # Fragmentos de contexto recuperados (colapsable)
        with st.expander("🧩 Fragmentos de contexto recuperados"):
            if entry["context"]:
                for j, ctx in enumerate(entry["context"], 1):
                    st.markdown(
                        f"**Fragmento {j}** — `{ctx['source']}` "
                        f"(similitud: {ctx['score']:.2f})"
                    )
                    st.code(ctx["chunk"], language=None)
            else:
                st.caption("No se recuperaron fragmentos.")

    if i < len(st.session_state.history) - 1:
        st.markdown("---")

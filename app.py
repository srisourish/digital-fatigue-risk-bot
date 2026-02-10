import streamlit as st
import tempfile
import json
from datetime import datetime, timedelta

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🧠 Digital Fatigue Risk Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .risk-high { background-color: #ff6b6b; }
    .risk-moderate { background-color: #ffd93d; }
    .risk-low { background-color: #6bcf7f; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Digital Fatigue Risk Detection Bot")
st.markdown("### Analyze your digital routine and get personalized recommendations")

# =========================
# INITIALIZE SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_assessment" not in st.session_state:
    st.session_state.current_assessment = None

# =========================
# LOAD OLLAMA LLM
# =========================
@st.cache_resource
def load_llm():
    return Ollama(
        model="phi3",
        temperature=0.3
    )

llm = load_llm()

# =========================
# RAG SETUP FUNCTIONS
# =========================
@st.cache_resource
def load_vectorstore(uploaded_file):
    """Load PDF and create vector store"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        if not documents:
            return None
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)
        
        embeddings = OllamaEmbeddings(model="phi3")
        vectordb = FAISS.from_documents(chunks, embeddings)
        
        return vectordb
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

def create_rag_chain(vectordb):
    """Create RAG chain from vector store"""
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    prompt = ChatPromptTemplate.from_template(
        """You are a digital wellness and fatigue expert assistant.
Use the provided context from health and productivity resources to answer questions about digital fatigue, screen time effects, and recovery strategies.

Context from knowledge base:
{context}

User Question:
{input}

Provide clear, practical, and evidence-based advice based on the context. If the context doesn't contain relevant information, provide general best practices for digital wellness."""
    )
    
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

# =========================
# FATIGUE RISK ASSESSMENT LOGIC
# =========================
def calculate_fatigue_risk(screen_hours, sleep_hours, breaks, exercise, blue_light_protection):
    """Calculate fatigue risk and return detailed metrics"""
    risk_score = 0
    risk_factors = []
    
    # Screen time assessment
    if screen_hours > 10:
        risk_score += 3
        risk_factors.append("❌ Excessive screen time (>10 hours)")
    elif screen_hours > 8:
        risk_score += 2
        risk_factors.append("⚠️ High screen time (8-10 hours)")
    elif screen_hours > 6:
        risk_score += 1
        risk_factors.append("🟡 Moderate screen time (6-8 hours)")
    else:
        risk_factors.append("✅ Healthy screen time (<6 hours)")
    
    # Sleep assessment
    if sleep_hours < 5:
        risk_score += 3
        risk_factors.append("❌ Severe sleep deprivation (<5 hours)")
    elif sleep_hours < 6:
        risk_score += 2
        risk_factors.append("⚠️ Inadequate sleep (5-6 hours)")
    elif sleep_hours < 7:
        risk_score += 1
        risk_factors.append("🟡 Below recommended sleep (6-7 hours)")
    else:
        risk_factors.append("✅ Healthy sleep duration (7+ hours)")
    
    # Breaks assessment
    if breaks == 0:
        risk_score += 3
        risk_factors.append("❌ No breaks taken")
    elif breaks < 2:
        risk_score += 2
        risk_factors.append("⚠️ Insufficient breaks (<2/day)")
    elif breaks < 4:
        risk_score += 1
        risk_factors.append("🟡 Below optimal breaks (2-4/day)")
    else:
        risk_factors.append("✅ Good break frequency (4+ breaks/day)")
    
    # Exercise assessment
    if exercise == 0:
        risk_score += 2
        risk_factors.append("❌ No physical activity")
    elif exercise < 30:
        risk_score += 1
        risk_factors.append("⚠️ Below recommended exercise (<30 min)")
    else:
        risk_factors.append("✅ Adequate physical activity (30+ min)")
    
    # Blue light protection
    if not blue_light_protection:
        risk_score += 1
        risk_factors.append("⚠️ No blue light protection")
    else:
        risk_factors.append("✅ Using blue light protection")
    
    # Determine risk level
    if risk_score <= 2:
        risk_level = "Low"
        color = "green"
    elif risk_score <= 5:
        risk_level = "Moderate"
        color = "orange"
    elif risk_score <= 8:
        risk_level = "High"
        color = "red"
    else:
        risk_level = "Critical"
        color = "darkred"
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "color": color,
        "factors": risk_factors
    }

def get_ai_recommendations(risk_data, screen_hours, sleep_hours, breaks, exercise, work_type):
    """Get AI-powered recommendations using RAG or fallback"""
    prompt = f"""Based on the following digital routine assessment, provide 5 specific, actionable recommendations:

User Profile:
- Risk Level: {risk_data['risk_level']} (Score: {risk_data['risk_score']}/11)
- Daily Screen Time: {screen_hours} hours
- Sleep Duration: {sleep_hours} hours
- Breaks per Day: {breaks}
- Physical Activity: {exercise} minutes
- Primary Activity: {work_type}

Risk Factors:
{chr(10).join(risk_data['factors'])}

Please provide:
1. Immediate actions (this week)
2. Long-term lifestyle changes
3. Specific tools or techniques
4. When to seek professional help
5. Success metrics to track progress

Make recommendations practical and achievable."""
    
    if st.session_state.rag_chain:
        try:
            response = st.session_state.rag_chain.invoke({"input": prompt})
            return response.get("answer", "")
        except Exception as e:
            st.warning(f"RAG chain error: {e}. Using default recommendations.")
            return get_default_recommendations(risk_data, work_type)
    else:
        return get_default_recommendations(risk_data, work_type)

def get_default_recommendations(risk_data, work_type):
    """Fallback recommendations when RAG is not available"""
    recommendations = {
        "Low": f"""
✅ Your digital routine is healthy! Maintain these habits:
1. Continue your current screen time pattern and break schedule
2. Schedule weekly digital detox activities (tech-free hours)
3. Monitor your metrics monthly to ensure consistency
4. Share your healthy habits with colleagues/friends
5. Track eye strain and cognitive performance to establish baseline
""",
        "Moderate": f"""
🟡 Your digital routine needs some adjustments:
1. Reduce daily screen time by 1-2 hours gradually
2. Implement the 20-20-20 rule (every 20 min, look away for 20 sec at 20ft distance)
3. Add 1-2 more breaks during your {work_type} sessions
4. Increase sleep duration by 1 hour
5. Start with 15-minute walks after intensive work sessions
""",
        "High": f"""
⚠️ Your digital routine is putting significant strain on your health:
1. URGENT: Increase sleep to at least 7-8 hours immediately
2. Reduce screen time by 2-3 hours daily through breaks or task switching
3. Take a 5-minute break every 25-30 minutes of {work_type}
4. Use blue light filters on all devices after 6 PM
5. Consult a health professional if symptoms persist
""",
        "Critical": f"""
❌ Your digital routine is critically unsustainable:
1. IMMEDIATE: Consult a healthcare professional
2. Implement a hard limit on device usage (8 hours maximum)
3. Get at least 8 hours of quality sleep nightly
4. Take mandatory 10-minute breaks every 30 minutes
5. Consider temporary work schedule modifications
"""
    }
    return recommendations.get(risk_data["risk_level"], recommendations["Moderate"])

# =========================
# SIDEBAR – KNOWLEDGE BASE & HISTORY
# =========================
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.markdown("Upload health/wellness PDFs to enhance AI recommendations")
    
    uploaded_pdf = st.file_uploader(
        "Upload PDF Knowledge Base",
        type=["pdf"],
        key="pdf_uploader"
    )
    
    if uploaded_pdf:
        with st.spinner("Loading knowledge base..."):
            vectordb = load_vectorstore(uploaded_pdf)
            if vectordb:
                st.session_state.vectordb = vectordb
                st.session_state.rag_chain = create_rag_chain(vectordb)
                st.success("✅ Knowledge base loaded successfully!")
                st.info(f"Loaded PDF: {uploaded_pdf.name}")
            else:
                st.error("Failed to load PDF. Please try another file.")
    
    if st.session_state.rag_chain:
        st.success("🎯 RAG mode active - Using knowledge base for recommendations")
    else:
        st.info("📖 Upload a PDF to enable AI-enhanced recommendations")
    
    # History
    st.divider()
    st.header("📊 Assessment History")
    if st.session_state.history:
        for i, assessment in enumerate(st.session_state.history[-5:], 1):
            st.write(f"{i}. {assessment['timestamp']} - **{assessment['risk_level']} Risk**")
    else:
        st.info("No assessments yet")
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
    
    st.divider()
    if st.button("💬 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# =========================
# MAIN CONTENT – INPUT FORM
# =========================
st.header("📝 Digital Routine Assessment")
st.markdown("Fill in your typical daily digital habits to get a personalized fatigue risk analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⏱️ Time Metrics")
    screen_hours = st.slider(
        "Daily Screen Time (hours)",
        0.0, 16.0, 8.0,
        help="Total hours spent on screens daily"
    )
    
    sleep_hours = st.slider(
        "Sleep Duration (hours)",
        2.0, 12.0, 7.0,
        help="Average nightly sleep duration"
    )
    
    breaks_per_day = st.number_input(
        "Breaks per Day",
        0, 20, 3,
        help="Number of breaks (5+ min) from screens daily"
    )

with col2:
    st.subheader("🏃 Wellness Metrics")
    exercise_minutes = st.slider(
        "Physical Activity (minutes/day)",
        0, 120, 30,
        help="Minutes of exercise daily"
    )
    
    work_type = st.selectbox(
        "Primary Digital Activity",
        ["Coding", "Studying", "Office Work", "Gaming", "Content Creation", "Remote Teaching", "Other"]
    )
    
    blue_light_protection = st.checkbox(
        "Using Blue Light Protection",
        value=False,
        help="Screen filters, blue light glasses, or apps"
    )

st.divider()

# =========================
# ANALYSIS & RESULTS
# =========================
if st.button("🔍 Analyze Digital Fatigue Risk", use_container_width=True):
    # Calculate risk
    risk_data = calculate_fatigue_risk(
        screen_hours, sleep_hours, breaks_per_day, exercise_minutes, blue_light_protection
    )
    
    # Store current assessment for chat context
    st.session_state.current_assessment = {
        "risk_level": risk_data["risk_level"],
        "risk_score": risk_data["risk_score"],
        "screen_hours": screen_hours,
        "sleep_hours": sleep_hours,
        "breaks": breaks_per_day,
        "exercise": exercise_minutes,
        "work_type": work_type,
        "blue_light_protection": blue_light_protection,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    
    # Add to history
    st.session_state.history.append({
        "timestamp": st.session_state.current_assessment["timestamp"],
        "risk_level": risk_data["risk_level"],
        "screen_time": screen_hours,
        "sleep": sleep_hours,
        "score": risk_data["risk_score"]
    })
    
    # Display Results
    st.markdown("---")
    st.header("📊 Assessment Results")
    
    # Risk Gauge
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Display risk score as metric
        risk_percentage = (risk_data["risk_score"] / 11) * 100
        st.metric(
            "Fatigue Risk Score",
            f"{risk_data['risk_score']}/11",
            delta=f"{risk_percentage:.0f}%",
            delta_color="inverse"
        )
    
    # Risk Level Summary
    st.markdown(f"""
    <div class='metric-card'>
    <h2 style='text-align: center;'>Risk Level: {risk_data['risk_level']}</h2>
    <p style='text-align: center; font-size: 16px;'>Score: {risk_data['risk_score']}/11</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk Factors
    st.subheader("📋 Risk Factor Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Contributing Factors")
        for factor in risk_data["factors"]:
            st.markdown(f"• {factor}")
    
    with col2:
        st.markdown("### Your Metrics")
        metrics_display = [
            ("Screen Time", f"{screen_hours}h", "✅" if screen_hours <= 6 else "⚠️" if screen_hours <= 8 else "❌"),
            ("Sleep Hours", f"{sleep_hours}h", "✅" if sleep_hours >= 7 else "⚠️" if sleep_hours >= 6 else "❌"),
            ("Daily Breaks", f"{breaks_per_day}", "✅" if breaks_per_day >= 4 else "⚠️" if breaks_per_day >= 2 else "❌"),
            ("Exercise", f"{exercise_minutes}m", "✅" if exercise_minutes >= 30 else "⚠️" if exercise_minutes >= 15 else "❌"),
            ("Blue Light Protection", "Yes" if blue_light_protection else "No", "✅" if blue_light_protection else "❌")
        ]
        for metric, value, status in metrics_display:
            st.write(f"{status} **{metric}**: {value}")
    
    st.divider()
    
    # AI Recommendations
    st.subheader("🤖 Personalized Recommendations")
    
    with st.spinner("Generating personalized recommendations..."):
        recommendations = get_ai_recommendations(
            risk_data, screen_hours, sleep_hours, breaks_per_day, exercise_minutes, work_type
        )
    
    st.markdown(recommendations)
    
    # Recovery Timeline
    st.subheader("📈 Recovery Timeline")
    
    timeline_data = {
        "Week": ["Wk 1", "Wk 2", "Wk 3", "Wk 4", "Mo 2", "Mo 3"],
        "Progress %": [10, 25, 40, 55, 75, 90]
    }
    
    st.bar_chart(data=timeline_data, x="Week", y="Progress %", use_container_width=True)
    
    # Action Plan
    st.subheader("✅ 7-Day Action Plan")
    
    action_plan = f"""
    **Day 1-2: Assessment & Planning**
    - Review your current digital habits
    - Set up monitoring tools (screen time apps)
    - Schedule your breaks in calendar
    
    **Day 3-4: Implementation**
    - Start with one recommended change (easiest first)
    - Try blue light filters on devices
    - Establish a {work_type} schedule with breaks
    
    **Day 5-7: Optimization**
    - Adjust sleep schedule by 30 minutes
    - Add physical activity to break periods
    - Monitor energy levels and eye strain
    
    """
    st.info(action_plan)
    
    # Export Results
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Assessment Report"):
            report = f"""
Digital Fatigue Risk Assessment Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SUMMARY
-------
Risk Level: {risk_data['risk_level']}
Risk Score: {risk_data['risk_score']}/11

METRICS
-------
Screen Time: {screen_hours} hours/day
Sleep Duration: {sleep_hours} hours
Daily Breaks: {breaks_per_day}
Physical Activity: {exercise_minutes} minutes
Primary Activity: {work_type}
Blue Light Protection: {'Yes' if blue_light_protection else 'No'}

RECOMMENDATIONS
---------------
{recommendations}

---
Next Assessment: {(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")}
"""
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"fatigue_assessment_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("🔄 New Assessment"):
            st.rerun()

# =========================
# INTERACTIVE CHATBOT SECTION
# =========================
st.divider()
st.header("💬 Ask Your Wellness Assistant")
st.markdown("Chat with the AI for personalized advice and clarifications about your assessment")

# Chat interface
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about your digital fatigue risk, recovery tips, or health recommendations...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Construct context-aware prompt
                context_prompt = user_input
                
                if st.session_state.current_assessment:
                    assessment = st.session_state.current_assessment
                    context_prompt = f"""Based on the user's latest assessment:
- Risk Level: {assessment['risk_level']}
- Screen Time: {assessment.get('screen_hours', 'N/A')} hours/day
- Sleep: {assessment.get('sleep_hours', 'N/A')} hours
- Daily Breaks: {assessment.get('breaks', 'N/A')}
- Activity Type: {assessment.get('work_type', 'N/A')}

User's question: {user_input}

Provide helpful, personalized advice based on their digital habits."""
                
                try:
                    # Use RAG chain if available, otherwise use direct LLM
                    if st.session_state.rag_chain:
                        response = st.session_state.rag_chain.invoke({"input": context_prompt})
                        bot_response = response.get("answer", "I couldn't generate a response.")
                    else:
                        # Direct LLM call
                        bot_response = llm.invoke(context_prompt)
                    
                    st.markdown(bot_response)
                    
                    # Add bot response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": bot_response
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again."
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Digital Fatigue Risk Bot v1.0</strong></p>
    <p>⚠️ <em>Disclaimer: This tool provides general guidance. For persistent symptoms, consult a healthcare professional.</em></p>
    <p>Powered by LangChain + Ollama + Streamlit</p>
</div>
""", unsafe_allow_html=True)

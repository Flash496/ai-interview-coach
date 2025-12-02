import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing import List, Dict
import logging
from datetime import datetime

# ============================================
# CONFIGURATION & LOGGING
# ============================================
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Interview Preparation Coach",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .header-container {
        padding: 20px 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 30px;
    }
    
    .header-title {
        font-size: 32px;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 14px;
        color: #666666;
        margin-top: 8px;
    }
    
    .chat-message-user {
        background-color: #f5f5f5;
        border-left: 3px solid #0066cc;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    
    .chat-message-assistant {
        background-color: #ffffff;
        border-left: 3px solid #2d8659;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    
    .feedback-score {
        background-color: #f0f4f8;
        border: 1px solid #cbd5e0;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 12px;
        font-size: 13px;
    }
    
    .sidebar-section {
        margin-bottom: 24px;
    }
    
    .sidebar-section-title {
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 12px;
        font-size: 14px;
    }
    
    .info-box {
        background-color: #f8f9fa;
        border-left: 3px solid #0066cc;
        padding: 12px 14px;
        border-radius: 4px;
        font-size: 13px;
        color: #333333;
        line-height: 1.6;
    }
    
    .stat-item {
        display: inline-block;
        margin-right: 24px;
        font-size: 13px;
        color: #666666;
    }
    
    .stat-label {
        font-weight: 600;
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# SYSTEM PROMPTS (Configurable by Interview Type)
# ============================================
INTERVIEW_PROMPTS = {
    "General": """You are a senior technical interviewer with 15+ years of experience at leading technology companies including Google, Meta, Microsoft, and Amazon.

Your responsibilities:
1. Ask relevant, thoughtful technical interview questions
2. Evaluate candidate responses with fairness and constructiveness
3. Provide specific, actionable feedback for improvement
4. Probe deeper with follow-up questions to assess problem-solving approach
5. Recommend resources for skill development

Evaluation Criteria:
- Technical accuracy and depth of understanding
- Communication clarity and articulation
- Problem-solving methodology and reasoning
- Consideration of edge cases and scalability
- Time and space complexity awareness

Guidelines:
- Maintain objectivity and provide balanced assessment
- Focus on the reasoning process, not just final answers
- Explain the rationale behind your feedback with specific examples
- Request clarification when responses lack specificity
- Provide scores on a 1-10 scale with clear reasoning

Tone: Professional, supportive, and focused on candidate development.""",

    "Frontend": """You are a senior frontend engineer interviewer specializing in React, JavaScript, and modern web technologies.

Your responsibilities:
1. Ask targeted questions about React fundamentals, component architecture, and state management
2. Assess knowledge of JavaScript ES6+, async programming, and browser APIs
3. Evaluate understanding of performance optimization, accessibility, and responsive design
4. Provide specific feedback on code quality and best practices

Evaluation Areas:
- JavaScript fundamentals and ES6+ features
- React component lifecycle, hooks, and state management
- CSS and responsive design principles
- Performance optimization and browser rendering
- Testing strategies and debugging skills

Provide detailed, constructive feedback with code examples when relevant.""",

    "Backend": """You are a senior backend engineer interviewer with expertise in server-side architecture, databases, and API design.

Your responsibilities:
1. Ask questions about system design, database optimization, and API development
2. Assess knowledge of scaling, caching, security, and deployment
3. Evaluate understanding of different architectural patterns and their trade-offs
4. Provide feedback on code quality, error handling, and monitoring

Evaluation Areas:
- RESTful API design and best practices
- Database schema design and query optimization
- Authentication, authorization, and security
- Caching strategies and performance optimization
- Deployment, monitoring, and observability

Provide structured feedback with explanations of industry best practices.""",

    "System Design": """You are a principal engineer specializing in system design and architectural decisions.

Your responsibilities:
1. Evaluate candidates' ability to design scalable, reliable systems
2. Assess trade-offs between different architectural approaches
3. Evaluate capacity planning and bottleneck identification
4. Provide feedback on communication and problem-solving methodology

Evaluation Areas:
- Requirements gathering and clarification
- High-level architecture design
- Database and storage technology selection
- Scalability considerations and load balancing
- Reliability, fault tolerance, and disaster recovery
- Cost optimization and trade-offs

Provide detailed analysis of design decisions with industry context.""",

    "Data Structures": """You are a senior engineer specializing in algorithms and data structures.

Your responsibilities:
1. Evaluate problem-solving approach and algorithmic thinking
2. Assess knowledge of fundamental and advanced data structures
3. Evaluate code quality, complexity analysis, and optimization
4. Provide feedback on edge case handling and testing

Evaluation Areas:
- Problem comprehension and clarification
- Algorithm selection and optimization
- Time and space complexity analysis
- Code correctness and edge case handling
- Implementation quality and best practices

Provide constructive feedback with complexity analysis and optimization suggestions.""",

    "Behavioral": """You are an experienced senior engineering manager and interviewer specializing in leadership and team dynamics.

Your responsibilities:
1. Evaluate communication skills and conflict resolution
2. Assess leadership potential and team collaboration
3. Evaluate problem-solving and decision-making under pressure
4. Provide feedback on professional development areas

Evaluation Areas:
- Communication clarity and listening skills
- Teamwork and collaboration
- Leadership and mentorship
- Problem-solving under pressure
- Conflict resolution and negotiation
- Adaptability and growth mindset

Provide balanced, supportive feedback focused on professional growth."""
}

# ============================================
# UTILITY FUNCTIONS
# ============================================
@st.cache_resource
def initialize_llm():
    """Initialize Groq LLM with error handling."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not found in environment variables")
        return None
    
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.7,
            max_tokens=2048
        )
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        return None


def format_conversation_history(messages: List[Dict]) -> List[Dict]:
    """Format conversation history for LLM."""
    formatted = []
    for msg in messages:
        formatted.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    return formatted


def get_system_prompt(interview_type: str) -> str:
    """Get the appropriate system prompt for the interview type."""
    return INTERVIEW_PROMPTS.get(interview_type, INTERVIEW_PROMPTS["General"])


def parse_response_with_score(response: str) -> tuple:
    """Parse response to extract main content and scoring if present."""
    if "Score:" in response or "Rating:" in response:
        parts = response.split("---")
        if len(parts) > 1:
            return parts[0].strip(), parts[1].strip()
    return response, ""


def log_interaction(interview_type: str, user_input: str, response: str, duration: float):
    """Log interview interactions for monitoring."""
    logger.info(
        f"Interaction - Type: {interview_type}, "
        f"Input length: {len(user_input)}, "
        f"Response length: {len(response)}, "
        f"Duration: {duration:.2f}s"
    )


# ============================================
# INITIALIZE SESSION STATE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "interview_type" not in st.session_state:
    st.session_state.interview_type = "General"

if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# ============================================
# MAIN LAYOUT
# ============================================
# Header
st.markdown(
    """
    <div class="header-container">
        <div class="header-title">Interview Preparation Coach</div>
        <div class="header-subtitle">
            Professional technical interview preparation powered by advanced AI
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================
# SIDEBAR CONFIGURATION
# ============================================
with st.sidebar:
    st.markdown("### Interview Settings")
    
    # Interview type selection
    interview_type = st.selectbox(
        "Interview Type",
        options=[
            "General",
            "Frontend",
            "Backend",
            "System Design",
            "Data Structures",
            "Behavioral"
        ],
        help="Select the type of interview you're preparing for"
    )
    st.session_state.interview_type = interview_type
    
    st.divider()
    
    # Session management
    st.markdown("### Session Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.rerun()
    
    with col2:
        if st.button("New Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.rerun()
    
    st.divider()
    
    # Information panel
    st.markdown("### About This Tool")
    
    st.markdown(
        """
        <div class="info-box">
        <strong>Purpose:</strong> Prepare for technical interviews through realistic 
        practice with AI feedback.
        
        <strong>Features:</strong>
        - Personalized interview questions
        - Detailed performance feedback
        - Follow-up questions for deeper evaluation
        - Targeted improvement recommendations
        
        <strong>Technology Stack:</strong>
        - Groq API (High-performance LLM inference)
        - LLaMA 3.3 70B (Advanced open-source model)
        - Streamlit (Professional web interface)
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    
    st.markdown("### Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='stat-item'><span class='stat-label'>Messages:</span> {len(st.session_state.messages)}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-item'><span class='stat-label'>Type:</span> {interview_type}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown(
        "<div style='font-size: 12px; color: #999999; text-align: center;'>v1.0 | Built for serious preparation</div>",
        unsafe_allow_html=True
    )

# ============================================
# MAIN CONTENT AREA
# ============================================
# Display configuration
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Current Focus:** {interview_type} Interview")
    with col2:
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")

st.divider()

# ============================================
# CHAT INTERFACE
# ============================================
# Initialize LLM
llm = initialize_llm()

if llm is None:
    st.error(
        "Configuration Error: Cannot initialize language model. "
        "Please ensure GROQ_API_KEY is properly set in your .env file."
    )
    st.info("Setup instructions: Visit https://console.groq.com to create your API key.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================
# CHAT INPUT & RESPONSE
# ============================================
if user_input := st.chat_input(
    "Enter your question or share your answer...",
    key="user_input"
):
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.message_count += 1
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get LLM response
    with st.chat_message("assistant"):
        with st.spinner("Processing response..."):
            try:
                import time
                start_time = time.time()
                
                # Build messages for LLM
                system_prompt = get_system_prompt(interview_type)
                messages = [
                    {"role": "system", "content": system_prompt}
                ]
                
                # Add conversation history
                messages.extend(format_conversation_history(st.session_state.messages[:-1]))
                
                # Get response from LLM
                response = llm.invoke(messages)
                output = response.content
                
                duration = time.time() - start_time
                
                # Parse response
                main_content, score_content = parse_response_with_score(output)
                
                # Display main response
                st.markdown(main_content)
                
                # Display score if present
                if score_content:
                    st.markdown(f"<div class='feedback-score'>{score_content}</div>", unsafe_allow_html=True)
                
                # Log interaction
                log_interaction(interview_type, user_input, output, duration)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": output
                })
                
            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")
                error_message = (
                    "An error occurred while processing your request. "
                    "Please verify your API key is valid and try again."
                )
                st.error(error_message)
                st.info(
                    "Troubleshooting:\n"
                    "1. Verify GROQ_API_KEY in your .env file\n"
                    "2. Check your internet connection\n"
                    "3. Ensure your API key is still valid\n"
                    "4. Try with a shorter input"
                )

# ============================================
# FOOTER
# ============================================
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='stat-item'><span class='stat-label'>Cost:</span> Free</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='stat-item'><span class='stat-label'>Performance:</span> High-speed inference</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='stat-item'><span class='stat-label'>Privacy:</span> Secure</div>", unsafe_allow_html=True)
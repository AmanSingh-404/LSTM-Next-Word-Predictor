import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Premium Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Global Reset and Typography */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
    background: radial-gradient(circle at 50% 50%, #1e1b4b 0%, #0f172a 50%, #020617 100%) !important;
    color: #f8fafc !important;
}

/* Glassmorphism Title Section */
.hero-container {
    text-align: center;
    padding: 3rem 2rem;
    margin-bottom: 2.5rem;
    background: rgba(30, 27, 75, 0.3);
    border: 1px solid rgba(99, 102, 241, 0.15);
    backdrop-filter: blur(12px);
    border-radius: 24px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
    text-shadow: 0 0 30px rgba(192, 132, 252, 0.2);
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #cbd5e1;
    font-weight: 300;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Glass Card containers */
.glass-card {
    background: rgba(15, 23, 42, 0.45);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 2.2rem;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    margin-bottom: 2rem;
}

.card-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0.75rem;
}

/* Custom styles for Streamlit text input box */
div[data-baseweb="input"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 14px !important;
    padding: 6px 12px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #a855f7 !important;
    box-shadow: 0 0 18px rgba(168, 85, 247, 0.25) !important;
    background-color: rgba(15, 23, 42, 0.85) !important;
}

input[data-testid="stTextInput-Input"] {
    color: #ffffff !important;
    font-size: 1.2rem !important;
    font-weight: 400 !important;
}

/* Styling for recommendation pills */
.stButton > button {
    background: rgba(255, 255, 255, 0.04) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 30px !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1 0%, #d946ef 100%) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-3px) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* Sidebar overrides */
[data-testid="stSidebar"] {
    background-color: #090d16 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.sidebar-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #a5b4fc;
    margin-bottom: 1.5rem;
    letter-spacing: -0.02em;
}

/* Metric text values */
.sidebar-metric {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 0.8rem;
}

.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.metric-value {
    font-size: 1.1rem;
    color: #f8fafc;
    font-weight: 700;
    margin-top: 0.2rem;
}

</style>
""", unsafe_allow_html=True)

# 3. Load Model and Tokenizer (with cache caching)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokinizer.pkl")
MAX_LEN_PATH = os.path.join(BASE_DIR, "max_len.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.h5")

@st.cache_resource
def load_resources():
    # Load max length
    with open(MAX_LEN_PATH, "rb") as f:
        max_len = pickle.load(f)
    
    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
        
    # Load Keras Model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    return model, tokenizer, max_len

# Loader feedback in case of error
try:
    model, tokenizer, max_len = load_resources()
    vocab_size = len(tokenizer.word_index) if tokenizer else 0
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_msg = str(e)

# 4. Callback Logic for Suggestions
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

def select_word(word):
    current = st.session_state.user_text.strip()
    if current:
        st.session_state.user_text = current + " " + word + " "
    else:
        st.session_state.user_text = word + " "

def clear_text():
    st.session_state.user_text = ""

# 5. Header UI rendering
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔮 LSTM Next Word Predictor</div>
    <div class="hero-subtitle">
        Type an English phrase and our deep learning LSTM network will analyze your syntax, 
        embeddings, and temporal structure to predict the most probable next word instantly.
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Failed to load model resources. Error details: {error_msg}")
    st.stop()

# 6. Sidebar Implementation
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 Model Architecture</div>', unsafe_allow_html=True)
    
    # Display details
    st.markdown(f"""
    <div class="sidebar-metric">
        <div class="metric-label">Neural Network Type</div>
        <div class="metric-value">LSTM (Sequential)</div>
    </div>
    <div class="sidebar-metric">
        <div class="metric-label">Max Sequence Length</div>
        <div class="metric-value">{max_len} words</div>
    </div>
    <div class="sidebar-metric">
        <div class="metric-label">Vocabulary Size</div>
        <div class="metric-value">{vocab_size:,} tokens</div>
    </div>
    <div class="sidebar-metric">
        <div class="metric-label">Model Weights File</div>
        <div class="metric-value">lstm_model.h5 (22.6 MB)</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="sidebar-title">⚙️ Control Panel</div>', unsafe_allow_html=True)
    num_suggestions = st.slider("Max suggestions to show", min_value=3, max_value=8, value=5)
    
    st.markdown("---")
    
    # Model layers summary inside sidebar expander
    with st.expander("🔍 View Layer-by-Layer Summary"):
        # We can extract layers details programmatically
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        st.code("\n".join(summary_list), language="text")

# 7. Main Interface Layout
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📝 Context Input</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Input box bound to the session state
    st.text_input(
        "Type your phrase here and press Space/Enter:",
        key="user_text",
        placeholder="e.g. what is the, do you want to, i am going..."
    )
    
    # Help/Action buttons
    col_clear, col_space = st.columns([1, 4])
    with col_clear:
        st.button("🧹 Clear", on_click=clear_text)
        
    st.markdown("---")
    
    # Determine the context text
    input_phrase = st.session_state.user_text.strip()
    
    # Run Prediction logic
    predictions = []
    if input_phrase:
        # Preprocessing: texts_to_sequences
        token_list = tokenizer.texts_to_sequences([input_phrase])[0]
        
        # Padding length based on model input shape or max_len
        # (usually embedding input length, here 745)
        input_len = model.input_shape[1] if model.input_shape[1] is not None else (max_len - 1)
        
        padded_seq = pad_sequences([token_list], maxlen=input_len, padding='pre')
        
        # Inference
        raw_preds = model.predict(padded_seq, verbose=0)[0]
        
        # Sort indices
        top_indices = np.argsort(raw_preds)[-num_suggestions:][::-1]
        
        # Decode and structure predictions
        for idx in top_indices:
            # check if valid token and in index_word map
            if idx in tokenizer.index_word:
                word = tokenizer.index_word[idx]
                prob = float(raw_preds[idx])
                # Filter empty word keys or formatting symbols
                if word.strip():
                    predictions.append((word, prob))
    else:
        # Default starting suggestions when input is empty
        default_words = ["the", "i", "how", "what", "it", "she", "he", "they"]
        predictions = [(w, 0.0) for w in default_words[:num_suggestions]]

    # Render suggestion buttons
    st.markdown('<div class="predictions-label">🔮 Click a Suggestion to append:</div>', unsafe_allow_html=True)
    
    if predictions:
        # Distribute pills in columns
        sub_cols = st.columns(len(predictions))
        for index, (word, prob) in enumerate(predictions):
            with sub_cols[index]:
                # On click, callback appends word and triggers rerun
                st.button(
                    word.upper(), 
                    key=f"btn_{word}_{index}", 
                    on_click=select_word, 
                    args=(word,)
                )
    else:
        st.info("No matching next-word predictions. Try a different context.")

with col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📊 Inference Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    if input_phrase:
        st.markdown(f"**Current Context:** `\"{input_phrase}\"`")
        st.markdown(f"**Input Tokens:** `{token_list}`")
        
        st.markdown('<div class="predictions-label" style="margin-top: 1rem;">Softmax Probability Breakdown:</div>', unsafe_allow_html=True)
        
        # Custom progress bar list
        for word, prob in predictions:
            percentage = prob * 100
            st.markdown(f"""
            <div style="margin-bottom: 1rem; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); border-radius: 10px; padding: 0.6rem 0.8rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.95rem; color: #e2e8f0; margin-bottom: 6px; font-weight: 500;">
                    <span style="color: #c084fc; font-weight: 600;">{word}</span>
                    <span>{percentage:.2f}%</span>
                </div>
                <div style="background-color: rgba(255,255,255,0.06); border-radius: 6px; height: 8px; width: 100%; overflow: hidden; border: 1px solid rgba(255,255,255,0.03);">
                    <div style="background: linear-gradient(90deg, #6366f1, #d946ef); width: {percentage}%; height: 100%; border-radius: 6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #94a3b8; font-style: italic;'>Start typing in the text box to display confidence score insights.</p>", unsafe_allow_html=True)
        
        # Show general info
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.1); border-radius: 12px; padding: 1.2rem; margin-top: 1rem;">
            <h4 style="color: #a5b4fc; margin-top: 0; margin-bottom: 0.5rem; font-size: 1rem;">How it works:</h4>
            <ol style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0; padding-left: 1.2rem;">
                <li style="margin-bottom: 0.4rem;">Your input is split into individual words.</li>
                <li style="margin-bottom: 0.4rem;">Tokenizer converts word strings to integer indices based on the training vocabulary.</li>
                <li style="margin-bottom: 0.4rem;">The sequence is pre-padded to fit the required 745 tokens context.</li>
                <li style="margin-bottom: 0.4rem;">The LSTM network analyzes word-embeddings and temporal sequences to predict the next word distribution.</li>
                <li>Top confidence predictions are displayed as interactive buttons.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

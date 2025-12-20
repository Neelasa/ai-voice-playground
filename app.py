import os
import io
import base64
import pandas as pd
from gtts import gTTS
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Please set your GROQ_API_KEY in a .env file.")
    st.stop()

client = Groq(api_key=api_key)

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "usage_data" not in st.session_state:
    st.session_state.usage_data = [] 
if "current_response" not in st.session_state:
    st.session_state.current_response = ""

st.set_page_config(page_title="Mega AI Playground", layout="wide", page_icon="🧪")

# --- SIDEBAR: HISTORY & MODES ---
with st.sidebar:
    st.title("🧪 AI Control Center")
    
    if st.button("➕ New Session", use_container_width=True):
        st.session_state.current_response = ""
        st.rerun()

    st.divider()
    st.subheader("🕒 Recent Chats")
    for idx, item in enumerate(reversed(st.session_state.history)):
        if st.button(f"💬 {item['title']}", key=f"hist_{idx}", use_container_width=True):
            st.session_state.current_response = item['assistant']

    st.divider()
    st.header("⚙️ Configuration")
    
    # Mode Selection
    app_mode = st.radio("Select Mode", ["Text & Voice", "Image Generation", "Vision (Analyze)"])
    
    # Model & Temp (temph)
    selected_model = st.selectbox("LLM Model", ["llama-3.3-70b-versatile", "llama-3.2-11b-vision-preview", "mixtral-8x7b-32768"])
    temp = st.slider("Temperature (Creativity)", 0.0, 2.0, 0.7, help="0.0 is factual, 1.0+ is creative.")

    if st.session_state.history:
        history_text = "\n\n".join([f"Q: {h['user']}\nA: {h['assistant']}" for h in st.session_state.history])
        st.download_button("📂 Export Lab Notes", history_text, file_name="playground_history.txt")

# --- MAIN INTERFACE ---
st.title("🧪 Mega Prompt Engineering Playground")

# Graphic UI: Token Usage
if st.session_state.usage_data:
    st.subheader("📊 Token Efficiency Chart")
    st.area_chart(pd.DataFrame(st.session_state.usage_data), x="Message", y="Tokens")

# --- MODE 1: TEXT & VOICE ---
if app_mode == "Text & Voice":
    tab_type, tab_voice = st.tabs(["⌨️ Type", "🎙️ Speak"])
    user_input = ""

    with tab_type:
        t_in = st.text_area("Your Question", height=100, placeholder="Ask Llama anything...")
        if t_in: user_input = t_in
    with tab_voice:
        audio = st.audio_input("Record your prompt")
        if audio:
            with st.spinner("Transcribing..."):
                trans = client.audio.transcriptions.create(
                    file=("audio.wav", audio.read()), model="whisper-large-v3-turbo", response_format="text"
                )
                user_input = trans
                st.info(f"Heard: {user_input}")

    if st.button("🚀 Run AI", type="primary"):
        if user_input:
            with st.spinner("Thinking..."):
                resp = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": user_input}],
                    temperature=temp
                )
                answer = resp.choices[0].message.content
                st.session_state.current_response = answer
                st.session_state.usage_data.append({"Message": len(st.session_state.usage_data)+1, "Tokens": len(answer.split())})
                st.session_state.history.append({"title": user_input[:20], "user": user_input, "assistant": answer})

# --- MODE 2: IMAGE GENERATION ---
elif app_mode == "Image Generation":
    st.subheader("🎨 Text-to-Image (Pollinations)")
    img_prompt = st.text_input("Describe the image you want to create:")
    if st.button("🖼️ Generate Image"):
        if img_prompt:
            encoded_prompt = img_prompt.replace(" ", "%20")
            img_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
            st.image(img_url, caption=f"Generated: {img_prompt}")
            st.session_state.history.append({"title": f"Img: {img_prompt[:15]}", "user": img_prompt, "assistant": f"Generated image at {img_url}"})

# --- MODE 3: VISION (ANALYZE IMAGES) ---
elif app_mode == "Vision (Analyze)":
    st.subheader("👁️ Llama 3.2 Vision")
    uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png"])
    vision_prompt = st.text_input("What should the AI look for?", value="Describe this image in detail.")
    
    if uploaded_file and st.button("🔍 Analyze Image"):
        base64_image = base64.b64encode(uploaded_file.read()).decode('utf-8')
        with st.spinner("Llama is looking at your image..."):
            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}]
            )
            analysis = response.choices[0].message.content
            st.session_state.current_response = analysis
            st.image(uploaded_file, width=300)

# --- FINAL OUTPUT & TTS ---
if st.session_state.current_response:
    st.divider()
    st.markdown(st.session_state.current_response)
    if st.button("🔊 Read Response Aloud"):
        tts = gTTS(text=st.session_state.current_response, lang='en')
        aud_io = io.BytesIO()
        tts.write_to_fp(aud_io)
        st.audio(aud_io, format="audio/mp3", autoplay=True)
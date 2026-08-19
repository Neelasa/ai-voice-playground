import os
import io
import base64
import pandas as pd
from gtts import gTTS
from dotenv import load_dotenv
from groq import Groq
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mega AI Playground",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file or Streamlit Cloud Secrets."
    )
    st.stop()

client = Groq(api_key=api_key)


# ============================================================
# CURRENT GROQ MODELS
# ============================================================

TEXT_MODELS = {
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "GPT-OSS 20B": "openai/gpt-oss-20b",
    "Qwen 3.6 27B": "qwen/qwen3.6-27b"
}

VISION_MODEL = "qwen/qwen3.6-27b"
WHISPER_MODEL = "whisper-large-v3-turbo"


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "usage_data" not in st.session_state:
    st.session_state.usage_data = []

if "current_response" not in st.session_state:
    st.session_state.current_response = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧪 AI Control Center")

    # --------------------------------------------------------
    # NEW SESSION
    # --------------------------------------------------------

    if st.button("➕ New Session", use_container_width=True):
        st.session_state.history = []
        st.session_state.current_response = ""
        st.session_state.usage_data = []
        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # RECENT CHATS
    # --------------------------------------------------------

    st.subheader("🕒 Recent Chats")

    if st.session_state.history:

        for idx, item in enumerate(
            reversed(st.session_state.history)
        ):

            if st.button(
                f"💬 {item['title']}",
                key=f"hist_{idx}",
                use_container_width=True
            ):

                st.session_state.current_response = item["assistant"]

    else:

        st.caption("No conversations yet.")

    st.divider()

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    st.header("⚙️ Configuration")

    app_mode = st.radio(
        "Select Mode",
        [
            "Text & Voice",
            "Image Generation",
            "Vision (Analyze)"
        ]
    )

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    if app_mode == "Text & Voice":

        selected_model_name = st.selectbox(
            "LLM Model",
            list(TEXT_MODELS.keys())
        )

        selected_model = TEXT_MODELS[selected_model_name]

    else:

        selected_model = VISION_MODEL

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    temp = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help=(
            "Lower values produce more focused responses. "
            "Higher values produce more creative responses."
        )
    )

    # --------------------------------------------------------
    # EXPORT HISTORY
    # --------------------------------------------------------

    if st.session_state.history:

        history_text = "\n\n".join(
            [
                f"Q: {h['user']}\nA: {h['assistant']}"
                for h in st.session_state.history
            ]
        )

        st.download_button(
            "📂 Export Lab Notes",
            history_text,
            file_name="playground_history.txt",
            mime="text/plain"
        )


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🧪 Mega Prompt Engineering Playground")

st.caption(
    "Experiment with LLMs, temperature, voice input, "
    "image generation and vision analysis."
)


# ============================================================
# TOKEN USAGE CHART
# ============================================================

if st.session_state.usage_data:

    st.subheader("📊 Token Efficiency Chart")

    usage_df = pd.DataFrame(
        st.session_state.usage_data
    )

    st.area_chart(
        usage_df,
        x="Message",
        y="Tokens"
    )


# ============================================================
# MODE 1: TEXT & VOICE
# ============================================================

if app_mode == "Text & Voice":

    tab_type, tab_voice = st.tabs(
        ["⌨️ Type", "🎙️ Speak"]
    )

    user_input = ""

    # --------------------------------------------------------
    # TYPE INPUT
    # --------------------------------------------------------

    with tab_type:

        t_in = st.text_area(
            "Your Question",
            height=100,
            placeholder="Ask anything..."
        )

        if t_in.strip():
            user_input = t_in.strip()

    # --------------------------------------------------------
    # VOICE INPUT
    # --------------------------------------------------------

    with tab_voice:

        audio = st.audio_input(
            "Record your prompt"
        )

        if audio:

            with st.spinner("🎙️ Transcribing..."):

                try:

                    transcription = client.audio.transcriptions.create(
                        file=(
                            "audio.wav",
                            audio.read()
                        ),
                        model=WHISPER_MODEL,
                        response_format="text"
                    )

                    user_input = transcription

                    st.info(
                        f"Heard: {user_input}"
                    )

                except Exception as e:

                    st.error(
                        f"Transcription failed: {str(e)}"
                    )

    # --------------------------------------------------------
    # RUN AI
    # --------------------------------------------------------

    if st.button(
        "🚀 Run AI",
        type="primary",
        use_container_width=True
    ):

        if not user_input:

            st.warning(
                "Please enter or record a question."
            )

        else:

            with st.spinner(
                f"🤖 {selected_model_name} is thinking..."
            ):

                try:

                    response = client.chat.completions.create(

                        model=selected_model,

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are Neela Nexus AI, "
                                    "a helpful, intelligent and "
                                    "concise AI assistant. "
                                    "Provide clear and accurate "
                                    "answers and explain concepts "
                                    "when necessary."
                                )
                            },
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ],

                        temperature=temp,

                        max_tokens=2048
                    )

                    answer = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                    # ------------------------------------------------
                    # SAVE CURRENT RESPONSE
                    # ------------------------------------------------

                    st.session_state.current_response = answer

                    # ------------------------------------------------
                    # TOKEN TRACKING
                    # ------------------------------------------------

                    if hasattr(response, "usage") and response.usage:

                        total_tokens = response.usage.total_tokens

                    else:

                        total_tokens = len(
                            answer.split()
                        )

                    st.session_state.usage_data.append(
                        {
                            "Message": (
                                len(
                                    st.session_state.usage_data
                                ) + 1
                            ),
                            "Tokens": total_tokens
                        }
                    )

                    # ------------------------------------------------
                    # SAVE CHAT HISTORY
                    # ------------------------------------------------

                    st.session_state.history.append(
                        {
                            "title": user_input[:30],
                            "user": user_input,
                            "assistant": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        "❌ AI request failed."
                    )

                    st.caption(
                        f"Error details: {str(e)}"
                    )


# ============================================================
# MODE 2: IMAGE GENERATION
# ============================================================

elif app_mode == "Image Generation":

    st.subheader(
        "🎨 Text-to-Image"
    )

    st.caption(
        "Image generation powered by Pollinations AI."
    )

    img_prompt = st.text_input(
        "Describe the image you want to create:"
    )

    if st.button(
        "🖼️ Generate Image",
        type="primary"
    ):

        if not img_prompt.strip():

            st.warning(
                "Please enter an image description."
            )

        else:

            encoded_prompt = (
                img_prompt
                .strip()
                .replace(" ", "%20")
            )

            img_url = (
                "https://image.pollinations.ai/prompt/"
                f"{encoded_prompt}"
                "?width=1024"
                "&height=1024"
                "&nologo=true"
            )

            try:

                st.image(
                    img_url,
                    caption=f"Generated: {img_prompt}"
                )

                st.session_state.history.append(
                    {
                        "title": f"Img: {img_prompt[:20]}",
                        "user": img_prompt,
                        "assistant": (
                            f"Generated image for: "
                            f"{img_prompt}"
                        )
                    }
                )

            except Exception as e:

                st.error(
                    f"Image generation failed: {str(e)}"
                )


# ============================================================
# MODE 3: VISION ANALYSIS
# ============================================================

elif app_mode == "Vision (Analyze)":

    st.subheader(
        "👁️ Vision Analysis"
    )

    st.caption(
        f"Powered by {VISION_MODEL}"
    )

    uploaded_file = st.file_uploader(
        "Upload an image to analyze",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    vision_prompt = st.text_input(
        "What should the AI look for?",
        value="Describe this image in detail."
    )

    if uploaded_file:

        # ----------------------------------------------------
        # DISPLAY UPLOADED IMAGE
        # ----------------------------------------------------

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            width=350
        )

    if uploaded_file and st.button(
        "🔍 Analyze Image",
        type="primary"
    ):

        with st.spinner(
            "👁️ AI is analyzing your image..."
        ):

            try:

                # ------------------------------------------------
                # READ IMAGE
                # ------------------------------------------------

                image_bytes = uploaded_file.getvalue()

                base64_image = base64.b64encode(
                    image_bytes
                ).decode("utf-8")

                # ------------------------------------------------
                # DETERMINE MIME TYPE
                # ------------------------------------------------

                mime_type = uploaded_file.type

                # ------------------------------------------------
                # VISION REQUEST
                # ------------------------------------------------

                response = client.chat.completions.create(

                    model=VISION_MODEL,

                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": vision_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            f"data:{mime_type};"
                                            f"base64,{base64_image}"
                                        )
                                    }
                                }
                            ]
                        }
                    ],

                    temperature=temp,

                    max_tokens=2048
                )

                analysis = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                st.session_state.current_response = analysis

                st.session_state.history.append(
                    {
                        "title": (
                            f"Vision: "
                            f"{vision_prompt[:20]}"
                        ),
                        "user": vision_prompt,
                        "assistant": analysis
                    }
                )

            except Exception as e:

                st.error(
                    "❌ Vision analysis failed."
                )

                st.caption(
                    f"Error details: {str(e)}"
                )


# ============================================================
# FINAL RESPONSE
# ============================================================

if st.session_state.current_response:

    st.divider()

    st.subheader(
        "🤖 AI Response"
    )

    st.markdown(
        st.session_state.current_response
    )

    # --------------------------------------------------------
    # TEXT TO SPEECH
    # --------------------------------------------------------

    if st.button(
        "🔊 Read Response Aloud"
    ):

        with st.spinner(
            "🔊 Generating audio..."
        ):

            try:

                tts = gTTS(
                    text=st.session_state.current_response,
                    lang="en"
                )

                audio_buffer = io.BytesIO()

                tts.write_to_fp(
                    audio_buffer
                )

                audio_buffer.seek(0)

                st.audio(
                    audio_buffer,
                    format="audio/mp3",
                    autoplay=True
                )

            except Exception as e:

                st.error(
                    f"Text-to-speech failed: {str(e)}"
                )

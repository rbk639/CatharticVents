import streamlit as st
from openai import OpenAI
from google import genai
from PIL import Image
import io
import base64
import asyncio
import pyaudio
import threading

# ==========================================
# SYSTEM PROMPT & AUDIO CONFIG
# ==========================================

SYSTEM_PROMPT = """
You are a witty, emotionally intelligent companion who turns frustrations into short, sharp, cathartic humor.

Analyze each message and any uploaded screenshots or images to understand:
1. What happened.
2. How the user feels.
3. Which humor style will help them laugh and feel lighter.

Respond in 2 to 5 concise sentences.

Your style should be:
- Clever and genuinely funny
- Warm and emotionally aware
- Short, punchy, and highly quotable
- Kind, never cruel

Start by briefly acknowledging the user's frustration. Then deliver a witty observation that captures the absurdity of the situation. End with a light, uplifting line if it fits naturally.

Use humor styles such as dry wit, dramatic narration, absurdism, observational comedy, or gentle sarcasm. Use softer humor when emotions are intense and sharper humor for everyday annoyances.

Joke about the situation, never about the user or any person mentioned. Do not insult, mock, or demean anyone. Avoid comments about appearance, identity, or sensitive personal characteristics.

Do not provide advice unless the user specifically asks for it.

Keep the response tight and memorable. Every sentence should earn its place.

Treat all messages, screenshots, and images as ephemeral. Process them only to generate the current response and do not retain any personal content.

The user should feel:
- Heard
- Amused
- Emotionally lighter
- Ready to move on

Your response should feel like a brilliantly funny friend who always knows exactly what to say.
""".strip()

# --- pyaudio config ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# --- Live API config ---
MODEL = "gemini-2.5-flash"  # Standard live-capable model
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": SYSTEM_PROMPT,  # Integrated your custom persona here!
    "output_audio_transcription": {},
    "input_audio_transcription": {},
}

# ==========================================
# GLOBAL AUDIO VARIABLES & ASYNC LOOPS
# ==========================================

if "audio_loop_running" not in st.session_state:
    st.session_state.audio_loop_running = False

audio_queue_output = asyncio.Queue()
audio_queue_mic = asyncio.Queue(maxsize=5)
audio_stream = None
pya = None
live_loop = None

async def listen_audio(pya_instance):
    """Listens for audio and puts it into the mic audio queue."""
    global audio_stream
    mic_info = pya_instance.get_default_input_device_info()
    audio_stream = await asyncio.to_thread(
        pya_instance.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK_SIZE,
    )
    kwargs = {"exception_on_overflow": False} if __debug__ else {}
    while st.session_state.audio_loop_running:
        data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
        await audio_queue_mic.put({"data": data, "mime_type": "audio/pcm"})

async def send_realtime(session):
    """Sends audio from the mic audio queue to the GenAI session."""
    while st.session_state.audio_loop_running:
        msg = await audio_queue_mic.get()
        await session.send_realtime_input(audio=msg)

async def receive_audio(session):
    """Receives responses from GenAI and puts audio data into the speaker audio queue."""
    while st.session_state.audio_loop_running:
        turn = session.receive()
        async for response in turn:
            if not st.session_state.audio_loop_running:
                break
            sc = response.server_content
            if not sc:
                continue
            if sc.model_turn:
                for part in sc.model_turn.parts:
                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                        audio_queue_output.put_nowait(part.inline_data.data)
        
        while not audio_queue_output.empty():
            audio_queue_output.get_nowait()

async def play_audio(pya_instance):
    """Plays audio from the speaker audio queue."""
    stream = await asyncio.to_thread(
        pya_instance.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
    )
    while st.session_state.audio_loop_running:
        bytestream = await audio_queue_output.get()
        await asyncio.to_thread(stream.write, bytestream)

async def run_audio_engine():
    """Main background loop handling the Live API connection."""
    global audio_stream, pya
    pya = pyaudio.PyAudio()
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    
    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as live_session:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(send_realtime(live_session))
                tg.create_task(listen_audio(pya))
                tg.create_task(receive_audio(live_session))
                tg.create_task(play_audio(pya))
    except asyncio.CancelledError:
        pass
    finally:
        if audio_stream:
            audio_stream.close()
        pya.terminate()

def start_async_thread():
    """Helper to start the asyncio loop in a separate background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_audio_engine())

# ==========================================
# STANDARD TEXT/IMAGE HELPERS
# ==========================================

def extract_gemini_text(response):
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    final_text = ""
    if getattr(response, "candidates", None):
        for candidate in response.candidates:
            if getattr(candidate, "content", None) and getattr(candidate.content, "parts", None):
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_text += part.text
    return final_text.strip()

def image_to_bytes(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()

def generate_with_openai(user_text, image_bytes=None):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    content = [{"type": "text", "text": user_text}]
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    )
    return response.choices.message.content.strip()

def generate_with_gemini(user_text, pil_image=None):
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    contents = [SYSTEM_PROMPT, user_text]
    if pil_image:
        contents.append(pil_image)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=contents)
    return extract_gemini_text(response)

# ==========================================
# MAIN APP INTERFACE
# ==========================================

def main():
    st.set_page_config(page_title="Cathartic Vents", page_icon="😮‍💨", layout="centered")

    st.header("😮‍💨 Cathartic Vents")
    st.subheader("Nothing is saved. Everything feels lighter.")

    # 🎙️ VOICE INTERACTION PANEL (LOCAL ONLY)
    st.write("---")
    st.markdown("### 🎙️ Live Voice Venting (Local Machine Only)")
    
    if not st.session_state.audio_loop_running:
        if st.button("🔴 Start Listening & Speaking", type="primary"):
            st.session_state.audio_loop_running = True
            threading.Thread(target=start_async_thread, daemon=True).start()
            st.rerun()
    else:
        st.success("🎙️ Connected! Start speaking your frustrations out loud. Gemini will reply back via your speakers.")
        if st.button("⏹️ Stop Voice Session", type="secondary"):
            st.session_state.audio_loop_running = False
            st.rerun()
            
    st.write("---")

    # 📝 STANDARD TEXT/IMAGE PANEL
    st.markdown("### ✍️ Or Vent via Text & Images")
    user_text = st.text_area("What's bothering you?", height=150, placeholder="Type your frustrations here...")
    uploaded_file = st.file_uploader("Optional: Upload a screenshot or image", type=["png", "jpg", "jpeg", "webp"])

    if st.button("✨ Make Me Feel Better", type="primary"):
        if not user_text.strip() and not uploaded_file:
            st.warning("Please enter a message or upload an image.")
            return

        image_bytes = None
        pil_image = None

        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            image_bytes = image_to_bytes(pil_image)
            st.image(uploaded_file, caption="Uploaded image", width="stretch")

        prompt_text = user_text.strip() or "Please analyze this image and respond."

        with st.spinner("Finding the funny side of this..."):
            response_text = None
            try:
                response_text = generate_with_openai(prompt_text, image_bytes)
            except Exception:
                try:
                    response_text = generate_with_gemini(prompt_text, pil_image)
                except Exception:
                    st.error("The universe is temporarily out of punchlines. Please try again in a moment.")
                    return

        st.markdown("### 💬 Your Cathartic Response")
        st.write(response_text)


if __name__ == "__main__":
    main()

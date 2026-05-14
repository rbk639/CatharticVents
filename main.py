import streamlit as st
from openai import OpenAI
from google import genai
from PIL import Image
import io
import base64

# ==========================================
# SYSTEM PROMPT
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


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_gemini_text(response):
    """Safely extract text from Gemini response."""
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


def image_to_bytes(uploaded_file):
    """Convert uploaded image to PNG bytes."""
    image = Image.open(uploaded_file)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ==========================================
# OPENAI GENERATION
# ==========================================

def generate_with_openai(user_text, image_bytes=None):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    content = [
        {
            "type": "text",
            "text": user_text
        }
    ]

    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}"
                }
            }
        )

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return response.choices[0].message.content.strip()


# ==========================================
# GEMINI GENERATION (FALLBACK)
# ==========================================

def generate_with_gemini(user_text, image_bytes=None):
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

    contents = [SYSTEM_PROMPT, user_text]

    if image_bytes:
        contents.append(
            {
                "mime_type": "image/png",
                "data": image_bytes
            }
        )

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=contents
    )

    return extract_gemini_text(response)


# ==========================================
# MAIN APP
# ==========================================

def main():
    st.set_page_config(
        page_title="Cathartic Vents",
        page_icon="😮‍💨",
        layout="centered"
    )

    st.header("😮‍💨 Cathartic Vents")
    st.subheader("Nothing is saved. Everything feels lighter.")

    st.write(
        "Unload your frustrations and get a kind, witty response "
        "that helps you laugh, let go, and move on."
    )

    user_text = st.text_area(
        "What's bothering you?",
        height=200,
        placeholder="Type your frustrations here..."
    )

    uploaded_file = st.file_uploader(
        "Optional: Upload a screenshot or image",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if st.button("✨ Make Me Feel Better", type="primary"):
        if not user_text.strip() and not uploaded_file:
            st.warning("Please enter a message or upload an image.")
            return

        image_bytes = None

        if uploaded_file:
            image_bytes = image_to_bytes(uploaded_file)
            st.image(
                uploaded_file,
                caption="Uploaded image",
                use_container_width=True
            )

        prompt_text = user_text.strip() or "Please analyze this image and respond."

        with st.spinner("Finding the funny side of this..."):
            response_text = None

            # Try OpenAI first
            try:
                response_text = generate_with_openai(
                    prompt_text,
                    image_bytes
                )

            # Silently fall back to Gemini
            except Exception:
                try:
                    response_text = generate_with_gemini(
                        prompt_text,
                        image_bytes
                    )
                except Exception:
                    st.error(
                        "The universe is temporarily out of punchlines. "
                        "Please try again in a moment."
                    )
                    return

        st.markdown("### 💬 Your Cathartic Response")
        st.write(response_text)

        st.caption(
            "Your message and any uploaded images are processed only for "
            "this response and are not stored by the app."
        )


if __name__ == "__main__":
    main()

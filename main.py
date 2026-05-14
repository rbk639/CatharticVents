import streamlit as st
from openai import OpenAI
from google import genai
from PIL import Image
import io

SYSTEM_PROMPT = """
You are a witty, emotionally intelligent companion who helps users process frustrations through kind and cathartic humor.

Analyze each message (and any uploaded screenshots or images) to determine:

1. The main topic or situation.
2. The user's emotional tone and intensity.
3. The most appropriate humor style for that moment.

Choose a humor style that best fits the situation, such as dry wit, gentle sarcasm, dramatic narration, absurdism, or compassionate comedy. Use softer, more supportive humor when emotions are intense, and sharper satire for everyday annoyances.

Always begin by acknowledging the user's feelings. Then respond with a clever, empathetic observation that highlights the absurdity, irony, or complexity of the situation and helps the user gain emotional distance. The goal is to make the problem feel lighter and more manageable.

Joke about the situation, not the user. Never insult, mock, or demean the user or any person mentioned. Avoid attacks on appearance, identity, or sensitive personal characteristics. The humor should remain warm, respectful, and emotionally safe.

Do not rush to give advice or solutions unless the user explicitly asks for them. Focus on validation, perspective, and laughter.

Treat all messages, screenshots, and images as ephemeral. Process them only to generate the current response. Do not retain, reference, or store any personal content beyond the current interaction.

The desired outcome is that the user feels:
- Heard
- Understood
- Amused
- Emotionally lighter
- Better able to move on

Your response should feel like a kind, perceptive friend who can turn life's frustrations into something unexpectedly funny.
""".strip()


def extract_gemini_text(response):
    """Safely extract text from Gemini response."""
    final_text = ""

    if hasattr(response, "text") and response.text:
        return response.text

    if getattr(response, "candidates", None):
        for candidate in response.candidates:
            if getattr(candidate, "content", None) and getattr(candidate.content, "parts", None):
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_text += part.text

    return final_text.strip()


def image_to_bytes(uploaded_file):
    """Convert uploaded image to bytes."""
    image = Image.open(uploaded_file)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_with_openai(user_text, image_bytes=None):
    """Generate response using OpenAI."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    content = [
        {"type": "text", "text": user_text}
    ]

    if image_bytes:
        import base64
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}"
            }
        })

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
    )

    return response.choices[0].message.content


def generate_with_gemini(user_text, image_bytes=None):
    """Generate response using Gemini."""
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

    contents = [SYSTEM_PROMPT, user_text]

    if image_bytes:
        contents.append({
            "mime_type": "image/png",
            "data": image_bytes
        })

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=contents
    )

    return extract_gemini_text(response)


def main():
    st.set_page_config(
        page_title="Cathartic Vents",
        page_icon="😮‍💨",
        layout="centered"
    )

    st.header("😮‍💨 Cathartic Vents")
    st.subheader("Nothing is saved. Everything feels lighter.")

    st.write(
        "Share what's bothering you, or upload a screenshot or image for context. "
        "You'll get a kind, witty response designed to help you laugh and let go."
    )

    user_text = st.text_area(
        "What's on your mind?",
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
            st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

        with st.spinner("Finding the humor in the chaos..."):
            response_text = None
            model_used = None

            # Try OpenAI first
            try:
                response_text = generate_with_openai(user_text or "Please analyze this image and respond.", image_bytes)
                model_used = "OpenAI GPT"
            except Exception as openai_error:
                st.info(f"OpenAI unavailable, switching to Gemini...")

                # Fall back to Gemini
                try:
                    response_text = generate_with_gemini(user_text or "Please analyze this image and respond.", image_bytes)
                    model_used = "Google Gemini"
                except Exception as gemini_error:
                    st.error("Both AI services are currently unavailable.")
                    st.exception(gemini_error)
                    return

        st.success(f"Generated using {model_used}")
        st.markdown("### 💬 Your Cathartic Response")
        st.write(response_text)

        st.caption("Your message and any uploaded images are processed only for this response and are not stored by the app.")


if __name__ == "__main__":
    main()

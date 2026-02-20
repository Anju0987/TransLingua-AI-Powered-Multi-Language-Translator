import streamlit as st
from google import genai
from google.genai import types
import time

# --- 1. Configuration ---
# Your API Key remains the same
API_KEY = "AIzaSyD1Nb7fymTcLrxgz2FtTfT09X9cT96Wgww" 

# Initialize the Client using 2026 stable v1 API
client = genai.Client(
    api_key=API_KEY,
    http_options={'api_version': 'v1'}
)

def process_and_translate(content_input, mime_type, source_lang, target_lang):
    """
    Multimodal translation: Handles Text, Image, and PDF extraction.
    Uses Gemini 3 and 2.5 Flash models.
    """
    
    # 2026 Optimized Prompt
    instruction = f"""
    Task: Act as a professional multimodal translator. 
    1. Extract all text from the provided source (Document/Image/Text).
    2. Translate the text from {source_lang} to {target_lang}.
    3. If it is a research paper or sign, maintain the structural layout.
    4. Provide only the translated text as output.
    """

    # Prepare data for Gemini
    if isinstance(content_input, bytes):
        # This 'Part' wrapper is required for Images and PDFs in the new SDK
        file_part = types.Part.from_bytes(
            data=content_input,
            mime_type=mime_type
        )
        contents = [file_part, instruction]
    else:
        # Standard text input
        contents = [instruction, content_input]

    # 2026 Model Priority List
    # gemini-3-flash-preview: The new standard for PhD-level document reasoning.
    # gemini-2.5-flash: The reliable high-speed workhorse.
    models_to_try = ["gemini-3-flash-preview", "gemini-2.5-flash"]
    
    last_error = ""

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            return response.text
            
        except Exception as e:
            last_error = str(e)
            # If 404 (Retired model) or 429 (Quota full), switch to next model
            if "404" in last_error or "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                continue 
            else:
                return f"Technical Error ({model_name}): {last_error}"

    return f"⚠️ Service Overloaded. Please wait 60 seconds and try again. \n(Error: {last_error})"

# --- 2. Streamlit UI Layout ---
def main():
    st.set_page_config(page_title="Universal TransLingua 2026", page_icon="🌎", layout="wide")

    st.title("🌎 Universal TransLingua AI")
    st.markdown("#### High-Speed Document & Vision Translation (Gemini 3 Powered)")
    st.divider()

    # Sidebar: Language Selection
    st.sidebar.header("Translation Settings")
    languages = ["English", "Hindi", "Telugu", "Spanish", "French", "German", "Japanese", "Chinese"]
    src_lang = st.sidebar.selectbox("Original Language", languages, index=0)
    tar_lang = st.sidebar.selectbox("Translate To", languages, index=1)
    
    st.sidebar.info("💡 **Travel Tip:** Use the 'Live Camera' tab on your phone to translate street signs instantly!")

    # Main UI Tabs
    tab1, tab2, tab3 = st.tabs(["📄 Research Papers / Media", "📸 Live Sign Capture", "✍️ Direct Text"])

    # --- TAB 1: Document Upload ---
    with tab1:
        st.subheader("Document & Image Translation")
        uploaded_file = st.file_uploader("Upload PDF or Image (max 10MB)", type=['pdf', 'png', 'jpg', 'jpeg'])
        if uploaded_file and st.button("Extract & Translate Document"):
            with st.spinner("Analyzing document structure..."):
                file_bytes = uploaded_file.getvalue()
                result = process_and_translate(file_bytes, uploaded_file.type, src_lang, tar_lang)
                st.success("Analysis Complete")
                st.markdown("### Translated Content:")
                st.info(result)

    # --- TAB 2: Live Camera ---
    with tab2:
        st.subheader("Camera Translator")
        camera_photo = st.camera_input("Point at a sign or document")
        if camera_photo:
            if st.button("Translate Capture"):
                with st.spinner("Processing image..."):
                    img_bytes = camera_photo.getvalue()
                    # Camera input is always treated as image/jpeg
                    result = process_and_translate(img_bytes, "image/jpeg", src_lang, tar_lang)
                    st.success("Translated Sign:")
                    st.info(result)

    # --- TAB 3: Text Input ---
    with tab3:
        st.subheader("Manual Text Translation")
        text_input = st.text_area("Enter text manually:", height=200)
        if st.button("Translate Text"):
            if text_input.strip():
                with st.spinner("Translating..."):
                    result = process_and_translate(text_input, None, src_lang, tar_lang)
                    st.write(result)
            else:
                st.warning("Please enter text first.")

if __name__ == "__main__":
    main()
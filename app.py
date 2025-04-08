import streamlit as st
from openai import OpenAI

# Initialize the OpenAI client using Streamlit secrets
client = OpenAI(api_key=st.secrets["openai_api_key"])

def translate_text(text, source_lang, target_lang):
    """
    Uses GPT-4 to translate text from source_lang to target_lang.
    """
    # Constructing the prompt with a friendly twist
    prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}"
    
    try:
        # Call GPT-4 via the new ChatCompletion API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a skilled language translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Keeping responses more deterministic
        )
        translation = response.choices[0].message.content.strip()
    except Exception as e:
        st.error("Uh-oh, something went wrong with the translation!")
        st.error(f"Error details: {e}")
        translation = ""
    
    return translation

def main():
    # App Title and Description
    st.title("GPT-4 Translator App")
    st.write("Welcome, friend! Translate text between languages using the mighty powers of GPT-4. No lost-in-translation moments on our watch!")

    # Text input for the content to be translated
    text_to_translate = st.text_area("Enter the text you want to translate:", "")

    # Input fields for specifying languages
    col1, col2 = st.columns(2)
    with col1:
        source_language = st.text_input("Source Language:", "English")
    with col2:
        target_language = st.text_input("Target Language:", "Hindi")

    # Translate button
    if st.button("Translate"):
        if text_to_translate.strip():
            with st.spinner("Translating..."):
                translated_text = translate_text(text_to_translate, source_language, target_language)
            if translated_text:
                st.subheader("Your Translation:")
                st.write(translated_text)
        else:
            st.warning("Please enter some text to translate!")

if __name__ == '__main__':
    main()
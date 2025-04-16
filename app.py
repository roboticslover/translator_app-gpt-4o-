import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Set page config
st.set_page_config(page_title="LangChain Translator", page_icon="🌐", layout="wide")

# Initialize the LLM using Streamlit secrets
def get_llm(streaming=True):
    return ChatOpenAI(
        api_key=st.secrets["openai_api_key"],
        model="gpt-4",
        temperature=0.3,
        streaming=streaming
    )

def translate_text(text, source_lang, target_lang):
    """
    Uses LangChain with GPT-4 to translate text from source_lang to target_lang.
    """
    # Create a prompt template
    template = """You are a skilled language translator.
    
    Translate the following text from {source_language} to {target_language}:
    
    {text_to_translate}
    
    Translation:"""
    
    prompt = PromptTemplate(
        input_variables=["source_language", "target_language", "text_to_translate"],
        template=template
    )
    
    try:
        # Create LangChain chain
        llm = get_llm(streaming=True)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Set up for streaming output
        result_area = st.empty()
        collected_text = ""
        
        # Create input dictionary for the chain
        inputs = {
            "source_language": source_lang,
            "target_language": target_lang,
            "text_to_translate": text
        }
        
        # Process the chain with streaming
        for chunk in chain.stream(inputs):
            if "text" in chunk:
                collected_text += chunk["text"]
                result_area.markdown(collected_text)
        
        return collected_text
        
    except Exception as e:
        st.error("Uh-oh, something went wrong with the translation!")
        st.error(f"Error details: {e}")
        return ""

def main():
    # App Title and Description
    st.title("LangChain Translator App")
    st.write("Welcome, friend! Translate text between languages using the power of LangChain and GPT-4.")
    
    # Create tabs for different features
    tab1, tab2 = st.tabs(["Translate", "About"])
    
    with tab1:
        # Text input for the content to be translated
        text_to_translate = st.text_area("Enter the text you want to translate:", "", height=150)
        
        # Input fields for specifying languages
        col1, col2 = st.columns(2)
        with col1:
            source_language = st.text_input("Source Language:", "English")
        with col2:
            target_language = st.text_input("Target Language:", "Hindi")
        
        # Translate button
        if st.button("Translate", type="primary"):
            if text_to_translate.strip():
                # Create container for output
                output_container = st.container()
                with output_container:
                    st.subheader("Your Translation:")
                    translated_text = translate_text(text_to_translate, source_language, target_language)
                    
                    if translated_text:
                        # Save to history feature
                        if "translation_history" not in st.session_state:
                            st.session_state.translation_history = []
                        
                        st.session_state.translation_history.append({
                            "original": text_to_translate,
                            "translated": translated_text,
                            "from": source_language,
                            "to": target_language
                        })
            else:
                st.warning("Please enter some text to translate!")
        
        # Display translation history
        if "translation_history" in st.session_state and st.session_state.translation_history:
            st.subheader("Translation History")
            for i, item in enumerate(reversed(st.session_state.translation_history[-5:])):
                with st.expander(f"Translation {len(st.session_state.translation_history) - i}: {item['from']} → {item['to']}"):
                    st.write("**Original:**")
                    st.write(item["original"])
                    st.write("**Translation:**")
                    st.write(item["translated"])
    
    with tab2:
        st.subheader("About LangChain Translator")
        st.write("""
        This application uses LangChain with OpenAI's GPT-4 to provide high-quality translations between languages.
        
        **Features:**
        - Translate text between any language pair
        - Streaming translation output
        - Translation history
        
        **LangChain components used:**
        - PromptTemplate: For structured prompting
        - LLMChain: For chaining the language model with the prompt
        - ChatOpenAI: For interaction with GPT-4
        
        The translation is streamed directly to the interface as it's being generated.
        """)

if __name__ == '__main__':
    main()

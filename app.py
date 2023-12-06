from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import requests 
import os
import streamlit as st
import streamlit.components.v1 as components

load_dotenv(find_dotenv())
HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
OPEN_AI_API_TOKEN = os.getenv("OPENAI_API_KEY")

def img2text(url):
    # first parameter pipeline is from https://huggingface.co/tasks
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    
    text = image_to_text(url)[0]["generated_text"]
    
    print(text)
    
    return text

def generate_story(scenario,image_desc):
        
    prompt_template = PromptTemplate.from_template(
     f"""
    You are a story teller, use the scenario {scenario} to describe {image_desc} use no more than 40 words.
    CONTEXT:
    """
    )
    prompt_template.format()
    
    llm = OpenAI(openai_api_key=OPEN_AI_API_TOKEN)
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    story = llm_chain.run({'CONTEXT':image_desc})
    return story

#using hugging face api
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer " + HUGGING_FACE_API_TOKEN }
    response = requests.post(API_URL, headers=headers, json=message)
    
    with open('audio.flac', 'wb') as file:
        file.write(response.content)



def main():
    st.set_page_config(page_title='img 2 audio story ⛔', page_icon='⛄')
    st.header("Turn your image into an audio story")
    context = None

    # Initialize session state variables
    if 'image_desc' not in st.session_state:
        st.session_state.image_desc = None

    uploaded_file = st.file_uploader("Choose an image", type='jpg')

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        
        st.image(uploaded_file, caption='Uploaded Image', width=200)

        # Check if the uploaded file has changed
        if st.session_state.image_desc is None or uploaded_file.name != st.session_state.uploaded_file_name:
            with st.spinner("Converting your image into text!"):
                st.session_state.image_desc = img2text(uploaded_file.name)
                st.session_state.uploaded_file_name = uploaded_file.name

        st.header("Your Image is: " + st.session_state.image_desc)

        context = st.text_input("Enter the context 👇",
                                label_visibility="visible",
                                disabled=False,
                                placeholder="Enter context",
                                )

    if context and st.session_state.image_desc:
        story = generate_story(context, st.session_state.image_desc)

        with st.spinner("Generating your story..."):
            text2speech(story)
        
        with st.expander("Image Description"):
            st.write(st.session_state.image_desc)
        with st.expander("story"):
            st.write(story)
        
        st.audio("audio.flac")
        
        
if __name__ == '__main__':
    main()
    









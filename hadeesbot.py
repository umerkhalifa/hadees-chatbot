import openai
import streamlit as st
from streamlit_chat import message
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os 


# Setting page title and header
st.set_page_config(page_title="Zayn", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Zayn - a totally harmless chatbot 😬</h1>", unsafe_allow_html=True)


ngrok_url = os.environ.get("NGROK_URL")

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.sidebar.title("Sidebar")
counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    model_path = './checkpoint-5000'
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    input_ids = tokenizer(f"""answer the question: {prompt}""", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=125)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.session_state['messages'].append({"role": "assistant", "content": response})
    return response

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    
    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))



if ngrok_url:
    st.write(f"Your Streamlit app is available at: {ngrok_url}")
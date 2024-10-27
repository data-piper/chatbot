import streamlit as st
from api.search_api.app import ingest_document, search
from api.chat_api.app import get_openai_client, chat_completion

# App title and OpenAI API key input
st.title("Interview-Genie")
openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Initialize OpenAI client
    client = get_openai_client(openai_api_key)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Document Upload Section
    st.write("### Upload Documents for Reference")
    uploaded_files = st.file_uploader("Upload PDF or text documents", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            doc_id = file.name
            text = file.read().decode("utf-8")
            st.write(ingest_document(doc_id, text))

    # Chat Interface
    st.write("### Chat with Interview-Genie")
    if prompt := st.chat_input("Ask a question..."):
        # Display user prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant context from search API
        context = search(prompt)

        # Generate response with chat completion API
        response = chat_completion(client, prompt, context, st.session_state.messages)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
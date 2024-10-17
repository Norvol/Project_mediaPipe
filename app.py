import streamlit as st
from log import login
from main import main

def app():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login()
    else:
        main()

if __name__ == "__main__":
    app()
    #streamlit run app.py
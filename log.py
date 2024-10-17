import streamlit as st
import hashlib
import json
import os

# Ensure the 'Data' directory exists
os.makedirs('Data', exist_ok=True)

# File path for storing user data
USER_DB = os.path.join('Data', 'users.json')

def load_users():
    """Load user data"""
    if os.path.exists(USER_DB):
        with open(USER_DB, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save user data"""
    with open(USER_DB, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    """Hash the password"""
    return hashlib.sha256(password.encode()).hexdigest()

def login():
    """Login page"""
    st.title("Login System")

    users = load_users()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("登錄"):
            if username in users and users[username] == hash_password(password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("登錄成功！")
                st.rerun()
                #st.query_params(logged_in="true")  
            else:
                st.error("用戶名或密碼錯誤")


    with col2:
        if st.button("Register"):
            if username in users:
                st.error("Username already exists")
            elif username and password:
                users[username] = hash_password(password)
                save_users(users)
                st.success("Registration successful, please log in")
            else:
                st.error("Please enter a username and password")

if __name__ == "__main__":
    login()

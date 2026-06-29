#!/usr/bin/env python
"""
=============================================================================
STREAMLIT DASHBOARD ENTRY POINT WITH AUTHENTICATION
=============================================================================
B.Sc. Computer Science Final Year Project
Nigerian Exchange Group (NGX) Volatility Prediction System

File: app.py
Purpose: Entry point for launching the interactive web application
         with user authentication layer.

Usage:
    $ streamlit run app.py

    Then open browser to: http://localhost:8501

This file serves as the user-facing entry point. It presents a login
screen before delegating to the dashboard.py controller module.
=============================================================================
"""

import sys
import os
import json
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')

AUTH_USERS_FILE = os.path.join(current_dir, 'auth_users.json')

DEFAULT_USERS: Dict[str, Dict[str, Any]] = {
    "admin": {
        "salt": "254a699480370ff10fac4314bd0ffc19",
        "hash": "8ecf35207b8181cc3fb3e9a694261066a54e1746626a01144ae5e95cf94ee80c",
        "iterations": 200000,
        "role": "Administrator",
        "full_name": "System Administrator"
    },
    "guest": {
        "salt": "9ed82d5d79663f6a9ab3a4622f37a236",
        "hash": "e34fab39750647a859aa937b79a71dff21632044b086bf771decb249505c5d8e",
        "iterations": 200000,
        "role": "Guest",
        "full_name": "Guest User"
    },
    "user": {
        "salt": "d8504bd2fec37222beb691360b029f48",
        "hash": "3e6b9034a62ebf33d04b874289309826c9cfb71deeda552c20372671c19c882b",
        "iterations": 200000,
        "role": "Analyst",
        "full_name": "Research Analyst"
    }
}


def load_users() -> Dict[str, Dict[str, Any]]:
    """Load user accounts from the auth JSON file."""
    if not os.path.exists(AUTH_USERS_FILE):
        save_users(DEFAULT_USERS)

    try:
        with open(AUTH_USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
            if not isinstance(users, dict):
                raise ValueError("User file is malformed")
            return users
    except Exception:
        return DEFAULT_USERS.copy()


def save_users(users: Dict[str, Dict[str, Any]]) -> None:
    """Save user accounts back to the auth JSON file."""
    with open(AUTH_USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)


def hash_password(password: str, salt: Optional[str] = None, iterations: int = 200000) -> Dict[str, Any]:
    """Hash a password using PBKDF2-SHA256."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), iterations)
    return {
        'salt': salt,
        'hash': digest.hex(),
        'iterations': iterations
    }


def verify_password(password: str, user_record: Dict[str, Any]) -> bool:
    """Verify a password against a stored user record."""
    if 'hash' in user_record and 'salt' in user_record and 'iterations' in user_record:
        check = hash_password(password, salt=user_record['salt'], iterations=user_record['iterations'])
        return check['hash'] == user_record['hash']
    if 'password_hash' in user_record:
        return hashlib.sha256(password.encode('utf-8')).hexdigest() == user_record['password_hash']
    return False


# Pre-defined users (username: {password_hash, role, full_name})
# Passwords are hashed with PBKDF2 for more realistic account management
USERS = load_users()

# Add src to Python path BEFORE importing
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Also add current directory to path for imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# =============================================================================
# AUTHENTICATION CONFIGURATION
# =============================================================================
# NOTE: the canonical user store is auth_users.json, managed via load_users()/
# save_users() above. The in-memory USERS dict loaded earlier just seeds that
# file on first run via DEFAULT_USERS. (A second hardcoded SHA-256 USERS dict
# used to live here and silently shadowed the JSON store — removed, since it
# meant newly registered accounts could be inconsistent with what
# verify_credentials() actually checked.)

# Session timeout in minutes
SESSION_TIMEOUT = 60


def verify_credentials(username: str, password: str) -> bool:
    """Verify username and password against stored credentials."""
    users = load_users()
    if username not in users:
        return False
    return verify_password(password, users[username])


def is_session_valid() -> bool:
    """Check if the current session is still valid (not expired)."""
    import streamlit as st

    if "authenticated" not in st.session_state or not st.session_state.authenticated:
        return False

    if "login_time" not in st.session_state:
        return False

    # Check session timeout
    login_time = st.session_state.login_time
    if isinstance(login_time, str):
        login_time = datetime.fromisoformat(login_time)

    elapsed = datetime.now() - login_time
    if elapsed > timedelta(minutes=SESSION_TIMEOUT):
        # Session expired
        st.session_state.authenticated = False
        st.session_state.pop("username", None)
        st.session_state.pop("login_time", None)
        return False

    return True


def login_user(username: str):
    """Set session state for authenticated user."""
    import streamlit as st

    users = load_users()
    user_record = users.get(username, {})

    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.login_time = datetime.now()
    st.session_state.user_role = user_record.get("role", "Guest")
    st.session_state.user_full_name = user_record.get("full_name", username)


def logout_user():
    """Clear session state to log out user."""
    import streamlit as st

    st.session_state.authenticated = False
    st.session_state.pop("username", None)
    st.session_state.pop("login_time", None)
    st.session_state.pop("user_role", None)
    st.session_state.pop("user_full_name", None)
    st.session_state.pop("login_error", None)


def render_login_page():
    """Render the login page UI."""
    import streamlit as st

    # Page config MUST be first Streamlit command
    st.set_page_config(
        page_title="Nigerian Exchange Group Volatility Prediction - Login",
        page_icon="Nigeria Exchange Group Volatility Prediction",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for login page with enhanced animations and styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Keyframe animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes glow {
        0% {
            box-shadow: 0 0 10px rgba(1,166,251,0.2), inset 0 0 10px rgba(1,166,251,0.1);
        }
        50% {
            box-shadow: 0 0 20px rgba(1,166,251,0.4), inset 0 0 20px rgba(1,166,251,0.2);
        }
        100% {
            box-shadow: 0 0 10px rgba(1,166,251,0.2), inset 0 0 10px rgba(1,166,251,0.1);
        }
    }

    @keyframes floatIcon {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-8px);
        }
    }

    @keyframes buttonHoverGlow {
        from {
            box-shadow: 0 4px 15px rgba(1,115,178,0.3);
        }
        to {
            box-shadow: 0 8px 30px rgba(1,166,251,0.5);
        }
    }

    /* Background gradient matching dashboard */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0F172A 50%, #1a1f3a 100%) !important;
    }

    /* Hide sidebar on login page */
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Center the main block */
    .block-container {
        padding-top: 50px !important;
        max-width: 820px !important;
    }

    /* Main login container */
    .login-container {
        max-width: 820px;
        margin: 0 auto;
        padding: 62px 48px 64px;
        min-height: 200px;
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.95));
        border-radius: 24px;
        border: 1px solid rgba(1,166,251,0.15);
        box-shadow: 0 25px 60px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.8s ease-out;
    }

    /* Logo section */
    .login-logo {
        text-align: center;
        margin-bottom: 40px;
        animation: fadeInDown 0.8s ease-out 0.1s both;
    }

    .login-logo-icon {
        width: 130px;
        height: 130px;
        margin: 0 auto 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, rgba(1,115,178,0.2), rgba(0,166,251,0.15));
        border-radius: 20px;
        border: 2px solid rgba(1,166,251,0.3);
        font-size: 60px;
        font-weight: 700;
        color: #00A6FB;
        animation: floatIcon 4s ease-in-out infinite, glow 3s ease-in-out infinite;
    }

    .login-title {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: -0.5px;
        animation: slideInLeft 0.8s ease-out 0.2s both;
    }

    .login-subtitle {
        font-size: 15px;
        color: rgba(255,255,255,0.6);
        text-align: center;
        margin-bottom: 40px;
        font-weight: 300;
        animation: slideInLeft 0.8s ease-out 0.3s both;
    }

    /* Form inputs */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.04) !important;
        border: 1.5px solid rgba(1,166,251,0.2) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        padding: 16px 20px !important;
        font-size: 16px !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(5px) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.4) !important;
    }

    .stTextInput > div > div > input:hover {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(1,166,251,0.4) !important;
        box-shadow: 0 4px 12px rgba(1,166,251,0.15) !important;
    }

    .stTextInput > div > div > input:focus {
        background: rgba(255,255,255,0.08) !important;
        border-color: #00A6FB !important;
        box-shadow: 0 0 0 3px rgba(1,166,251,0.25), 0 4px 20px rgba(1,166,251,0.2) !important;
    }

    /* Sign In button */
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(90deg, #0173B2 0%, #00A6FB 50%, #00D4FF 100%) !important;
        background-size: 200% 100% !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 18px 28px !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1) !important;
        margin-top: 12px !important;
        letter-spacing: 0.5px;
        animation: slideInLeft 0.8s ease-out 0.4s both;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover {
        background-position: 200% 0 !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(1,115,178,0.45), 0 0 20px rgba(0,166,251,0.3) !important;
        letter-spacing: 1px;
    }

    .stButton > button:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(1,115,178,0.35) !important;
    }

    /* Error message */
    .login-error {
        background: linear-gradient(135deg, rgba(231,76,60,0.15), rgba(192,57,43,0.1));
        border: 1.5px solid rgba(231,76,60,0.4);
        border-radius: 12px;
        padding: 14px 18px;
        color: #ff6b6b;
        font-size: 14px;
        margin-bottom: 22px;
        text-align: center;
        font-weight: 500;
        animation: fadeInDown 0.5s ease-out;
        backdrop-filter: blur(5px);
    }

    /* Success message */
    .login-success {
        background: linear-gradient(135deg, rgba(46,204,113,0.15), rgba(39,174,96,0.1));
        border: 1.5px solid rgba(46,204,113,0.4);
        border-radius: 12px;
        padding: 14px 18px;
        color: #2ecc71;
        font-size: 14px;
        margin-bottom: 22px;
        text-align: center;
        font-weight: 500;
        animation: fadeInDown 0.5s ease-out;
        backdrop-filter: blur(5px);
    }

    /* Footer */
    .login-footer {
        text-align: center;
        margin-top: 32px;
        padding-top: 24px;
        border-top: 1px solid rgba(1,166,251,0.1);
        font-size: 12px;
        color: rgba(255,255,255,0.5);
        font-weight: 300;
        line-height: 1.6;
        animation: fadeInUp 0.8s ease-out 0.5s both;
    }

    /* Toggle link button (Create account / Back to sign in) */
    div[data-testid="stButton"] button[kind="secondary"] {
        background: transparent !important;
        color: #00A6FB !important;
        border: none !important;
        box-shadow: none !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        text-decoration: underline;
        padding: 4px 0 !important;
        width: auto !important;
        margin-top: 4px !important;
    }

    div[data-testid="stButton"] button[kind="secondary"]:hover {
        color: #00D4FF !important;
        transform: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    .auth-toggle-row {
        text-align: center;
        margin-top: 18px;
        font-size: 14px;
        color: rgba(255,255,255,0.55);
    }

    /* Role selector radio styling */
    div[role="radiogroup"] label {
        color: rgba(255,255,255,0.85) !important;
    }

    /* Ensure all controls use Poppins */
    input, textarea, label, button {
        font-family: 'Poppins', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Centered login container
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-logo">
                <div class="login-logo-icon" style="width:180px; height:180px; margin:0 auto 24px; background:rgba(1,115,178,0.14); border-radius:24px; border:1px solid rgba(1,166,251,0.22); display:flex; align-items:center; justify-content:center;">
                    <svg viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg" style="width: 108px; height: 108px;">
                        <defs>
                            <linearGradient id="lg1" x1="0" y1="0" x2="1" y2="1">
                                <stop offset="0%" stop-color="#0173B2"/>
                                <stop offset="60%" stop-color="#00A6FB"/>
                                <stop offset="100%" stop-color="#00D4FF"/>
                            </linearGradient>
                            <linearGradient id="lg2" x1="0" y1="1" x2="0" y2="0">
                                <stop offset="0%" stop-color="#0DAFE8" stop-opacity="0.9"/>
                                <stop offset="100%" stop-color="#00D4FF" stop-opacity="1"/>
                            </linearGradient>
                        </defs>
                        <circle cx="80" cy="80" r="74" fill="rgba(1,166,251,0.08)" stroke="url(#lg1)" stroke-opacity="0.2" stroke-width="2"/>
                        <g transform="translate(18,46)">
                            <rect x="0" y="58" width="16" height="42" rx="4" fill="#083144"/>
                            <rect x="24" y="42" width="16" height="58" rx="4" fill="#0b3b54"/>
                            <rect x="48" y="22" width="16" height="78" rx="4" fill="#0e5f7f"/>
                            <rect x="72" y="34" width="16" height="66" rx="4" fill="url(#lg2)"/>
                            <rect x="96" y="50" width="16" height="50" rx="4" fill="#0173B2"/>
                        </g>
                        <polyline points="20,118 46,76 72,58 98,52 124,36" fill="none" stroke="url(#lg1)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M126 34 L118 44 L124 40 L126 34" fill="#00D4FF"/>
                    </svg>
                </div>
                <div class="login-title">NGX Volatility Prediction System</div>
                <div class="login-subtitle">Secure Access Portal</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Decide which form to show: sign-in or create-account
        if "auth_view" not in st.session_state:
            st.session_state.auth_view = "login"

        if st.session_state.auth_view == "register":
            render_registration_form()
        else:
            render_login_form()

        st.markdown("""
        <div class="login-footer">
            B.Sc. Computer Science Final Year Project<br>
            Crawford University | © 2026 NGX Volatility Prediction System
        </div>
        """, unsafe_allow_html=True)


def render_login_form():
    """Render the sign-in form and the link to switch to registration."""
    import streamlit as st

    # Show error message if any
    if "login_error" in st.session_state and st.session_state.login_error:
        st.markdown(f"""
        <div class="login-error">
            {st.session_state.login_error}
        </div>
        """, unsafe_allow_html=True)
        st.session_state.login_error = None

    # Show session expired message
    if "session_expired" in st.session_state and st.session_state.session_expired:
        st.markdown("""
        <div class="login-error">
            Your session has expired. Please log in again.
        </div>
        """, unsafe_allow_html=True)
        st.session_state.session_expired = False

    # Show post-registration success message
    if "register_success" in st.session_state and st.session_state.register_success:
        st.markdown("""
        <div class="login-success">
            Account created successfully. You can now sign in below.
        </div>
        """, unsafe_allow_html=True)
        st.session_state.register_success = False

    # Login form
    username = st.text_input("Username", placeholder="Enter your username", key="login_username")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

    login_clicked = st.button("Sign In", use_container_width=True, key="login_submit")

    # Link to switch to the registration view
    st.markdown('<div class="auth-toggle-row">New to the platform?</div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        create_clicked = st.button("Create account", key="goto_register", type="secondary", use_container_width=True)

    if create_clicked:
        st.session_state.auth_view = "register"
        st.rerun()

    # Handle login
    if login_clicked:
        if not username or not password:
            st.session_state.login_error = "Please enter both username and password."
            st.rerun()
        elif verify_credentials(username, password):
            login_user(username)
            st.session_state.login_error = None
            st.rerun()
        else:
            st.session_state.login_error = "Invalid username or password. Please try again."
            st.rerun()


def render_registration_form():
    """Render the create-account form (Guest or Analyst roles only)."""
    import streamlit as st

    st.markdown(
        '<div class="login-title" style="font-size:24px; margin-bottom:6px;">Create your account</div>'
        '<div class="login-subtitle" style="margin-bottom:28px;">New users join as Guest or Research Analyst</div>',
        unsafe_allow_html=True
    )

    if "register_error" in st.session_state and st.session_state.register_error:
        st.markdown(f"""
        <div class="login-error">
            {st.session_state.register_error}
        </div>
        """, unsafe_allow_html=True)
        st.session_state.register_error = None

    full_name = st.text_input("Full name", placeholder="e.g. Ada Lovelace", key="reg_full_name")
    new_username = st.text_input("Choose a username", placeholder="Enter a username", key="reg_username")
    new_password = st.text_input("Choose a password", type="password", placeholder="At least 8 characters", key="reg_password")
    confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter your password", key="reg_confirm_password")

    role_choice = st.radio(
        "Account type",
        options=["Guest", "Analyst"],
        index=0,
        horizontal=True,
        key="reg_role",
        help="Guests have read-only/demo access. Analysts can run full research workflows. Administrator accounts cannot be self-registered."
    )

    register_clicked = st.button("Create account", use_container_width=True, key="register_submit")

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        back_clicked = st.button("Back to sign in", key="goto_login", type="secondary", use_container_width=True)

    if back_clicked:
        st.session_state.auth_view = "login"
        st.rerun()

    if register_clicked:
        error = validate_new_account(full_name, new_username, new_password, confirm_password)
        if error:
            st.session_state.register_error = error
            st.rerun()
        else:
            create_user_account(new_username.strip(), new_password, role_choice, full_name.strip())
            st.session_state.auth_view = "login"
            st.session_state.register_success = True
            st.rerun()


def validate_new_account(full_name: str, username: str, password: str, confirm_password: str) -> Optional[str]:
    """Validate registration form input. Returns an error string, or None if valid."""
    full_name = (full_name or "").strip()
    username = (username or "").strip()

    if not full_name or not username or not password or not confirm_password:
        return "Please fill in every field."

    if not username.isascii() or not username.replace("_", "").replace(".", "").isalnum():
        return "Username may only contain letters, numbers, underscores, and periods."

    if username.lower() in {"admin", "administrator", "root"}:
        return "That username is reserved. Please choose another."

    if len(username) < 3:
        return "Username must be at least 3 characters."

    if len(password) < 8:
        return "Password must be at least 8 characters."

    if password != confirm_password:
        return "Passwords do not match."

    users = load_users()
    if username.lower() in {u.lower() for u in users.keys()}:
        return "That username is already taken. Please choose another."

    return None


def create_user_account(username: str, password: str, role: str, full_name: str) -> None:
    """Persist a new Guest or Analyst account to the users store."""
    if role not in {"Guest", "Analyst"}:
        role = "Guest"  # enforced fallback — registration can never create Administrators

    users = load_users()
    credentials = hash_password(password)
    users[username] = {
        "salt": credentials["salt"],
        "hash": credentials["hash"],
        "iterations": credentials["iterations"],
        "role": role,
        "full_name": full_name
    }
    save_users(users)


def render_dashboard():
    """Render the main dashboard after successful authentication."""
    # Import and run the existing dashboard
    try:
        from dashboard import create_streamlit_app
        create_streamlit_app()
    except ImportError as e:
        import streamlit as st
        st.error(f"ERROR: Cannot import from dashboard.py: {e}")
        st.stop()


def main():
    """Main entry point with authentication gate."""
    import streamlit as st

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Check if user is already authenticated and session is valid
    if st.session_state.authenticated and is_session_valid():
        # User is logged in — show dashboard
        render_dashboard()
    else:
        # Session expired while user was away
        if st.session_state.authenticated and not is_session_valid():
            st.session_state.session_expired = True

        # Show login page
        render_login_page()


if __name__ == "__main__":
    main()
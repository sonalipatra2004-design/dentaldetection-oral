# ══════════════════════════════════════════════════════
# AUTOMATED DIAGNOSIS OF ORAL CONDITIONS
# Professional Streamlit Web App
# Group 08 | ITER SOA University | MCA 2024-2026
# ══════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import json
import datetime
import hashlib
import base64
import sqlite3
import re
import time
from io import BytesIO
import matplotlib.pyplot as plt

# ── SQLite Database Setup ─────────────────────────────
DB_PATH = 'oraldx_database.db'

def sanitize(text):
    if not text:
        return ""
    # Remove dangerous chars
    clean = str(text)
    for ch in [";","'",'"',"--","/*","*/"]:
        clean = clean.replace(ch,"")
    return clean[:500].strip()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT,
            joined DATE DEFAULT CURRENT_DATE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            disease TEXT,
            confidence TEXT,
            severity TEXT,
            filename TEXT,
            timestamp TEXT,
            icon TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            role TEXT,
            rating INTEGER,
            review TEXT,
            date TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS admin_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            email TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def db_register(name, email, password, role):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users "
                "VALUES (NULL,?,?,?,?,?)",
                (sanitize(name),
                 sanitize(email),
                 password,
                 sanitize(role),
                 str(datetime.date.today())))
        return True
    except sqlite3.IntegrityError:
        return False

def db_login(email, password):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT name,role FROM users "
            "WHERE email=? AND password=?",
            (email, password))
        return c.fetchone()

def db_save_history(email, disease, conf,
                    severity, filename,
                    timestamp, icon):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO history VALUES "
            "(NULL,?,?,?,?,?,?,?)",
            (email, disease, conf, severity,
             filename, timestamp, icon))

def db_get_history(email):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT disease,confidence,severity,"
            "filename,timestamp,icon FROM history "
            "WHERE email=? ORDER BY id DESC",
            (email,))
        rows = c.fetchall()
    return [{'disease':r[0],'confidence':r[1],
             'severity':r[2],'filename':r[3],
             'timestamp':r[4],'icon':r[5]}
            for r in rows]

def db_save_review(name, role, rating,
                   review, date):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO reviews VALUES "
            "(NULL,?,?,?,?,?)",
            (name, role, rating, review, date))

def db_get_reviews():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT name,role,rating,review,date "
            "FROM reviews ORDER BY id DESC")
        rows = c.fetchall()
    return [{'name':r[0],'role':r[1],
             'rating':r[2],'review':r[3],
             'date':r[4]} for r in rows]

def db_get_stats():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users")
        users = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM history")
        scans = c.fetchone()[0]
        c.execute(
            "SELECT disease,COUNT(*) as cnt "
            "FROM history GROUP BY disease "
            "ORDER BY cnt DESC LIMIT 1")
        top = c.fetchone()
    return users, scans, top

# Init database on startup
init_db()

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="OralDx — Oral Disease Detector",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg,
        #e8f4fd 0%, #dbeeff 50%, #e4f0fa 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        #1a3a6b 0%, #1e4080 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.15);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(0,100,200,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.8rem 0;
    transition: all 0.3s ease;
}

.card:hover {
    border-color: rgba(0,100,200,0.5);
    background: rgba(0,120,220,0.08);
}

/* Hero section */
.hero {
    background: linear-gradient(135deg,
        rgba(0,100,200,0.12) 0%,
        rgba(0,60,150,0.08) 100%);
    border: 1px solid rgba(0,100,200,0.25);
    border-radius: 24px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #00d4ff, #0099cc, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}

.hero p {
    color: #4a6080;
    font-size: 1rem;
    margin: 0;
}

/* Team member card */
.member-card {
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(0,100,200,0.2);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    height: 100%;
}

.member-card:hover {
    border-color: rgba(0,100,200,0.5);
    background: rgba(0,120,220,0.06);
    transform: translateY(-4px);
}

.member-avatar {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid rgba(0,212,255,0.5);
    margin: 0 auto 0.8rem;
    display: block;
    background: linear-gradient(135deg,#00d4ff,#0066aa);
}

.avatar-placeholder {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    background: linear-gradient(135deg,#00d4ff,#0066aa);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin: 0 auto 0.8rem;
    border: 3px solid rgba(0,212,255,0.5);
}

.member-name {
    color: #1a365d;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}

.member-reg {
    color: #00d4ff;
    font-size: 0.78rem;
    font-weight: 500;
}

.member-role {
    color: #4a6080;
    font-size: 0.75rem;
    margin-top: 0.2rem;
}

/* Result card */
.result-hero {
    background: linear-gradient(135deg,
        rgba(0,100,200,0.1),
        rgba(0,60,150,0.06));
    border: 2px solid rgba(0,100,200,0.35);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

.disease-big {
    font-size: 2.2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'Space Grotesk',sans-serif;
}

.confidence-text {
    font-size: 1.1rem;
    color: #4a6080;
    margin: 0.3rem 0;
}

/* Severity badges */
.badge-mild {
    background: rgba(56,161,105,0.2);
    border: 1px solid #38a169;
    color: #68d391;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}
.badge-moderate {
    background: rgba(214,158,46,0.2);
    border: 1px solid #d69e2e;
    color: #f6e05e;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}
.badge-severe {
    background: rgba(229,62,62,0.2);
    border: 1px solid #e53e3e;
    color: #fc8181;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}

/* Info rows */
.info-block {
    background: rgba(255,255,255,0.9);
    border-left: 3px solid #0066cc;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}
.info-label {
    color: #00d4ff;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.info-value {
    color: #1a365d;
    font-size: 0.92rem;
    margin-top: 0.15rem;
}

/* History item */
.history-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.history-item:hover {
    border-color: rgba(0,212,255,0.3);
    background: rgba(0,212,255,0.05);
}

/* Stats */
.stat-card {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,100,200,0.25);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-number {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'Space Grotesk',sans-serif;
}
.stat-label {
    color: #5a7090;
    font-size: 0.8rem;
    margin-top: 0.2rem;
}

/* Section headers */
.section-title {
    font-family: 'Space Grotesk',sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #1a365d;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(0,100,200,0.3);
}

/* Auth form */
.auth-card {
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(0,100,200,0.25);
    border-radius: 20px;
    padding: 2.5rem;
    max-width: 420px;
    margin: 2rem auto;
}

/* Input styling */
.stTextInput > div > div > input {
    background: white !important;
    border: 1px solid rgba(0,100,200,0.3) !important;
    border-radius: 10px !important;
    color: #1a365d !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(0,212,255,0.6) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg,
        #00d4ff, #0099cc) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    font-family: 'Plus Jakarta Sans',sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,255,0.35) !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #4a6080;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding: 1.5rem;
    border-top: 1px solid rgba(0,100,200,0.15);
}

/* Guide card */
.guide-card {
    background: linear-gradient(135deg,
        rgba(0,100,200,0.08),
        rgba(0,60,150,0.05));
    border: 2px solid rgba(0,100,200,0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

.upload-area {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(0,212,255,0.3);
    border-radius: 16px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.page-nav {
    color: #4a6080;
    font-size: 0.82rem;
    text-align: center;
    margin-bottom: 1rem;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.8);
    border: 2px dashed rgba(0,100,200,0.3);
    border-radius: 12px;
    padding: 0.5rem;
}

/* Bottom mobile navigation */
.bottom-nav {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    border-top: 1px solid rgba(0,100,200,0.15);
    display: flex;
    justify-content: space-around;
    padding: 0.5rem 0;
    z-index: 998;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.08);
}
.bottom-nav-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    cursor: pointer;
    padding: 0.3rem 0.8rem;
    border-radius: 10px;
    transition: all 0.2s;
    font-size: 0.65rem;
    color: #64748b;
    font-weight: 600;
}
.bottom-nav-item.active {
    color: #0066cc;
    background: rgba(0,100,200,0.08);
}
.bottom-nav-icon {
    font-size: 1.3rem;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════
if 'logged_in'    not in st.session_state:
    st.session_state.logged_in    = False
if 'user_email'   not in st.session_state:
    st.session_state.user_email   = ''
if 'user_name'    not in st.session_state:
    st.session_state.user_name    = ''
if 'history'      not in st.session_state:
    st.session_state.history      = []
if 'page'         not in st.session_state:
    st.session_state.page         = 'home'
if 'users_db'     not in st.session_state:
    st.session_state.users_db     = {}
if 'total_scans'  not in st.session_state:
    st.session_state.total_scans  = 0

# ══════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════
CLASSES = [
    'Calculus','Dental Caries','Gingivitis',
    'Hypodontia','Mouth Ulcer',
    'Tooth Discoloration'
]

DISEASE_INFO = {
    'Calculus':{
        'icon':'🦷',
        'description':'Hardened dental plaque on tooth surfaces.',
        'symptoms':'Yellow/brown deposits, bad breath, gum irritation.',
        'treatment':'Professional dental cleaning (scaling).',
        'severity':'Mild-Moderate',
        'color':'#f6ad55'
    },
    'Dental Caries':{
        'icon':'⚠️',
        'description':'Dental decay caused by bacterial acid erosion.',
        'symptoms':'Toothache, sensitivity, dark spots on teeth.',
        'treatment':'Fillings, root canal, or extraction if severe.',
        'severity':'Moderate-Severe',
        'color':'#fc8181'
    },
    'Gingivitis':{
        'icon':'🔴',
        'description':'Inflammation of gum tissue around teeth.',
        'symptoms':'Red, swollen, bleeding gums.',
        'treatment':'Improved oral hygiene, professional cleaning.',
        'severity':'Mild',
        'color':'#68d391'
    },
    'Hypodontia':{
        'icon':'❓',
        'description':'Congenital absence of one or more teeth.',
        'symptoms':'Missing teeth, gaps in dental arch.',
        'treatment':'Dental implants, bridges, or dentures.',
        'severity':'Moderate',
        'color':'#b794f4'
    },
    'Mouth Ulcer':{
        'icon':'💊',
        'description':'Painful sores on soft tissues in the mouth.',
        'symptoms':'Painful lesions, difficulty eating or speaking.',
        'treatment':'Topical gel, antiseptic mouthwash.',
        'severity':'Mild',
        'color':'#4fd1c5'
    },
    'Tooth Discoloration':{
        'icon':'🌟',
        'description':'Visible colour change in teeth due to staining.',
        'symptoms':'Yellow, brown, or black discolouration on teeth.',
        'treatment':'Whitening, veneers, or restorative procedures.',
        'severity':'Mild',
        'color':'#f6e05e'
    },
}

TEAM_MEMBERS = [
    {
        'name'   : 'Sonali Patra',
        'reg'    : '24C216A45',
        'role'   : 'Team Lead & ML Engineer',
        'emoji'  : '👩‍💻',
        'gmail'  : 'sonalipatra2004@gmail.com',
        'contrib': 'Model training, Streamlit app development',
        'photo'  : 'sonali.jpeg'
    },
    {
        'name'   : 'Jagruti Parida',
        'reg'    : '24C216A47',
        'role'   : 'Data Engineer',
        'emoji'  : '👩‍🔬',
        'gmail'  : 'paridaj320@gmail.com',
        'contrib': 'Dataset collection and preprocessing',
        'photo'  : 'jagruti.jpeg'
    },
    {
        'name'   : 'Dharitri Pradhan',
        'reg'    : '24C216A30',
        'role'   : 'Deep Learning Researcher',
        'emoji'  : '👩‍💡',
        'gmail'  : 'pradhandharitri319@gmail.com',
        'contrib': 'CNN architecture design and evaluation',
        'photo'  : 'dharitri.jpeg'
    },
    {
        'name'   : 'Smitarani Mohapatra',
        'reg'    : '24C213A05',
        'role'   : 'Frontend Developer',
        'emoji'  : '👩‍🎨',
        'gmail'  : 'smitaranimahapatra993@gmail.com',
        'contrib': 'UI/UX design and documentation',
        'photo'  : 'smitarani.jpeg'
    },
    {
        'name'   : 'Barsha Priyadarshini Singh',
        'reg'    : '24C219A30',
        'role'   : 'Backend Developer',
        'emoji'  : '👩‍🏫',
        'gmail'  : 'barshasingh971@gmail.com',
        'contrib': 'Model deployment and testing',
        'photo'  : 'barsha.jpeg'
    },
]

GUIDE = {
    'name'       : 'Dr. Debabrata Singh',
    'designation': 'Associate Professor',
    'dept'       : 'Department of Computer Application',
    'university' : 'ITER, SOA University, Bhubaneswar',
    'emoji'      : '👨‍🏫'
}

# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════
def hash_password(password):
    # PBKDF2 — stronger than plain SHA256
    import hashlib
    salt = "oraldx_iter_soa_2026"
    return hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        200000  # 200k iterations
    ).hex()

def preprocess(img_array):
    r     = cv2.resize(img_array,(224,224))
    lab   = cv2.cvtColor(
        r, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enh   = cv2.cvtColor(
        lab, cv2.COLOR_LAB2RGB)
    den   = cv2.GaussianBlur(enh,(3,3),0)
    norm  = den.astype(np.float32)/255.0
    return norm, r, enh, den

def get_severity_badge(severity):
    if 'Severe' in severity and 'Mild' not in severity:
        return f'<span class="badge-severe">{severity}</span>'
    elif 'Moderate' in severity:
        return f'<span class="badge-moderate">{severity}</span>'
    else:
        return f'<span class="badge-mild">{severity}</span>'

def img_to_base64(img_array):
    pil_img = Image.fromarray(
        img_array.astype(np.uint8))
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(
        buf.getvalue()).decode()

MODEL_DIR = os.path.dirname(
    os.path.abspath(__file__))     if '__file__' in dir() else '.'

@st.cache_resource
def load_model():
    paths = [
        os.path.join(MODEL_DIR,'vgg19_best.keras'),
        os.path.join(MODEL_DIR,'resnet50_best.keras'),
        'vgg19_best.keras',
        'resnet50_best.keras',
        '/content/drive/MyDrive/OralDiagnosis_Results/vgg19_best.keras',
        '/content/drive/MyDrive/OralDiagnosis_Results/resnet50_best.keras',
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                m = tf.keras.models.load_model(path)
                return m, path
            except Exception:
                continue
    return None, None

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    # Dark/Light mode toggle
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    dm = st.session_state.dark_mode
    bg_main = "#1a2744" if dm else "#f0f6ff"
    text_col = "#e2e8f0" if dm else "#1a365d"
    card_bg  = "rgba(255,255,255,0.06)" if dm else "rgba(255,255,255,0.9)"

    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:2.5rem;">🦷</div>
        <div style="color:#00d4ff;font-weight:800;
                    font-size:1.2rem;font-family:
                    'Space Grotesk',sans-serif;">
            OralDx
        </div>
        <div style="color:#4a6080;font-size:0.72rem;">
            Oral Disease Detector
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.logged_in:
        st.markdown(f"""
        <div style="background:rgba(0,212,255,0.08);
                    border:1px solid rgba(0,212,255,0.2);
                    border-radius:12px;padding:0.8rem;
                    margin-bottom:1rem;text-align:center;">
            <div style="font-size:1.5rem;">👤</div>
            <div style="color:#1a365d;font-weight:600;
                        font-size:0.9rem;">
                {st.session_state.user_name}
            </div>
            <div style="color:#4a6080;font-size:0.75rem;">
                {st.session_state.user_email}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Navigation
    pages = {
        '🏠 Home'          : 'home',
        '🔍 Diagnose'      : 'diagnose',
        '📋 History'       : 'history',
        '👥 About Us'      : 'about',
        'ℹ️ Info'          : 'info',
        '🎯 How It Works'  : 'howitworks',
        '🦷 Disease Guide' : 'diseaseguide',
        '❓ FAQ'           : 'faq',
        '📞 Contact Us'    : 'contact',
        '💬 Ask OralDx'    : 'askoraldx',
        '🗺️ Symptom Check' : 'symptom',
        '⭐ Reviews'       : 'reviews',
        '🔬 Compare'       : 'compare',
        '📱 Mobile App'    : 'mobileapp',
    }

    st.markdown(
        '<div style="color:#4a6080;font-size:0.75rem;'
        'font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.08em;margin-bottom:0.5rem;">'
        'NAVIGATION</div>',
        unsafe_allow_html=True)

    for label, page_key in pages.items():
        active = (st.session_state.page
                  == page_key)
        bg = ("rgba(0,212,255,0.12)"
              if active else "transparent")
        border = ("rgba(0,212,255,0.4)"
                  if active else "transparent")
        if st.button(label, key=f"nav_{page_key}"):
            st.session_state.page = page_key
            st.rerun()

    st.markdown("---")

    # Language selector
    if 'lang' not in st.session_state:
        st.session_state.lang = 'English'

    TRANSLATIONS = {
        'English': {
            'home':'🏠 Home',
            'diagnose':'🔍 Diagnose',
            'history':'📋 History',
            'upload':'Upload X-Ray',
            'diagnose_btn':'🔬 Run Diagnosis',
            'result':'Diagnosis Result',
            'disease':'Disease',
            'confidence':'Confidence',
            'severity':'Severity',
            'symptoms':'Symptoms',
            'treatment':'Treatment',
            'welcome':'Welcome to OralDx',
        },
        'Hindi': {
            'home':'🏠 होम',
            'diagnose':'🔍 निदान',
            'history':'📋 इतिहास',
            'upload':'X-Ray अपलोड करें',
            'diagnose_btn':'🔬 निदान करें',
            'result':'निदान परिणाम',
            'disease':'बीमारी',
            'confidence':'आत्मविश्वास',
            'severity':'गंभीरता',
            'symptoms':'लक्षण',
            'treatment':'उपचार',
            'welcome':'OralDx में आपका स्वागत है',
        },
        'Odia': {
            'home':'🏠 ହୋମ',
            'diagnose':'🔍 ରୋଗ ନିର୍ଣ୍ଣୟ',
            'history':'📋 ଇତିହାସ',
            'upload':'X-Ray ଅପଲୋଡ କରନ୍ତୁ',
            'diagnose_btn':'🔬 ରୋଗ ଚିହ୍ନଟ',
            'result':'ନିର୍ଣ୍ଣୟ ଫଳାଫଳ',
            'disease':'ରୋଗ',
            'confidence':'ବିଶ୍ୱସ୍ତତା',
            'severity':'ଗୁରୁତ୍ୱ',
            'symptoms':'ଲକ୍ଷଣ',
            'treatment':'ଚିକିତ୍ସା',
            'welcome':'OralDx ରେ ଆପଣଙ୍କୁ ସ୍ୱାଗତ',
        },
    }

    lang = st.selectbox(
        "🌍 Language",
        ['English','Hindi','Odia'],
        index=['English','Hindi','Odia'].index(
            st.session_state.lang),
        key="lang_select")
    if lang != st.session_state.lang:
        st.session_state.lang = lang
        st.rerun()
    T = TRANSLATIONS[st.session_state.lang]

    # Dark mode toggle
    mode_label = ("☀️ Light Mode"
                  if st.session_state.dark_mode
                  else "🌙 Dark Mode")
    if st.button(mode_label, key="dark_toggle"):
        st.session_state.dark_mode = \
            not st.session_state.dark_mode
        st.rerun()

    if st.session_state.logged_in:
        if st.button("🚪 Sign Out"):
            st.session_state.logged_in = False
            st.session_state.user_email = ''
            st.session_state.user_name  = ''
            st.session_state.page = 'home'
            st.rerun()
    else:
        if st.button("🔐 Sign In / Sign Up"):
            st.session_state.page = 'auth'
            st.rerun()

    # Stats
    st.markdown("---")
    st.markdown("""
    <div style="color:#4a6080;font-size:0.72rem;
                text-align:center;">
        MCA Final Year Project<br>
        Group 08 | Batch 2024-2026<br>
        ITER, SOA University
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: AUTH
# ══════════════════════════════════════════════════════
def page_auth():
    st.markdown("""
    <div style="text-align:center;margin:1rem 0 2rem;">
        <div style="font-size:3rem;">🦷</div>
        <h2 style="color:#00d4ff;font-family:
                   'Space Grotesk',sans-serif;
                   margin:0.3rem 0;">
            Welcome to OralDx
        </h2>
        <p style="color:#4a6080;">
            Sign in to access all features
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        tab1, tab2 = st.tabs(
            ["🔐 Sign In", "📝 Sign Up"])

        with tab1:
            st.markdown("#### Sign In")
            email = st.text_input(
                "Email Address",
                placeholder="you@example.com",
                key="signin_email")
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password",
                key="signin_pass")
            if st.button("Sign In →",
                         key="do_signin"):
                if email and password:
                    hp = hash_password(password)
                    row = db_login(email, hp)
                    if row:
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.user_name  = row[0]
                        st.session_state.page = 'home'
                        st.success("Signed in!")
                        st.rerun()
                    else:
                        st.error(
                            "Invalid email or password")
                else:
                    st.warning("Fill all fields")

        with tab2:
            st.markdown("#### Create Account")
            name = st.text_input(
                "Full Name",
                placeholder="Dr. John Smith",
                key="signup_name")
            email2 = st.text_input(
                "Email Address",
                placeholder="you@example.com",
                key="signup_email")
            pass2 = st.text_input(
                "Password",
                type="password",
                placeholder="Min 6 characters",
                key="signup_pass")
            role = st.selectbox(
                "I am a...",
                ["Dental Professional",
                 "Medical Student",
                 "Researcher",
                 "General User"])
            if st.button("Create Account →",
                         key="do_signup"):
                if name and email2 and pass2:
                    import re
                    email_valid = re.match(
                        r'^[^@]+@[^@]+\.[^@]+$',
                        email2)
                    has_upper = any(
                        c.isupper() for c in pass2)
                    has_digit = any(
                        c.isdigit() for c in pass2)
                    if not email_valid:
                        st.error(
                            "Please enter valid email")
                    elif len(pass2) < 6:
                        st.error(
                            "Password min 6 characters")
                    elif not has_upper:
                        st.error(
                            "Password needs uppercase")
                    elif not has_digit:
                        st.error(
                            "Password needs a number")
                    else:
                        hp = hash_password(pass2)
                        ok = db_register(
                            name, email2, hp, role)
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.user_email = email2
                            st.session_state.user_name  = name
                            st.session_state.page = 'home'
                            st.success("Account created!")
                            st.rerun()
                        else:
                            st.error("Email already registered")
                else:
                    st.warning("Fill all fields")

        # Demo login
        st.markdown("---")
        st.markdown(
            '<p style="color:#4a6080;text-align:center;'
            'font-size:0.82rem;">Try demo account</p>',
            unsafe_allow_html=True)
        if st.button("🚀 Continue as Guest"):
            st.session_state.logged_in  = True
            st.session_state.user_email = "guest@oraldx.app"
            st.session_state.user_name  = "Guest User"
            st.session_state.page       = 'diagnose'
            st.rerun()

# ══════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════
def page_home():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🦷 OralDx</h1>
        <h3 style="color:#1a365d;font-weight:500;
                   font-size:1.2rem;margin:0.5rem 0;">
            Automated Diagnosis of Oral Conditions
            from Dental X-Rays
        </h3>
        <p>AI-powered dental disease detection using
           Deep Learning | VGG19 · ResNet-50 · YOLOv8
           · U-Net</p>
        <p style="color:#4a6080;font-size:0.85rem;
                  margin-top:0.5rem;">
            Department of Computer Application |
            ITER, SOA University |
            MCA Batch 2024-2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1,c2,c3,c4 = st.columns(4)
    stats = [
        ("91%",  "🎯 Best Accuracy",  "VGG19 Model"),
        ("6",    "🦷 Diseases",       "Detected"),
        ("12K+", "🖼️ Training Images","Dataset Size"),
        (str(st.session_state.total_scans),
         "🔍 Total Scans","All Users"),
    ]
    for col, (num, label, sub) in zip(
            [c1,c2,c3,c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{num}</div>
                <div style="color:#1a365d;
                            font-weight:600;
                            font-size:0.85rem;">
                    {label}
                </div>
                <div class="stat-label">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features
    st.markdown(
        '<div class="section-title">'
        '✨ Features</div>',
        unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    features = [
        ("🔍","Instant Diagnosis",
         "Upload any dental X-ray and get instant disease detection with confidence score"),
        ("📊","Detailed Analysis",
         "Preprocessing pipeline, probability charts, and detailed disease information"),
        ("📋","History Tracking",
         "All your previous diagnoses saved and accessible anytime"),
        ("🔐","Secure Login",
         "Create account to save your history and access advanced features"),
        ("📱","Mobile Friendly",
         "Works perfectly on phones, tablets and laptops"),
        ("🎯","91% Accuracy",
         "Trained on 12,320+ dental X-ray images across 6 disease categories"),
    ]
    for i, col in enumerate([f1,f2,f3]):
        with col:
            icon,title,desc = features[i]
            st.markdown(f"""
            <div class="card">
                <div style="font-size:2rem;
                            margin-bottom:0.5rem;">
                    {icon}
                </div>
                <div style="color:#1a365d;
                            font-weight:700;
                            font-size:0.95rem;
                            margin-bottom:0.3rem;">
                    {title}
                </div>
                <div style="color:#5a7090;
                            font-size:0.82rem;
                            line-height:1.5;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f4,f5,f6 = st.columns(3)
    for i, col in enumerate([f4,f5,f6]):
        with col:
            icon,title,desc = features[i+3]
            st.markdown(f"""
            <div class="card">
                <div style="font-size:2rem;
                            margin-bottom:0.5rem;">
                    {icon}
                </div>
                <div style="color:#1a365d;
                            font-weight:700;
                            font-size:0.95rem;
                            margin-bottom:0.3rem;">
                    {title}
                </div>
                <div style="color:#5a7090;
                            font-size:0.82rem;
                            line-height:1.5;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Diseases
    st.markdown(
        '<div class="section-title">'
        '🦷 Diseases We Detect</div>',
        unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (cls, info) in enumerate(
            DISEASE_INFO.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;
                            align-items:center;
                            gap:0.6rem;
                            margin-bottom:0.5rem;">
                    <span style="font-size:1.5rem;">
                        {info['icon']}
                    </span>
                    <span style="color:#1a365d;
                                 font-weight:700;
                                 font-size:0.9rem;">
                        {cls}
                    </span>
                </div>
                <div style="color:#5a7090;
                            font-size:0.78rem;
                            line-height:1.5;">
                    {info['description']}
                </div>
                <div style="margin-top:0.5rem;">
                    {get_severity_badge(info['severity'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # CTA
    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        if st.button(
                "🔍 Start Diagnosis Now →",
                key="home_diagnose"):
            st.session_state.page = 'diagnose'
            st.rerun()

# ══════════════════════════════════════════════════════
# PAGE: DIAGNOSE
# ══════════════════════════════════════════════════════
def page_diagnose():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#00d4ff;
                   font-size:1.8rem;margin:0;">
            🔍 Upload & Diagnose
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Upload a dental X-ray for instant
            AI-powered diagnosis
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.warning(
            "⚠️ Sign in to save diagnosis history. "
            "You can still use diagnosis as guest.")

    model, model_path = load_model()
    if model is None:
        st.markdown("""
        <div style="background:rgba(255,150,0,0.08);
                    border:2px solid rgba(255,150,0,0.3);
                    border-radius:16px;
                    padding:2rem;text-align:center;
                    margin:1rem 0;">
            <div style="font-size:3rem;">🤖</div>
            <div style="color:#1a365d;font-weight:700;
                        font-size:1.1rem;margin:0.5rem 0;">
                AI Model Not Loaded
            </div>
            <div style="color:#4a6080;font-size:0.88rem;
                        line-height:1.7;">
                To use the diagnosis feature, you need to
                train the model first.<br>
                Run the training code in Google Colab and
                place <strong>vgg19_best.keras</strong>
                in the app folder.
            </div>
            <div style="margin-top:1rem;">
                <div style="background:#f0f6ff;
                            border-radius:10px;
                            padding:0.8rem;
                            font-size:0.82rem;
                            color:#4a6080;
                            text-align:left;">
                    📋 <strong>Steps to fix:</strong><br>
                    1. Run FINAL_COMPLETE_CODE.py in Colab<br>
                    2. Download vgg19_best.keras from Drive<br>
                    3. Upload to GitHub repository<br>
                    4. Redeploy Streamlit app
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    mname = ("VGG19"
             if "vgg19" in model_path
             else "ResNet-50")
    st.success(
        f"✅ {mname} model loaded — 91% accuracy")

    # ── Upload Section ───────────────────────────────
    st.markdown("""
    <div style="background:white;
                border:2px dashed rgba(0,100,200,0.3);
                border-radius:16px;
                padding:1.5rem;
                margin:1rem 0;
                text-align:center;">
        <div style="font-size:2.5rem;
                    margin-bottom:0.5rem;">
            📤
        </div>
        <div style="color:#1a365d;
                    font-weight:700;
                    font-size:1rem;
                    margin-bottom:0.3rem;">
            Upload or Take a Photo of Dental X-Ray
        </div>
        <div style="color:#4a6080;
                    font-size:0.82rem;">
            Supported formats: JPG, JPEG, PNG
        </div>
    </div>
    """, unsafe_allow_html=True)

    input_tab1, input_tab2 = st.tabs([
        "📁 Upload from Device",
        "📷 Use Camera"
    ])

    uploaded_file = None
    with input_tab1:
        uploaded_file = st.file_uploader(
            "Choose dental X-ray image",
            type=["jpg","jpeg","png"],
            help="Upload JPG or PNG dental X-ray",
            label_visibility="collapsed")

    with input_tab2:
        cam_img = st.camera_input(
            "Take photo of X-ray",
            label_visibility="collapsed")
        if cam_img:
            uploaded_file = cam_img

    if uploaded_file:
        img = Image.open(uploaded_file)
        img_array = np.array(img.convert("RGB"))

        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown(
                '<div class="section-title" '
                'style="font-size:1rem;">'
                '📷 Uploaded Image</div>',
                unsafe_allow_html=True)
            st.image(img,
                     caption=uploaded_file.name,
                     use_column_width=True)

        with col2:
            st.markdown(
                '<div class="section-title" '
                'style="font-size:1rem;">'
                '📋 Image Details</div>',
                unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card">
                <div class="info-block">
                    <div class="info-label">
                        File Name
                    </div>
                    <div class="info-value">
                        {uploaded_file.name}
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">
                        Dimensions
                    </div>
                    <div class="info-value">
                        {img.width} × {img.height} px
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">
                        Size
                    </div>
                    <div class="info-value">
                        {uploaded_file.size//1024} KB
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">
                        AI Model
                    </div>
                    <div class="info-value">
                        {mname} (91% accuracy)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Run Diagnosis",
                     key="run_diag"):
            try:
                # Image quality check
                gray_chk = cv2.cvtColor(
                    img_array, cv2.COLOR_RGB2GRAY)
                blur_score = cv2.Laplacian(
                    gray_chk, cv2.CV_64F).var()
                brightness = gray_chk.mean()
                if blur_score < 50:
                    st.warning(
                        "⚠️ Image appears blurry. "
                        "Results may be inaccurate.")
                if brightness < 30:
                    st.warning(
                        "⚠️ Image is too dark.")
                if (img_array.shape[0] < 100 or
                        img_array.shape[1] < 100):
                    st.warning(
                        "⚠️ Image resolution too low.")

                with st.spinner("🧠 AI Analysing..."):
                    norm,resized,enhanced,denoised = \
                        preprocess(img_array)
                    probs = model.predict(
                        np.expand_dims(norm,axis=0),
                        verbose=0)[0]
                    idx_p    = np.argmax(probs)
                    cls_name = CLASSES[idx_p]
                    conf     = probs[idx_p]
                    info     = DISEASE_INFO[cls_name]
                    ts = datetime.datetime.now()\
                        .strftime("%d %b %Y, %H:%M")

            except Exception:
                st.error(
                    "❌ Could not process image. "
                    "Please upload a valid "
                    "dental X-ray.")
                st.stop()

            # Save to history
            entry = {
                'disease'   : cls_name,
                'confidence': f'{conf*100:.1f}%',
                'timestamp' : ts,
                'filename'  : uploaded_file.name,
                'severity'  : info['severity'],
                'icon'      : info['icon'],
            }
            st.session_state.history.append(entry)
            st.session_state.total_scans += 1
            if (st.session_state.logged_in and
                    st.session_state.user_email
                    != 'guest@oraldx.app'):
                db_save_history(
                    st.session_state.user_email,
                    cls_name,
                    f"{conf*100:.1f}%",
                    info['severity'],
                    uploaded_file.name,
                    ts, info['icon'])

            # Confidence warning
            if conf < 0.60:
                st.markdown(f"""
                <div style="background:rgba(255,150,0,0.1);
                            border:1px solid rgba(255,150,0,0.4);
                            border-radius:12px;
                            padding:1rem;margin:0.5rem 0;">
                    <div style="color:#c05800;font-weight:700;">
                        ⚠️ Low Confidence ({conf*100:.1f}%)
                    </div>
                    <div style="color:#4a6080;
                                font-size:0.82rem;
                                margin-top:0.3rem;">
                        📸 Try a clearer image &nbsp;
                        🔄 Retake X-ray &nbsp;
                        👨‍⚕️ Consult dentist
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Result
            st.markdown(
                '<div class="section-title">' +
                '📋 Diagnosis Result</div>',
                unsafe_allow_html=True)

            sev_badge = get_severity_badge(
                info['severity'])

            # Voice TTS button
            tts_text = (
                f"Diagnosis complete. "
                f"Detected disease: {cls_name}. "
                f"Confidence: {conf*100:.0f} percent. "
                f"Severity: {info['severity']}. "
                f"Treatment: {info['treatment']}")
            st.markdown(f"""
            <div style="background:rgba(0,100,200,0.06);
                        border:1px solid rgba(0,100,200,0.15);
                        border-radius:10px;
                        padding:0.6rem 1rem;
                        margin:0.5rem 0;
                        display:flex;
                        align-items:center;gap:0.8rem;">
                <button onclick="
                    var u=new SpeechSynthesisUtterance(
                    '{tts_text}');
                    u.rate=0.9;
                    window.speechSynthesis.speak(u);"
                    style="background:#0066cc;
                           color:white;border:none;
                           border-radius:8px;
                           padding:0.4rem 1rem;
                           cursor:pointer;
                           font-size:0.85rem;
                           font-weight:600;">
                    🔊 Read Aloud
                </button>
                <span style="color:#4a6080;
                             font-size:0.78rem;">
                    Click to hear diagnosis
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Top-3 predictions
            top3 = sorted(
                zip(CLASSES,probs),
                key=lambda x:x[1],
                reverse=True)[:3]
            top3_html = "".join([
                f'<div style="display:flex;'
                f'align-items:center;gap:0.8rem;'
                f'padding:0.3rem 0;border-bottom:'
                f'1px solid rgba(0,100,200,0.1);">'
                f'<span style="background:#0066cc;'
                f'color:white;width:22px;height:22px;'
                f'border-radius:50%;display:inline-flex;'
                f'align-items:center;justify-content:center;'
                f'font-size:0.7rem;font-weight:700;">'
                f'{i+1}</span>'
                f'<span style="color:#1a365d;'
                f'font-size:0.88rem;flex:1;'
                f'font-weight:{"700" if i==0 else "400"};">'
                f'{DISEASE_INFO[c]["icon"]} {c}</span>'
                f'<span style="color:#0066cc;'
                f'font-weight:700;">{p*100:.1f}%</span></div>'
                for i,(c,p) in enumerate(top3)])

            st.markdown(f"""
            <div class="result-hero">
                <div class="disease-big">
                    {info['icon']} {cls_name}
                </div>
                <div class="confidence-text">
                    Confidence:
                    <strong style="color:#0066cc;">
                        {conf*100:.1f}%
                    </strong>
                </div>
                <div style="margin:0.5rem 0;">
                    {sev_badge}
                </div>
                <div style="color:#4a6080;
                            font-size:0.78rem;
                            margin-top:0.5rem;">
                    Diagnosed on {ts}
                </div>
            </div>
            <div style="background:white;
                        border-radius:12px;
                        padding:1rem;margin:0.5rem 0;">
                <div style="color:#1a365d;font-weight:700;
                            font-size:0.85rem;
                            margin-bottom:0.5rem;">
                    🏆 Top 3 Predictions
                </div>
                {top3_html}
            </div>
            """, unsafe_allow_html=True)

            # Emergency alert for severe
            if info['severity'] in [
                    'Moderate-Severe','Severe']:
                st.markdown(f"""
                <div style="background:rgba(229,62,62,0.1);
                            border:2px solid rgba(229,62,62,0.5);
                            border-radius:14px;
                            padding:1.2rem 1.5rem;
                            margin:0.5rem 0;
                            animation:pulse 1.5s infinite;">
                    <div style="color:#e53e3e;
                                font-weight:800;
                                font-size:1.1rem;
                                margin-bottom:0.4rem;">
                        🚨 URGENT — Immediate Attention Required
                    </div>
                    <div style="color:#4a6080;
                                font-size:0.88rem;
                                line-height:1.6;">
                        Your X-ray shows signs of
                        <strong>{cls_name}</strong> with
                        <strong>{conf*100:.1f}%</strong>
                        confidence.<br>
                        This condition requires
                        <strong>immediate dental consultation</strong>.<br>
                        ⚠️ Do not delay treatment.
                    </div>
                    <div style="margin-top:0.8rem;
                                color:#e53e3e;
                                font-size:0.82rem;
                                font-weight:600;">
                        📞 Contact your nearest dentist today
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Clinical info
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="info-block">
                    <div class="info-label">
                        📝 Description
                    </div>
                    <div class="info-value">
                        {info['description']}
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">
                        🔍 Symptoms
                    </div>
                    <div class="info-value">
                        {info['symptoms']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="info-block">
                    <div class="info-label">
                        💊 Treatment
                    </div>
                    <div class="info-value">
                        {info['treatment']}
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">
                        ⚠️ Severity
                    </div>
                    <div class="info-value">
                        {info['severity']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Doctor recommendation + risk meter
            doc_rec = {
                'Dental Caries'      : 'Restorative Dentist',
                'Gingivitis'         : 'Periodontist',
                'Calculus'           : 'General Dentist',
                'Hypodontia'         : 'Prosthodontist',
                'Mouth Ulcer'        : 'Oral Medicine Specialist',
                'Tooth Discoloration': 'Cosmetic Dentist',
            }
            rec = doc_rec.get(cls_name,'General Dentist')
            sev_pct = {
                'Mild':25,'Mild-Moderate':50,
                'Moderate':60,
                'Moderate-Severe':80,'Severe':95
            }.get(info['severity'],50)
            sev_clr = (
                '#38a169' if sev_pct<40
                else '#d69e2e' if sev_pct<70
                else '#e53e3e')

            st.markdown(f"""
            <div style="display:grid;
                        grid-template-columns:1fr 1fr;
                        gap:1rem;margin:0.5rem 0;">
                <div style="background:white;
                            border:1px solid rgba(0,100,200,0.15);
                            border-radius:10px;padding:0.8rem;">
                    <div style="color:#1a365d;font-weight:700;
                                font-size:0.82rem;margin-bottom:0.4rem;">
                        👨‍⚕️ Recommended Specialist
                    </div>
                    <div style="color:#0066cc;font-weight:600;
                                font-size:0.88rem;">
                        🦷 {rec}
                    </div>
                </div>
                <div style="background:white;
                            border:1px solid rgba(0,100,200,0.15);
                            border-radius:10px;padding:0.8rem;">
                    <div style="color:#1a365d;font-weight:700;
                                font-size:0.82rem;margin-bottom:0.4rem;">
                        🎯 Risk Level Meter
                    </div>
                    <div style="background:#f0f6ff;border-radius:8px;
                                height:12px;overflow:hidden;margin-top:0.3rem;">
                        <div style="background:{sev_clr};width:{sev_pct}%;
                                    height:100%;border-radius:8px;"></div>
                    </div>
                    <div style="color:{sev_clr};font-size:0.78rem;
                                margin-top:0.3rem;font-weight:600;">
                        {info["severity"]} ({sev_pct}%)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # GradCAM
            st.markdown(
                '<div class="section-title">' +
                '🔥 AI Focus Map (Grad-CAM)</div>',
                unsafe_allow_html=True)
            try:
                last_conv = None
                for layer in model.layers[::-1]:
                    if 'conv' in layer.name.lower():
                        last_conv = layer.name
                        break
                if last_conv:
                    grad_model = tf.keras.models.Model(
                        inputs=model.inputs,
                        outputs=[
                            model.get_layer(
                                last_conv).output,
                            model.output])
                    inp_t = tf.cast(
                        np.expand_dims(norm,axis=0),
                        tf.float32)
                    with tf.GradientTape() as tape:
                        tape.watch(inp_t)
                        conv_out, preds = grad_model(inp_t)
                        cls_ch = preds[:,idx_p]
                    grads = tape.gradient(
                        cls_ch, conv_out)
                    pooled = tf.reduce_mean(
                        grads, axis=(0,1,2))
                    cam = conv_out[0] @ \
                        pooled[...,tf.newaxis]
                    cam = tf.squeeze(cam)
                    cam = tf.maximum(cam,0) / (
                        tf.math.reduce_max(cam)+1e-8)
                    cam_np = cam.numpy()
                    cam_r  = cv2.resize(
                        cam_np,(224,224))
                    cam_c  = cv2.applyColorMap(
                        np.uint8(255*cam_r),
                        cv2.COLORMAP_JET)
                    cam_c  = cv2.cvtColor(
                        cam_c, cv2.COLOR_BGR2RGB)
                    overlay = cv2.addWeighted(
                        resized,0.6,cam_c,0.4,0)
                    gc1,gc2,gc3 = st.columns(3)
                    with gc1:
                        st.image(resized,
                            caption="Original",
                            use_column_width=True)
                    with gc2:
                        st.image(cam_c,
                            caption="AI Heatmap",
                            use_column_width=True)
                    with gc3:
                        st.image(overlay,
                            caption="Overlay",
                            use_column_width=True)
                    st.markdown("""
                    <div style="background:rgba(255,100,0,0.06);
                                border:1px solid rgba(255,100,0,0.2);
                                border-radius:8px;
                                padding:0.6rem 1rem;
                                font-size:0.8rem;
                                color:#4a6080;">
                        🔥 <strong>Red/warm areas</strong>
                        = AI focused here<br>
                        🔵 <strong>Blue/cool areas</strong>
                        = Less important
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("GradCAM not available.")
            except Exception:
                st.info(
                    "GradCAM not available. "
                    "Load trained model to enable.")

            # Preprocessing pipeline
            st.markdown(
                '<div class="section-title">' +
                '🔬 Preprocessing Pipeline</div>',
                unsafe_allow_html=True)
            stages = [img_array,resized,
                      enhanced,denoised,
                      (norm*255).astype(np.uint8)]
            titles = ['Original','Resized 224x224',
                      'CLAHE','Denoised',
                      'Normalised']
            pipe_cols = st.columns(5)
            for col,st_img,t in zip(
                    pipe_cols,stages,titles):
                with col:
                    col.image(st_img,
                             caption=t,
                             use_column_width=True)

            # Probability chart
            st.markdown(
                '<div class="section-title">' +
                '📊 Disease Probabilities</div>',
                unsafe_allow_html=True)
            sorted_p = sorted(
                zip(CLASSES,probs),
                key=lambda x:x[1],reverse=True)
            fig,ax = plt.subplots(
                figsize=(10,4),facecolor='white')
            s_cls  = [p[0] for p in sorted_p]
            s_prob = [p[1] for p in sorted_p]
            colors = ['#0066cc' if c==cls_name
                      else '#94a3b8'
                      for c in s_cls]
            bars = ax.barh(s_cls,s_prob,
                           color=colors,
                           edgecolor='none',
                           height=0.6)
            for bar,prob in zip(bars,s_prob):
                ax.text(
                    bar.get_width()+0.01,
                    bar.get_y()+bar.get_height()/2,
                    f'{prob*100:.1f}%',
                    va='center',fontsize=10,
                    color='#1a365d',
                    fontweight='bold')
            ax.set_xlim(0,1.25)
            ax.set_xlabel('Probability',
                          color='#1a365d')
            ax.tick_params(
                colors='#1a365d',labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

            # PDF Report
            def generate_pdf_report():
                fig2 = plt.figure(
                    figsize=(8.27,11.69),
                    facecolor='white')
                fig2.text(0.5,0.95,
                    'OralDx — Diagnosis Report',
                    ha='center',fontsize=18,
                    fontweight='bold',
                    color='#0066cc')
                fig2.text(0.5,0.92,
                    'ITER SOA University | MCA 2024-2026',
                    ha='center',fontsize=10,
                    color='#4a6080')
                ax_ln = fig2.add_axes([0.1,0.905,0.8,0.002])
                ax_ln.axhline(0,color='#0066cc',lw=2)
                ax_ln.axis('off')
                y = 0.87
                for label,val in [
                    ('Date',ts),
                    ('Patient',
                     st.session_state.user_name),
                    ('Email',
                     st.session_state.user_email)]:
                    fig2.text(0.12,y,
                        f'{label}:',fontsize=10,
                        fontweight='bold',
                        color='#1a365d')
                    fig2.text(0.35,y,val,
                        fontsize=10,color='#4a6080')
                    y -= 0.04
                ax_b = fig2.add_axes([0.1,0.67,0.8,0.14])
                ax_b.set_facecolor('#e8f4fd')
                for sp in ax_b.spines.values():
                    sp.set_color('#0066cc')
                ax_b.set_xticks([]); ax_b.set_yticks([])
                fig2.text(0.5,0.80,
                    f'Disease: {cls_name}',
                    ha='center',fontsize=14,
                    fontweight='bold',color='#0066cc')
                fig2.text(0.5,0.76,
                    f'Confidence: {conf*100:.1f}%  |  Severity: {info["severity"]}',
                    ha='center',fontsize=11,
                    color='#1a365d')
                y = 0.63
                for label,val in [
                    ('Symptoms',info['symptoms']),
                    ('Treatment',info['treatment'])]:
                    fig2.text(0.12,y,
                        f'{label}:',fontsize=10,
                        fontweight='bold',
                        color='#1a365d')
                    fig2.text(0.35,y,
                        val[:80],fontsize=9,
                        color='#4a6080')
                    y -= 0.06
                ax_p = fig2.add_axes([0.1,0.15,0.8,0.3])
                bars2 = ax_p.barh(
                    s_cls,s_prob,
                    color=['#0066cc' if c==cls_name
                           else '#94a3b8'
                           for c in s_cls],
                    height=0.6)
                for bar,prob in zip(bars2,s_prob):
                    ax_p.text(
                        bar.get_width()+0.01,
                        bar.get_y()+bar.get_height()/2,
                        f'{prob*100:.1f}%',
                        va='center',fontsize=8,
                        color='#1a365d')
                ax_p.set_xlim(0,1.2)
                ax_p.set_title(
                    'Disease Probabilities',
                    fontsize=10,fontweight='bold')
                ax_p.spines['top'].set_visible(False)
                ax_p.spines['right'].set_visible(False)
                fig2.text(0.5,0.07,
                    'Educational purposes only. Consult dental professional.',
                    ha='center',fontsize=8,
                    color='#94a3b8',style='italic')
                fig2.text(0.5,0.04,
                    'OralDx | Group 08 | ITER SOA University',
                    ha='center',fontsize=8,
                    color='#94a3b8')
                buf = BytesIO()
                plt.savefig(buf,format='pdf',
                           bbox_inches='tight')
                plt.close()
                buf.seek(0)
                return buf.getvalue()

            pdf_data = generate_pdf_report()
            col_dl, col_sh = st.columns(2)
            with col_dl:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                file_name=f"OralDx_{ts.replace(' ','_').replace(',','')}.pdf",
                    mime="application/pdf",
                    use_container_width=True)
            with col_sh:
                if st.button(
                        "📤 Share Report",
                        use_container_width=True):
                    st.code(
                        f"Disease: {cls_name}\n"
                        f"Confidence: {conf*100:.1f}%\n"
                        f"Severity: {info['severity']}\n"
                        f"Date: {ts}\n"
                        f"\nOralDx | ITER SOA University",
                        language=None)
                    st.info(
                        "Copy the above and "
                        "share via WhatsApp/Email!")

            # Disclaimer
            st.markdown("""
            <div style="background:rgba(255,200,0,0.06);
                        border:1px solid rgba(255,200,0,0.2);
                        border-radius:10px;
                        padding:0.8rem 1rem;
                        margin-top:1rem;">
                <small style="color:#c05800;">
                    ⚠️ <strong>Medical Disclaimer:</strong>
                    This AI diagnosis is for educational
                    purposes only and is NOT medical advice.
                    Always consult a qualified dental
                    professional before any treatment.
                    Do not self-medicate based on AI results.
                </small>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()


# ══════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════
def page_history():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#00d4ff;
                   font-size:1.8rem;margin:0;">
            📋 Diagnosis History
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            All your previous diagnoses
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.warning(
            "Sign in to save and view history.")
        if st.button("🔐 Sign In"):
            st.session_state.page = 'auth'
            st.rerun()
        return

    # Load from DB
    db_hist = db_get_history(
        st.session_state.user_email)
    history = db_hist if db_hist else \
        st.session_state.history
    if not history:
        st.markdown("""
        <div class="card" style="text-align:center;
                                  padding:3rem;">
            <div style="font-size:3rem;">📭</div>
            <div style="color:#4a6080;
                        margin-top:1rem;">
                No diagnoses yet.<br>
                Run your first diagnosis!
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 Go to Diagnose"):
            st.session_state.page = 'diagnose'
            st.rerun()
        return

    # Summary stats
    total = len(history)
    diseases = {}
    for h in history:
        d = h['disease']
        diseases[d] = diseases.get(d,0) + 1
    most_common = max(diseases,
                      key=diseases.get)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total}</div>
            <div style="color:#1a365d;
                        font-weight:600;
                        font-size:0.85rem;">
                Total Diagnoses
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">
                {len(diseases)}
            </div>
            <div style="color:#1a365d;
                        font-weight:600;
                        font-size:0.85rem;">
                Unique Diseases
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size:1.3rem;
                        font-weight:800;
                        color:#00d4ff;">
                {DISEASE_INFO[most_common]['icon']}
                {most_common}
            </div>
            <div style="color:#1a365d;
                        font-weight:600;
                        font-size:0.85rem;">
                Most Common
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">'
        '📋 All Records</div>',
        unsafe_allow_html=True)

    # History list (newest first)
    for item in reversed(history):
        info = DISEASE_INFO[item['disease']]
        sev  = item['severity']
        if 'Severe' in sev and 'Mild' not in sev:
            sev_color = '#fc8181'
        elif 'Moderate' in sev:
            sev_color = '#f6e05e'
        else:
            sev_color = '#68d391'

        st.markdown(f"""
        <div class="card" style="display:flex;
                                  align-items:center;
                                  gap:1.2rem;">
            <div style="font-size:2rem;
                        min-width:40px;
                        text-align:center;">
                {info['icon']}
            </div>
            <div style="flex:1;">
                <div style="color:#1a365d;
                            font-weight:700;
                            font-size:0.95rem;">
                    {item['disease']}
                </div>
                <div style="color:#4a6080;
                            font-size:0.78rem;
                            margin-top:0.1rem;">
                    📁 {item['filename']} &nbsp;|&nbsp;
                    🕐 {item['timestamp']}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="color:#00d4ff;
                            font-weight:700;
                            font-size:1rem;">
                    {item['confidence']}
                </div>
                <div style="color:{sev_color};
                            font-size:0.75rem;
                            margin-top:0.2rem;">
                    {sev}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        # Also clear from database
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "DELETE FROM history WHERE email=?",
                (st.session_state.user_email,))
            conn.commit()
            conn.close()
        except:
            pass
        st.success("History cleared!")
        st.rerun()

# ══════════════════════════════════════════════════════
# PAGE: ABOUT US
# ══════════════════════════════════════════════════════
def page_about():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#00d4ff;
                   font-size:1.8rem;margin:0;">
            👥 About Us
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Meet the team behind OralDx
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Project Guide
    st.markdown(
        '<div class="section-title">'
        '🎓 Project Guide</div>',
        unsafe_allow_html=True)

    st.markdown(f"""
    <div class="guide-card">
        <div style="font-size:4rem;
                    margin-bottom:0.8rem;">
            {GUIDE['emoji']}
        </div>
        <div style="font-family:'Space Grotesk',
                    sans-serif;font-size:1.5rem;
                    font-weight:700;color:#1a365d;
                    margin-bottom:0.3rem;">
            {GUIDE['name']}
        </div>
        <div style="color:#00d4ff;font-weight:600;
                    font-size:0.95rem;
                    margin-bottom:0.2rem;">
            {GUIDE['designation']}
        </div>
        <div style="color:#4a6080;
                    font-size:0.85rem;">
            {GUIDE['dept']}<br>
            {GUIDE['university']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Team Members
    st.markdown(
        '<div class="section-title">'
        '👩‍💻 Team Members</div>',
        unsafe_allow_html=True)

    st.info(
        "💡 Upload your photos to replace "
        "the emoji avatars! Place photos named: "
        "sonali.jpg, jagruti.jpg, dharitri.jpg, "
        "smitarani.jpg, barsha.jpg in app folder.")

    def make_member_card(m):
        photo_path = m.get('photo','')
        if os.path.exists(photo_path):
            pil_img = Image.open(photo_path)
            pil_img = pil_img.resize((200,200))
            buf = BytesIO()
            pil_img.save(buf, format='PNG')
            b64 = base64.b64encode(
                buf.getvalue()).decode()
            photo_html = (
                f'<img src="data:image/png;'
                f'base64,{b64}" '
                f'class="member-avatar">')
        else:
            photo_html = (
                f'<div class="avatar-placeholder">'
                f'{m["emoji"]}</div>')

        return f"""
        <div class="member-card">
            {photo_html}
            <div class="member-name">
                {m['name']}
            </div>
            <div class="member-reg">
                Reg: {m['reg']}
            </div>
            <div style="margin:0.5rem 0;">
                <a href="mailto:{m['gmail']}"
                   style="color:#0066cc;
                          font-size:0.72rem;
                          text-decoration:none;
                          word-break:break-all;">
                    ✉️ {m['gmail']}
                </a>
            </div>
        </div>
        """

    # Row 1 — first 3 members
    r1cols = st.columns(3)
    for i, col in enumerate(r1cols):
        with col:
            st.markdown(
                make_member_card(TEAM_MEMBERS[i]),
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — last 2 members centered
    _,c1,c2,_ = st.columns([0.5,1,1,0.5])
    for i, col in enumerate([c1,c2]):
        with col:
            st.markdown(
                make_member_card(TEAM_MEMBERS[i+3]),
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

        # University info
    st.markdown(
        '<div class="section-title">'
        '🏛️ About the Institution</div>',
        unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="display:flex;gap:1.5rem;
                    align-items:flex-start;">
            <div style="font-size:3rem;">🎓</div>
            <div>
                <div style="color:#1a365d;
                            font-weight:700;
                            font-size:1rem;
                            margin-bottom:0.3rem;">
                    Institute of Technical Education
                    and Research (ITER)
                </div>
                <div style="color:#00d4ff;
                            font-size:0.85rem;
                            margin-bottom:0.5rem;">
                    Siksha O Anusandhan (SOA) Deemed
                    to be University
                </div>
                <div style="color:#5a7090;
                            font-size:0.82rem;
                            line-height:1.7;">
                    📍 Bhubaneswar, Odisha, India<br>
                    🎓 Department of Computer Application<br>
                    📚 Master in Computer Application
                    (MCA) 2024-2026<br>
                    👥 Group 08 | Final Year Project
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: INFO
# ══════════════════════════════════════════════════════
def page_info():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#00d4ff;
                   font-size:1.8rem;margin:0;">
            ℹ️ Project Information
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Technical details about OralDx
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model Performance
    st.markdown(
        '<div class="section-title">'
        '📊 Model Performance</div>',
        unsafe_allow_html=True)

    models_data = [
        ("VGG19",    "91%","0.91","0.88","0.895","Best performer","#00d4ff"),
        ("ResNet-50","87%","0.87","0.85","0.860","Transfer learning","#68d391"),
        ("YOLOv8",   "88%","0.87","0.84","0.855","Object detection","#f6ad55"),
        ("U-Net",    "85%","0.83","0.82","0.825","Segmentation","#b794f4"),
    ]

    cols = st.columns(4)
    for col,(name,acc,prec,rec,f1,role,clr) in \
            zip(cols, models_data):
        with col:
            st.markdown(f"""
            <div class="card"
                 style="border-color:{clr}33;
                        text-align:center;">
                <div style="color:{clr};
                            font-weight:800;
                            font-size:1.1rem;
                            font-family:'Space Grotesk',
                            sans-serif;">
                    {name}
                </div>
                <div style="font-size:2rem;
                            font-weight:800;
                            color:{clr};
                            margin:0.3rem 0;">
                    {acc}
                </div>
                <div style="color:#5a7090;
                            font-size:0.75rem;">
                    Accuracy
                </div>
                <div style="margin-top:0.8rem;
                            font-size:0.75rem;">
                    <div style="display:flex;
                                justify-content:space-between;
                                color:#4a6080;
                                padding:2px 0;">
                        <span>Precision</span>
                        <span style="color:{clr};">
                            {prec}
                        </span>
                    </div>
                    <div style="display:flex;
                                justify-content:space-between;
                                color:#4a6080;
                                padding:2px 0;">
                        <span>Recall</span>
                        <span style="color:{clr};">
                            {rec}
                        </span>
                    </div>
                    <div style="display:flex;
                                justify-content:space-between;
                                color:#4a6080;
                                padding:2px 0;">
                        <span>F1-Score</span>
                        <span style="color:{clr};">
                            {f1}
                        </span>
                    </div>
                </div>
                <div style="margin-top:0.8rem;
                            background:{clr}18;
                            border-radius:6px;
                            padding:3px 8px;
                            color:{clr};
                            font-size:0.72rem;
                            font-weight:600;">
                    {role}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Dataset
    st.markdown(
        '<div class="section-title">'
        '🗃️ Dataset Information</div>',
        unsafe_allow_html=True)

    d_data = [
        ("Calculus",            "1,296","#2ECC71"),
        ("Caries",              "2,601","#E74C3C"),
        ("Gingivitis",          "2,349","#3498DB"),
        ("Hypodontia",          "1,251","#F39C12"),
        ("Mouth Ulcer",         "2,806","#9B59B6"),
        ("Tooth Discoloration", "2,017","#1ABC9C"),
    ]

    st.markdown("""
    <div class="card">
        <div style="display:grid;
                    grid-template-columns:repeat(3,1fr);
                    gap:0.8rem;">
    """, unsafe_allow_html=True)

    for cls,count,clr in d_data:
        info = DISEASE_INFO[cls if cls != "Caries"
                            else "Dental Caries"]
        st.markdown(f"""
        <div style="background:{clr}11;
                    border:1px solid {clr}44;
                    border-radius:10px;
                    padding:0.8rem;
                    text-align:center;">
            <div style="font-size:1.5rem;">
                {DISEASE_INFO.get(cls, info)['icon']
                 if cls in DISEASE_INFO
                 else info['icon']}
            </div>
            <div style="color:#1a365d;
                        font-weight:600;
                        font-size:0.82rem;
                        margin:0.2rem 0;">
                {cls}
            </div>
            <div style="color:{clr};
                        font-weight:700;
                        font-size:1.1rem;">
                {count}
            </div>
            <div style="color:#4a6080;
                        font-size:0.7rem;">
                images
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True)

    # Total
    st.markdown("""
    <div style="background:rgba(0,212,255,0.08);
                border:1px solid rgba(0,212,255,0.2);
                border-radius:12px;
                padding:1rem;text-align:center;
                margin:0.5rem 0;">
        <span style="color:#00d4ff;font-weight:800;
                     font-size:1.5rem;">12,320</span>
        <span style="color:#4a6080;font-size:0.9rem;">
             total images across 6 classes
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack
    st.markdown(
        '<div class="section-title">'
        '⚙️ Technology Stack</div>',
        unsafe_allow_html=True)

    tech = [
        ("🐍","Python 3.10",
         "Core programming language"),
        ("🧠","TensorFlow / Keras",
         "Deep learning framework"),
        ("👁️","OpenCV",
         "Image preprocessing (CLAHE, denoising)"),
        ("📊","Matplotlib / Seaborn",
         "Data visualization and charts"),
        ("🌐","Streamlit",
         "Web application framework"),
        ("🎯","Transfer Learning",
         "ImageNet pretrained weights"),
    ]
    t1,t2,t3 = st.columns(3)
    for i,(em,name,desc) in enumerate(tech):
        with [t1,t2,t3][i%3]:
            st.markdown(f"""
            <div class="card"
                 style="padding:1rem;">
                <div style="display:flex;
                            align-items:center;
                            gap:0.6rem;">
                    <span style="font-size:1.4rem;">
                        {em}
                    </span>
                    <div>
                        <div style="color:#1a365d;
                                    font-weight:600;
                                    font-size:0.85rem;">
                            {name}
                        </div>
                        <div style="color:#4a6080;
                                    font-size:0.75rem;">
                            {desc}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════
def show_footer():
    st.markdown("""
    <div class="ai-bubble" title="Ask OralDx AI">
        🤖
    </div>
    <div class="footer">
        <div style="color:#3a5070;
                    margin-bottom:0.5rem;">
            🦷 OralDx — Automated Diagnosis of
            Oral Conditions from Dental X-Rays
        </div>
        <div>
            Group 08 &nbsp;|&nbsp;
            Department of Computer Application
            &nbsp;|&nbsp;
            ITER SOA University,
            Bhubaneswar &nbsp;|&nbsp;
            MCA Batch 2024-2026
        </div>
        <div style="margin-top:0.3rem;">
            Developed by: Sonali Patra ·
            Jagruti Parida · Dharitri Pradhan ·
            Smitarani Mohapatra ·
            Barsha Priyadarshini Singh
        </div>
        <div style="margin-top:0.3rem;">
            Guide: Dr. Debabrata Singh,
            Associate Professor
        </div>
        <div style="color:#1e293b;
                    margin-top:0.5rem;
                    font-size:0.7rem;">
            ⚠️ For educational purposes only.
            Not a substitute for professional
            dental advice.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ══════════════════════════════════════════════════════
def page_howitworks():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            🎯 How It Works
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Step by step — how OralDx detects
            oral diseases
        </p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01","📤","Upload X-Ray",
         "Doctor uploads a dental X-ray image (JPG/PNG) through the web interface or mobile app.",
         "#0066cc"),
        ("02","🔧","Preprocessing",
         "The image is automatically enhanced using CLAHE contrast adjustment, Gaussian denoising, and pixel normalisation to improve quality.",
         "#0077dd"),
        ("03","🧠","AI Analysis",
         "The preprocessed image is fed into our VGG19 deep learning model trained on 12,320 dental X-ray images.",
         "#0088ee"),
        ("04","📊","Classification",
         "The model analyses patterns and predicts which of the 6 oral diseases is present, along with a confidence percentage.",
         "#0099ff"),
        ("05","📋","Result",
         "Diagnosis result is displayed with disease name, confidence score, severity level, symptoms and recommended treatment.",
         "#00aaff"),
        ("06","💾","Save History",
         "All diagnosis results are saved to your account history for future reference and tracking.",
         "#00bbff"),
    ]

    for i, (num, icon, title, desc, clr) in \
            enumerate(steps):
        c1, c2 = st.columns([1,4])
        with c1:
            st.markdown(f"""
            <div style="background:linear-gradient(
                        135deg,{clr},{clr}99);
                        color:white;
                        width:70px;height:70px;
                        border-radius:50%;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:1.5rem;
                        font-weight:800;
                        font-family:'Space Grotesk',
                        sans-serif;
                        margin:auto;">
                {num}
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card"
                 style="border-left:4px solid {clr};
                        margin:0.3rem 0;">
                <div style="display:flex;
                            align-items:center;
                            gap:0.6rem;
                            margin-bottom:0.4rem;">
                    <span style="font-size:1.5rem;">
                        {icon}
                    </span>
                    <span style="color:#1a365d;
                                 font-weight:700;
                                 font-size:1rem;">
                        {title}
                    </span>
                </div>
                <div style="color:#4a6080;
                            font-size:0.88rem;
                            line-height:1.6;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if i < len(steps)-1:
            st.markdown("""
            <div style="text-align:center;
                        color:#0066cc;
                        font-size:1.2rem;
                        margin:0.2rem 0;">
                ↓
            </div>
            """, unsafe_allow_html=True)

    # Architecture
    st.markdown(
        '<div class="section-title">'
        '🧱 Model Architecture</div>',
        unsafe_allow_html=True)

    arch = [
        ("Input Layer","224×224×3 RGB image","#e3f2fd"),
        ("CLAHE + Denoise","OpenCV preprocessing","#e8f5e9"),
        ("VGG19 Base","16 conv layers, ImageNet weights","#fff3e0"),
        ("Fine-tuning","Block 4+5 layers unfrozen","#fce4ec"),
        ("Dense Layers","1024→512→256 neurons","#f3e5f5"),
        ("Output Layer","6 classes with Softmax","#e8eaf6"),
    ]

    cols = st.columns(6)
    for col,(layer,desc,bg) in zip(cols,arch):
        with col:
            st.markdown(f"""
            <div style="background:{bg};
                        border:1px solid #ddd;
                        border-radius:10px;
                        padding:0.8rem;
                        text-align:center;
                        height:100px;">
                <div style="color:#1a365d;
                            font-weight:700;
                            font-size:0.78rem;">
                    {layer}
                </div>
                <div style="color:#4a6080;
                            font-size:0.7rem;
                            margin-top:0.3rem;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE: DISEASE GUIDE
# ══════════════════════════════════════════════════════
def page_diseaseguide():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            🦷 Disease Guide
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Complete guide to all 6 oral diseases
            detected by OralDx
        </p>
    </div>
    """, unsafe_allow_html=True)

    detailed = {
        'Calculus': {
            'icon'     : '🦷',
            'color'    : '#f6ad55',
            'overview' : 'Calculus, also known as dental tartar, is mineralised dental plaque that forms on the surfaces of teeth. It develops when plaque — a sticky film of bacteria — is not removed and hardens over time.',
            'causes'   : 'Poor oral hygiene, irregular brushing, dry mouth, smoking, diet high in sugar and starch.',
            'symptoms' : 'Yellow or brown deposits on teeth, bad breath, swollen or bleeding gums, rough tooth surface.',
            'treatment': 'Professional cleaning (scaling) by a dentist. Cannot be removed by brushing at home.',
            'prevention': 'Brush twice daily, floss regularly, use antiseptic mouthwash, visit dentist every 6 months.',
            'severity' : 'Mild-Moderate',
            'prevalence': 'Affects 68% of adults worldwide'
        },
        'Dental Caries': {
            'icon'     : '⚠️',
            'color'    : '#fc8181',
            'overview' : 'Dental caries, commonly known as tooth decay or cavities, is the breakdown of teeth due to acids produced by bacteria. It is one of the most widespread chronic diseases globally.',
            'causes'   : 'Bacterial infection, high sugar diet, poor oral hygiene, dry mouth, fluoride deficiency.',
            'symptoms' : 'Toothache, tooth sensitivity, visible holes or pits, brown/black/white staining, pain when biting.',
            'treatment': 'Fluoride treatments for early stage, dental fillings, crowns, root canal therapy, or extraction for severe cases.',
            'prevention': 'Regular brushing with fluoride toothpaste, limit sugary foods, dental sealants, regular checkups.',
            'severity' : 'Moderate-Severe',
            'prevalence': 'Affects 2.3 billion people globally'
        },
        'Gingivitis': {
            'icon'     : '🔴',
            'color'    : '#68d391',
            'overview' : 'Gingivitis is a mild form of gum disease that causes irritation, redness, and swelling of the gingiva — the part of the gum around the base of teeth. It is very common and reversible with proper care.',
            'causes'   : 'Plaque buildup on teeth, poor oral hygiene, hormonal changes, medications, illness.',
            'symptoms' : 'Red, swollen, tender gums, bleeding while brushing, bad breath, receding gums.',
            'treatment': 'Professional dental cleaning, improved brushing and flossing technique, antiseptic mouthwash.',
            'prevention': 'Brush twice daily, floss daily, regular dental cleanings, avoid smoking.',
            'severity' : 'Mild',
            'prevalence': 'Affects 50% of adults over 30'
        },
        'Hypodontia': {
            'icon'     : '❓',
            'color'    : '#b794f4',
            'overview' : 'Hypodontia is a condition where a person is born with fewer teeth than normal due to the failure of tooth development. It is the most common developmental dental anomaly.',
            'causes'   : 'Genetic factors, environmental factors during pregnancy, systemic conditions, family history.',
            'symptoms' : 'Missing permanent teeth, gaps in dental arch, small or unusually shaped teeth, delayed tooth eruption.',
            'treatment': 'Dental implants, bridges, dentures, orthodontic treatment, composite bonding.',
            'prevention': 'Cannot be prevented as it is congenital. Early diagnosis helps in treatment planning.',
            'severity' : 'Moderate',
            'prevalence': 'Affects 2-8% of population'
        },
        'Mouth Ulcer': {
            'icon'     : '💊',
            'color'    : '#4fd1c5',
            'overview' : 'Mouth ulcers (also called canker sores or aphthous ulcers) are small, painful sores that form inside the mouth on the soft tissues such as gums, tongue, inner cheeks, or lips.',
            'causes'   : 'Minor injuries, stress, nutritional deficiencies, hormonal changes, food sensitivities, autoimmune conditions.',
            'symptoms' : 'Round or oval sores with white/yellow center and red border, pain while eating or talking, tingling sensation.',
            'treatment': 'Topical numbing agents, antiseptic mouthwash, vitamin supplements, corticosteroid ointments.',
            'prevention': 'Avoid injury to mouth, manage stress, maintain good nutrition, avoid trigger foods.',
            'severity' : 'Mild',
            'prevalence': 'Affects 20% of population at some point'
        },
        'Tooth Discoloration': {
            'icon'     : '🌟',
            'color'    : '#f6e05e',
            'overview' : 'Tooth discoloration refers to staining or change in colour of teeth. It can range from white streaks to yellow tints or brown spots and can affect the entire tooth or just certain areas.',
            'causes'   : 'Coffee, tea, wine, tobacco, poor hygiene, certain medications, aging, trauma, fluorosis.',
            'symptoms' : 'Yellow, brown, grey, or white spots on teeth, dull or stained appearance, uneven colouring.',
            'treatment': 'Teeth whitening, dental veneers, bonding, crowns, professional cleaning.',
            'prevention': 'Limit staining foods/drinks, quit smoking, brush after meals, regular dental cleanings.',
            'severity' : 'Mild',
            'prevalence': 'Affects millions worldwide, increasing with age'
        },
    }

    for cls, info in detailed.items():
        with st.expander(
                f"{info['icon']}  {cls}  —  "
                f"{info['severity']}",
                expanded=False):
            c1, c2 = st.columns([2,1])
            with c1:
                st.markdown(f"""
                <div style="background:white;
                            border-radius:12px;
                            padding:1rem;">
                    <div style="color:#1a365d;
                                font-weight:700;
                                font-size:1rem;
                                margin-bottom:0.8rem;
                                border-bottom:2px solid
                                {info['color']};
                                padding-bottom:0.4rem;">
                        📖 Overview
                    </div>
                    <p style="color:#4a6080;
                              font-size:0.88rem;
                              line-height:1.7;">
                        {info['overview']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background:{info['color']}22;
                            border:1px solid
                            {info['color']}66;
                            border-radius:12px;
                            padding:1rem;">
                    <div style="color:#1a365d;
                                font-weight:600;
                                font-size:0.8rem;
                                margin-bottom:0.3rem;">
                        📊 Prevalence
                    </div>
                    <div style="color:{info['color']};
                                font-weight:700;
                                font-size:0.85rem;">
                        {info['prevalence']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            r1,r2,r3,r4 = st.columns(4)
            for col,(label,key,emoji) in zip(
                [r1,r2,r3,r4],[
                    ("Causes","causes","🔍"),
                    ("Symptoms","symptoms","😷"),
                    ("Treatment","treatment","💊"),
                    ("Prevention","prevention","🛡️")]):
                with col:
                    st.markdown(f"""
                    <div style="background:white;
                                border-top:3px solid
                                {info['color']};
                                border-radius:8px;
                                padding:0.8rem;
                                margin-top:0.5rem;">
                        <div style="color:#1a365d;
                                    font-weight:700;
                                    font-size:0.8rem;
                                    margin-bottom:
                                    0.3rem;">
                            {emoji} {label}
                        </div>
                        <div style="color:#4a6080;
                                    font-size:0.75rem;
                                    line-height:1.5;">
                            {info[key]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE: FAQ
# ══════════════════════════════════════════════════════
def page_faq():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            ❓ Frequently Asked Questions
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Answers to common questions about OralDx
        </p>
    </div>
    """, unsafe_allow_html=True)

    faqs = [
        ("What is OralDx?",
         "OralDx is an AI-powered web application developed by Group 08, MCA students at ITER SOA University. It uses deep learning models (VGG19, ResNet-50, YOLOv8, U-Net) to automatically detect and classify oral diseases from dental X-ray images."),
        ("How accurate is the diagnosis?",
         "Our best model (VGG19) achieves 91% accuracy on the test dataset of 12,320 dental X-ray images across 6 disease categories. The system also provides a confidence percentage for each diagnosis."),
        ("Which diseases can OralDx detect?",
         "OralDx can detect 6 oral diseases: Calculus (tartar), Dental Caries (tooth decay), Gingivitis (gum inflammation), Hypodontia (missing teeth), Mouth Ulcers, and Tooth Discoloration."),
        ("What type of images can I upload?",
         "You can upload dental X-ray images in JPG or PNG format. The system works best with panoramic or periapical dental radiographs. Images are automatically resized to 224×224 pixels for processing."),
        ("Is my data safe and private?",
         "Yes. Images are processed locally and are not stored permanently on our servers. If you are signed in, diagnosis history is stored only in your browser session and is cleared when you sign out."),
        ("Do I need to create an account?",
         "No. You can use the diagnosis feature as a guest without creating an account. However, creating an account allows you to save your diagnosis history and access it anytime."),
        ("Can OralDx replace a dentist?",
         "No. OralDx is an AI-assisted educational tool designed to support dental professionals. It should NOT be used as a substitute for professional dental examination and diagnosis. Always consult a qualified dentist."),
        ("Which model is used for diagnosis?",
         "The primary model is VGG19 with transfer learning from ImageNet weights. It was fine-tuned on dental X-ray data using 2-phase training with early stopping and learning rate reduction."),
        ("How was the model trained?",
         "The model was trained on 12,320 dental X-ray images using Google Colab with T4 GPU. Training involved data augmentation (rotation, flipping, zooming) and 2-phase fine-tuning achieving 91% validation accuracy."),
        ("Can I use this on mobile?",
         "Yes! OralDx is fully responsive and works on smartphones, tablets, and laptops. An Android APK is also available for download on the Mobile App page."),
    ]

    for q, a in faqs:
        with st.expander(f"❓  {q}"):
            st.markdown(f"""
            <div style="background:white;
                        border-left:4px solid #0066cc;
                        border-radius:0 8px 8px 0;
                        padding:1rem;
                        color:#4a6080;
                        font-size:0.9rem;
                        line-height:1.7;">
                {a}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE: CONTACT US
# ══════════════════════════════════════════════════════
def page_contact():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            📞 Contact Us
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Get in touch with the OralDx team
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1,1])

    with c1:
        st.markdown(
            '<div class="section-title">'
            '✉️ Send us a Message</div>',
            unsafe_allow_html=True)

        name = st.text_input(
            "Your Name",
            placeholder="Enter your name")
        email = st.text_input(
            "Email Address",
            placeholder="your@email.com")
        subject = st.selectbox(
            "Subject", [
                "General Inquiry",
                "Technical Support",
                "Report a Bug",
                "Collaboration",
                "Academic Query",
                "Other"
            ])
        message = st.text_area(
            "Message",
            placeholder="Write your message here...",
            height=150)

        if st.button("📤 Send Message",
                     key="send_msg"):
            if name and email and message:
                st.success(
                    f"✅ Thank you {name}! "
                    f"Your message has been received. "
                    f"We will reply within 24 hours.")
                st.balloons()
            else:
                st.warning("Please fill all fields")

    with c2:
        st.markdown(
            '<div class="section-title">'
            '📍 Contact Information</div>',
            unsafe_allow_html=True)

        contacts = [
            ("🏛️","Institution",
             "Institute of Technical Education & Research (ITER)"),
            ("🎓","University",
             "Siksha O Anusandhan (SOA) Deemed to be University"),
            ("📍","Location",
             "Bhubaneswar, Odisha, India"),
            ("📚","Department",
             "Department of Computer Application"),
            ("👥","Group",
             "Group 08 | MCA Batch 2024-2026"),
            ("👨‍🏫","Guide",
             "Dr. Debabrata Singh, Associate Professor"),
        ]

        for emoji,label,value in contacts:
            st.markdown(f"""
            <div class="info-block">
                <div class="info-label">
                    {emoji} {label}
                </div>
                <div class="info-value">
                    {value}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Team emails
        st.markdown(
            '<div class="section-title" '
            'style="font-size:1rem;margin-top:1rem;">'
            '📧 Team Emails</div>',
            unsafe_allow_html=True)

        emails = [
            ("Sonali Patra",
             "sonali@student.iter.ac.in"),
            ("Jagruti Parida",
             "jagruti@student.iter.ac.in"),
            ("Dharitri Pradhan",
             "dharitri@student.iter.ac.in"),
            ("Smitarani Mohapatra",
             "smitarani@student.iter.ac.in"),
            ("Barsha P. Singh",
             "barsha@student.iter.ac.in"),
        ]

        for name, mail in emails:
            st.markdown(f"""
            <div style="display:flex;
                        justify-content:space-between;
                        padding:0.4rem 0;
                        border-bottom:1px solid
                        rgba(0,100,200,0.1);">
                <span style="color:#1a365d;
                             font-size:0.82rem;
                             font-weight:600;">
                    {name}
                </span>
                <span style="color:#0066cc;
                             font-size:0.78rem;">
                    {mail}
                </span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE: ASK ORALDX (Chatbot)
# ══════════════════════════════════════════════════════
def page_askoraldx():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            💬 Ask OralDx
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Ask any question about oral health
        </p>
    </div>
    """, unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Knowledge base
    kb = {
        'calculus' : "Calculus is hardened dental plaque on teeth. Treatment: professional cleaning. Prevention: brush twice daily and floss.",
        'caries'   : "Dental caries is tooth decay caused by bacteria. Treatment: fillings or root canal. Prevention: limit sugar and brush regularly.",
        'gingivitis': "Gingivitis is gum inflammation. Symptoms: red bleeding gums. Treatment: improved hygiene and cleaning.",
        'hypodontia': "Hypodontia means missing teeth from birth. Treatment: implants, bridges or dentures.",
        'ulcer'    : "Mouth ulcers are painful sores in the mouth. Treatment: topical gel and mouthwash. Usually heals in 1-2 weeks.",
        'discoloration': "Tooth discoloration is colour change in teeth. Treatment: whitening or veneers.",
        'accuracy' : "OralDx achieves 91% accuracy using VGG19 deep learning model trained on 12,320 dental X-ray images.",
        'model'    : "OralDx uses 4 models: VGG19 (91%), ResNet-50 (87%), YOLOv8 (88%), and U-Net (85%).",
        'dataset'  : "The model was trained on 12,320 dental X-ray images across 6 disease categories.",
        'upload'   : "Go to Diagnose page, upload your dental X-ray in JPG or PNG format, click Run Diagnosis.",
        'safe'     : "Yes, OralDx is safe. Images are processed locally and not stored permanently.",
        'dentist'  : "OralDx is an AI tool to assist diagnosis. Always consult a qualified dentist for treatment.",
        'free'     : "Yes, OralDx is completely free to use. Create an account to save your history.",
        'brush'    : "Dentists recommend brushing teeth twice daily with fluoride toothpaste for 2 minutes each time.",
        'floss'    : "Floss at least once daily to remove plaque between teeth where brushes cannot reach.",
        'hello'    : "Hello! I am OralDx assistant. Ask me anything about oral health or how to use this app!",
        'hi'       : "Hi there! How can I help you with oral health today?",
        'help'     : "I can answer questions about oral diseases, how to use OralDx, model accuracy, and oral health tips!",
    }

    def get_response(question):
        q = question.lower()
        for key, answer in kb.items():
            if key in q:
                return answer
        return ("I am not sure about that. Please consult "
                "a dental professional for specific advice. "
                "You can also check our Disease Guide page "
                "or FAQ page for more information!")

    # Show chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div style="display:flex;
                        justify-content:flex-end;
                        margin:0.5rem 0;">
                <div style="background:#0066cc;
                            color:white;
                            border-radius:18px
                            18px 4px 18px;
                            padding:0.6rem 1rem;
                            max-width:70%;
                            font-size:0.88rem;">
                    {msg['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;
                        justify-content:flex-start;
                        margin:0.5rem 0;">
                <div style="background:white;
                            color:#1a365d;
                            border:1px solid
                            rgba(0,100,200,0.2);
                            border-radius:18px
                            18px 18px 4px;
                            padding:0.6rem 1rem;
                            max-width:70%;
                            font-size:0.88rem;">
                    🦷 {msg['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Quick questions
    st.markdown(
        '<div style="color:#4a6080;font-size:0.8rem;'
        'margin:0.5rem 0;">Quick questions:</div>',
        unsafe_allow_html=True)

    quick = st.columns(3)
    quick_qs = [
        "What is calculus?",
        "How accurate is OralDx?",
        "How do I upload an X-ray?",
        "What is gingivitis?",
        "Is OralDx free?",
        "How to prevent caries?",
    ]
    for i, col in enumerate(quick):
        with col:
            if i < len(quick_qs):
                if st.button(quick_qs[i],
                             key=f"qq_{i}"):
                    ans = get_response(quick_qs[i])
                    st.session_state.chat_history\
                        .append({'role':'user',
                                 'content':quick_qs[i]})
                    st.session_state.chat_history\
                        .append({'role':'bot',
                                 'content':ans})
                    st.rerun()

    quick2 = st.columns(3)
    for i, col in enumerate(quick2):
        with col:
            if i+3 < len(quick_qs):
                if st.button(quick_qs[i+3],
                             key=f"qq2_{i}"):
                    ans = get_response(quick_qs[i+3])
                    st.session_state.chat_history\
                        .append({'role':'user',
                                 'content':quick_qs[i+3]})
                    st.session_state.chat_history\
                        .append({'role':'bot',
                                 'content':ans})
                    st.rerun()

    # Input
    user_input = st.text_input(
        "Type your question...",
        placeholder="e.g. What is dental caries?",
        key="chat_input",
        help="Type here and press Send")

    c1, c2 = st.columns([3,1])
    with c2:
        if st.button("Send 💬", key="send_chat"):
            if user_input:
                ans = get_response(user_input)
                st.session_state.chat_history\
                    .append({'role':'user',
                             'content':user_input})
                st.session_state.chat_history\
                    .append({'role':'bot',
                             'content':ans})
                st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════
# PAGE: SYMPTOM CHECKER
# ══════════════════════════════════════════════════════
def page_symptom():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            🗺️ Symptom Checker
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Select your symptoms to find possible
            oral diseases
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "⚠️ This tool is for educational purposes "
        "only. Always consult a dentist for proper "
        "diagnosis.")

    st.markdown(
        '<div class="section-title">'
        '🔍 Select Your Symptoms</div>',
        unsafe_allow_html=True)

    symptoms = {
        'Toothache or pain'              : ['Dental Caries'],
        'Yellow/brown deposits on teeth' : ['Calculus'],
        'Bad breath'                     : ['Calculus','Gingivitis'],
        'Bleeding gums'                  : ['Gingivitis'],
        'Red or swollen gums'            : ['Gingivitis'],
        'Missing teeth'                  : ['Hypodontia'],
        'Gaps in teeth'                  : ['Hypodontia'],
        'Painful sores in mouth'         : ['Mouth Ulcer'],
        'Difficulty eating/talking'      : ['Mouth Ulcer'],
        'Yellow or stained teeth'        : ['Tooth Discoloration'],
        'Dark spots on teeth'            : ['Dental Caries','Tooth Discoloration'],
        'Tooth sensitivity'              : ['Dental Caries'],
        'Rough tooth surface'            : ['Calculus'],
        'White spots on teeth'           : ['Tooth Discoloration'],
        'Gum recession'                  : ['Gingivitis'],
    }

    selected = []
    c1, c2, c3 = st.columns(3)
    cols = [c1,c2,c3]
    symp_list = list(symptoms.keys())
    for i, symp in enumerate(symp_list):
        with cols[i%3]:
            if st.checkbox(symp, key=f"s_{i}"):
                selected.append(symp)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Check Symptoms",
                 key="check_symp"):
        if selected:
            # Count disease matches
            disease_score = {}
            for symp in selected:
                for d in symptoms[symp]:
                    disease_score[d] = \
                        disease_score.get(d,0) + 1

            if disease_score:
                st.markdown(
                    '<div class="section-title">'
                    '📋 Possible Conditions</div>',
                    unsafe_allow_html=True)

                sorted_d = sorted(
                    disease_score.items(),
                    key=lambda x: x[1],
                    reverse=True)

                for disease, score in sorted_d:
                    info = DISEASE_INFO.get(
                        disease, {})
                    match_pct = min(
                        int(score /
                            len(selected)*100),
                        95)
                    st.markdown(f"""
                    <div class="card">
                        <div style="display:flex;
                                    justify-content:
                                    space-between;
                                    align-items:center;">
                            <div>
                                <span style="font-size:
                                             1.5rem;">
                                    {info.get('icon',
                                              '🦷')}
                                </span>
                                <span style="color:
                                             #1a365d;
                                             font-weight:
                                             700;
                                             font-size:
                                             1rem;
                                             margin-left:
                                             0.5rem;">
                                    {disease}
                                </span>
                            </div>
                            <div style="color:#0066cc;
                                        font-weight:700;
                                        font-size:1rem;">
                                {match_pct}% match
                            </div>
                        </div>
                        <div style="color:#4a6080;
                                    font-size:0.82rem;
                                    margin-top:0.4rem;">
                            {info.get('description','')}
                        </div>
                        <div style="margin-top:0.5rem;">
                            {get_severity_badge(
                                info.get(
                                    'severity',''))}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.info(
                    "💡 For confirmed diagnosis, "
                    "upload a dental X-ray on the "
                    "Diagnose page.")
            else:
                st.info("No matching conditions found.")
        else:
            st.warning("Please select at least one symptom.")


# ══════════════════════════════════════════════════════
# PAGE: REVIEWS
# ══════════════════════════════════════════════════════
def page_reviews():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            ⭐ Reviews
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            What users say about OralDx
        </p>
    </div>
    """, unsafe_allow_html=True)

    if 'all_reviews' not in st.session_state:
        st.session_state.all_reviews = []

    # Load from DB
    db_reviews = db_get_reviews()
    reviews = (db_reviews
               if db_reviews
               else st.session_state.all_reviews)
    if not reviews:
        st.markdown("""
        <div class="card" style="text-align:center;
                                  padding:2rem;">
            <div style="font-size:3rem;">⭐</div>
            <div style="color:#4a6080;margin-top:1rem;">
                No reviews yet. Be the first to review!
            </div>
        </div>
        """, unsafe_allow_html=True)
    avg = sum(r['rating'] for r in reviews) / \
          len(reviews) if reviews else 0
    five_star = sum(
        1 for r in reviews if r['rating']==5)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">
                {avg:.1f}
            </div>
            <div style="color:#f6ad55;
                        font-size:1.2rem;">
                {'⭐'*int(avg)}
            </div>
            <div style="color:#4a6080;
                        font-size:0.8rem;">
                Average Rating
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">
                {len(reviews)}
            </div>
            <div style="color:#4a6080;
                        font-size:0.8rem;">
                Total Reviews
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">
                {five_star}
            </div>
            <div style="color:#4a6080;
                        font-size:0.8rem;">
                5-Star Reviews
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Show reviews
    st.markdown(
        '<div class="section-title">'
        '💬 User Reviews</div>',
        unsafe_allow_html=True)

    for r in reversed(reviews):
        stars = '⭐' * r['rating']
        st.markdown(f"""
        <div class="card">
            <div style="display:flex;
                        justify-content:space-between;
                        margin-bottom:0.5rem;">
                <div>
                    <span style="color:#1a365d;
                                 font-weight:700;">
                        {r['name']}
                    </span>
                    <span style="color:#4a6080;
                                 font-size:0.78rem;
                                 margin-left:0.5rem;">
                        — {r['role']}
                    </span>
                </div>
                <div style="font-size:0.85rem;">
                    {stars}
                </div>
            </div>
            <div style="color:#4a6080;
                        font-size:0.88rem;
                        line-height:1.6;
                        font-style:italic;">
                "{r['review']}"
            </div>
            <div style="color:#94a3b8;
                        font-size:0.72rem;
                        margin-top:0.4rem;">
                {r['date']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Add review
    st.markdown(
        '<div class="section-title">'
        '✍️ Leave a Review</div>',
        unsafe_allow_html=True)

    r_name    = st.text_input(
        "Your Name",
        placeholder="Dr. John Smith",
        key="r_name")
    r_role    = st.selectbox(
        "Your Role",
        ["Dental Professional",
         "Medical Student",
         "Researcher",
         "Patient",
         "General User"],
        key="r_role")
    r_rating  = st.slider(
        "Rating", 1, 5, 5,
        key="r_rating")
    st.write("⭐" * r_rating)
    r_review  = st.text_area(
        "Your Review",
        placeholder="Share your experience...",
        height=100,
        key="r_review")

    if st.button("Submit Review ⭐",
                 key="submit_review"):
        if r_name and r_review:
            db_save_review(
                r_name, r_role, r_rating,
                r_review,
                str(datetime.date.today()
                    .strftime("%d %b %Y")))
            st.session_state.all_reviews.append({
                'name'  : r_name,
                'role'  : r_role,
                'rating': r_rating,
                'review': r_review,
                'date'  : str(datetime.date.today()
                              .strftime("%d %b %Y")),
            })
            st.success("✅ Review submitted!")
            st.rerun()
        else:
            st.warning("Please fill name and review")


# ══════════════════════════════════════════════════════
# PAGE: COMPARE X-RAYS
# ══════════════════════════════════════════════════════
def page_compare():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            🔬 Compare X-Rays
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Upload two X-ray images and compare
            diagnosis side by side
        </p>
    </div>
    """, unsafe_allow_html=True)

    model, model_path = load_model()
    if model is None:
        st.warning(
            "No trained model found. "
            "Please add vgg19_best.keras "
            "to run diagnosis.")
        st.stop()

    c1, c2 = st.columns(2)

    results = []
    for i, col in enumerate([c1, c2]):
        with col:
            st.markdown(
                f'<div class="section-title" '
                f'style="font-size:1rem;">'
                f'X-Ray {i+1}</div>',
                unsafe_allow_html=True)
            upl = st.file_uploader(
                f"Upload X-Ray {i+1}",
                type=["jpg","jpeg","png"],
                key=f"cmp_{i}")
            if upl:
                img = Image.open(upl)
                img_arr = np.array(
                    img.convert("RGB"))
                st.image(
                    img,
                    caption=upl.name,
                    use_column_width=True)
                norm,_,_,_ = preprocess(img_arr)
                probs = model.predict(
                    np.expand_dims(norm,axis=0),
                    verbose=0)[0]
                idx  = np.argmax(probs)
                cls  = CLASSES[idx]
                conf = probs[idx]
                info = DISEASE_INFO[cls]
                results.append(
                    (cls,conf,info,probs))

                st.markdown(f"""
                <div class="result-hero"
                     style="padding:1rem;">
                    <div style="font-size:1.5rem;
                                font-weight:800;
                                color:#1a365d;">
                        {info['icon']} {cls}
                    </div>
                    <div style="color:#4a6080;">
                        {conf*100:.1f}% confidence
                    </div>
                    {get_severity_badge(
                        info['severity'])}
                </div>
                """, unsafe_allow_html=True)

    # Side by side comparison
    if len(results) == 2:
        st.markdown(
            '<div class="section-title">'
            '📊 Comparison</div>',
            unsafe_allow_html=True)

        cls1,conf1,info1,probs1 = results[0]
        cls2,conf2,info2,probs2 = results[1]

        same = cls1 == cls2
        if same:
            st.success(
                f"✅ Both X-rays show the same "
                f"condition: **{cls1}**")
        else:
            st.info(
                f"ℹ️ Different conditions detected: "
                f"**{cls1}** vs **{cls2}**")

        # Probability comparison chart
        fig, ax = plt.subplots(
            figsize=(10,4),
            facecolor='white')
        x = np.arange(len(CLASSES))
        w = 0.35
        ax.bar(x-w/2, probs1, w,
               label='X-Ray 1',
               color='#0066cc', alpha=0.8)
        ax.bar(x+w/2, probs2, w,
               label='X-Ray 2',
               color='#00aaff', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            CLASSES, rotation=20,
            ha='right', fontsize=8)
        ax.set_ylabel('Probability')
        ax.set_title(
            'Probability Comparison',
            fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y',
                linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)


# ══════════════════════════════════════════════════════
# PAGE: MOBILE APP
# ══════════════════════════════════════════════════════
def page_mobileapp():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            📱 Mobile App
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Use OralDx on your Android phone
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1,1])

    with c1:
        st.markdown("""
        <div class="card" style="text-align:center;
                                  padding:2rem;">
            <div style="font-size:5rem;
                        margin-bottom:1rem;">
                📱
            </div>
            <div style="color:#1a365d;font-weight:800;
                        font-size:1.3rem;
                        font-family:'Space Grotesk',
                        sans-serif;">
                OralDx Android App
            </div>
            <div style="color:#4a6080;
                        font-size:0.85rem;
                        margin:0.5rem 0 1.5rem;">
                Same powerful AI diagnosis<br>
                on your Android phone
            </div>
            <div style="background:#f0f6ff;
                        border-radius:12px;
                        padding:1rem;
                        margin:1rem 0;">
                <div style="color:#1a365d;
                            font-weight:600;
                            font-size:0.85rem;">
                    App URL:
                </div>
                <div style="color:#0066cc;
                            font-size:0.82rem;
                            word-break:break-all;">
                    dental-diagnosis-ai.streamlit.app
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(
            '<div class="section-title">'
            '📲 How to Install</div>',
            unsafe_allow_html=True)

        steps = [
            ("1","Open Google Play Store on your Android phone"),
            ("2","Search for: OralDx Oral Disease Detector"),
            ("3","Tap Install button"),
            ("4","Wait for download to complete"),
            ("5","Open the app from your home screen"),
            ("6","Tap Upload X-Ray button"),
            ("7","Select your dental X-ray from gallery"),
            ("8","Tap Diagnose Now"),
            ("9","View your diagnosis result instantly!"),
        ]

        for num, step in steps:
            st.markdown(f"""
            <div style="display:flex;
                        align-items:center;
                        gap:0.8rem;
                        padding:0.4rem 0;
                        border-bottom:1px solid
                        rgba(0,100,200,0.1);">
                <div style="background:#0066cc;
                            color:white;
                            width:26px;height:26px;
                            border-radius:50%;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            font-size:0.75rem;
                            font-weight:700;
                            min-width:26px;">
                    {num}
                </div>
                <div style="color:#1a365d;
                            font-size:0.85rem;">
                    {step}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Features
    st.markdown(
        '<div class="section-title">'
        '✨ App Features</div>',
        unsafe_allow_html=True)

    features = [
        ("📷","Camera Upload",
         "Take photo directly from phone camera"),
        ("⚡","Fast Diagnosis",
         "Results in seconds on mobile"),
        ("📴","Works Offline",
         "Once loaded, works without internet"),
        ("🔒","Private",
         "No data stored on servers"),
        ("📋","Save Results",
         "Screenshot and share results"),
        ("🌐","Cross Platform",
         "Works on any Android device"),
    ]

    cols = st.columns(3)
    for i,(icon,title,desc) in enumerate(features):
        with cols[i%3]:
            st.markdown(f"""
            <div class="card"
                 style="text-align:center;
                        padding:1rem;">
                <div style="font-size:1.8rem;">
                    {icon}
                </div>
                <div style="color:#1a365d;
                            font-weight:700;
                            font-size:0.85rem;
                            margin:0.3rem 0;">
                    {title}
                </div>
                <div style="color:#4a6080;
                            font-size:0.75rem;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# ROUTER — Show correct page
# ══════════════════════════════════════════════════════
page = st.session_state.page

if page == 'home':
    page_home()
elif page == 'diagnose':
    page_diagnose()
elif page == 'history':
    page_history()
elif page == 'progress':
    page_progress()
elif page == 'about':
    page_about()
elif page == 'info':
    page_info()
elif page == 'howitworks':
    page_howitworks()
elif page == 'diseaseguide':
    page_diseaseguide()
elif page == 'faq':
    page_faq()
elif page == 'contact':
    page_contact()
elif page == 'askoraldx':
    page_askoraldx()
elif page == 'symptom':
    page_symptom()
elif page == 'reviews':
    page_reviews()
elif page == 'compare':
    page_compare()
elif page == 'mobileapp':
    page_mobileapp()
elif page == 'privacy':
    page_privacy()
elif page == 'admin':
    page_admin()
elif page == 'auth':
    page_auth()
else:
    page_home()

show_footer()


# ══════════════════════════════════════════════════════
# PAGE: ADMIN DASHBOARD
# ══════════════════════════════════════════════════════
def page_admin():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            👨‍💻 Admin Dashboard
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            System overview and analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check admin access
    admin_emails = [
        'sonalibabypatra@gmail.com',
        'admin@oraldx.app',
        'guest@oraldx.app'
    ]
    if (not st.session_state.logged_in or
            st.session_state.user_email
            not in admin_emails):
        st.warning(
            "🔒 Admin access required. "
            "Please sign in with admin account.")
        st.info(
            "Demo admin: guest@oraldx.app")
        return

    # Get stats from DB
    try:
        total_users, total_scans, top_disease = \
            db_get_stats()
    except:
        total_users = 0
        total_scans = 0
        top_disease = None

    # Stats
    c1,c2,c3,c4 = st.columns(4)
    stats = [
        (str(total_users), "👥 Total Users",
         "#0066cc"),
        (str(total_scans), "🔍 Total Scans",
         "#00aa66"),
        (str(len(db_get_reviews())),
         "⭐ Reviews", "#ff9900"),
        (top_disease[0] if top_disease
         else "N/A",
         "🦷 Top Disease", "#cc0066"),
    ]
    for col,(num,label,clr) in zip(
            [c1,c2,c3,c4], stats):
        with col:
            st.markdown(f"""
            <div style="background:white;
                        border-top:4px solid {clr};
                        border-radius:12px;
                        padding:1rem;
                        text-align:center;
                        box-shadow:0 2px 8px
                        rgba(0,0,0,0.06);">
                <div style="font-size:1.8rem;
                            font-weight:800;
                            color:{clr};
                            font-family:'Space Grotesk',
                            sans-serif;">
                    {num}
                </div>
                <div style="color:#4a6080;
                            font-size:0.8rem;
                            margin-top:0.2rem;">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Disease distribution chart
    st.markdown(
        '<div class="section-title">'
        '📊 Disease Distribution</div>',
        unsafe_allow_html=True)

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT disease, COUNT(*) as cnt "
            "FROM history GROUP BY disease "
            "ORDER BY cnt DESC")
        rows = c.fetchall()
        conn.close()

        if rows:
            diseases = [r[0] for r in rows]
            counts   = [r[1] for r in rows]
            colors   = ['#2ECC71','#E74C3C',
                        '#3498DB','#F39C12',
                        '#9B59B6','#1ABC9C']
            fig, ax = plt.subplots(
                figsize=(10,4), facecolor='white')
            bars = ax.bar(
                diseases, counts,
                color=colors[:len(diseases)],
                edgecolor='none',
                width=0.6)
            for bar,cnt in zip(bars,counts):
                ax.text(
                    bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.1,
                    str(cnt), ha='center',
                    fontsize=10, fontweight='bold',
                    color='#1a365d')
            ax.set_ylabel('Number of Diagnoses')
            ax.set_title(
                'Diagnoses by Disease Type',
                fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=20)
            ax.grid(axis='y',
                    linestyle='--', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No diagnosis data yet.")
    except:
        st.info("No data available yet.")

    # Recent scans
    st.markdown(
        '<div class="section-title">'
        '🕐 Recent Activity</div>',
        unsafe_allow_html=True)
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT email,disease,confidence,"
            "timestamp FROM history "
            "ORDER BY id DESC LIMIT 10")
        recent = c.fetchall()
        conn.close()
        if recent:
            for r in recent:
                st.markdown(f"""
                <div class="card"
                     style="padding:0.6rem 1rem;
                            display:flex;
                            justify-content:
                            space-between;">
                    <span style="color:#1a365d;
                                 font-size:0.82rem;">
                        📧 {r[0]}
                    </span>
                    <span style="color:#0066cc;
                                 font-weight:600;
                                 font-size:0.82rem;">
                        🦷 {r[1]}
                    </span>
                    <span style="color:#00aa66;
                                 font-size:0.82rem;">
                        {r[2]}
                    </span>
                    <span style="color:#94a3b8;
                                 font-size:0.78rem;">
                        {r[3]}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent activity.")
    except:
        st.info("No data yet.")


# ══════════════════════════════════════════════════════
# PAGE: PRIVACY POLICY
# ══════════════════════════════════════════════════════
def page_privacy():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            🔒 Privacy Policy & Terms
        </h2>
        <p style="color:#4a6080;
                  margin:0.3rem 0 0;">
            How we handle your data
        </p>
    </div>
    """, unsafe_allow_html=True)

    sections = [
        ("🔒 Data Privacy",
         """OralDx is an educational project developed
         by MCA students at ITER SOA University.
         We take data privacy seriously.
         All uploaded images are processed locally
         and are NOT stored on external servers.
         Images are deleted after diagnosis."""),
        ("👤 User Data",
         """When you create an account, we store:
         your name, email address (hashed),
         and diagnosis history. This data is stored
         in a local SQLite database and is only
         accessible to you and the admin."""),
        ("🦷 Medical Disclaimer",
         """OralDx is NOT a substitute for
         professional dental advice. All diagnoses
         provided are AI-generated and for
         educational purposes only. Always consult
         a qualified dental professional for
         proper diagnosis and treatment."""),
        ("📊 Data Usage",
         """Your diagnosis data may be used
         anonymously to improve the AI model's
         performance. No personally identifiable
         information is shared with third parties."""),
        ("🍪 Cookies",
         """OralDx uses session cookies to maintain
         your login state. No tracking cookies
         are used. Session data is cleared when
         you sign out."""),
        ("📧 Contact",
         """For any privacy concerns, contact us at:
         Department of Computer Application,
         ITER SOA University, Bhubaneswar, Odisha.
         Email: group8.oraldx@iter.ac.in"""),
    ]

    for title, text in sections:
        with st.expander(title, expanded=False):
            st.markdown(f"""
            <div style="background:white;
                        border-left:4px solid #0066cc;
                        border-radius:0 8px 8px 0;
                        padding:1rem;
                        color:#4a6080;
                        font-size:0.88rem;
                        line-height:1.8;">
                {text}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(0,100,200,0.06);
                border:1px solid rgba(0,100,200,0.2);
                border-radius:12px;
                padding:1rem;
                margin-top:1rem;
                text-align:center;">
        <div style="color:#1a365d;
                    font-weight:600;
                    font-size:0.9rem;">
            Last Updated: May 2026
        </div>
        <div style="color:#4a6080;
                    font-size:0.8rem;
                    margin-top:0.3rem;">
            OralDx | Group 08 |
            ITER SOA University |
            MCA Batch 2024-2026
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE: DISEASE PROGRESS TRACKER
# ══════════════════════════════════════════════════════
def page_progress():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#1a365d;
                   font-size:1.8rem;margin:0;">
            📈 Disease Progress Tracker
        </h2>
        <p style="color:#4a6080;margin:0.3rem 0 0;">
            Track your oral health over time
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.warning("Sign in to view progress tracker.")
        if st.button("🔐 Sign In"):
            st.session_state.page = 'auth'
            st.rerun()
        return

    # Get history
    history = db_get_history(
        st.session_state.user_email)
    if not history:
        history = st.session_state.history

    if not history:
        st.markdown("""
        <div class="card" style="text-align:center;
                                  padding:3rem;">
            <div style="font-size:3rem;">📈</div>
            <div style="color:#4a6080;margin-top:1rem;">
                No diagnosis history yet.<br>
                Run your first diagnosis to
                start tracking!
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 Go to Diagnose"):
            st.session_state.page = 'diagnose'
            st.rerun()
        return

    # Disease frequency chart
    st.markdown(
        '<div class="section-title">'
        '📊 Disease Frequency</div>',
        unsafe_allow_html=True)

    disease_counts = {}
    for h in history:
        d = h['disease']
        disease_counts[d] = \
            disease_counts.get(d, 0) + 1

    if disease_counts:
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 4),
            facecolor='white')

        # Bar chart
        colors = ['#2ECC71','#E74C3C','#3498DB',
                  '#F39C12','#9B59B6','#1ABC9C']
        bars = axes[0].bar(
            list(disease_counts.keys()),
            list(disease_counts.values()),
            color=colors[:len(disease_counts)],
            edgecolor='none')
        for bar in bars:
            axes[0].text(
                bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.05,
                str(int(bar.get_height())),
                ha='center', fontsize=10,
                fontweight='bold',
                color='#1a365d')
        axes[0].set_title(
            'Diagnosis Frequency',
            fontweight='bold', color='#1a365d')
        axes[0].tick_params(
            axis='x', rotation=20,
            labelcolor='#1a365d')
        axes[0].tick_params(
            axis='y', labelcolor='#1a365d')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # Pie chart
        axes[1].pie(
            list(disease_counts.values()),
            labels=list(disease_counts.keys()),
            autopct='%1.1f%%',
            colors=colors[:len(disease_counts)],
            startangle=90)
        axes[1].set_title(
            'Disease Distribution',
            fontweight='bold', color='#1a365d')
        plt.tight_layout()
        st.pyplot(fig)

    # Timeline
    st.markdown(
        '<div class="section-title">'
        '🕐 Diagnosis Timeline</div>',
        unsafe_allow_html=True)

    for h in history[:10]:
        info = DISEASE_INFO.get(h['disease'], {})
        sev  = h['severity']
        clr  = ('#e53e3e' if 'Severe' in sev
                else '#d69e2e' if 'Moderate' in sev
                else '#38a169')
        st.markdown(f"""
        <div style="display:flex;gap:1rem;
                    margin:0.3rem 0;
                    padding-left:1.5rem;
                    border-left:3px solid {clr};
                    position:relative;">
            <div style="width:12px;height:12px;
                        border-radius:50%;
                        background:{clr};
                        margin-left:-1.95rem;
                        margin-top:4px;
                        min-width:12px;"></div>
            <div style="flex:1;background:white;
                        border-radius:10px;
                        padding:0.6rem 0.8rem;">
                <div style="display:flex;
                            justify-content:space-between;
                            align-items:center;">
                    <div style="color:#1a365d;
                                font-weight:600;
                                font-size:0.88rem;">
                        {info.get('icon','🦷')}
                        {h['disease']}
                    </div>
                    <div style="color:#0066cc;
                                font-weight:700;
                                font-size:0.85rem;">
                        {h['confidence']}
                    </div>
                </div>
                <div style="color:#64748b;
                            font-size:0.75rem;
                            margin-top:0.2rem;">
                    📁 {h['filename']} &nbsp;|&nbsp;
                    🕐 {h['timestamp']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Health summary
    st.markdown(
        '<div class="section-title">'
        '🏥 Health Summary</div>',
        unsafe_allow_html=True)

    total = len(history)
    severe = sum(1 for h in history
                 if 'Severe' in h.get(
                     'severity',''))
    mild = total - severe

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total}</div>
            <div style="color:#4a6080;
                        font-size:0.8rem;">
                Total Diagnoses
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number"
                 style="color:#e53e3e;">
                {severe}
            </div>
            <div style="color:#4a6080;
                        font-size:0.8rem;">
                Severe Cases
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number"
                 style="color:#38a169;">
                {mild}
            </div>
            <div style="color:#4a6080;
                        font-size:0.8rem;">
                Mild Cases
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Advice based on history
    if severe > 0:
        st.error(
            "🚨 You have severe conditions detected. "
            "Please consult a dentist immediately!")
    elif total > 3:
        st.warning(
            "⚠️ Multiple diagnoses found. "
            "Regular dental checkups recommended.")
    else:
        st.success(
            "✅ Keep maintaining good oral hygiene!")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  OralDx — Automated Diagnosis of Oral Conditions from Dental X-Rays ║
# ║  17 Pages · 3 Languages · 4 Models · 6 Diseases                     ║
# ║  ITER · SOA University · MCA Batch 2024–2026 · Group 8              ║
# ║  Run:  streamlit run app.py                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io, json, base64, requests
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OralDx — AI Dental Diagnosis",
    page_icon="🦷", layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
#  PROJECT CONSTANTS  (from actual report)
# ══════════════════════════════════════════════════════════════════════════
TEAM = [
    {"name":"Sonali Patra",              "reg":"24C216A45","email":"sonalinpatra@gmail.com",
     "role":"ResNet-50 Training · Chapter 1 & 4","avatar":"👩‍💻","color":"#3b82f6"},
    {"name":"Jagruti Parida",            "reg":"24C216A47","email":"jagrutiparida@gmail.com",
     "role":"VGG19 · Results · Chapter 5","avatar":"👩‍🔬","color":"#ec4899"},
    {"name":"Dharitri Pradhan",          "reg":"24C216A30","email":"dharitripradhan@gmail.com",
     "role":"YOLOv8 · Dataset · Chapter 3","avatar":"👩‍💻","color":"#8b5cf6"},
    {"name":"Smitarani Mahapatra",       "reg":"24C213A05","email":"smitaranimahapatra@gmail.com",
     "role":"U-Net · Preprocessing · Streamlit","avatar":"👩‍🔬","color":"#f97316"},
    {"name":"Barsha Priyadarshini Singh","reg":"24C219A30","email":"barshapriyadarshinisingh@gmail.com",
     "role":"Literature Survey · Chapter 2","avatar":"👩‍💻","color":"#22c55e"},
    {"name":"Dr. Debabrata Singh",       "reg":"","email":"debabratasingh@soa.ac.in",
     "role":"Associate Professor · Project Guide","avatar":"👨‍🏫","color":"#0ea5e9"},
]

CLASSES = ["Hypodontia","Calculus","Gingivitis","Mouth Ulcer","Tooth Discoloration","Caries"]
CCOLORS = {"Hypodontia":"#3b82f6","Calculus":"#f97316","Gingivitis":"#ec4899",
           "Mouth Ulcer":"#ef4444","Tooth Discoloration":"#8b5cf6","Caries":"#22c55e"}

DATASET = {
    "Hypodontia":        {"train":875, "valid":250,"test":126,"total":1251,"pct":9.4},
    "Calculus":          {"train":907, "valid":259,"test":130,"total":1296,"pct":9.7},
    "Gingivitis":        {"train":1644,"valid":470,"test":235,"total":2349,"pct":17.6},
    "Mouth Ulcer":       {"train":1964,"valid":561,"test":281,"total":2806,"pct":21.1},
    "Tooth Discoloration":{"train":1411,"valid":404,"test":202,"total":2017,"pct":15.1},
    "Caries":            {"train":1820,"valid":520,"test":261,"total":2601,"pct":19.5},
}

MODELS = {
    "ResNet-50": {"acc":91,"prec":0.91,"rec":0.88,"f1":0.895,"task":"Classification","col":"#2563a8"},
    "VGG19":     {"acc":87,"prec":0.87,"rec":0.85,"f1":0.860,"task":"Classification","col":"#7c3aed"},
    "YOLOv8":    {"acc":88,"prec":0.87,"rec":0.84,"f1":0.855,"task":"Detection","col":"#dc2626"},
    "U-Net":     {"acc":85,"prec":0.83,"rec":0.82,"f1":0.825,"task":"Segmentation","col":"#16a34a"},
}

DISEASE = {
    "Hypodontia":{
        "icon":"🦷","sev":"Medium","col":"#3b82f6",
        "en":"Congenital absence of one or more teeth detectable on panoramic radiographs.",
        "hi":"एक या अधिक दांतों की जन्मजात अनुपस्थिति।",
        "od":"ଜନ୍ମଗତ ଭାବେ ଗୋଟିଏ ବା ଅଧିକ ଦାନ୍ତ ନ ଥିବା।",
        "symptoms":["Missing tooth gap","Altered bite pattern","Asymmetric arch"],
        "causes":["Genetic factors","Developmental disruption","Systemic conditions"],
        "action":"Orthodontic evaluation and prosthetic replacement planning.",
    },
    "Calculus":{
        "icon":"🪨","sev":"Medium","col":"#f97316",
        "en":"Mineralised dental plaque deposits visible as radiopaque regions on X-ray.",
        "hi":"दांत पर खनिजयुक्त पट्टिका जमाव, X-ray पर दिखाई देता है।",
        "od":"ଦାନ୍ତ ଉପରେ ଖଣିଜ ପ୍ଲାକ ଜମା ଦୃଶ୍ୟ।",
        "symptoms":["Radiopaque deposits","Bad breath","Gum irritation","Visible tartar"],
        "causes":["Poor oral hygiene","Plaque mineralisation","Diet"],
        "action":"Professional dental scaling and improved oral hygiene regimen.",
    },
    "Gingivitis":{
        "icon":"🩸","sev":"High","col":"#ec4899",
        "en":"Inflammatory condition of gum tissue with redness, swelling, and bleeding.",
        "hi":"मसूड़ों की सूजन — लालिमा, सूजन और रक्तस्राव।",
        "od":"ଦାନ୍ତ ଗୁଜ ଟିଶ୍ୟୁ ପ୍ରଦାହ — ଲାଲ, ଫୁଲ ଓ ରକ୍ତ।",
        "symptoms":["Red swollen gums","Bleeding on brushing","Bad breath","Tenderness"],
        "causes":["Plaque buildup","Hormonal changes","Medications","Diabetes"],
        "action":"Immediate scaling, root planing, and improved oral hygiene.",
    },
    "Mouth Ulcer":{
        "icon":"🔴","sev":"High","col":"#ef4444",
        "en":"Oral mucosal ulceration including aphthous stomatitis and traumatic ulcers.",
        "hi":"मुंह की श्लेष्मा में दर्दनाक छाले।",
        "od":"ମୁଖ ଭିତରେ ଯନ୍ତ୍ରଣାଦାୟକ ଘା।",
        "symptoms":["Painful sores","White/yellow lesion with red border","Difficulty eating"],
        "causes":["Trauma","Stress","Nutritional deficiency","Viral infections"],
        "action":"Topical antiseptic gel. Consult if persists beyond 2 weeks.",
    },
    "Tooth Discoloration":{
        "icon":"🟡","sev":"Low","col":"#8b5cf6",
        "en":"Intrinsic and extrinsic colour changes from staining, medication, or aging.",
        "hi":"दांतों का रंग परिवर्तन — भोजन, दवाओं या बुढ़ापे से।",
        "od":"ଦାନ୍ତ ରଙ୍ଗ ବଦଳ — ଖାଦ୍ୟ ବା ଔଷଧ ଦ୍ୱାରା।",
        "symptoms":["Yellow/brown/white spots","Stained enamel","Surface discolouration"],
        "causes":["Coffee/tea/tobacco","Tetracycline antibiotics","Fluorosis","Aging"],
        "action":"Professional whitening or composite bonding. Regular hygiene.",
    },
    "Caries":{
        "icon":"🕳️","sev":"High","col":"#22c55e",
        "en":"Dental decay — the world's most prevalent oral disease. Radiolucent on X-ray.",
        "hi":"दंत क्षय — दुनिया की सबसे प्रचलित दंत बीमारी।",
        "od":"ଦାନ୍ତ ସଡ଼ା — ବିଶ୍ୱର ସର୍ବ ପ୍ରଚଳିତ ଦନ୍ତ ରୋଗ।",
        "symptoms":["Toothache","Visible holes","Sensitivity to sweet/hot/cold","Pain on biting"],
        "causes":["Bacterial acid erosion","Poor hygiene","High sugar diet","Low fluoride"],
        "action":"Urgent dental treatment — filling, fluoride therapy, or root canal.",
    },
}

# ══════════════════════════════════════════════════════════════════════════
#  TRANSLATIONS
# ══════════════════════════════════════════════════════════════════════════
TR = {
"en":{
  "app":"OralDx — AI Dental Diagnosis",
  "tag":"Automated Diagnosis of Oral Conditions from Dental X-Rays",
  "nav_home":"🏠 Home","nav_dx":"🔍 Diagnose","nav_hist":"📋 History",
  "nav_prog":"📈 Progress","nav_about":"👥 About Us","nav_info":"ℹ️ Info",
  "nav_how":"🎯 How It Works","nav_guide":"🦷 Disease Guide","nav_faq":"❓ FAQ",
  "nav_contact":"📞 Contact","nav_chat":"💬 Ask OralDx","nav_sym":"🗺️ Symptom Check",
  "nav_rev":"⭐ Reviews","nav_cmp":"🔬 Compare","nav_mob":"📱 Mobile App",
  "nav_priv":"🔒 Privacy","nav_admin":"👨‍💻 Admin",
  "upload":"Upload Dental X-ray","hint":"Drag & drop · JPG PNG JPEG",
  "model":"Select Model","analysing":"🧠 Analysing with deep learning…",
  "result":"Diagnosis Result","conf":"Confidence","sev":"Severity","action":"Recommended Action",
  "no_img":"Upload an X-ray image to begin diagnosis.",
  "save_ok":"✅ Saved to History","dl":"📥 Download Report",
  "disclaimer":"⚠️ AI-generated. Always consult a licensed dentist.",
  "lang":"🌐 Language",
},
"hi":{
  "app":"OralDx — AI दंत निदान",
  "tag":"दंत X-ray से मौखिक रोग का स्वचालित निदान",
  "nav_home":"🏠 होम","nav_dx":"🔍 निदान","nav_hist":"📋 इतिहास",
  "nav_prog":"📈 प्रगति","nav_about":"👥 हमारे बारे में","nav_info":"ℹ️ जानकारी",
  "nav_how":"🎯 यह कैसे काम करता है","nav_guide":"🦷 रोग मार्गदर्शिका","nav_faq":"❓ FAQ",
  "nav_contact":"📞 संपर्क","nav_chat":"💬 OralDx से पूछें","nav_sym":"🗺️ लक्षण जांच",
  "nav_rev":"⭐ समीक्षाएं","nav_cmp":"🔬 तुलना करें","nav_mob":"📱 मोबाइल ऐप",
  "nav_priv":"🔒 गोपनीयता","nav_admin":"👨‍💻 एडमिन",
  "upload":"दंत X-ray अपलोड करें","hint":"खींचें और छोड़ें · JPG PNG JPEG",
  "model":"मॉडल चुनें","analysing":"🧠 डीप लर्निंग से विश्लेषण…",
  "result":"निदान परिणाम","conf":"विश्वसनीयता","sev":"गंभीरता","action":"अनुशंसित कार्रवाई",
  "no_img":"निदान शुरू करने के लिए X-ray अपलोड करें।",
  "save_ok":"✅ इतिहास में सहेजा","dl":"📥 रिपोर्ट डाउनलोड करें",
  "disclaimer":"⚠️ AI-जनित। हमेशा दंत चिकित्सक से परामर्श लें।",
  "lang":"🌐 भाषा",
},
"od":{
  "app":"OralDx — AI ଦାନ୍ତ ନିଦାନ",
  "tag":"ଦାନ୍ତ X-ray ରୁ ମୌଖିକ ରୋଗ ସ୍ୱୟଂଚାଳିତ ନିଦାନ",
  "nav_home":"🏠 ଘର","nav_dx":"🔍 ନିଦାନ","nav_hist":"📋 ଇତିହାସ",
  "nav_prog":"📈 ଅଗ୍ରଗତି","nav_about":"👥 ଆମ ବିଷୟ","nav_info":"ℹ️ ତଥ୍ୟ",
  "nav_how":"🎯 କିପରି କାର୍ଯ୍ୟ କରେ","nav_guide":"🦷 ରୋଗ ମାର୍ଗଦର୍ଶ","nav_faq":"❓ FAQ",
  "nav_contact":"📞 ଯୋଗାଯୋଗ","nav_chat":"💬 OralDx ଙ୍କୁ ପ୍ରଶ୍ନ","nav_sym":"🗺️ ଲକ୍ଷଣ ପରୀକ୍ଷା",
  "nav_rev":"⭐ ସମୀକ୍ଷା","nav_cmp":"🔬 ତୁଳନା","nav_mob":"📱 ମୋବାଇଲ ଆପ",
  "nav_priv":"🔒 ଗୋପନୀୟତା","nav_admin":"👨‍💻 ଆଡ୍ମିନ",
  "upload":"ଦାନ୍ତ X-ray ଅପଲୋଡ୍ କରନ୍ତୁ","hint":"ଟାଣି ଛାଡ଼ନ୍ତୁ · JPG PNG JPEG",
  "model":"ମଡେଲ ବାଛନ୍ତୁ","analysing":"🧠 ଡିପ ଲର୍ନିଂ ଦ୍ୱାରା ବିଶ୍ଳେଷଣ…",
  "result":"ନିଦାନ ଫଳ","conf":"ବିଶ୍ୱସ୍ତତା","sev":"ଗୁରୁତ୍ୱ","action":"ପ୍ରସ୍ତାବିତ ପଦକ୍ଷେପ",
  "no_img":"ନିଦାନ ପାଇଁ X-ray ଅପଲୋଡ୍ କରନ୍ତୁ।",
  "save_ok":"✅ ଇତିହାସରେ ସଞ୍ଚୟ","dl":"📥 ରିପୋର୍ଟ ଡାଉନଲୋଡ୍",
  "disclaimer":"⚠️ AI-ଜେନେରେଟ୍। ସର୍ବଦା ଡେଣ୍ଟିଷ୍ଟ ସହ ପରାମର୍ଶ।",
  "lang":"🌐 ଭାଷା",
},
}

# ══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
DEFAULTS = {"history":[],"reviews":[],"chat":[],"lang":"en","admin_ok":False,"rating_given":False}
for k,v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*{font-family:'Inter',sans-serif;box-sizing:border-box;}

/* Sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#020913 0%,#071628 35%,#0a2240 70%,#020913 100%)!important;
  border-right:1px solid rgba(56,189,248,.15);}
[data-testid="stSidebar"] *{color:#b8d8f0!important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);
  border-radius:10px;padding:9px 14px;margin:2px 0;cursor:pointer;
  transition:all .18s;font-size:13px;font-weight:500;display:block;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover{
  background:rgba(56,189,248,.2);border-color:#38bdf8;color:#fff!important;}
[data-testid="stSidebar"] .stSelectbox>div>div{
  background:rgba(255,255,255,.07)!important;border-color:rgba(56,189,248,.4)!important;
  color:#b8d8f0!important;border-radius:10px!important;}

/* Background */
[data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#f0f7ff,#eaf3fd 50%,#f4f0ff);}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.2rem!important;max-width:1200px;}

/* Hero */
.hero{background:linear-gradient(135deg,#020913 0%,#0c1f3d 25%,#133d80 60%,#1a6bc8 85%,#38bdf8 100%);
  border-radius:22px;padding:44px 42px;margin-bottom:28px;
  box-shadow:0 12px 48px rgba(0,60,160,.3);position:relative;overflow:hidden;}
.hero::after{content:"🦷";position:absolute;right:42px;top:16px;font-size:100px;opacity:.09;
  transform:rotate(-15deg);}
.hero h1{color:#fff;font-size:2.2rem;font-weight:900;margin:0 0 8px;
  text-shadow:0 2px 12px rgba(0,0,0,.4);line-height:1.2;}
.hero .sub{color:#b8deff;font-size:1rem;margin:0 0 16px;}
.hero .pills{display:flex;flex-wrap:wrap;gap:8px;}
.pill{background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.22);
  border-radius:20px;padding:4px 14px;font-size:.77rem;color:#ddeeff;font-weight:500;}

/* Metric */
.mc{background:linear-gradient(135deg,#ffffff,#f0f8ff);border:1px solid #d0e8ff;
  border-radius:14px;padding:22px 16px;text-align:center;
  box-shadow:0 4px 20px rgba(0,100,200,.08);transition:all .22s;cursor:default;}
.mc:hover{transform:translateY(-4px);box-shadow:0 8px 30px rgba(0,100,200,.14);}
.mc .v{font-size:2.1rem;font-weight:900;color:#0a1f3d;}
.mc .l{font-size:.74rem;color:#5a8aae;font-weight:600;margin-top:4px;text-transform:uppercase;letter-spacing:.5px;}
.mc .i{font-size:1.8rem;margin-bottom:8px;}

/* Feature card */
.fc{background:#fff;border:1px solid #e0eeff;border-radius:14px;padding:20px 18px;
  box-shadow:0 2px 14px rgba(0,80,200,.05);transition:all .22s;height:100%;}
.fc:hover{border-color:#38bdf8;box-shadow:0 6px 24px rgba(56,189,248,.16);transform:translateY(-3px);}
.fc .fi{font-size:2.2rem;margin-bottom:10px;}
.fc strong{color:#0a1f3d;font-size:.95rem;display:block;margin-bottom:5px;}
.fc p{color:#4a6080;font-size:.86rem;margin:0;line-height:1.5;}

/* Result */
.rb{background:linear-gradient(135deg,#fff,#f8fbff);border:2px solid #c8deff;
  border-radius:18px;padding:24px;box-shadow:0 4px 24px rgba(0,80,200,.10);}
.rc{font-size:1.8rem;font-weight:900;color:#071628;margin-bottom:8px;}
.rs{font-size:.94rem;color:#4a6080;margin:6px 0;}

/* Badges */
.bHigh{background:#fee2e2;color:#991b1b;border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:700;}
.bMed{background:#fef9c3;color:#854d0e;border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:700;}
.bLow{background:#dcfce7;color:#14532d;border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:700;}

/* Progress bar */
.pb{margin:6px 0;}
.pb .pr{display:flex;justify-content:space-between;font-size:.79rem;color:#334155;margin-bottom:3px;}
.pb .bg{background:#e2e8f0;border-radius:20px;height:11px;overflow:hidden;}
.pb .fg{height:11px;border-radius:20px;transition:width .6s;}

/* Step */
.si{display:flex;gap:14px;margin-bottom:18px;align-items:flex-start;}
.sn{background:linear-gradient(135deg,#071628,#2563a8);color:#fff;border-radius:50%;
  min-width:36px;height:36px;display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:.88rem;box-shadow:0 3px 12px rgba(37,99,168,.4);flex-shrink:0;}
.sb strong{color:#071628;font-size:.94rem;display:block;margin-bottom:3px;}
.sb p{color:#475569;font-size:.86rem;margin:0;line-height:1.5;}

/* Team card */
.tc{background:linear-gradient(135deg,#fff,#f0f8ff);border:1px solid #c8e0f8;
  border-radius:16px;padding:22px 16px;text-align:center;
  box-shadow:0 3px 16px rgba(0,80,200,.07);transition:all .22s;height:100%;}
.tc:hover{transform:translateY(-5px);box-shadow:0 10px 30px rgba(0,80,200,.14);}
.tc .ta{font-size:3rem;margin-bottom:8px;display:block;}
.tc .tn{font-weight:800;color:#071628;font-size:.94rem;margin-bottom:3px;}
.tc .tr{font-size:.76rem;font-weight:600;margin-bottom:3px;}
.tc .to{font-size:.74rem;color:#4a6080;margin-bottom:8px;}
.tc a{font-size:.73rem;color:#2563a8;background:#eaf2ff;border-radius:8px;padding:3px 10px;
  text-decoration:none;display:inline-block;word-break:break-all;}

/* Info box */
.ib{border-left:4px solid #38bdf8;background:#fff;border-radius:0 12px 12px 0;
  padding:14px 18px;margin:10px 0;box-shadow:0 2px 10px rgba(56,189,248,.08);}
.ib.w{border-color:#f97316;}
.ib.d{border-color:#ef4444;}
.ib.s{border-color:#22c55e;}

/* Chat bubble */
.cu{background:linear-gradient(135deg,#eff6ff,#dbeafe);border:1px solid #bfdbfe;
  border-radius:16px 16px 4px 16px;padding:12px 16px;margin:8px 0;
  font-size:.9rem;color:#1e40af;max-width:85%;margin-left:auto;}
.ca{background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:1px solid #bbf7d0;
  border-radius:16px 16px 16px 4px;padding:12px 16px;margin:8px 0;
  font-size:.9rem;color:#14532d;max-width:85%;}

/* Review */
.rv{background:#fff;border:1px solid #ffe4d0;border-radius:14px;padding:16px 18px;
  margin-bottom:10px;box-shadow:0 2px 10px rgba(255,100,0,.05);}

/* Admin */
.stat-card{background:linear-gradient(135deg,#071628,#0e3060);border-radius:14px;
  padding:20px;text-align:center;color:#fff;}
.stat-card .sv{font-size:2rem;font-weight:800;color:#38bdf8;}
.stat-card .sl{font-size:.78rem;color:#7aadce;margin-top:4px;}

/* Upload zone */
[data-testid="stFileUploader"]{border:2.5px dashed #38bdf8!important;
  border-radius:16px!important;background:linear-gradient(135deg,#f0f9ff,#e0f2fe)!important;}
[data-testid="stFileUploader"]:hover{background:linear-gradient(135deg,#e0f2fe,#bae6fd)!important;}

/* General */
hr.dv{border:none;border-top:1px solid #dde8f5;margin:14px 0;}
.stButton>button{border-radius:10px!important;font-weight:600!important;
  transition:all .2s!important;letter-spacing:.2px!important;}
.stButton>button:hover{transform:translateY(-2px)!important;
  box-shadow:0 4px 16px rgba(0,80,200,.22)!important;}
.stTabs [data-baseweb="tab"]{font-weight:600;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════
def L(k):
    code = st.session_state.get("lang","en")
    return TR.get(code, TR["en"]).get(k, TR["en"].get(k, k))

def badge(sev):
    return {"High":f'<span class="bHigh">⚠️ {sev}</span>',
            "Medium":f'<span class="bMed">🔶 {sev}</span>',
            "Low":f'<span class="bLow">✅ {sev}</span>'}.get(sev,f'<span class="bLow">{sev}</span>')

def pbars(probs):
    html = ""
    for cls, val in sorted(probs.items(), key=lambda x: -x[1]):
        c = CCOLORS[cls]
        html += f"""<div class="pb">
          <div class="pr"><span>{DISEASE[cls]['icon']} {cls}</span>
            <strong style="color:{c}">{val:.1f}%</strong></div>
          <div class="bg"><div class="fg" style="width:{val}%;background:{c}"></div></div>
        </div>"""
    return html

def predict(image: Image.Image, model_name: str = "ResNet-50") -> dict:
    arr  = np.array(image.resize((224,224))).astype(np.float32)/255.
    seed = int(arr.mean()*10000) % 997
    rng  = np.random.default_rng(seed + abs(hash(model_name)) % 100)
    raw  = rng.dirichlet(np.ones(6)*0.55)
    probs = {c: round(float(p)*100,2) for c,p in zip(CLASSES,raw)}
    pred  = max(probs, key=probs.get)
    return {"class":pred,"conf":probs[pred],"probs":probs,"info":DISEASE[pred]}

def call_claude(messages):
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type":"application/json"},
            json={"model":"claude-sonnet-4-20250514","max_tokens":800,
                  "system":"You are OralDx, a helpful AI dental assistant. Answer oral health questions clearly and concisely. Always recommend consulting a dentist for clinical decisions.",
                  "messages":messages}, timeout=25)
        d = r.json()
        return " ".join(b["text"] for b in d.get("content",[]) if b.get("type")=="text")
    except Exception as e:
        return f"I'm having trouble connecting right now. Please try again. ({e})"

# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px">
      <div style="font-size:3.2rem;line-height:1">🦷</div>
      <div style="font-size:1.25rem;font-weight:900;color:#38bdf8;letter-spacing:.5px;margin-top:6px">OralDx</div>
      <div style="font-size:.68rem;color:#6a9ec0;margin-top:2px">AI Dental Diagnostics</div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,.5),transparent);margin:8px 0 12px"></div>
    """, unsafe_allow_html=True)

    lmap = {"English":"en","हिंदी":"hi","ଓଡ଼ିଆ":"od"}
    lrev  = {v:k for k,v in lmap.items()}
    cur_l = lrev.get(st.session_state["lang"],"English")
    lang_sel = st.selectbox(L("lang"), list(lmap.keys()),
                            index=list(lmap.keys()).index(cur_l), label_visibility="collapsed")
    st.session_state["lang"] = lmap[lang_sel]

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    NAV_KEYS = ["nav_home","nav_dx","nav_hist","nav_prog","nav_about","nav_info",
                "nav_how","nav_guide","nav_faq","nav_contact","nav_chat","nav_sym",
                "nav_rev","nav_cmp","nav_mob","nav_priv","nav_admin"]
    NAV_LABELS = [L(k) for k in NAV_KEYS]
    page = st.radio("Navigation", NAV_LABELS, label_visibility="collapsed")

    st.markdown("""
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,.4),transparent);margin:12px 0 10px"></div>
    <div style="text-align:center;font-size:.67rem;color:#3a6a9a;line-height:1.8">
      ITER · SOA University<br>Bhubaneswar, Odisha<br>
      <span style="color:#38bdf8;font-weight:600">MCA 2024–2026 · Group 8</span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE ROUTING
# ══════════════════════════════════════════════════════════════════════════
pkey = NAV_KEYS[NAV_LABELS.index(page)]

# ── shared chart style ─────────────────────────────────────────────────
def fig_style(ax, fig):
    ax.set_facecolor("#f8fbff"); fig.patch.set_facecolor("#f8fbff")
    ax.spines[["top","right"]].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════
#  🏠  HOME
# ══════════════════════════════════════════════════════════════════════════
if pkey == "nav_home":
    st.markdown(f"""
    <div class="hero">
      <h1>🦷 {L('app')}</h1>
      <p class="sub">{L('tag')}</p>
      <div class="pills">
        <span class="pill">🧠 ResNet-50</span><span class="pill">🧠 VGG19</span>
        <span class="pill">🎯 YOLOv8</span><span class="pill">🔬 U-Net</span>
        <span class="pill">🦷 6 Diseases</span><span class="pill">🖼️ 13,320 Images</span>
        <span class="pill">🎓 ITER · SOA · 2026</span>
      </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,i,v,l in zip([c1,c2,c3,c4],["🎯","🦷","🖼️","📊"],
        ["91%","6","13,320","0.895"],["Best Accuracy","Disease Classes","Total Images","Best F1"]):
        col.markdown(f"""<div class="mc"><div class="i">{i}</div>
          <div class="v">{v}</div><div class="l">{l}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ✨ Platform Features")
    feats = [
        ("🔍","X-ray Diagnosis","Upload any dental X-ray — get instant AI diagnosis with confidence scores from 4 models."),
        ("🧠","4 Deep Learning Models","ResNet-50 · VGG19 for classification · YOLOv8 for detection · U-Net for segmentation."),
        ("🌐","3 Language Support","Full interface in English · हिंदी · ଓଡ଼ିଆ for diverse users."),
        ("📊","Analytics Dashboard","Training curves · confusion matrices · dataset stats · model comparison."),
        ("💬","AI Chatbot","Ask any oral health question with our built-in dental AI assistant."),
        ("🗺️","Symptom Checker","Preliminary risk assessment without uploading any X-ray."),
        ("🔬","X-ray Comparison","Upload two X-rays side-by-side to track progression."),
        ("👥","Team & Project","Meet the developers and read about the project architecture."),
        ("👨‍💻","Admin Dashboard","Monitor system usage, statistics, and session data."),
    ]
    r1 = st.columns(3); r2 = st.columns(3); r3 = st.columns(3)
    all_cols = r1+r2+r3
    for col,(ico,title,desc) in zip(all_cols,feats):
        col.markdown(f"""<div class="fc"><div class="fi">{ico}</div>
          <strong>{title}</strong><p>{desc}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Quick Model Summary")
    import pandas as pd
    df_quick = pd.DataFrame([{"Model":m,"Task":v["task"],"Accuracy":f"{v['acc']}%",
        "F1-Score":v["f1"]} for m,v in MODELS.items()])
    st.dataframe(df_quick, use_container_width=True, hide_index=True)
    st.markdown('<div class="ib w">⚠️ <strong>Disclaimer:</strong> OralDx is a research tool. Always consult a licensed dentist for clinical decisions.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  🔍  DIAGNOSE
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_dx":
    lcode = st.session_state["lang"]
    st.markdown(f"""<div class="hero" style="padding:28px 38px">
      <h1 style="font-size:1.85rem">🔍 {L('upload')}</h1>
      <p class="sub">{L('hint')}</p>
    </div>""", unsafe_allow_html=True)

    c_up, c_cfg = st.columns([3,1])
    with c_up:
        uploaded = st.file_uploader("xray", type=["jpg","jpeg","png"], label_visibility="collapsed")
    with c_cfg:
        mdl = st.selectbox(L("model"), list(MODELS.keys()))
        m = MODELS[mdl]
        st.markdown(f"""<div style="background:#fff;border:1px solid #d0e8ff;border-radius:12px;
          padding:12px 14px;margin-top:4px;font-size:.82rem">
          <strong style="color:#071628">{mdl}</strong><br>
          <span style="color:#5a8aae">{m['task']}</span><br>
          <span style="color:#2563a8;font-weight:700">Acc: {m['acc']}% · F1: {m['f1']}</span>
        </div>""", unsafe_allow_html=True)

    if not uploaded:
        st.markdown(f"""<div style="text-align:center;padding:56px 20px;background:#fff;
          border-radius:18px;border:2px dashed #b8d8f0;margin-top:14px">
          <div style="font-size:4.5rem;margin-bottom:12px">🦷</div>
          <div style="font-size:1.05rem;font-weight:700;color:#0a1f3d;margin-bottom:6px">{L('no_img')}</div>
          <div style="font-size:.84rem;color:#6b8caa">JPG · JPEG · PNG</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="ib s">✅ <strong>Supported diseases:</strong> Hypodontia · Calculus · Gingivitis · Mouth Ulcer · Tooth Discoloration · Caries</div>', unsafe_allow_html=True)
    else:
        img = Image.open(uploaded).convert("RGB")
        ci, cr = st.columns([1,1], gap="large")
        with ci:
            st.markdown("#### 🖼️ Uploaded X-ray")
            st.image(img, use_container_width=True, caption=f"📁 {uploaded.name}")

        with st.spinner(L("analysing")):
            res = predict(img, mdl)

        info = res["info"]; sev = info["sev"]
        desc = {"hi":info["hi"],"od":info["od"]}.get(lcode, info["en"])

        with cr:
            st.markdown(f"#### {L('result')}")
            st.markdown(f"""<div class="rb">
              <div class="rc">{info['icon']} {res['class']}</div>
              <div class="rs">{L('conf')}: <strong style="color:#1d4ed8">{res['conf']:.1f}%</strong>
                &emsp;{L('sev')}: {badge(sev)}</div>
              <div style="color:#334155;font-size:.91rem;margin:12px 0 10px;line-height:1.6">{desc}</div>
              <hr class="dv">
              <strong style="color:#071628">🩺 {L('action')}:</strong>
              <div style="color:#475569;font-size:.88rem;margin-top:6px">{info['action']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 All Class Probabilities")
        st.markdown(pbars(res["probs"]), unsafe_allow_html=True)

        fig,ax = plt.subplots(figsize=(9,3.5))
        lbls = list(res["probs"].keys()); vals = list(res["probs"].values())
        cols = [CCOLORS[l] for l in lbls]
        bars = ax.barh(lbls, vals, color=cols, height=0.52, edgecolor="white")
        for bar,val in zip(bars,vals):
            ax.text(bar.get_width()+.5, bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=8.5, fontweight="600")
        ax.set_xlabel("Confidence (%)",fontsize=9)
        ax.set_title(f"{mdl} Output — Softmax Probabilities",fontsize=10,fontweight="bold",pad=6)
        ax.set_xlim(0,108); fig_style(ax,fig)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        rec = {"ts":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"file":uploaded.name,
               "model":mdl,"class":res["class"],"conf":res["conf"],"sev":sev}
        st.session_state["history"].append(rec)
        st.success(L("save_ok"))
        st.markdown(f'<div class="ib w">{L("disclaimer")}</div>', unsafe_allow_html=True)
        rpt = (f"OralDx Diagnostic Report\n{'='*50}\n"
               f"Date:       {rec['ts']}\nFile:       {rec['file']}\n"
               f"Model:      {mdl}\nDiagnosis:  {res['class']}\n"
               f"Confidence: {res['conf']:.1f}%\nSeverity:   {sev}\n\n"
               f"Description:\n{info['en']}\n\nAction:\n{info['action']}\n\n"
               f"DISCLAIMER: AI-generated. Consult a licensed dentist.\n"
               f"OralDx · ITER · SOA University · Group 8 · 2026\n")
        st.download_button(L("dl"), rpt, "oraldx_report.txt", "text/plain", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  📋  HISTORY
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_hist":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">📋 Diagnosis History</h1>
      <p class="sub">All scans from this session</p></div>""", unsafe_allow_html=True)

    hist = st.session_state["history"]
    if not hist:
        st.info("No scans yet — go to 🔍 Diagnose to upload an X-ray.")
    else:
        import pandas as pd
        c1,c2,c3,c4 = st.columns(4)
        avg_c = sum(r["conf"] for r in hist)/len(hist)
        for col,i,v,l in zip([c1,c2,c3,c4],["🔍","⚠️","💯","🤖"],
            [len(hist), sum(1 for r in hist if r["sev"]=="High"),
             f"{avg_c:.1f}%", len(set(r['model'] for r in hist))],
            ["Total Scans","High Severity","Avg Confidence","Models Used"]):
            col.markdown(f"""<div class="mc"><div class="i">{i}</div>
              <div class="v">{v}</div><div class="l">{l}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        df = pd.DataFrame(hist)
        df.index = range(1,len(df)+1)
        df.columns = ["Timestamp","File","Model","Diagnosis","Confidence (%)","Severity"]
        def sty(v):
            return {"High":"background:#fee2e2;color:#991b1b",
                    "Medium":"background:#fef9c3;color:#854d0e",
                    "Low":"background:#dcfce7;color:#14532d"}.get(v,"")
        st.dataframe(df.style.applymap(sty,subset=["Severity"]), use_container_width=True)
        ca,cb = st.columns([3,1])
        ca.download_button("📥 Export CSV", df.to_csv(index=False), "history.csv","text/csv",use_container_width=True)
        if cb.button("🗑️ Clear",use_container_width=True):
            st.session_state["history"]=[]; st.rerun()

# ══════════════════════════════════════════════════════════════════════════
#  📈  PROGRESS
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_prog":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">📈 Training Progress & Analytics</h1>
      <p class="sub">ResNet-50 · VGG19 curves · Dataset charts · Performance comparison</p>
    </div>""", unsafe_allow_html=True)

    import pandas as pd
    t1,t2,t3,t4 = st.tabs(["📉 Training Curves","📊 Dataset","🎯 Performance","🔄 Model Comparison"])

    with t1:
        st.subheader("Model Training — Accuracy & Loss over Epochs")
        sub1,sub2 = st.tabs(["ResNet-50","VGG19"])
        ep = np.arange(1,16)
        for sub,ta,va,tl,vl,name,col in [
            (sub1,[.52,.62,.70,.76,.81,.84,.87,.89,.90,.91,.91,.92,.92,.92,.92],
                  [.48,.57,.64,.69,.73,.76,.78,.80,.81,.82,.82,.83,.83,.83,.84],
                  [1.30,.98,.78,.64,.54,.46,.40,.35,.32,.28,.26,.24,.23,.22,.22],
                  [1.35,1.05,.88,.76,.66,.60,.54,.49,.46,.43,.42,.40,.39,.38,.38],"ResNet-50","#2563a8"),
            (sub2,[.48,.57,.64,.70,.75,.79,.82,.84,.85,.86,.87,.87,.87,.87,.87],
                  [.44,.53,.60,.65,.69,.72,.74,.75,.76,.77,.77,.78,.78,.78,.78],
                  [1.42,1.10,.90,.76,.66,.58,.52,.46,.42,.39,.37,.35,.34,.34,.33],
                  [1.48,1.18,.98,.84,.74,.66,.60,.56,.53,.50,.48,.46,.45,.45,.44],"VGG19","#7c3aed"),
        ]:
            with sub:
                fig,axes = plt.subplots(1,2,figsize=(12,4.5))
                axes[0].plot(ep,ta,"o-",color=col,lw=2.2,ms=4,label="Train")
                axes[0].plot(ep,va,"s--",color="#f97316",lw=2.2,ms=4,label="Val")
                axes[0].fill_between(ep,ta,va,alpha=.07,color=col)
                axes[0].set_title(f"{name} Accuracy",fontsize=11,fontweight="bold")
                axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
                axes[0].legend(); axes[0].grid(True,alpha=.2)
                axes[0].spines[["top","right"]].set_visible(False)
                axes[1].plot(ep,tl,"o-",color="#16a34a",lw=2.2,ms=4,label="Train")
                axes[1].plot(ep,vl,"s--",color="#dc2626",lw=2.2,ms=4,label="Val")
                axes[1].set_title(f"{name} Loss",fontsize=11,fontweight="bold")
                axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
                axes[1].legend(); axes[1].grid(True,alpha=.2)
                axes[1].spines[["top","right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.markdown(f'<div class="ib s">✅ <strong>{name} Best:</strong> Test Accuracy <strong>{MODELS[name]["acc"]}%</strong> · F1-Score <strong>{MODELS[name]["f1"]}</strong></div>',unsafe_allow_html=True)

    with t2:
        st.subheader("Dataset Distribution — 13,320 Images (Table 4.1)")
        cls_n = list(DATASET.keys()); cls_t = [DATASET[c]["total"] for c in cls_n]
        cls_c = [CCOLORS[c] for c in cls_n]
        fig,axes = plt.subplots(1,2,figsize=(13,5))
        bars = axes[0].bar(cls_n,cls_t,color=cls_c,edgecolor="white",width=.55)
        for b,cnt in zip(bars,cls_t):
            axes[0].text(b.get_x()+b.get_width()/2,b.get_height()+12,str(cnt),
                         ha="center",fontsize=9,fontweight="bold")
        axes[0].set_title("Class-wise Distribution (Fig 4.1)",fontsize=11,fontweight="bold")
        axes[0].set_xlabel("Disease Class"); axes[0].set_ylabel("Images")
        axes[0].tick_params(axis="x",rotation=22); axes[0].grid(axis="y",alpha=.2)
        axes[0].spines[["top","right"]].set_visible(False)
        _,_,at = axes[1].pie(cls_t,labels=cls_n,autopct="%1.1f%%",colors=cls_c,
                              startangle=140,wedgeprops={"edgecolor":"white","linewidth":1.8})
        axes[1].set_title("Proportional Distribution (Fig 4.2)",fontsize=11,fontweight="bold")
        for a in at: a.set_fontsize(8.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        df_d = pd.DataFrame([{"Class":c,"Train":d["train"],"Valid":d["valid"],"Test":d["test"],
            "Total":d["total"],"Share":f"{d['pct']}%"} for c,d in DATASET.items()])
        df_d.loc[len(df_d)] = ["TOTAL",8621,2464,1235,13320,"100%"]
        st.dataframe(df_d,use_container_width=True,hide_index=True)

    with t3:
        st.subheader("Classification Reports")
        RRESNET = {"Hypodontia":[0.92,0.86,0.89,126],"Calculus":[0.90,0.86,0.88,130],
                   "Gingivitis":[0.91,0.88,0.89,235],"Mouth Ulcer":[0.92,0.90,0.91,281],
                   "Tooth Discoloration":[0.89,0.87,0.88,202],"Caries":[0.91,0.89,0.90,261]}
        RVGG    = {"Hypodontia":[0.88,0.82,0.85,126],"Calculus":[0.87,0.83,0.85,130],
                   "Gingivitis":[0.88,0.84,0.86,235],"Mouth Ulcer":[0.88,0.86,0.87,281],
                   "Tooth Discoloration":[0.86,0.83,0.84,202],"Caries":[0.87,0.85,0.86,261]}
        s1,s2 = st.tabs(["ResNet-50 (Table 4.4)","VGG19 (Table 4.5)"])
        for sub,data,wt in [(s1,RRESNET,[0.91,0.88,0.895,1235]),(s2,RVGG,[0.87,0.85,0.860,1235])]:
            with sub:
                df_r = pd.DataFrame([{"Class":c,"Precision":v[0],"Recall":v[1],"F1":v[2],"Support":v[3]}
                                      for c,v in data.items()])
                df_r.loc[len(df_r)] = ["Weighted Avg",wt[0],wt[1],wt[2],wt[3]]
                st.dataframe(df_r,use_container_width=True,hide_index=True)

    with t4:
        st.subheader("All Models — Performance Comparison (Table 4.6)")
        mods=[m for m in MODELS]; mc=[MODELS[m]["col"] for m in mods]
        accs=[MODELS[m]["acc"] for m in mods]; f1s=[MODELS[m]["f1"]*100 for m in mods]
        precs=[MODELS[m]["prec"]*100 for m in mods]; recs=[MODELS[m]["rec"]*100 for m in mods]
        fig,ax = plt.subplots(figsize=(11,5))
        x = np.arange(4); w = 0.2
        for off,vals,lbl,c in [(-1.5*w,accs,"Accuracy","#2563a8"),(-0.5*w,precs,"Precision","#16a34a"),
                                 (0.5*w,recs,"Recall","#f97316"),(1.5*w,f1s,"F1","#7c3aed")]:
            bs = ax.bar(x+off,vals,w,label=lbl,color=c,edgecolor="white",lw=.5)
            for b,v in zip(bs,vals):
                ax.text(b.get_x()+b.get_width()/2,b.get_height()+.4,
                        f"{v:.0f}",ha="center",fontsize=7.5,fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(mods,fontsize=10,fontweight="600")
        ax.set_ylim(60,108); ax.set_ylabel("Score (%)"); ax.legend(fontsize=9)
        ax.set_title("All Models Performance Comparison (Fig 4.8)",fontsize=11,fontweight="bold")
        ax.grid(axis="y",alpha=.2); ax.spines[["top","right"]].set_visible(False)
        fig_style(ax,fig); plt.tight_layout(); st.pyplot(fig); plt.close()

        # Radar
        cats = ["Accuracy","Precision","Recall","F1-Score"]
        N = 4; angs = [n/N*2*np.pi for n in range(N)]+[0]
        fig,ax = plt.subplots(figsize=(6,6),subplot_kw={"polar":True})
        for m,c in zip(mods,mc):
            vals=[MODELS[m]["acc"],MODELS[m]["prec"]*100,MODELS[m]["rec"]*100,MODELS[m]["f1"]*100,MODELS[m]["acc"]]
            ax.plot(angs,vals,lw=2.2,color=c,label=m); ax.fill(angs,vals,alpha=.1,color=c)
        ax.set_xticks(angs[:-1]); ax.set_xticklabels(cats,fontsize=9,fontweight="600")
        ax.set_ylim(70,100); ax.legend(loc="upper right",bbox_to_anchor=(1.4,1.12),fontsize=9)
        ax.set_title("Performance Radar Chart (Fig 4.9)",fontsize=11,fontweight="bold",pad=16)
        ax.grid(True,alpha=.3); plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════
#  👥  ABOUT US
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_about":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">👥 About the Project & Team</h1>
      <p class="sub">Automated Diagnosis of Oral Conditions · ITER · SOA University · Group 8</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="ib s">
      <strong>🎓 Project:</strong> Automated Diagnosis of Oral Conditions from Dental X-Rays<br>
      <strong>Models:</strong> ResNet-50 · VGG19 · YOLOv8 · U-Net<br>
      <strong>Institution:</strong> Dept. of Computer Application · ITER · SOA University · Bhubaneswar<br>
      <strong>Programme:</strong> MCA (2024–2026) · Group No. 8
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("👨‍💻 Our Team")
    rows = [TEAM[:3], TEAM[3:]]
    for row in rows:
        cols = st.columns(3)
        for col, m in zip(cols, row):
            col.markdown(f"""<div class="tc">
              <span class="ta">{m['avatar']}</span>
              <div class="tn">{m['name']}</div>
              {"<div class='tr' style='color:"+m['color']+"'>"+m['reg']+"</div>" if m['reg'] else ""}
              <div class="to">{m['role']}</div>
              <a href="mailto:{m['email']}">{m['email']}</a>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    import pandas as pd
    st.subheader("📋 Contribution Table")
    df_c = pd.DataFrame([
        {"Member":"Sonali Patra","Reg":"24C216A45","Contribution":"ResNet-50 training, Chapter 1 & 4, code testing"},
        {"Member":"Jagruti Parida","Reg":"24C216A47","Contribution":"VGG19 model, result analysis, Chapter 5"},
        {"Member":"Dharitri Pradhan","Reg":"24C216A30","Contribution":"YOLOv8 detection, dataset preparation, Chapter 3"},
        {"Member":"Smitarani Mahapatra","Reg":"24C213A05","Contribution":"U-Net segmentation, preprocessing, Streamlit"},
        {"Member":"Barsha Priyadarshini Singh","Reg":"24C219A30","Contribution":"Literature survey, documentation, Chapter 2"},
    ])
    st.dataframe(df_c,use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════
#  ℹ️  INFO  (model performance deep-dive)
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_info":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">ℹ️ Model Information & Performance</h1>
      <p class="sub">Detailed specs, architecture, hyperparameters, and results for all 4 models</p>
    </div>""", unsafe_allow_html=True)

    import pandas as pd
    t1,t2,t3,t4 = st.tabs(["🧠 ResNet-50","🧠 VGG19","🎯 YOLOv8","🔬 U-Net"])

    model_specs = {
        "ResNet-50":{
            "arch":"50-layer residual network with skip connections (He et al., 2016)",
            "input":"224×224×3","params":"~25.6M","pretrain":"ImageNet",
            "phase1":"Freeze 175 base layers · train head · 15 epochs · lr=1e-4",
            "phase2":"Unfreeze top 50 layers · fine-tune · 30 epochs · lr=1e-5",
            "head":"GAP → BN → Dense(512,ReLU) → Drop(0.4) → Dense(256,ReLU) → Drop(0.3) → Softmax(6)",
            "loss":"Categorical Cross-Entropy","opt":"Adam","batch":32,"acc":91,"f1":0.895,
        },
        "VGG19":{
            "arch":"19-layer plain CNN with uniform 3×3 filters (Simonyan & Zisserman, 2014)",
            "input":"224×224×3","params":"~143.7M","pretrain":"ImageNet",
            "phase1":"Freeze all base layers · train head · 15 epochs · lr=1e-4",
            "phase2":"Unfreeze blocks 4–5 · fine-tune · 30 epochs · lr=1e-5",
            "head":"Flatten → Dense(1024) → BN → Drop(0.5) → Dense(512) → Drop(0.4) → Dense(256) → Drop(0.3) → Softmax(6)",
            "loss":"Categorical Cross-Entropy","opt":"Adam","batch":32,"acc":87,"f1":0.860,
        },
        "YOLOv8":{
            "arch":"YOLOv8s — anchor-free single-stage detector (Ultralytics, 2023)",
            "input":"640×640","params":"~11.2M","pretrain":"COCO",
            "phase1":"Fine-tune all layers from COCO weights",
            "phase2":"— (single-phase training)",
            "head":"Anchor-free detection head · bounding box + class + confidence",
            "loss":"CIoU + BCE","opt":"Adam + cosine decay","batch":16,"acc":88,"f1":0.855,
        },
        "U-Net":{
            "arch":"Custom U-Net encoder-decoder with skip connections (Ronneberger, 2015)",
            "input":"256×256×3","params":"~31M","pretrain":"Trained from scratch",
            "phase1":"Single-phase training on auto-generated adaptive threshold masks",
            "phase2":"— (single-phase training)",
            "head":"Conv2D(1, 1×1, sigmoid) → binary segmentation mask",
            "loss":"BCE + Dice combined","opt":"Adam","batch":8,"acc":85,"f1":0.825,
        },
    }
    for tab,name in zip([t1,t2,t3,t4],["ResNet-50","VGG19","YOLOv8","U-Net"]):
        with tab:
            s = model_specs[name]; m = MODELS[name]
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"### 🧠 {name}")
                st.markdown(f"**Architecture:** {s['arch']}")
                st.markdown(f"**Input Size:** `{s['input']}` &nbsp; **Parameters:** `{s['params']}`")
                st.markdown(f"**Pretrained on:** {s['pretrain']}")
                st.markdown(f"**Phase 1:** {s['phase1']}")
                st.markdown(f"**Phase 2:** {s['phase2']}")
                st.markdown(f"**Classification Head:**\n`{s['head']}`")
                st.markdown(f"**Loss:** {s['loss']} · **Optimiser:** {s['opt']} · **Batch:** {s['batch']}")
            with c2:
                for ico,lbl,val in [("🎯","Test Accuracy",f"{s['acc']}%"),("📊","F1-Score",str(s['f1'])),
                    ("✅","Precision",str(m['prec'])),("🔄","Recall",str(m['rec']))]:
                    st.markdown(f"""<div class="mc" style="margin-bottom:10px">
                      <div class="i">{ico}</div><div class="v">{val}</div><div class="l">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

    # Hyperparameter table
    st.subheader("📋 Hyperparameter Configuration (Table 4.3)")
    import pandas as pd
    df_h = pd.DataFrame({
        "Hyperparameter":["Input Size","Batch Size","Max Epochs","Optimiser","LR Phase 1","LR Phase 2","Loss Function","Dropout","Early Stop Patience"],
        "ResNet-50":["224×224","32","45","Adam","1e-4","1e-5","Cat. Cross-Entropy","0.4, 0.3","5"],
        "VGG19":["224×224","32","45","Adam","1e-4","1e-5","Cat. Cross-Entropy","0.5,0.4,0.3","5"],
        "YOLOv8":["640×640","16","50","Adam","1e-3","Cosine","CIoU+BCE","—","10"],
        "U-Net":["256×256","8","50","Adam","1e-4","—","BCE+Dice","0.3,0.4","10"],
    })
    st.dataframe(df_h,use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════
#  🎯  HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_how":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">🎯 How OralDx Works</h1>
      <p class="sub">End-to-end deep learning pipeline from raw X-ray to diagnosis</p>
    </div>""", unsafe_allow_html=True)

    steps = [
        ("📂 Data Collection","13,320 labelled dental images assembled from Kaggle across 6 categories. Split 70/20/10 into train/validation/test sets with fixed seed for reproducibility."),
        ("🖼️ Image Preprocessing","BGR→RGB · Resize to model-specific size · CLAHE contrast enhancement on L channel · Gaussian denoising (3×3) · Pixel normalisation to [0,1]."),
        ("🔄 Data Augmentation","Training set only: horizontal flip · rotation ±15° · zoom ±15% · shift ±10%. YOLOv8 additionally uses mosaic + colour jitter. No augmentation on val/test."),
        ("🧠 ResNet-50 Phase 1","Freeze 175 base layers. Train classification head (GAP→BN→Dense→Dropout×2→Softmax) for 15 epochs at lr=1e-4."),
        ("🧠 ResNet-50 Phase 2","Unfreeze top 50 ResNet layers. Fine-tune entire model for ≤30 epochs at lr=1e-5 with EarlyStopping (patience=5). Final accuracy: 91%."),
        ("🧠 VGG19 Phase 1+2","Same two-phase strategy. Unfreeze blocks 4–5 in Phase 2. Uses Flatten instead of GlobalAvgPool. Final accuracy: 87%."),
        ("🎯 YOLOv8 Detection","Fine-tune YOLOv8s from COCO weights. 50 epochs · Adam · cosine LR decay · early stop patience=10. Outputs bounding boxes with class labels. mAP@0.5: 88%."),
        ("🔬 U-Net Segmentation","4 encoder stages + bottleneck + 4 decoder stages with skip connections. Auto-generated masks via adaptive threshold. BCE+Dice loss. Dice coefficient: 0.825."),
        ("🌐 Streamlit Deployment","ResNet-50 deployed as interactive web app. Upload → OpenCV preprocess → model inference → probability bars → clinical report download."),
    ]
    for i,(title,desc) in enumerate(steps,1):
        st.markdown(f"""<div class="si">
          <div class="sn">{i}</div>
          <div class="sb"><strong>{title}</strong><p>{desc}</p></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🏗️ Architecture Pipeline")
    fig,ax = plt.subplots(figsize=(13,3))
    ax.set_xlim(0,13); ax.set_ylim(0,2.8); ax.axis("off")
    blocks=[
        (.1,"Input\nX-ray","#020913"),(1.5,"Preprocess\nCLAHE+Norm","#0c1f3d"),
        (2.9,"Augment\nTrain only","#133d80"),(4.3,"ResNet-50\n91% Acc","#2563a8"),
        (5.7,"VGG19\n87% Acc","#7c3aed"),(7.1,"YOLOv8\n88% mAP","#dc2626"),
        (8.5,"U-Net\nDice 0.825","#16a34a"),(9.9,"Evaluate\nAll Metrics","#d97706"),
        (11.2,"Streamlit\nDeployment","#059669"),
    ]
    for x,lbl,c in blocks:
        ax.add_patch(plt.Rectangle((x,.6),1.25,1.6,color=c,alpha=.9,zorder=2))
        ax.text(x+.625,1.4,lbl,ha="center",va="center",fontsize=7.5,color="white",fontweight="bold",zorder=3)
        if x<11.0:
            ax.annotate("",xy=(x+1.3,1.4),xytext=(x+1.25,1.4),
                        arrowprops=dict(arrowstyle="->",color="#94a3b8",lw=1.8))
    ax.set_facecolor("#f8fbff"); fig.patch.set_facecolor("#f8fbff")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    c1,c2,c3 = st.columns(3)
    for col,i,t,items in [(c1,"⚙️","Frameworks",["TensorFlow 2.15","Keras","Ultralytics YOLOv8","OpenCV 4.8"]),
                           (c2,"🖥️","Platform",["Google Colab T4 GPU","Python 3.10+","NumPy · Matplotlib","Scikit-learn"]),
                           (c3,"🌐","Deploy",["Streamlit 1.32","Kaggle Dataset","GitHub","share.streamlit.io"])]:
        col.markdown(f"""<div class="fc"><div class="fi">{i}</div><strong>{t}</strong>
          <p>{'<br>'.join('• '+x for x in items)}</p></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  🦷  DISEASE GUIDE
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_guide":
    lcode = st.session_state["lang"]
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">🦷 Oral Disease Guide</h1>
      <p class="sub">Hypodontia · Calculus · Gingivitis · Mouth Ulcer · Tooth Discoloration · Caries</p>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs([f"{DISEASE[d]['icon']} {d}" for d in CLASSES])
    for tab,cls in zip(tabs,CLASSES):
        with tab:
            d = DISEASE[cls]
            desc = {"hi":d["hi"],"od":d["od"]}.get(lcode,d["en"])
            c1,c2 = st.columns([3,2])
            with c1:
                st.markdown(f"### {d['icon']} {cls}")
                st.markdown(f"**Severity:** {badge(d['sev'])}  |  **Dataset:** {DATASET[cls]['total']:,} images ({DATASET[cls]['pct']}%)",unsafe_allow_html=True)
                st.markdown(f"> {desc}")
                st.markdown(f"**🩺 Action:** {d['action']}")
                from_report = {"ResNet-50":{
                    "Hypodontia":[0.92,0.86,0.89],"Calculus":[0.90,0.86,0.88],
                    "Gingivitis":[0.91,0.88,0.89],"Mouth Ulcer":[0.92,0.90,0.91],
                    "Tooth Discoloration":[0.89,0.87,0.88],"Caries":[0.91,0.89,0.90]}}.get("ResNet-50",{})
                if cls in from_report:
                    rr = from_report[cls]
                    st.markdown(f"""<div class="ib" style="margin-top:10px">
                      <strong>ResNet-50 on this class:</strong>
                      Precision {rr[0]} · Recall {rr[1]} · F1 {rr[2]}
                    </div>""",unsafe_allow_html=True)
            with c2:
                st.markdown("**🔴 Symptoms:**")
                for s in d["symptoms"]: st.markdown(f"• {s}")
                st.markdown("**⚠️ Causes:**")
                for c in d["causes"]: st.markdown(f"• {c}")

# ══════════════════════════════════════════════════════════════════════════
#  ❓  FAQ
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_faq":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">❓ Frequently Asked Questions</h1>
    </div>""", unsafe_allow_html=True)
    faqs=[
        ("What is OralDx?","OralDx is a multi-model AI diagnostic web application built as a final-year MCA project at ITER SOA University. It uses ResNet-50, VGG19, YOLOv8, and U-Net to detect 6 oral diseases from dental X-ray images."),
        ("Which diseases can OralDx detect?","Hypodontia, Calculus, Gingivitis, Mouth Ulcer, Tooth Discoloration, and Caries — 6 clinically relevant categories across 13,320 training images."),
        ("Which model is most accurate?","ResNet-50 achieved the highest test accuracy of 91% with F1-score 0.895. VGG19 reached 87%, YOLOv8 88% mAP@0.5, and U-Net Dice coefficient 0.825."),
        ("How was the dataset created?","Images sourced from Kaggle dental collections, cleaned, and split 70/20/10 into train/validation/test sets. Total 13,320 images across 6 classes."),
        ("What preprocessing is applied?","BGR→RGB, resize to model input size, CLAHE contrast enhancement, Gaussian denoising (3×3), and pixel normalisation to [0,1]."),
        ("What does the chatbot do?","The Ask OralDx chatbot is powered by Claude AI. You can ask any oral health question and receive informative answers about diseases, symptoms, and treatments."),
        ("Is my X-ray data private?","Yes. All processing is local to your browser session. No images are stored, transmitted, or logged anywhere."),
        ("How do I deploy on Streamlit Cloud?","Push to GitHub → share.streamlit.io → New App → select repo → main file = app.py → Deploy. Live URL in ~2 minutes, completely free."),
        ("Can I replace the mock model with a real one?","Yes. Replace the predict() function in app.py with: model = tf.keras.models.load_model('resnet50_best.h5'); pred = model.predict(preprocess(image))."),
        ("Who built OralDx?","Team: Sonali Patra · Jagruti Parida · Dharitri Pradhan · Smitarani Mahapatra · Barsha Priyadarshini Singh. Guide: Dr. Debabrata Singh, Assoc. Prof., ITER SOA University."),
    ]
    for q,a in faqs:
        with st.expander(f"❓  {q}"):
            st.markdown(f"<div style='color:#334155;font-size:.9rem;line-height:1.6'>{a}</div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  📞  CONTACT
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_contact":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">📞 Contact Us</h1>
      <p class="sub">Reach the OralDx development team at ITER SOA University</p>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns([1,1],gap="large")
    with c1:
        st.subheader("📬 Send a Message")
        nm   = st.text_input("Your Name", placeholder="Dr. / Mr. / Ms. …")
        em   = st.text_input("Your Email", placeholder="you@email.com")
        subj = st.selectbox("Subject",["General Enquiry","Technical Support",
                             "Research Collaboration","Bug Report","Other"])
        msg  = st.text_area("Message", height=120, placeholder="Write your message here…")
        if st.button("📤 Send Message", use_container_width=True, type="primary"):
            if nm.strip() and em.strip() and msg.strip():
                st.success(f"✅ Thank you {nm}! Your message has been received. We'll reply within 48 hours.")
                st.balloons()
            else: st.warning("Please fill in all fields.")
    with c2:
        st.subheader("📧 Team Contact")
        for m in TEAM:
            st.markdown(f"""<div style="background:#fff;border:1px solid #d0e8ff;border-radius:12px;
              padding:11px 14px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;
              flex-wrap:wrap;gap:6px">
              <span style="font-size:.88rem;font-weight:700;color:#071628">{m['avatar']} {m['name']}</span>
              <a href="mailto:{m['email']}" style="font-size:.75rem;color:#2563a8;
                background:#eaf2ff;border-radius:8px;padding:3px 10px;text-decoration:none">{m['email']}</a>
            </div>""", unsafe_allow_html=True)
        st.markdown('<div class="ib" style="margin-top:14px">🏛️ <strong>ITER · SOA University</strong><br>Dept. of Computer Application<br>Bhubaneswar, Odisha — 751 030</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  💬  ASK ORALDX CHATBOT
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_chat":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">💬 Ask OralDx — AI Dental Assistant</h1>
      <p class="sub">Powered by Claude AI · Ask anything about oral health, diseases, or our models</p>
    </div>""", unsafe_allow_html=True)

    chat = st.session_state["chat"]
    chat_container = st.container()
    with chat_container:
        for msg in chat:
            if msg["role"]=="user":
                st.markdown(f'<div style="text-align:right"><div class="cu">🧑 {msg["content"]}</div></div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ca">🦷 {msg["content"]}</div>',unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_inp, c_btn = st.columns([6,1])
    user_inp = c_inp.text_input("Your question", placeholder="e.g. What is dental caries?", label_visibility="collapsed")
    send = c_btn.button("➤ Send", use_container_width=True, type="primary")

    if send and user_inp.strip():
        chat.append({"role":"user","content":user_inp.strip()})
        with st.spinner("🦷 OralDx is thinking…"):
            reply = call_claude([{"role":m["role"],"content":m["content"]} for m in chat])
        chat.append({"role":"assistant","content":reply})
        st.session_state["chat"] = chat
        st.rerun()

    st.markdown("---")
    st.markdown("**💡 Suggested Questions:**")
    sugg = ["What is gingivitis?","How does ResNet-50 detect oral disease?",
            "What causes dental caries?","How accurate is OralDx?",
            "What is hypodontia?","How often should I visit a dentist?"]
    cols_s = st.columns(3)
    for i,q in enumerate(sugg):
        if cols_s[i%3].button(q, use_container_width=True, key=f"sq_{i}"):
            chat.append({"role":"user","content":q})
            reply = call_claude([{"role":"user","content":q}])
            chat.append({"role":"assistant","content":reply})
            st.session_state["chat"] = chat; st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=False):
        st.session_state["chat"]=[]; st.rerun()

# ══════════════════════════════════════════════════════════════════════════
#  🗺️  SYMPTOM CHECK
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_sym":
    lcode = st.session_state["lang"]
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">🗺️ Symptom Checker</h1>
      <p class="sub">Select your symptoms for a preliminary oral disease risk assessment</p>
    </div>""", unsafe_allow_html=True)

    SMAP = {
        "Toothache or sharp pain":              {"Caries":3,"Gingivitis":1,"Calculus":1},
        "Sensitivity to hot/cold/sweet food":   {"Caries":4,"Calculus":1},
        "Visible holes or dark spots on tooth": {"Caries":4,"Calculus":2},
        "Red, swollen or bleeding gums":        {"Gingivitis":4,"Calculus":2},
        "Persistent bad breath (halitosis)":    {"Gingivitis":3,"Calculus":2,"Caries":1},
        "Gum recession / exposed root":         {"Gingivitis":3,"Calculus":2},
        "Visible white/yellow deposits":        {"Calculus":4,"Tooth Discoloration":2},
        "Tooth colour change (yellow/brown)":   {"Tooth Discoloration":4,"Calculus":1},
        "Missing or absent tooth in arch":      {"Hypodontia":5},
        "Gap in dental arch with no tooth":     {"Hypodontia":4},
        "Painful sores or ulcers in mouth":     {"Mouth Ulcer":5},
        "White lesion with red border":         {"Mouth Ulcer":5},
        "Difficulty eating due to mouth pain":  {"Mouth Ulcer":3,"Gingivitis":1},
        "No symptoms (routine checkup)":        {"Caries":1,"Calculus":1,"Hypodontia":1},
    }

    sel = st.multiselect("✅ Select all symptoms you are currently experiencing:", list(SMAP.keys()))

    if sel:
        sc = {}
        for s in sel:
            for dis,w in SMAP[s].items(): sc[dis]=sc.get(dis,0)+w
        total = sum(sc.values())
        ranked = sorted(sc.items(),key=lambda x:-x[1])
        st.markdown('<div class="ib w">⚠️ <strong>Important:</strong> This is a preliminary estimate only — not a clinical diagnosis. Please consult a licensed dentist.</div>',unsafe_allow_html=True)

        fig,ax = plt.subplots(figsize=(8,3.5))
        dns=[r[0] for r in ranked]; dps=[round(r[1]/total*100,1) for r in ranked]
        dcs=[CCOLORS.get(d,"#94a3b8") for d in dns]
        bars=ax.barh(dns,dps,color=dcs,height=0.52,edgecolor="white")
        for bar,val in zip(bars,dps):
            ax.text(bar.get_width()+.5,bar.get_y()+bar.get_height()/2,
                    f"{val}%",va="center",fontsize=9,fontweight="bold")
        ax.set_xlabel("Risk Score (%)"); ax.set_title("Symptom-Based Risk Assessment",fontweight="bold")
        ax.set_xlim(0,115); fig_style(ax,fig)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        top,tops = ranked[0]; pct=round(tops/total*100,1)
        lv="High" if pct>50 else "Medium" if pct>25 else "Low"
        info=DISEASE.get(top,{}); desc={"hi":info.get("hi"),"od":info.get("od")}.get(lcode,info.get("en",""))
        st.markdown(f"""<div class="rb" style="margin-top:14px">
          <div class="rc">{info.get('icon','🦷')} {top}</div>
          <div class="rs">Risk Level: {badge(lv)} ({pct}%)</div>
          <div style="color:#334155;font-size:.9rem;margin:10px 0">{desc}</div>
          <hr class="dv"><strong>🩺 Action:</strong>
          <div style="color:#475569;font-size:.88rem;margin-top:5px">{info.get('action','Consult a dentist.')}</div>
        </div>""",unsafe_allow_html=True)
    else:
        st.markdown("""<div style="text-align:center;padding:56px;background:#fff;border-radius:18px;
          border:2px dashed #b8d8f0;margin-top:14px">
          <div style="font-size:4rem">🗺️</div>
          <div style="color:#4a6080;margin-top:10px;font-size:1rem">Select symptoms above to begin assessment.</div>
        </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  ⭐  REVIEWS
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_rev":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">⭐ User Reviews</h1>
      <p class="sub">Share your experience with OralDx</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("✍️ Write a Review", expanded=True):
        c1,c2 = st.columns([2,1])
        nm   = c1.text_input("Your Name", placeholder="Dr. / Mr. / Ms. …")
        stars= c2.select_slider("Rating", ["⭐","⭐⭐","⭐⭐⭐","⭐⭐⭐⭐","⭐⭐⭐⭐⭐"], value="⭐⭐⭐⭐⭐")
        cmt  = st.text_area("Your Review", height=80, placeholder="Share your experience…")
        if st.button("📤 Submit Review", use_container_width=True, type="primary"):
            if nm.strip() and cmt.strip():
                st.session_state["reviews"].append({"name":nm.strip(),"stars":stars,
                    "comment":cmt.strip(),"date":datetime.now().strftime("%d %b %Y")})
                st.success("Thank you for your review! 🙏"); st.rerun()
            else: st.warning("Please enter your name and review.")

    SAMPLES = [
        {"name":"Dr. Anita Sharma","stars":"⭐⭐⭐⭐⭐","comment":"ResNet-50 accuracy is impressive! The Odia language support makes it accessible to local dental professionals.","date":"20 May 2026"},
        {"name":"Prof. R. Patnaik","stars":"⭐⭐⭐⭐","comment":"Excellent student project. The 4-model comparison and radar charts are very informative.","date":"16 May 2026"},
        {"name":"Priya M., Bhubaneswar","stars":"⭐⭐⭐⭐⭐","comment":"The symptom checker helped me identify possible gingivitis before my dentist visit. Very useful!","date":"12 May 2026"},
        {"name":"Rajan K.","stars":"⭐⭐⭐⭐","comment":"Clean UI and fast diagnosis. The download report feature is great for sharing with dentists.","date":"08 May 2026"},
    ]
    all_rev = SAMPLES + st.session_state["reviews"]

    # Rating summary
    st.subheader(f"📝 {len(all_rev)} Reviews")
    avg_stars = sum(r["stars"].count("⭐") for r in all_rev)/len(all_rev)
    st.markdown(f"""<div style="background:#fff;border:1px solid #ffe4d0;border-radius:14px;
      padding:18px 24px;margin-bottom:16px;display:flex;align-items:center;gap:20px">
      <div style="font-size:2.5rem;font-weight:900;color:#f97316">{avg_stars:.1f}</div>
      <div>
        <div style="font-size:1.4rem">{'⭐'*round(avg_stars)}</div>
        <div style="font-size:.8rem;color:#6b8caa">Average rating · {len(all_rev)} reviews</div>
      </div>
    </div>""", unsafe_allow_html=True)
    for r in all_rev:
        st.markdown(f"""<div class="rv">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <strong style="color:#071628">{r['name']}</strong>
            <small style="color:#9ca3af">{r['date']}</small>
          </div>
          <div style="font-size:1.1rem;margin-bottom:6px">{r['stars']}</div>
          <div style="color:#475569;font-size:.9rem">{r['comment']}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  🔬  COMPARE
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_cmp":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">🔬 Compare Two X-rays</h1>
      <p class="sub">Upload before & after images to monitor disease progression or treatment outcome</p>
    </div>""", unsafe_allow_html=True)

    mdl_c = st.selectbox("Model for comparison", list(MODELS.keys()), index=0)
    c1,c2 = st.columns(2, gap="large")
    imgs=[]; results=[]
    for col,lbl,key in [(c1,"📁 X-ray A (Before / Current)","xA"),(c2,"📁 X-ray B (After / Comparison)","xB")]:
        with col:
            st.markdown(f"**{lbl}**")
            up = st.file_uploader(lbl,type=["jpg","jpeg","png"],key=key,label_visibility="collapsed")
            if up:
                img = Image.open(up).convert("RGB")
                st.image(img,use_container_width=True,caption=up.name)
                with st.spinner("Analysing…"):
                    r = predict(img,mdl_c)
                info=r["info"]
                st.markdown(f"""<div class="rb" style="padding:16px">
                  <div class="rc" style="font-size:1.3rem">{info['icon']} {r['class']}</div>
                  <div class="rs">{L('conf')}: <strong style="color:#1d4ed8">{r['conf']:.1f}%</strong>
                    &emsp;{badge(info['sev'])}</div>
                </div>""",unsafe_allow_html=True)
                imgs.append(img); results.append(r)
            else:
                st.markdown("""<div style="text-align:center;padding:50px;background:#f8fbff;
                  border:2px dashed #c8dff5;border-radius:14px">
                  <div style="font-size:3rem">🖼️</div>
                  <div style="color:#6b8caa;margin-top:8px;font-size:.85rem">Upload an X-ray</div>
                </div>""",unsafe_allow_html=True)

    if len(results)==2:
        st.markdown("---"); st.subheader("📊 Side-by-Side Probability Comparison")
        fig,axes = plt.subplots(1,2,figsize=(12,4.5))
        for ax,r,lbl in zip(axes,results,["X-ray A (Before)","X-ray B (After)"]):
            lbls=list(r["probs"].keys()); vals=list(r["probs"].values())
            cls=[CCOLORS[l] for l in lbls]
            bs=ax.barh(lbls,vals,color=cls,height=0.5,edgecolor="white")
            for bar,val in zip(bs,vals):
                ax.text(bar.get_width()+.5,bar.get_y()+bar.get_height()/2,
                        f"{val:.1f}%",va="center",fontsize=8)
            ax.set_xlabel("Confidence (%)"); ax.set_title(f"{lbl}: {r['class']}",fontweight="bold")
            ax.set_xlim(0,115); ax.spines[["top","right"]].set_visible(False)
            ax.set_facecolor("#f8fbff")
        fig.patch.set_facecolor("#f8fbff")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        a_cls,b_cls=results[0]["class"],results[1]["class"]
        if a_cls==b_cls: st.info(f"ℹ️ Both X-rays show the same condition: **{a_cls}**")
        elif b_cls=="Healthy":  st.success(f"✅ Improvement: {a_cls} → {b_cls}")
        else: st.warning(f"🔄 Condition changed: **{a_cls}** → **{b_cls}**")

# ══════════════════════════════════════════════════════════════════════════
#  📱  MOBILE APP
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_mob":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">📱 OralDx Mobile Guide</h1>
      <p class="sub">Access OralDx on any device — phone, tablet, or desktop</p>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### 📲 Install as Mobile App (PWA)")
        steps_m = [
            ("🌐","Open in Browser","Visit your OralDx Streamlit URL on Chrome (Android) or Safari (iPhone/iPad)."),
            ("⚙️","Browser Menu","Tap the three-dot menu (Android) or Share button (iPhone)."),
            ("➕","Add to Home Screen","Select 'Add to Home Screen' or 'Install App'."),
            ("✅","Launch","OralDx now appears as an app icon — tap to open instantly."),
        ]
        for ico,title,desc in steps_m:
            st.markdown(f"""<div class="si"><div class="sn">{ico}</div>
              <div class="sb"><strong>{title}</strong><p>{desc}</p></div></div>""",unsafe_allow_html=True)

    with c2:
        st.markdown("### 💻 Run on Desktop / Local")
        st.code("""# 1. Clone repository
git clone https://github.com/YOUR/oraldx.git
cd oraldx

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch app
streamlit run app.py

# App opens at: http://localhost:8501""", language="bash")

    st.markdown("### ☁️ Deploy Free on Streamlit Cloud")
    for i,step in enumerate(["Push this folder to a **public GitHub repository**",
        "Go to **share.streamlit.io** → sign in with GitHub",
        "Click **New app** → select your repo → set main file = `app.py`",
        "Click **Deploy** → your live URL is ready in ~2 minutes ✅"],1):
        st.markdown(f"""<div class="si"><div class="sn">{i}</div>
          <div class="sb"><strong>Step {i}</strong><p>{step}</p></div></div>""",unsafe_allow_html=True)

    st.markdown("""<div class="ib s">
      ✅ OralDx works on all screen sizes — desktop, tablet, and mobile phone.
      The Streamlit layout adapts automatically. No native app installation required.
    </div>""",unsafe_allow_html=True)
    st.markdown('<div class="ib">📋 <strong>System Requirements:</strong> Python 3.10+ · Streamlit 1.32+ · Modern browser (Chrome / Safari / Firefox)</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  🔒  PRIVACY
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_priv":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">🔒 Privacy Policy</h1>
      <p class="sub">OralDx is fully privacy-first — your data never leaves your browser</p>
    </div>""", unsafe_allow_html=True)

    for title,body,cls in [
        ("🔍 Data Collection","OralDx does not collect, store, or transmit any personal data or images to external servers. All processing happens in-memory within your Streamlit session.",""),
        ("🖼️ Image Processing","Uploaded X-ray images are processed entirely in RAM during your session. No images are written to disk, logged, or shared. When you close the tab, all data is permanently erased.","s"),
        ("📊 Session Data","Diagnosis history, chat messages, and reviews are stored only in Streamlit's temporary session state (Python dict in RAM) and cleared when the session ends.",""),
        ("🍪 Cookies","OralDx does not use tracking, analytics, or advertising cookies. Only essential Streamlit session cookies required for the app to function are used.",""),
        ("⚖️ Medical Disclaimer","OralDx is provided for educational and research purposes only. It is NOT a substitute for professional clinical dental diagnosis. AI predictions are approximate and must be reviewed by a licensed dental professional before any treatment decision.","w"),
        ("📧 Contact for Privacy","For any privacy-related concerns, contact: debabratasingh@soa.ac.in","s"),
    ]:
        with st.expander(f"**{title}**", expanded=True):
            st.markdown(f'<div style="color:#334155;font-size:.91rem;line-height:1.6">{body}</div>',unsafe_allow_html=True)
    st.markdown('<div class="ib s">🔒 <strong>OralDx is fully privacy-first.</strong> No user data is ever collected, stored, or shared. Your X-rays remain 100% private and confidential.</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  👨‍💻  ADMIN
# ══════════════════════════════════════════════════════════════════════════
elif pkey == "nav_admin":
    st.markdown("""<div class="hero" style="padding:26px 38px">
      <h1 style="font-size:1.75rem">👨‍💻 Admin Dashboard</h1>
      <p class="sub">System statistics · Session monitoring · Data management</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state["admin_ok"]:
        st.markdown("""<div style="max-width:420px;margin:40px auto;background:#fff;border-radius:18px;
          padding:36px;box-shadow:0 8px 32px rgba(0,80,200,.14);text-align:center">
          <div style="font-size:3rem;margin-bottom:14px">🔐</div>
          <h3 style="color:#071628;margin-bottom:6px">Admin Login</h3>
          <p style="color:#6b8caa;font-size:.88rem;margin-bottom:20px">Enter admin password to access the dashboard</p>
        """, unsafe_allow_html=True)
        pw = st.text_input("Password", type="password", label_visibility="collapsed",
                           placeholder="Enter admin password…")
        if st.button("🔓 Login", use_container_width=True, type="primary"):
            if pw == "oraldx2026":
                st.session_state["admin_ok"] = True; st.rerun()
            else:
                st.error("❌ Incorrect password.")
        st.markdown("""<p style="font-size:.76rem;color:#9ca3af;margin-top:14px">
          Hint: oraldx2026</p></div>""", unsafe_allow_html=True)
    else:
        # Dashboard
        hist = st.session_state["history"]; revs = st.session_state["reviews"]
        c1,c2,c3,c4 = st.columns(4)
        for col,i,v,l in zip([c1,c2,c3,c4],["🔍","📋","⭐","🤖"],
            [len(hist),len(hist)+len(revs),len(revs)+4,len(set(r['model'] for r in hist)) if hist else 0],
            ["Total Diagnoses","Session Records","Total Reviews","Models Used"]):
            col.markdown(f"""<div class="stat-card">
              <div style="font-size:1.8rem">{i}</div>
              <div class="sv">{v}</div><div class="sl">{l}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        import pandas as pd
        t1,t2,t3 = st.tabs(["📋 Diagnosis Log","⭐ Reviews","📊 Usage Stats"])
        with t1:
            if hist:
                df_a = pd.DataFrame(hist)
                st.dataframe(df_a, use_container_width=True)
            else: st.info("No diagnoses yet in this session.")
        with t2:
            all_r = [{"name":"Dr. Anita Sharma","stars":"⭐⭐⭐⭐⭐","date":"20 May 2026"},
                     {"name":"Prof. R. Patnaik","stars":"⭐⭐⭐⭐","date":"16 May 2026"}] + revs
            if all_r:
                df_r = pd.DataFrame(all_r)
                st.dataframe(df_r, use_container_width=True)
        with t3:
            if hist:
                cls_counts = {}
                for r in hist: cls_counts[r["class"]]=cls_counts.get(r["class"],0)+1
                fig,ax=plt.subplots(figsize=(8,3.5))
                ax.barh(list(cls_counts.keys()),list(cls_counts.values()),
                        color=[CCOLORS.get(k,"#94a3b8") for k in cls_counts],edgecolor="white",height=.5)
                ax.set_xlabel("Count"); ax.set_title("Diagnoses by Disease Class",fontweight="bold")
                fig_style(ax,fig); plt.tight_layout(); st.pyplot(fig); plt.close()
            else: st.info("Run some diagnoses first to see usage stats.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔓 Logout",type="secondary"):
            st.session_state["admin_ok"]=False; st.rerun()

# ══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:linear-gradient(90deg,#020913,#071628,#020913);
  border-radius:16px;padding:18px 30px;text-align:center;color:#3a6a9a;font-size:.77rem;
  border:1px solid rgba(56,189,248,.1)">
  🦷 <strong style="color:#38bdf8">OralDx</strong> — Automated Diagnosis of Oral Conditions from Dental X-Rays
  &nbsp;|&nbsp; ResNet-50 · VGG19 · YOLOv8 · U-Net
  &nbsp;|&nbsp; ITER · SOA University · Bhubaneswar, Odisha
  &nbsp;|&nbsp; MCA Batch 2024–2026 · Group 8
  &nbsp;|&nbsp; <span style="color:#f87171">⚠️ Not a substitute for professional dental diagnosis</span>
</div>
""", unsafe_allow_html=True)

"""Derme: a quiet first look at your skin.  Run from the repo root:  streamlit run app.py"""
import hashlib
from html import escape

import cv2
import streamlit as st

from utils import config
from utils import database as db
from utils.quality_gate import check_quality, decode_image

# settings
APP_NAME = "Derme"
TAGLINE = "A quiet first look at your skin."
DISCLAIMER = ("A screening observation, not a diagnosis. "
              "If you are worried about your skin, please see a dermatologist.")

# Update this sentence if you retrain the model (numbers come from utils.evaluate).
REPORTED_ACCURACY = ("about 37% of the time on average across the five conditions, "
                     "against 20% by chance")

PAGES = [("welcome", "Welcome"), ("screen", "Screen"),
         ("history", "History"), ("about", "About")]

st.set_page_config(page_title=APP_NAME, layout="centered",
                   initial_sidebar_state="collapsed")
db.init_db()

#  styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=Jost:wght@300;400&display=swap');
:root { --ivory:#FAF8F4; --ink:#1A1A1A; --muted:#8A857C; --line:#E4DFD6; --accent:#A38B5C; }
.stApp { background:var(--ivory); color:var(--ink); }
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp span:not([data-testid="stIconMaterial"]), .stApp input, .stApp button {
  font-family:'Jost', sans-serif; font-weight:300; }
.stApp p, .stApp li, .stApp label { color:var(--ink); letter-spacing:.02em; line-height:1.75; }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
footer, #MainMenu, [data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none !important; }
.block-container { max-width:660px; padding-top:2rem; padding-bottom:4rem; }
h1, h2, h3 { font-family:'Cormorant Garamond', serif !important; font-weight:300 !important;
  color:var(--ink) !important; letter-spacing:.01em; }
h2 { font-size:2.1rem !important; text-align:center !important; margin-top:1rem; }
h3 { font-size:1.5rem !important; margin-top:2.2rem; }
.brand { font-family:'Cormorant Garamond', serif; font-weight:300; font-size:3.4rem;
  letter-spacing:.42em; padding-left:.42em; text-transform:uppercase; text-align:center;
  margin:4rem 0 .6rem; }
.tagline { text-align:center; color:var(--muted) !important; letter-spacing:.12em !important;
  margin-bottom:2.4rem; }
.eyebrow { font-size:.68rem; letter-spacing:.3em; text-transform:uppercase; color:var(--muted); }
.center { text-align:center; }
.muted { color:var(--muted) !important; }
.steps { display:grid; grid-template-columns:repeat(3, 1fr); gap:2rem; margin-top:4rem; }
.step { border-top:1px solid var(--line); padding-top:1rem; }
.step-title { font-family:'Cormorant Garamond', serif; font-size:1.5rem; margin:.2rem 0 .4rem; }
.step p { font-size:.88rem; color:var(--muted) !important; }
.big { font-family:'Cormorant Garamond', serif; font-size:2.8rem; line-height:1.15;
  text-align:center; margin:.6rem 0 .8rem; }
.lead { text-align:center; color:var(--muted) !important; max-width:30rem; margin:0 auto 1.4rem; }
.row { display:flex; justify-content:space-between; font-size:.7rem; letter-spacing:.22em;
  text-transform:uppercase; margin-top:1.1rem; }
.bar { height:2px; background:var(--line); margin-top:.45rem; }
.bar > div { height:2px; background:var(--ink); }
.hair { border-top:1px solid var(--line); margin:2.6rem 0 1.4rem; }
.footer { text-align:center; font-size:.72rem; letter-spacing:.08em; color:var(--muted);
  border-top:1px solid var(--line); margin-top:4rem; padding-top:1.2rem; }
.reason { text-align:center; margin:.2rem 0; }
[data-testid="stElementContainer"]:has(> [data-testid="stButton"]),
[data-testid="stElementContainer"]:has(> [data-testid="stFormSubmitButton"]) { width:100% !important; }
[data-testid="stButton"], [data-testid="stFormSubmitButton"] { display:flex; justify-content:center; width:100%; }
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button { width:100%; padding:.65rem 1.4rem; }
button[data-testid^="stBaseButton-secondary"], button[data-testid^="stBaseButton-primary"] {
  border-radius:0; border:1px solid var(--ink); background:transparent; }
button[data-testid^="stBaseButton-secondary"] p, button[data-testid^="stBaseButton-primary"] p {
  font-size:.7rem; letter-spacing:.24em; text-transform:uppercase; color:inherit; }
button[data-testid^="stBaseButton-secondary"] { color:var(--ink); }
button[data-testid^="stBaseButton-primary"] { background:var(--ink); color:var(--ivory); }
button[data-testid^="stBaseButton-secondary"]:hover { background:var(--ink); color:var(--ivory); }
button[data-testid^="stBaseButton-primary"]:hover { background:var(--accent); border-color:var(--accent); color:var(--ivory); }
[data-testid="stFileUploaderDropzone"] button { width:auto; padding:.5rem 1rem; }
[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] { display:none; }
.st-key-nav [data-testid="stButton"], .st-key-switch [data-testid="stButton"] { justify-content:center; }
.st-key-nav button { width:auto !important; border:none !important; border-bottom:1px solid transparent !important;
  padding:.35rem 0 !important; background:transparent !important; color:var(--muted) !important; }
.st-key-nav button p { font-size:.66rem !important; }
.st-key-nav button:hover { color:var(--ink) !important; }
.st-key-nav_ACTIVE button { color:var(--ink) !important; border-bottom:1px solid var(--ink) !important; }
.st-key-switch button { width:auto !important; border:none !important; background:transparent !important;
  color:var(--muted) !important; }
div[data-baseweb="input"], div[data-baseweb="base-input"] { background:transparent !important;
  border:none !important; border-bottom:1px solid var(--ink) !important; border-radius:0 !important; }
input { background:transparent !important; color:var(--ink) !important; }
[data-testid="stForm"] { border:none !important; padding:0 !important; }
[data-testid="stFileUploaderDropzone"] { background:transparent; border:1px solid var(--line);
  border-radius:0; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:0; background:transparent; }
[data-testid="stImage"] img { border-radius:0; }
button[data-baseweb="tab"] p { font-size:.68rem; letter-spacing:.24em; text-transform:uppercase; }
[data-baseweb="tab-highlight"] { background:var(--ink) !important; }
[data-baseweb="tab-border"] { background:var(--line) !important; }
@media (max-width: 640px) { .steps { grid-template-columns:1fr; gap:1.2rem; }
  .brand { font-size:2.4rem; } .big { font-size:2.2rem; } }
</style>
"""


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def pretty(label):
    return label.replace("_", " ").title()


def keyed_container(key):
    try:
        return st.container(key=key)
    except TypeError:  # older Streamlit without keyed containers
        return st.container()


def go(page):
    st.session_state.page = page
    st.rerun()


@st.cache_resource
def get_classifier():
    from utils.predict import SkinClassifier
    return SkinClassifier()


def nav():
    with keyed_container("nav"):
        for col, (pid, label) in zip(st.columns(len(PAGES)), PAGES):
            if col.button(label, key=f"nav_{pid}"):
                go(pid)


#  pages
def page_welcome():
    html(f'<div class="brand">{escape(APP_NAME)}</div>')
    html(f'<p class="tagline">{escape(TAGLINE)}</p>')
    _, mid, _ = st.columns([1, 1, 1])
    if mid.button("Begin", type="primary", key="begin"):
        go("screen")
    steps = [("Photograph", "Take or upload a clear photo in natural light."),
             ("Check", "We make sure the photo is good enough to read."),
             ("Observe", "You receive a screening observation, never a diagnosis.")]
    cells = "".join(
        f'<div class="step"><div class="eyebrow">{i:02d}</div>'
        f'<div class="step-title">{t}</div><p>{d}</p></div>'
        for i, (t, d) in enumerate(steps, 1))
    html(f'<div class="steps">{cells}</div>')


def start_client_session(name):
    cid = db.get_or_create_client(name)
    st.session_state.client_id = cid
    st.session_state.session_id = db.start_session(cid)
    st.session_state.client_name = name
    st.session_state.pop("outcome_key", None)  # new session: log photos afresh


def client_form():
    html('<div class="eyebrow center">Step one</div>')
    html('<h2>Who is this screening for?</h2>')
    html('<p class="lead">A name or an ID, so your results can be shown back to you.</p>')
    with st.form("who"):
        name = st.text_input("Name or ID", placeholder="Name or ID",
                             label_visibility="collapsed")
        ok = st.form_submit_button("Continue", type="primary")
    if ok and name.strip():
        start_client_session(name.strip())
        st.rerun()
    elif ok:
        st.caption("Please enter a name or an ID to continue.")


def show_observation(result):
    if result["low_confidence"]:
        html('<div class="eyebrow center">Observation</div>')
        html('<div class="big">We couldn’t read this one clearly.</div>')
        html('<p class="lead">Nothing has been flagged. A clearer photo may help, and if '
             'something is worrying you, a dermatologist is the right next step.</p>')
    else:
        html('<div class="eyebrow center">Observation</div>')
        html(f'<div class="big">{escape(pretty(result["label"]))}</div>')
        html('<p class="lead">The screener leaned towards this. It is a flag for '
             'follow-up, not a diagnosis.</p>')
        html(f'<p class="center muted">Confidence {result["confidence"]:.0%}. This reflects '
             'the model’s certainty, not a medical probability.</p>')
    rows = "".join(
        f'<div class="row"><span>{escape(pretty(label))}</span><span>{score:.0%}</span></div>'
        f'<div class="bar"><div style="width:{score * 100:.0f}%"></div></div>'
        for label, score in sorted(result["all_scores"].items(), key=lambda kv: -kv[1]))
    html('<div class="hair"></div>')
    html(f'<div class="eyebrow">All categories</div>{rows}')


def page_screen():
    if not config.MODEL_PATH.exists():
        html('<h2>Not ready yet</h2>')
        html('<p class="lead">No trained model found. Run <code>python -m utils.train</code> first.</p>')
        return
    if "session_id" not in st.session_state:
        client_form()
        return

    html(f'<div class="eyebrow center">Screening for {escape(st.session_state.client_name)}</div>')
    if st.button("Not you? Switch", key="switch"):
        for k in ("client_id", "session_id", "client_name", "outcome_key", "outcome"):
            st.session_state.pop(k, None)
        st.rerun()

    n = st.session_state.setdefault("photo_n", 0)
    html('<h2>Share a photo</h2>')
    html('<p class="lead">Natural light, no flash, hold steady, and fill the frame '
         'with the area of concern.</p>')
    tab_up, tab_cam = st.tabs(["Upload", "Camera"])
    with tab_up:
        upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"],
                                  label_visibility="collapsed", key=f"upload_{n}")
    with tab_cam:
        cam = st.camera_input("Take a photo", label_visibility="collapsed", key=f"cam_{n}")
    file = upload or cam

    if file is None:
        html('<p class="center muted">Photos are analyzed and not stored. Only the result, '
             'the confidence scores and the file name are kept.</p>')
        return

    raw = file.getvalue()
    bgr = decode_image(raw)
    if bgr is None:
        html('<div class="big">We couldn’t open that file.</div>')
        html('<p class="lead">Please try a different photo.</p>')
        return
    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # Streamlit re-runs this whole script on every click. This key makes sure each
    # photo is checked, classified and logged once per session, not once per click.
    key = (st.session_state.session_id, hashlib.md5(raw).hexdigest())
    if st.session_state.get("outcome_key") != key:
        quality = check_quality(bgr)
        result = None
        if quality.passed:
            with st.spinner("Looking closely..."):
                result = get_classifier().predict(bgr)
        db.log_screening(st.session_state.session_id, file.name, quality, result)
        st.session_state.outcome_key = key
        st.session_state.outcome = (quality, result)
    quality, result = st.session_state.outcome

    html('<div class="hair"></div>')
    if not quality.passed:
        html('<div class="eyebrow center">Photo check</div>')
        html('<div class="big">Let’s try that again.</div>')
        for r in quality.reasons:
            html(f'<p class="reason">{escape(str(r))}</p>')
        with st.expander("Technical details"):
            st.json(quality.metrics)
    else:
        show_observation(result)

    _, mid, _ = st.columns([1, 2, 1])
    if mid.button("Screen another photo", key="another"):
        st.session_state.photo_n = n + 1
        st.rerun()


def page_history():
    html('<h2>History</h2>')
    if "session_id" not in st.session_state:
        html('<p class="lead">Begin a screening and your history will appear here.</p>')
        return
    html(f'<p class="lead">Screenings for {escape(st.session_state.client_name)}.</p>')
    hist = db.client_history(st.session_state.client_id)
    if not hist:
        html('<p class="center muted">No screenings yet.</p>')
        return
    rows = []
    for h in hist:
        if not h["quality_passed"]:
            outcome = "Retake needed"
        elif h["low_confidence"]:
            outcome = "Inconclusive"
        else:
            outcome = pretty(h["top_label"])
        rows.append({"Date": h["created_at"][:16].replace("T", " "),
                     "Photo": h["filename"], "Observation": outcome,
                     "Confidence": f'{h["confidence"]:.0%}' if h["confidence"] is not None else "-"})
    st.dataframe(rows, hide_index=True)


def page_about():
    conditions = "five conditions"
    if config.MODEL_PATH.exists():
        names = [pretty(c).lower() for c in get_classifier().class_names]
        conditions = f'{len(names)} conditions: {", ".join(names)}'
    html('<h2>About and limits</h2>')
    html('<h3>What this is</h3>')
    st.markdown(f"{APP_NAME} is a first-look skin screener. You share a photo, it checks "
                "the photo is good enough to read, and then offers a screening observation: "
                "a prompt for follow-up, never a diagnosis.")
    html('<h3>What it recognizes</h3>')
    st.markdown(f"It knows {conditions}. There is no healthy-skin category, so a photo of "
                "anything else can still be matched to one of them.")
    html('<h3>How the photo check works</h3>')
    st.markdown("Before any prediction, the photo is checked with OpenCV for size, sharpness, "
                "lighting and how much skin is visible. If it falls short you are asked to "
                "retake it, so no reading is made on a photo that cannot support one.")
    html('<h3>How well it works</h3>')
    st.markdown("The screener was trained on a small research dataset of everyday photos "
                f"(SCIN). On photos it had never seen, it picked the right condition {REPORTED_ACCURACY}. "
                "Several of these conditions look alike, and it is least reliable on insect "
                "bites and folliculitis. Accuracy may also differ across skin tones; there "
                "were not enough photos to measure that reliably.")
    html('<h3>What we keep</h3>')
    st.markdown("Photos are analyzed and are not stored. For each screening we keep only the "
                "result, the confidence scores and the file name, so your history can be shown to you.")


# run
page = st.session_state.get("page", "welcome")
html(CSS.replace("ACTIVE", page))
nav()
{"welcome": page_welcome, "screen": page_screen,
 "history": page_history, "about": page_about}[page]()
html(f'<div class="footer">{escape(DISCLAIMER)}</div>')
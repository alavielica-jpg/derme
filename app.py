"""Streamlit UI. Run from the repo root:  streamlit run app.py"""
import hashlib

import cv2
import streamlit as st

from utils import config
from utils import database as db
from utils.quality_gate import check_quality, decode_image

st.set_page_config(page_title="Virtual Skin Screener", page_icon="🔍")
db.init_db()


@st.cache_resource
def get_classifier():
    from utils.predict import SkinClassifier
    return SkinClassifier()


st.title("Virtual Skin Screener")
st.warning(
    "**This is a screening tool, not a diagnosis.** It can flag a photo for follow-up "
    "but it can be wrong. It is not a substitute for a dermatologist or doctor. "
    "If a spot is changing, bleeding, painful, or worrying you, see a professional."
)

# ---- Sidebar: client / session ----
with st.sidebar:
    st.header("Client")
    name = st.text_input("Client name or ID")
    if st.button("Start session") and name.strip():
        cid = db.get_or_create_client(name.strip())
        st.session_state.client_id = cid
        st.session_state.session_id = db.start_session(cid)
        st.session_state.client_name = name.strip()
        st.session_state.pop("outcome_key", None)  # new session: log photos afresh
    if "session_id" in st.session_state:
        st.success(f"Session active: {st.session_state.client_name}")
        if st.checkbox("Show history"):
            hist = db.client_history(st.session_state.client_id)
            if hist:
                st.dataframe(hist)
            else:
                st.caption("No screenings yet.")

if "session_id" not in st.session_state:
    st.info("Enter a client name or ID in the sidebar and start a session.")
    st.stop()

# ---- Model availability ----
if not config.MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m utils.train` first.")
    st.stop()
clf = get_classifier()
conditions = ", ".join(c.replace("_", " ") for c in clf.class_names)

# ---- Upload ----
tab_up, tab_cam = st.tabs(["Upload", "Camera"])
with tab_up:
    upload = st.file_uploader("Upload a clear, well-lit photo of the area",
                              type=["jpg", "jpeg", "png"])
with tab_cam:
    cam = st.camera_input("Or take a photo")
file = upload or cam

if file is None:
    st.caption("Tips: natural light, no flash, hold steady, fill the frame with the area of concern.")
    st.stop()

raw = file.getvalue()
bgr = decode_image(raw)
if bgr is None:
    st.error("That file couldn't be read as an image. Please try a different photo.")
    st.stop()
st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Your photo")

# ---- Quality gate + classification, done once per photo ----
# Streamlit re-runs this whole script on every click (e.g. ticking "Show history").
# Without this check, each click would log the same photo again.
key = (st.session_state.session_id, hashlib.md5(raw).hexdigest())
if st.session_state.get("outcome_key") != key:
    quality = check_quality(bgr)
    result = None
    if quality.passed:
        with st.spinner("Analyzing..."):
            result = clf.predict(bgr)
    db.log_screening(st.session_state.session_id, file.name, quality, result)
    st.session_state.outcome_key = key
    st.session_state.outcome = (quality, result)
quality, result = st.session_state.outcome

if not quality.passed:
    st.error("Please retake the photo:")
    for r in quality.reasons:
        st.write(f"• {r}")
    with st.expander("Technical details"):
        st.json(quality.metrics)
    st.stop()

# ---- Result ----
st.subheader("Screening result")
if result["low_confidence"]:
    st.warning(
        f"**Inconclusive.** The model isn't confident enough to flag anything "
        f"(top guess was below {result['threshold']:.0%} confidence). "
        "Try a clearer photo, or speak to a professional if you're concerned."
    )
else:
    pretty = result["label"].replace("_", " ").title()
    st.info(f"**Flagged for follow-up: {pretty}** ({result['confidence']:.0%} confidence). "
            "This is not a diagnosis.")

st.write("Model confidence by category:")
for label, score in sorted(result["all_scores"].items(), key=lambda kv: -kv[1]):
    st.progress(score, text=f"{label.replace('_', ' ').title()}: {score:.0%}")

st.caption(
    f"This tool only knows {len(clf.class_names)} conditions ({conditions}) and has no "
    "'healthy skin' category, so a photo of anything else can still be matched to one of "
    "them. It was trained on a small research dataset, so accuracy may be lower on "
    "everyday photos and may vary across skin tones."
)
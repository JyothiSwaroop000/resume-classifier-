import re
import pickle
import numpy as np
import streamlit as st
from pypdf import PdfReader

# ---------------- Page config ----------------
st.set_page_config(page_title="Resume Category Classifier", page_icon="📄", layout="centered")

st.title("📄 Resume Category Classifier")
st.write("Upload a resume PDF — the model predicts the job category (NLP: TF-IDF + SVC).")

# Minimum extracted-text length below which predictions are considered unreliable.
# A real resume is typically 2,000+ characters; scanned/near-empty PDFs often
# extract to well under this.
MIN_RELIABLE_CHARS = 300


# ---------------- Load model artifacts ----------------
@st.cache_resource
def load_artifacts():
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    clf = pickle.load(open("clf.pkl", "rb"))
    le = pickle.load(open("encoder.pkl", "rb"))
    return tfidf, clf, le


try:
    tfidf, clf, le = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e.filename}. Keep tfidf.pkl, clf.pkl and encoder.pkl in the same folder as app.py.")
    st.stop()


# ---------------- Cleaning (same as training notebook) ----------------
def clean_resume(txt):
    clean_text = re.sub(r"http\S+\s?", " ", txt)
    clean_text = re.sub(r"RT|cc", " ", clean_text)
    clean_text = re.sub(r"#\S+\s?", " ", clean_text)
    clean_text = re.sub(r"@\S+", " ", clean_text)
    clean_text = re.sub(
        r"[%s]" % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", clean_text
    )
    clean_text = re.sub(r"[^\x00-\x7f]", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text.strip()


# ---------------- Read uploaded PDF ----------------
def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return " ".join(page.extract_text() or "" for page in reader.pages)


# ---------------- Confidence from decision_function ----------------
def scores_to_confidence(scores):
    """
    LinearSVC's decision_function returns signed distances to each class's
    hyperplane, not probabilities — they can be negative and aren't on a
    0-100% scale. Softmax turns them into a readable relative-confidence
    distribution across classes. This is NOT a calibrated probability
    (that would require retraining with CalibratedClassifierCV); treat it
    as "how much more this class stands out than the others."
    """
    exp_scores = np.exp(scores - np.max(scores))  # subtract max for numerical stability
    return exp_scores / exp_scores.sum()


# ---------------- Input ----------------
uploaded = st.file_uploader("Upload a resume (PDF)", type=["pdf"])

resume_text = ""
if uploaded is not None:
    resume_text = read_pdf(uploaded)
    char_count = len(resume_text)

    if char_count < MIN_RELIABLE_CHARS:
        st.warning(
            f"⚠️ Only {char_count} characters were extracted from this PDF. "
            "That's low for a resume (real resumes are usually 2,000+ characters) "
            "and predictions below this length are often unreliable. "
            "This can happen with scanned/image-based PDFs — try a text-based PDF, "
            "or run OCR first."
        )
    else:
        st.success(f"Loaded {uploaded.name} ({char_count} characters)")


# ---------------- Predict ----------------
if st.button("Predict Category", type="primary"):
    if not resume_text.strip():
        st.warning("Please upload a resume PDF first.")
    else:
        cleaned = clean_resume(resume_text)

        # transform -> dense (model was trained on dense arrays)
        features = tfidf.transform([cleaned]).toarray()

        pred = clf.predict(features)[0]

        # model predicts category names directly; decode only if numeric
        if isinstance(pred, (int, np.integer)):
            pred = le.inverse_transform([pred])[0]

        st.subheader("Predicted Category")
        st.success(f"🎯 **{pred}**")

        # Top 3 categories using decision_function scores, shown as
        # relative confidence percentages (see scores_to_confidence docstring)
        scores = clf.decision_function(features)[0]
        confidence = scores_to_confidence(scores)
        top3_idx = np.argsort(scores)[::-1][:3]

        st.subheader("Top 3 Matches")
        for rank, i in enumerate(top3_idx, 1):
            st.write(f"{rank}. {clf.classes_[i]} — {confidence[i]:.1%} relative confidence")

        st.caption(
            "Note: these are relative-confidence percentages derived from the model's "
            "decision boundaries, not calibrated probabilities."
        )

        with st.expander("View cleaned text sent to the model"):
            st.text(cleaned[:3000])
# ☕ Starbucks Review Sentiment Analyzer

A production-ready machine learning and Streamlit application that analyzes and visualizes customer sentiments from Starbucks reviews. It uses a tuned **Logistic Regression model with TF-IDF vectorization** to classify reviews as Positive or Negative, providing granular word-level weight explanations.

## 🚀 Quick Start (Local Run)

To run the Streamlit web application locally, follow these steps:

### 1. Clone or Download this Project
Ensure you have the following files in the same directory:
- `app.py` (The main Streamlit interface)
- `text_utils.py` (Custom text preprocessing library preserving fused negations)
- `sentiment_model.joblib` (The serialized trained model pipeline)
- `model_info.json` (Model metadata, metrics, and word importance datasets)
- `requirements.txt` (List of dependencies)

### 2. Install Dependencies
Create a virtual environment (optional but recommended) and install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Server
Execute the following command in your terminal:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🌐 Deploy to Get a Public URL (Streamlit Community Cloud)

If you want a **free public URL** to share your Sentiment Analyzer web app with others, deployment is straightforward:

1. **Push your code to GitHub**:
   Create a GitHub repository containing all 5 essential assets listed in the local run section.
2. **Sign in to Streamlit**:
   Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in using your GitHub account.
3. **Deploy the App**:
   - Click **"New app"**.
   - Choose your repository, branch (usually `main` or `master`), and set the main file path to `app.py`.
   - Click **"Deploy!"**.
4. **Your Live URL**:
   Streamlit will build the container and provide you with a permanent public URL (e.g., `https://<your-app-name>.streamlit.app`).

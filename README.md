# Basic Sentiment Analysis

## Run in VS Code

1. Open this folder in VS Code.
2. Open Terminal > New Terminal.
3. Install libraries: `pip install -r requirements.txt`
4. Train the model: `python train.py`
5. Run the app: `streamlit run app.py`

## Deploy

Upload `app.py`, `train.py`, `requirements.txt`, `model.pkl`, and `vectorizer.pkl` to GitHub.
Then visit https://share.streamlit.io/, create an app, choose your repository, and select `app.py`.

This is a small learning example trained on a tiny sample dataset, so predictions may be wrong on unfamiliar text.

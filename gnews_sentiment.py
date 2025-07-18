# ✅ Modulo: gnews_sentiment.py
# Estrae news generaliste da GNews.io, applica sentiment NLP (FinBERT) e genera feature storiche compatibili

import requests
import pandas as pd
import datetime
import time
from tqdm import tqdm
import os
from urllib.parse import urlencode
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import numpy as np

# ✅ Inserisci la tua API Key di GNews
GNEWS_API_KEY = "794b0a2fa2a4931549f25e4be5d73ea4"
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"

# ✅ Carica FinBERT
_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# ✅ Funzione per stimare il sentiment

def estimate_sentiment(text):
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    probs = softmax(outputs.logits.numpy()[0])
    return probs[2] - probs[0]  # positive - negative

# ✅ Scarica news da GNews per una certa data (query ottimizzata per notizie finanziarie)
def get_news_for_date_gnews(date, query=None):
    if query is None:
        query = "Market OR Earnings OR rates OR inflation OR tariffs OR Economy"

    from_str = pd.to_datetime(date).strftime("%Y-%m-%dT00:00:00Z")
    to_str = pd.to_datetime(date).strftime("%Y-%m-%dT23:59:59Z")

    params = {
        "q": query,
        "from": from_str,
        "to": to_str,
        "max": 100,
        "token": GNEWS_API_KEY
    }

    url = f"{GNEWS_ENDPOINT}?{urlencode(params)}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            print(f"⚠️ Errore {response.status_code} per la data {date}")
            return []
    except Exception as e:
        print(f"❌ Errore durante richiesta per {date}: {e}")
        return []

# ✅ Estrai le feature per una lista di articoli, tenendo solo i più impattanti

def extract_gnews_sentiment_features(articles, top_n=5):
    if not articles:
        return {
            "news_count": 0,
            "relevant_news_count": 0,
            "avg_sentiment": 0.0,
            "avg_sentiment_weighted": 0.0,
            "avg_sentiment_strong": 0.0,
            "final_sentiment_score": 0.0,
            "impact_score": 0.0,
            "high_impact_news": 0
        }

    scored_articles = []
    for a in articles:
        text = a["title"] + " " + a.get("description", "")
        s = estimate_sentiment(text)
        scored_articles.append((s, text))

    # Ordina per impatto assoluto
    top_articles = sorted(scored_articles, key=lambda x: abs(x[0]), reverse=True)[:top_n]
    scores = [s for s, _ in top_articles]

    abs_scores = [abs(s) for s in scores]
    strong_scores = [s for s in scores if abs(s) >= 0.6]

    avg_sentiment = np.mean(scores)
    avg_sentiment_weighted = np.average(scores, weights=abs_scores) if sum(abs_scores) > 0 else 0.0
    avg_sentiment_strong = np.mean(strong_scores) if strong_scores else 0.0
    impact_score = sum(abs_scores)
    high_impact = len(strong_scores)

    final_sentiment_score = 0.7 * avg_sentiment_weighted + 0.3 * (high_impact / max(len(scores), 1))

    return {
        "news_count": len(articles),
        "relevant_news_count": len(scores),
        "avg_sentiment": avg_sentiment,
        "avg_sentiment_weighted": avg_sentiment_weighted,
        "avg_sentiment_strong": avg_sentiment_strong,
        "final_sentiment_score": final_sentiment_score,
        "impact_score": impact_score,
        "high_impact_news": high_impact
    }

# ✅ Funzione principale per applicare al DataFrame

def apply_gnews_features(df, cache_path=None):
    df["Date"] = pd.to_datetime(df["Date"])
    unique_dates = df["Date"].dt.normalize().unique()
    features_by_date = []

    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        cached["Date"] = pd.to_datetime(cached["Date"])
        known_dates = set(cached["Date"].dt.normalize().unique())
    else:
        cached = pd.DataFrame()
        known_dates = set()

    for date in tqdm(unique_dates, desc="🗅️ GNews Processing"):
        if date in known_dates:
            features = cached[cached["Date"].dt.normalize() == date].iloc[0].to_dict()
        else:
            articles = get_news_for_date_gnews(date)
            features = extract_gnews_sentiment_features(articles, top_n=5)
            features["Date"] = pd.to_datetime(date)
            cached = pd.concat([cached, pd.DataFrame([features])], ignore_index=True)
        features_by_date.append(features)

    if cache_path:
        cached.drop_duplicates(subset=["Date"], keep="last").to_csv(cache_path, index=False)

    features_df = pd.DataFrame(features_by_date)
    features_df["Date"] = pd.to_datetime(features_df["Date"])
    df = df.merge(features_df, on="Date", how="left")
    df.fillna({
        "news_count": 0,
        "relevant_news_count": 0,
        "avg_sentiment": 0.0,
        "avg_sentiment_weighted": 0.0,
        "avg_sentiment_strong": 0.0,
        "final_sentiment_score": 0.0,
        "impact_score": 0.0,
        "high_impact_news": 0
    }, inplace=True)
    return df

import pandas as pd

# Puoi cambiare questi limiti per evitare rate limit
start_date = "2017-10-26"
end_date = "2025-12-31"  # esempio per test iniziale

df_dates = pd.DataFrame({
    "Date": pd.date_range(start=start_date, end=end_date, freq="D")
})
df_with_sentiment = apply_gnews_features(df_dates, cache_path="gnews_sentiment_five.csv")

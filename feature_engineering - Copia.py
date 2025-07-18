# ✅ IMPORTAZIONI
import pandas as pd
import numpy as np
import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import seaborn as sns
import optuna
import xgboost as xgb
import shap
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pywt
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from catboost import CatBoostClassifier
import ta  # importa una sola volta


def load_and_prepare_data(path):
    # ✅ 1. CARICAMENTO FILE
    df = pd.read_csv(path)

    # ✅ 2. PULIZIA DATI
    df.rename(columns={"Price": "Close", "Vol.": "Volume", "Change %": "Change_Pct"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Volume"] = df["Volume"].str.replace("M", "").astype(float) * 1_000_000
    df["Change_Pct"] = df["Change_Pct"].str.replace("%", "").astype(float)

    # ✅ 3. FEATURE ENGINEERING
    # Inizializzazione delle liste delle feature
    feature_list = []
    full_feature_list = []

    # ✅ DEFINIZIONE START E END DATE (prima di web.DataReader)
    import yfinance as yf
    print("✅ Download lista ticker SP500...")
    sp500_tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()
    sp500_tickers = [ticker.replace('.', '-') for ticker in sp500_tickers]
    valid_dates = None
    start_date = df["Date"].min()
    end_date = df["Date"].max()
    
    # ✅ ETF FLOWS: SPY, IVV, VOO volume features
    print("✅ Integrazione ETF Flows da Yahoo Finance...")
    etf_tickers = {"SPY": "SPY", "IVV": "IVV", "VOO": "VOO"}
    etf_volumes = {}
    for name, ticker in etf_tickers.items():
        try:
            etf_data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)["Volume"]
            etf_data.name = f"Volume_{name}"
            etf_volumes[name] = etf_data
            df = df.merge(etf_data, how="left", left_on="Date", right_index=True)
            print(f"✅ ETF {name} integrato con successo")
        except Exception as e:
            print(f"⚠️ Errore ETF {name}: {e}")

    # Deriva feature: flusso relativo + z-score
    try:
        if all(f"Volume_{k}" in df.columns for k in etf_tickers):
            df["ETF_Flow_Total"] = df[[f"Volume_{k}" for k in etf_tickers]].sum(axis=1)
            df["ETF_Flow_Z"] = (df["ETF_Flow_Total"] - df["ETF_Flow_Total"].rolling(20).mean()) / df["ETF_Flow_Total"].rolling(20).std()
            df["ETF_Flow_Ratio"] = df["Volume_SPY"] / (df["Volume_IVV"] + df["Volume_VOO"] + 1)
            df["ETF_Flow_Diff_1d"] = df["ETF_Flow_Total"].diff(1)
            df["ETF_Flow_Z5"] = (df["ETF_Flow_Total"] - df["ETF_Flow_Total"].rolling(5).mean()) / df["ETF_Flow_Total"].rolling(5).std()
            df["ETF_Flow_Change_Pct"] = df["ETF_Flow_Total"].pct_change(1)
            feature_list += ["ETF_Flow_Total", "ETF_Flow_Z", "ETF_Flow_Ratio", "ETF_Flow_Diff_1d", "ETF_Flow_Z5", "ETF_Flow_Change_Pct"]
            full_feature_list += ["ETF_Flow_Total", "ETF_Flow_Z", "ETF_Flow_Ratio", "ETF_Flow_Diff_1d", "ETF_Flow_Z5", "ETF_Flow_Change_Pct"]
            print("✅ Feature ETF Flow derivate calcolate")
    except Exception as e:
        print(f"⚠️ Errore nel calcolo delle feature ETF Flow: {e}")

    # =============================
    # ✅ MARKET INTERNALS: % Stocks Above MA200 da Yahoo
    # 🔁 Preparazione anche per TRIN Index: salvataggio volume
    close_data = {}
    volume_data = {}
    for ticker in sp500_tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)[["Close", "Volume"]]
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            close_data[ticker] = close

            volume = data["Volume"]
            if isinstance(volume, pd.DataFrame):
                volume = volume.squeeze()
            volume_data[ticker] = volume
        except Exception as e:
            print(f"⚠️ Skip TRIN prep {ticker}: {e}")

    # 🔁 Verifica numero di ticker validi
    print(f"✅ Ticker validi TRIN: {len(close_data)} su {len(sp500_tickers)} totali")

    # 🔍 Diagnostica avanzata su ticker scaricati
    empty_close = [ticker for ticker, series in close_data.items() if series.isna().all().all() or series.empty]
    empty_volume = [ticker for ticker, series in volume_data.items() if series.isna().all().all() or series.empty]
    print(f"📉 Ticker con Close completamente vuoti o NaN: {len(empty_close)}")
    print(f"📉 Ticker con Volume completamente vuoti o NaN: {len(empty_volume)}")
    if empty_close:
        print("Esempi Close vuoti:", empty_close[:10])
    if empty_volume:
        print("Esempi Volume vuoti:", empty_volume[:10])
    

    # 🔁 Calcolo TRIN Index direttamente dai dati già scaricati
    print("🔍 DEBUG: controllo tipo oggetti in close_data e volume_data")
    for k, v in close_data.items():
        if not isinstance(v, pd.Series):
            print(f"❌ close_data[{k}] è di tipo {type(v)}")
        elif v.empty or v.isna().all():
            print(f"⚠️ close_data[{k}] è vuoto o tutto NaN")
    
    for k, v in volume_data.items():
        if not isinstance(v, pd.Series):
            print(f"❌ volume_data[{k}] è di tipo {type(v)}")
        elif v.empty or v.isna().all():
            print(f"⚠️ volume_data[{k}] è vuoto o tutto NaN")

    try:
        close_df = pd.DataFrame(close_data)
        volume_df = pd.DataFrame(volume_data)

        returns = close_df.pct_change()
        adv_issues = (returns > 0).astype(int)
        dec_issues = (returns < 0).astype(int)
        adv_volume = volume_df * adv_issues
        dec_volume = volume_df * dec_issues

        trin_index = (adv_volume.sum(axis=1) / dec_volume.sum(axis=1)) / (adv_issues.sum(axis=1) / dec_issues.sum(axis=1))
        trin_index = trin_index.reset_index()
        trin_index.columns = ["Date", "TRIN_Index"]
        df = df.merge(trin_index, on="Date", how="left")
        df["TRIN_Index"] = df["TRIN_Index"].ffill()
        feature_list += ["TRIN_Index", "TRIN_Z20", "TRIN_Momentum_5d"]
        full_feature_list += ["TRIN_Index", "TRIN_Z20", "TRIN_Momentum_5d"]
        print("✅ TRIN Index calcolato e integrato")

        # ➕ Feature derivate dal TRIN Index
        df["TRIN_Z20"] = (df["TRIN_Index"] - df["TRIN_Index"].rolling(20).mean()) / df["TRIN_Index"].rolling(20).std()
        df["TRIN_Momentum_5d"] = df["TRIN_Index"].diff(5)
        print("✅ Feature TRIN_Z20 e TRIN_Momentum_5d calcolate e integrate")
    except Exception as e:
        print(f"⚠️ Errore TRIN Index: {e}")
    # ➕ Calcolo anche % Stocks Above MA50
    ma50_data = {}
    for ticker in sp500_tickers:
        try:
            data = close_data[ticker]
            ma50 = data.rolling(50).mean()
            ma50_data[ticker] = (data.squeeze() > ma50.squeeze()).astype(int)
        except Exception as inner_e:
            print(f"⚠️ Skip MA50 {ticker}: {inner_e}")

    ma50_df = pd.DataFrame(ma50_data, index=valid_dates)
    pct_above_ma50 = ma50_df.sum(axis=1).astype(float) / ma50_df.shape[1] * 100
    pct_above_ma50 = pct_above_ma50.reset_index()
    pct_above_ma50.columns = ["Date", "Pct_Stocks_Above_MA50"]
    df = df.merge(pct_above_ma50, on="Date", how="left")
    df["Pct_Stocks_Above_MA50"] = df["Pct_Stocks_Above_MA50"].ffill()
    feature_list += ["Pct_Stocks_Above_MA50"]
    full_feature_list += ["Pct_Stocks_Above_MA50"]
    print("✅ Integrazione completata: Pct_Stocks_Above_MA50")
    print("✅ Calcolo % Stocks Above MA200...")
    try:
        ma200_data = {}
        valid_dates = None
        for ticker in sp500_tickers:
            try:
                data = close_data[ticker]
                ma200 = data.rolling(200).mean()
                ma200_data[ticker] = (data.squeeze() > ma200.squeeze()).astype(int)
                if valid_dates is None:
                    valid_dates = data.index
            except Exception as inner_e:
                print(f"⚠️ Skip {ticker}: {inner_e}")
        ma200_df = pd.DataFrame(ma200_data, index=valid_dates)
        pct_above_ma200 = ma200_df.sum(axis=1).astype(float) / ma200_df.shape[1] * 100
        pct_above_ma200 = pct_above_ma200.reset_index()
        pct_above_ma200.columns = ["Date", "Pct_Stocks_Above_MA200"]
        df = df.merge(pct_above_ma200, on="Date", how="left")
        df["Pct_Stocks_Above_MA200"] = df["Pct_Stocks_Above_MA200"].ffill()
        feature_list += ["Pct_Stocks_Above_MA200"]
        full_feature_list += ["Pct_Stocks_Above_MA200"]
        print("✅ Integrazione completata: Pct_Stocks_Above_MA200")
    except Exception as e:
        print("⚠️ Errore durante il calcolo di Pct_Stocks_Above_MA200:", e)
    
    # ✅ MARKET INTERNALS: Advance/Decline Line da Yahoo Finance
    # ➕ Ricostruzione alternativa dell'A/D Line dai dati già scaricati
    print("📈 Calcolo Advance/Decline Line dai dati MA200...")
    try:
        ad_line_data = {}
        for ticker in sp500_tickers:
            try:
                data = close_data[ticker]
                returns = data.pct_change()
                advance = (returns > 0).astype(int)
                decline = (returns < 0).astype(int)
                ad_diff = advance - decline
                ad_line_data[ticker] = ad_diff
            except Exception as inner_e:
                print(f"⚠️ Skip AD-Line {ticker}: {inner_e}")

        ad_line_df = pd.DataFrame(ad_line_data)
        ad_line = ad_line_df.sum(axis=1).cumsum()
        ad_line.name = "AdvanceDeclineLine_Recalc"
        ad_line.index.name = "Date"
        ad_line = ad_line.reset_index()
        df = df.merge(ad_line, on="Date", how="left")
        df["AdvanceDeclineLine_Recalc"] = df["AdvanceDeclineLine_Recalc"].ffill().bfill()

        # ➕ Z-Score e Momentum A/D Line
        df["ADLine_Z20"] = (df["AdvanceDeclineLine_Recalc"] - df["AdvanceDeclineLine_Recalc"].rolling(20).mean()) / df["AdvanceDeclineLine_Recalc"].rolling(20).std()
        df["ADLine_Momentum_5d"] = df["AdvanceDeclineLine_Recalc"].diff(5)

        # ➕ McClellan Oscillator (EMA19 - EMA39 dell'A/D Line)
        ema19 = df["AdvanceDeclineLine_Recalc"].ewm(span=19, adjust=False).mean()
        ema39 = df["AdvanceDeclineLine_Recalc"].ewm(span=39, adjust=False).mean()
        df["McClellan_Oscillator"] = ema19 - ema39
        feature_list += ["McClellan_Oscillator"]
        full_feature_list += ["McClellan_Oscillator"]
        df["ADLine_Z20"] = (df["AdvanceDeclineLine_Recalc"] - df["AdvanceDeclineLine_Recalc"].rolling(20).mean()) / df["AdvanceDeclineLine_Recalc"].rolling(20).std()
        df["ADLine_Momentum_5d"] = df["AdvanceDeclineLine_Recalc"].diff(5)

        feature_list += ["AdvanceDeclineLine_Recalc", "ADLine_Z20", "ADLine_Momentum_5d"]
        full_feature_list += ["AdvanceDeclineLine_Recalc", "ADLine_Z20", "ADLine_Momentum_5d"]
        print("✅ A/D Line alternativa calcolata con successo e integrata")
    except Exception as e:
        print("⚠️ Errore durante il calcolo dell'A/D Line alternativa:", e)

    # ✅ FEATURE ESTERNE: SENTIMENT INDEX (UMCSENT)
    print("✅ Integrazione Sentiment Index da FRED...")
    original_dates = set(pd.to_datetime(pd.read_csv(path)["Date"], format="%m/%d/%Y"))
    import pandas_datareader.data as web

    try:
        sentiment = web.DataReader("UMCSENT", "fred", start_date, end_date)
        sentiment.rename(columns={"UMCSENT": "UMich_Sentiment"}, inplace=True)
        df = df.merge(sentiment, how="left", left_on="Date", right_index=True)
        dates_after = set(df["Date"])
        missing = original_dates - dates_after
        print(f"📉 Merge [Sentiment UMCSENT] ha perso {len(missing)} date")
        df.fillna(method="ffill", inplace=True)
        sentiment_features = ["UMich_Sentiment"]
        feature_list += sentiment_features
        full_feature_list += sentiment_features
        print("✅ Feature sentiment UMCSENT aggiunta con successo:", sentiment_features)
    except Exception as e:
        print("⚠️ Errore durante il caricamento del Sentiment Index da FRED:", e)

    # ✅ FEATURE MACROECONOMICHE - FRED + INTERMARKET
    print("✅ Integrazione macroeconomiche da FRED...")
    macro_extra = pd.DataFrame()
    macro_series = {
        "CPI": "CPIAUCSL",
        "CoreCPI": "CPILFESL",
        "NFP": "PAYEMS",
        "PPI": "PPIACO",
        "RetailSales": "RSAFS",
        "GDP_QoQ": "A191RL1Q225SBEA",
        "JOLTS": "JTSJOL"
    }

    for name, code in macro_series.items():
        try:
            macro_extra[name] = web.DataReader(code, "fred", start_date, end_date)
            print(f"✅ {name} acquisito da FRED")
        except Exception as e:
            print(f"⚠️ Errore nel download di {name} ({code}):", e)
            macro_extra[name] = np.nan

    # ✅ Merge macro_extra mantenendo tutte le date SPY e fill progressivo
    print("📊 Merge [macro_extra] in corso...")
    for col in macro_extra.columns:
        print(f"📊 Merge macro_extra: {col}")
        df = df.merge(macro_extra[[col]], how="left", left_on="Date", right_index=True)
        df[col] = df[col].ffill().bfill()

    macro_features = list(macro_series.keys())
    feature_list += macro_features
    full_feature_list += macro_features
    import datetime

    start_date = df["Date"].min()
    end_date = df["Date"].max()

    # Scarica i dati macro da FRED
    macro = pd.DataFrame()
    macro["FEDFUNDS"] = web.DataReader("FEDFUNDS", "fred", start_date, end_date)
    macro["GS10"] = web.DataReader("GS10", "fred", start_date, end_date)
    macro["GS2"] = web.DataReader("GS2", "fred", start_date, end_date)
    macro["VIX"] = web.DataReader("VIXCLS", "fred", start_date, end_date)
    macro["DXY"] = web.DataReader("DTWEXBGS", "fred", start_date, end_date)

    # Calcolo spread 10Y - 2Y
    macro["10Y-2Y"] = macro["GS10"] - macro["GS2"]

    # Forward-fill e allineamento alle date del DataFrame principale
    df = df.merge(macro, how="left", left_on="Date", right_index=True)
    dates_after = set(df["Date"])
    missing = original_dates - dates_after
    print(f"📉 Merge [macro principali FED/VIX] ha perso {len(missing)} date")
    df.fillna(method="ffill", inplace=True)

    # ✅ TRASFORMAZIONI TEMPORALI SULLE MACRO
    print("✅ Aggiunta trasformazioni temporali sulle macro...")
    df["CPI_Change_1m"] = df["CPI"].diff(21)
    df["RetailSales_Z"] = (df["RetailSales"] - df["RetailSales"].rolling(21).mean()) / df["RetailSales"].rolling(21).std()
    df["NFP_Diff_1m"] = df["NFP"].diff(21)

    macro_transformed_features = [
        "CPI_Change_1m", "RetailSales_Z", "NFP_Diff_1m"]
    feature_list += macro_transformed_features
    full_feature_list += macro_transformed_features
    print("✅ Trasformazioni temporali macro aggiunte:", macro_transformed_features)

    # Feature Intermarket derivate direttamente

    # ✅ FEATURE CROSS-ASSET (Gold, Oil, BTC, USDJPY)
    print("✅ Integrazione feature cross-asset...")
    import yfinance as yf
    cross_assets = {
        "Gold": "GC=F",
        "WTI_Oil": "CL=F",
        "BTC": "BTC-USD",
        "USDJPY": "JPY=X"
    }
    for name, ticker in cross_assets.items():
        try:
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

            if isinstance(data.columns, pd.MultiIndex):
                try:
                    data = data.xs('Close', axis=1, level=0, drop_level=True)
                except KeyError:
                    print(f"⚠️ Livello 'Close' non trovato in colonne per {name} ({ticker}) - Livelli: {data.columns.levels}")
                    continue

            # Rinomina la colonna di chiusura con il nome desiderato (es. Gold, BTC)
            if isinstance(data, pd.Series):
                data = data.to_frame(name=name)
            else:
                data.columns = [name]

            data.index = pd.to_datetime(data.index)
            df = df.merge(data[[name]], how="left", left_on="Date", right_index=True)
            dates_after = set(df["Date"])
            missing = original_dates - dates_after
            print(f"📉 Merge [cross-asset: {name}] ha perso {len(missing)} date")
            df[name] = df[name].ffill()

            # ✅ TRASFORMAZIONI TEMPORALI CROSS-ASSET
            df[f"{name}_Change_5d"] = df[name].diff(5)
            df[f"{name}_Z20"] = (df[name] - df[name].rolling(20).mean()) / df[name].rolling(20).std()
            df[f"{name}_MA10"] = df[name].rolling(10).mean()
            df[f"{name}_Volatility_20d"] = df[name].rolling(20).std()

            feature_list += [
                name,
                f"{name}_Change_5d",
                f"{name}_Z20",
                f"{name}_MA10",
                f"{name}_Volatility_20d"
            ]
            full_feature_list += [
                name,
                f"{name}_Change_5d",
                f"{name}_Z20",
                f"{name}_MA10",
                f"{name}_Volatility_20d"
            ]

            print(f"✅ {name} aggiunto da Yahoo Finance")
        except Exception as e:
            print(f"⚠️ Errore nel download di {name} ({ticker}):", e)
    print("✅ Aggiunta feature intermarket...")
    df["SPY_VIX"] = df["Close"] / df["VIX"]
    df["SPY_DXY"] = df["Close"] / df["DXY"]
    df["VIX_GS10"] = df["VIX"] / df["GS10"]

    intermarket_features = ["SPY_VIX", "SPY_DXY", "VIX_GS10"]
    feature_list += intermarket_features
    full_feature_list += intermarket_features
    print("✅ Feature intermarket aggiunte:", intermarket_features)
    
    # ✅ Aggiunta delle news storiche al DataFrame
    gnews_df = pd.read_csv("gnews_sentiment_five.csv")
    gnews_df["Date"] = pd.to_datetime(gnews_df["Date"])
    df = df.merge(gnews_df[["Date", "final_sentiment_score"]], on="Date", how="left")
    dates_after = set(df["Date"])
    missing = original_dates - dates_after
    print(f"📉 Merge [news sentiment GNews] ha perso {len(missing)} date")
    df["final_sentiment_score"] = df["final_sentiment_score"].shift(1)
    df["final_sentiment_score"].fillna(0.0, inplace=True)
    full_feature_list += ["final_sentiment_score"]
    
    # ✅ FEATURE MINIMI/MASSIMI STORICI
    print("✅ Aggiunta feature minimi/massimi storici...")

    # Massimo/minimo assoluto su 252 e 504 giorni
    lookbacks = [252, 504]
    for lb in lookbacks:
        df[f"Dist_from_Max_{lb}d"] = df["Close"] - df["Close"].rolling(window=lb).max()
        df[f"Dist_from_Min_{lb}d"] = df["Close"] - df["Close"].rolling(window=lb).min()
        df[f"Near_Max_{lb}d"] = (df["Close"] >= df["Close"].rolling(window=lb).max() * 0.99).astype(int)
        df[f"Near_Min_{lb}d"] = (df["Close"] <= df["Close"].rolling(window=lb).min() * 1.01).astype(int)

    long_term_extrema_features = [
        f"Dist_from_Max_{lb}d" for lb in lookbacks
    ] + [
        f"Dist_from_Min_{lb}d" for lb in lookbacks
    ] + [
        f"Near_Max_{lb}d" for lb in lookbacks
    ] + [
        f"Near_Min_{lb}d" for lb in lookbacks
    ]

    feature_list += long_term_extrema_features
    full_feature_list += long_term_extrema_features
    print("✅ Feature di minimi/massimi storici aggiunte con successo:", long_term_extrema_features)

    # ✅ FEATURE TEMPORALI CICLICHE
    print("✅ Aggiunta feature temporali...")
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    weekday_dummies = pd.get_dummies(df["DayOfWeek"], prefix="Weekday")
    df = pd.concat([df, weekday_dummies], axis=1)

    df["Month"] = df["Date"].dt.month
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    temporal_features = list(weekday_dummies.columns) + ["Month_sin", "Month_cos"]
    feature_list += temporal_features
    full_feature_list += temporal_features
    print("✅ Feature temporali cicliche aggiunte:", temporal_features)

    # ✅ FEATURE TEMPORALI
    df['Quarter'] = df['Date'].dt.quarter
    df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
    df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end.astype(int)

    # Calcolo Days_Since_High e Days_Since_Low corretti
    window = 100
    days_since_high = np.full(len(df), np.nan)
    days_since_low = np.full(len(df), np.nan)
    for i in range(window, len(df)):
        window_df = df.iloc[i - window:i]
        max_idx = window_df["Close"].idxmax()
        min_idx = window_df["Close"].idxmin()
        days_since_high[i] = (df.iloc[i]["Date"] - df.loc[max_idx, "Date"]).days
        days_since_low[i] = (df.iloc[i]["Date"] - df.loc[min_idx, "Date"]).days
    df["Days_Since_High"] = days_since_high
    df["Days_Since_Low"] = days_since_low

    # ✅ AGGIUNTA A feature_list
    temporal_features = ["DayOfWeek", "Month", "Quarter", "Is_Month_End", "Is_Quarter_End", "Days_Since_High", "Days_Since_Low"]
    feature_list += temporal_features
    full_feature_list += temporal_features
    print("✅ Feature temporali aggiunte con successo:", temporal_features)

    # ✅ FEATURE CANDLE PATTERN
    print("✅ Aggiunta feature candle pattern...")

    df['Bullish_Engulfing'] = ((df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))).astype(int)
    df['Bearish_Engulfing'] = ((df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))).astype(int)
    df['Doji'] = (np.abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1).astype(int)

    df['Hammer'] = ((df['High'] - df['Close'] > 2 * (df['Open'] - df['Low'])) & (df['Close'] > df['Open'])).astype(int)
    df['Inverted_Hammer'] = ((df['Close'] - df['Low'] > 2 * (df['High'] - df['Open'])) & (df['Close'] < df['Open'])).astype(int)

    candle_features = ["Bullish_Engulfing", "Bearish_Engulfing", "Doji", "Hammer", "Inverted_Hammer"]
    feature_list += candle_features
    full_feature_list += candle_features
    print("✅ Feature candle pattern aggiunte con successo:", candle_features)

    # ✅ FEATURE TECNICHE AVANZATE
    print("✅ Aggiunta feature tecniche avanzate...")
    df["CCI"] = ta.trend.CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20).cci()
    df["Stochastic_K"] = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14).stoch()
    df["Stochastic_D"] = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14).stoch_signal()
    df["Williams_%R"] = ta.momentum.WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"], lbp=14).williams_r()
    df["TRIX"] = ta.trend.TRIXIndicator(close=df["Close"], window=15).trix()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()

    tech_features = ["CCI", "Stochastic_K", "Stochastic_D", "Williams_%R", "TRIX", "OBV"]
    feature_list += tech_features
    full_feature_list += tech_features
    print("✅ Feature tecniche avanzate aggiunte con successo:", tech_features)

    # ✅ FEATURE STATISTICHE DERIVATE
    print("✅ Aggiunta feature statistiche derivate...")
    df["Z_score_Close"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).std()
    df["Rolling_Skew_5d"] = df["Close"].rolling(window=5).skew()
    df["Rolling_Skew_20d"] = df["Close"].rolling(window=20).skew()
    df["Rolling_Kurt_5d"] = df["Close"].rolling(window=5).kurt()
    df["Rolling_Kurt_20d"] = df["Close"].rolling(window=20).kurt()
    df["Autocorr_1d"] = df["Close"].pct_change().rolling(5).apply(lambda x: pd.Series(x).autocorr(lag=1))

    # ✅ AGGIUNTA A feature_list
    stat_features = [
        "Z_score_Close", "Rolling_Skew_5d", "Rolling_Skew_20d",
        "Rolling_Kurt_5d", "Rolling_Kurt_20d", "Autocorr_1d"
    ]
    feature_list += stat_features
    full_feature_list += stat_features
    print("✅ Feature statistiche aggiunte con successo:", stat_features)


    import pywt
    base_feature_list = [
        "RSI_14", "MACD", "MACD_signal", "MACD_diff", "ADX_14",
        "BB_upper", "BB_lower", "BB_percent", "Volatility_10d", "Volatility_20d",
        "LogRet_1d", "LogRet_5d", "LogRet_10d", "Volume_Z20",
        "Momentum_5d", "Momentum_10d", "MA200", "Pct_from_MA200",
        "Volume_MA20", "Volatility_ratio", "Open_Close", "High_Low", "Gap",
        "TR", "Return_Open", "Momentum_3d", "Close/Open"
    ]


    # ✅ WAVELET DECOMPOSITION (livello 3, tipo 'db1')
    def get_wavelet_features(series, wavelet='db1', level=3):
        coeffs = pywt.wavedec(series, wavelet, level=level)
        features = {}
        for i, coef in enumerate(coeffs):
            coef = coef[:len(series)] if len(coef) > len(series) else np.pad(coef, (0, len(series) - len(coef)))
            features[f'wavelet_L{i}'] = pd.Series(coef)
        return pd.concat(features.values(), axis=1).fillna(0)

    wavelet_feats = get_wavelet_features(df['Close'].fillna(method='ffill'))
    wavelet_feats.columns = [f"Wavelet_{i+1}" for i in range(wavelet_feats.shape[1])]
    wavelet_feats.index = df.index

    # Unisci le feature wavelet al DataFrame principale
    df = pd.concat([df, wavelet_feats], axis=1)

    # ✅ Regimi di Mercato (Trend, Volatilità, Momentum)

    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()
    adx = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ADX_14"] = adx.adx()
    boll = ta.volatility.BollingerBands(close=df["Close"], window=20)
    df["BB_upper"] = boll.bollinger_hband()
    df["BB_lower"] = boll.bollinger_lband()
    df["BB_percent"] = boll.bollinger_pband()
    df["Volatility_10d"] = df["Close"].rolling(window=10).std()
    df["Volatility_20d"] = df["Close"].rolling(window=20).std()
    df["LogRet_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    df["LogRet_5d"] = np.log(df["Close"] / df["Close"].shift(5))
    df["LogRet_10d"] = np.log(df["Close"] / df["Close"].shift(10))
    df["Volume_Z20"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    df["Momentum_5d"] = df["Close"] - df["Close"].shift(5)
    df["Momentum_10d"] = df["Close"] - df["Close"].shift(10)
    df["MA200"] = df["Close"].rolling(200).mean()
    df["Pct_from_MA200"] = (df["Close"] - df["MA200"]) / df["MA200"]
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    df["Volatility_ratio"] = df["Volatility_10d"] / df["Volatility_20d"]
    df["Open_Close"] = df["Close"] - df["Open"]
    df["High_Low"] = df["High"] - df["Low"]
    df["Gap"] = df["Open"] - df["Close"].shift(1)
    df["TR"] = df[["High", "Close"]].max(axis=1) - df[["Low", "Close"]].min(axis=1)
    df["Return_Open"] = np.log(df["Open"] / df["Close"].shift(1))
    df["Momentum_3d"] = df["Close"] - df["Close"].shift(3)
    df["Close/Open"] = df["Close"] / df["Open"]

    df.dropna(subset=base_feature_list, inplace=True)

    df["Trend_Regime"] = np.where(df["Close"] > df["MA200"], 1, 0)
    df["Volatility_Regime"] = np.where(df["Volatility_20d"] > df["Volatility_20d"].rolling(200).median(), 1, 0)
    df["Momentum_Regime"] = np.where(df["Momentum_5d"] > 0, 1, 0)

    # ✅ NUOVE FEATURE AVANZATE
    print("✅ Aggiunta nuove feature avanzate...")

    df["ATR_14"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
    df["Relative_Volume"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["BB_Distance"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    df["Gap_Strength"] = df["Gap"] / df["Volatility_10d"]
    # 🔧 Pulizia robusta del Noise_Ratio
    df["Noise_Ratio"] = df["High_Low"] / np.abs(df["Close"] - df["Open"])
    df["Noise_Ratio"] = df["Noise_Ratio"].replace([np.inf, -np.inf], np.nan)
    df["Noise_Ratio"] = df["Noise_Ratio"].fillna(df["Noise_Ratio"].rolling(10, min_periods=1).median())
    q99 = df["Noise_Ratio"].quantile(0.99)
    df["Noise_Ratio"] = df["Noise_Ratio"].clip(upper=q99)

    df["Up_Days_5d"] = (df["Close"].diff() > 0).rolling(5).sum()
    df["Down_Days_5d"] = (df["Close"].diff() < 0).rolling(5).sum()
    df["Local_Sharpe_20d"] = df["LogRet_1d"].rolling(20).mean() / df["LogRet_1d"].rolling(20).std()

    advanced_features = [
        "ATR_14", "Relative_Volume", "BB_Distance", "Gap_Strength", "Noise_Ratio",
        "Up_Days_5d", "Down_Days_5d", "Local_Sharpe_20d"
    ]
    feature_list += advanced_features
    full_feature_list += advanced_features
    print("✅ Nuove feature avanzate aggiunte con successo:", advanced_features)

    # ✅ CLUSTERING DEI REGIMI CON GMM
    print("✅ Calcolo dei regimi di mercato con Gaussian Mixture Model...")
    gmm_features = [
        "Volatility_20d", "Momentum_5d", "Return_Open",
        "Pct_from_MA200", "Rolling_Skew_5d", "Rolling_Kurt_5d"
    ]

    clustering_df = df[gmm_features].replace([np.inf, -np.inf], np.nan).dropna()
    scaler_gmm = StandardScaler()
    X_gmm = scaler_gmm.fit_transform(clustering_df)

    # Fit GMM
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm_probs = gmm.fit_predict(X_gmm)
    df.loc[clustering_df.index, "Market_Regime"] = gmm_probs

    # ➕ Soft clustering probabilities
    soft_probs = gmm.predict_proba(X_gmm)
    for i in range(soft_probs.shape[1]):
        df.loc[clustering_df.index, f"GMM_Prob_{i}"] = soft_probs[:, i]

    feature_list += [f"GMM_Prob_{i}" for i in range(soft_probs.shape[1])]
    full_feature_list += [f"GMM_Prob_{i}" for i in range(soft_probs.shape[1])]
    df["Market_Regime"] = df["Market_Regime"].astype(int)

    # Aggiunta ai feature set
    feature_list += ["Market_Regime"]
    full_feature_list += ["Market_Regime"]
    print("✅ Regimi di mercato con GMM aggiunti con successo.")


    # ➕ ANALISI GRAFICA DEI REGIMI
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values())

    plt.figure(figsize=(14, 5))
    for i, (start, group) in enumerate(df.groupby((df['Market_Regime'] != df['Market_Regime'].shift()).cumsum())):
        regime = group['Market_Regime'].iloc[0]
        plt.plot(group['Date'], group['Close'], label=f"Regime {regime}" if f"Regime {regime}" not in plt.gca().get_legend_handles_labels()[1] else "", color=colors[regime % len(colors)])
    plt.title("Prezzo SPY Segmentato per Regime di Mercato (GMM)")
    plt.xlabel("Date")
    plt.ylabel("Prezzo Close")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ➕ ANALISI METRICHE DEI REGIMI
    regime_stats = df.groupby("Market_Regime")["LogRet_1d"].agg(["mean", "std", "count"])
    regime_stats["Sharpe"] = regime_stats["mean"] / regime_stats["std"] * np.sqrt(252)
    print("\n📊 Statistiche per Regime:\n", regime_stats)

    # ➕ RINOMINA REGIMI PER INTERPRETAZIONE
    # ➕ Feature temporale: cambi di regime
    df["Is_Regime_Switch"] = (df["Market_Regime"] != df["Market_Regime"].shift(1)).astype(int)
    df["Days_Since_Regime_Change"] = df["Is_Regime_Switch"].cumsum()
    df["Days_Since_Regime_Change"] = df.groupby("Days_Since_Regime_Change").cumcount()

    feature_list += ["Is_Regime_Switch", "Days_Since_Regime_Change"]
    full_feature_list += ["Is_Regime_Switch", "Days_Since_Regime_Change"]
    regime_mapping = regime_stats["Sharpe"].sort_values().reset_index()
    regime_labels = {regime_mapping.loc[0, "Market_Regime"]: "Bear",
                     regime_mapping.loc[1, "Market_Regime"]: "Sideways",
                     regime_mapping.loc[2, "Market_Regime"]: "Bull"}
    df["Market_Regime_Label"] = df["Market_Regime"].map(regime_labels)
    print("\n📌 Etichette assegnate:", regime_labels)

    # ➕ HEATMAP MEDIE DELLE FEATURE DI CLUSTERING PER REGIME
    clustering_cols = ["Volatility_20d", "Momentum_5d", "Pct_from_MA200"]
    heatmap_data = df.groupby("Market_Regime_Label")[clustering_cols].mean()
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm")
    plt.title("Profilo Medio delle Feature per Regime")
    plt.tight_layout()
    plt.show()

    # ✅ AUTOENCODER SULLE FEATURE SHAP + REGIMI + WAVELET
    feature_list += ["Trend_Regime", "Volatility_Regime", "Momentum_Regime"] + [f"Wavelet_{i+1}" for i in range(wavelet_feats.shape[1])]

    # ✅ Aggiunta AE al feature set completo (ALL)
    ae_features = []
    feature_list += ae_features

    full_feature_list = feature_list.copy()
    feature_list = full_feature_list
  
    # ✅ FILL STRATEGICO DELLE NUOVE FEATURE ESTERNE E MACRO
    fill_neutral = {
        "UMich_Sentiment": df["UMich_Sentiment"].median(),
        "Pct_Stocks_Above_MA200": -1.0,
        "Gold": df["Gold"].mean(),
        "WTI_Oil": df["WTI_Oil"].mean(),
        "BTC": df["BTC"].mean(),
        "USDJPY": df["USDJPY"].mean(),
        "SPY_VIX": df["SPY_VIX"].mean(),
        "SPY_DXY": df["SPY_DXY"].mean(),
        "VIX_GS10": df["VIX_GS10"].mean()
    }
    
    for col, val in fill_neutral.items():
        if col in df.columns:
            df[col].fillna(val, inplace=True)
    
    # ✅ Riempimento macro con la mediana
    macro_columns = ["CPI", "CoreCPI", "NFP", "PPI", "RetailSales", "GDP_QoQ", "JOLTS", "FEDFUNDS", "GS10", "GS2", "VIX", "DXY", "10Y-2Y"]
    for col in macro_columns:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # ✅ FILL DEDICATO PER FEATURE PROBLEMATICHE
    neutral_fill = {
        "RetailSales_Z": 0.0,
        "BTC_Change_5d": 0.0,
        "BTC_Z20": 0.0,
        "BTC_MA10": df["BTC"].rolling(10).mean().mean() if "BTC" in df.columns else 0.0,
        "BTC_Volatility_20d": df["BTC"].rolling(20).std().mean() if "BTC" in df.columns else 0.0,
        "Dist_from_Max_252d": 0.0,
        "Dist_from_Max_504d": 0.0,
        "Dist_from_Min_252d": 0.0,
        "Dist_from_Min_504d": 0.0,
        "Relative_Volume": 1.0,
        "Up_Days_5d": 0.0,
        "Down_Days_5d": 0.0,
        "Local_Sharpe_20d": 0.0
    }
    for col, val in neutral_fill.items():
        if col in df.columns:
            df[col].fillna(val, inplace=True)

    # ✅ RIMOZIONE RIGHE RESIDUE CON NAN
    print("🔍 Diagnostica: quante righe hanno almeno un NaN per ogni feature:")
    for col in full_feature_list:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"🛑 {col}: {missing} NaN")
    nan_count_before = len(df)
    df.dropna(subset=full_feature_list, inplace=True)
    print(f"✅ Drop finale dei NaN completato: {nan_count_before - len(df)} righe eliminate")
    # 🔎 DIAGNOSTICA: quali feature causano i NaN
    nan_feature_counts = df[full_feature_list].isna().sum()
    nan_feature_counts = nan_feature_counts[nan_feature_counts > 0].sort_values(ascending=False)
    print("📉 Feature con NaN residui prima del dropna:\n", nan_feature_counts)
    
    # 🔎 Controllo continuità temporale
    df_sorted = df.sort_values("Date")
    date_diffs = df_sorted["Date"].diff().value_counts().sort_index()
    print("📆 Distribuzione differenze tra date consecutive:", date_diffs.head(10))
    
    # 🔎 Diagnostica: quali feature causano più eliminazioni
    elimination_impact = {col: df[col].isna().sum() for col in full_feature_list if df[col].isna().sum() > 0}
    elimination_impact = dict(sorted(elimination_impact.items(), key=lambda x: x[1], reverse=True))
    print("🧹 Feature che causano più eliminazioni:", elimination_impact)

    # 🔎 Verifica se mancano date rispetto al file SPY originale
    spy_calendar = pd.read_csv(path)["Date"]
    spy_calendar = pd.to_datetime(spy_calendar, format="%m/%d/%Y")
    spy_calendar = pd.Series(spy_calendar.sort_values().unique(), name="Date")
    
    missing_dates = set(spy_calendar) - set(df["Date"])
    print(f"📆 Date perse rispetto al file SPY originale: {len(missing_dates)}")
    if missing_dates:
        print("Esempio date mancanti:", sorted(list(missing_dates))[:5])

    return df, full_feature_list

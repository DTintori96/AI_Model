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
import ta

# ✅ MODULI CUSTOM
from feature_engineering import load_and_prepare_data
from trading_env import FullTradingEnv
from exit_env import ExitEnv
from training_pipeline import train_rl_pipeline
from metrics_analysis import analyze_results

# ✅ 1. CARICAMENTO E FEATURE ENGINEERING
df, full_feature_list = load_and_prepare_data("/content/SPY ETF Stock Price History.csv")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=full_feature_list)

# ✅ 2. NORMALIZZAZIONE E SPLIT
df = df.sort_values("Date")
split_date = pd.to_datetime("2022-01-01")
df_train = df[df["Date"] < split_date].copy()
df_test = df[df["Date"] >= split_date].copy()

scaler = StandardScaler()
df_train[full_feature_list] = scaler.fit_transform(df_train[full_feature_list])
df_test[full_feature_list] = scaler.transform(df_test[full_feature_list])

# ✅ 3. TRAINING RL CON ENTRY/EXIT
trade_df, equity_curve, model, exit_model = train_rl_pipeline(df_train, df_test, full_feature_list)

# ✅ 4. ANALISI DEI RISULTATI
analyze_results(trade_df, df_test, equity_curve)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(trade_df, df_test, equity_curve):

    if not trade_df.empty:
        # 📈 CURVA EQUITY CORRETTA — GIORNALIERA CON PnL REALIZZATI
        all_dates = pd.date_range(start=trade_df['entry_date'].min(), end=trade_df['exit_date'].max())
        equity_series = pd.Series(1.0, index=all_dates)

        # Ordina trade per exit
        trade_df = trade_df.sort_values("exit_date")

        # PnL solo alla chiusura del trade
        for _, row in trade_df.iterrows():
            exit_date = pd.to_datetime(row['exit_date'])
            pnl = row['return'] * row['capital_allocated']
            equity_series.loc[exit_date:] += pnl

        true_equity = equity_series
        # 📊 METRICHE SINTETICHE
        num_trades = len(trade_df)
        win_rate = (trade_df['return'] > 0).mean()
        avg_return = trade_df['return'].mean()
        avg_loss = trade_df.loc[trade_df['return'] < 0, 'return'].mean()
        avg_gain = trade_df.loc[trade_df['return'] > 0, 'return'].mean()
        Reward_Risk = avg_gain / abs(avg_loss) if avg_loss != 0 else 0
        sharpe = avg_return / trade_df['return'].std() * np.sqrt(252 / trade_df['duration'].mean()) if trade_df['return'].std() > 0 else 0
        max_dd = (true_equity / true_equity.cummax() - 1).min()
        days_passed = (trade_df['exit_date'].max() - trade_df['entry_date'].min()).days
        cagr = true_equity.iloc[-1] ** (365 / days_passed) - 1
        downside_std = trade_df['return'][trade_df['return'] < 0].std()
        sortino = avg_return / downside_std * np.sqrt(252 / trade_df['duration'].mean()) if downside_std > 0 else 0

        print("📊 METRICHE RL")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Avg return per trade: {avg_return:.4f}")
        print(f"Sharpe ratio per trade: {sharpe:.2f}")
        print(f"CAGR: {cagr:.2%}")
        print(f"Max drawdown: {max_dd:.2%}")
        print(f"Durata media trade: {trade_df['duration'].mean():.2f} giorni")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Average gain: {avg_gain:.4f}")
        print(f"Reward/Risk: {Reward_Risk:.4f}")
        print(f"Sortino Ratio: {sortino:.2f}")
        print(f"Numero totale di trade: {num_trades}")

        # 📁 SALVATAGGIO
        trade_df.to_excel("rl_trades_debug.xlsx", index=False)
        from google.colab import files
        files.download("rl_trades_debug.xlsx")

        # 📊 DISTRIBUZIONE RITORNI
        plt.figure(figsize=(10, 4))
        sns.histplot(trade_df['return'], bins=30, kde=True, color='steelblue')
        plt.title("Distribuzione dei Ritorni per Trade")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 📊 PERFORMANCE PER REGIME
        if 'Market_Regime_Label' in df_test.columns:
            trade_df['entry_date'] = pd.to_datetime(trade_df['entry_date'])
            df_indexed = df_test.set_index('Date') if 'Date' in df_test.columns else df_test
            trade_df['Regime'] = trade_df['entry_date'].apply(lambda d: df_indexed.loc[d, 'Market_Regime_Label'] if d in df_indexed.index else 'Unknown')

            regime_perf = trade_df.groupby('Regime')['return'].agg(['count', 'mean', 'std'])
            regime_perf['Sharpe'] = regime_perf['mean'] / regime_perf['std']
            print("📈 Performance per Regime:", regime_perf)

        # 📊 DURATA vs RITORNO (scatter)
        plt.figure(figsize=(10, 4))
        sns.scatterplot(data=trade_df, x='duration', y='return', alpha=0.6)
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        plt.title("Ritorno in funzione della Durata del Trade")
        plt.xlabel("Durata (giorni)")
        plt.ylabel("Return")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 📊 PLOT EQUITY CURVE
        # Buy & Hold (su dati test)
        daily_returns = df_test.set_index('Date')['Close'].pct_change().fillna(0)
        buyhold = (1 + daily_returns).cumprod()

        # 📊 PLOT
        plt.figure(figsize=(14, 5))
        plt.plot(true_equity, label='RL Strategy (Corretto)', linewidth=2, color='orange')
        plt.plot(buyhold, label='Buy & Hold', linestyle='--', color='blue')
        plt.title("Equity Curve Corretta: RL vs Buy & Hold")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
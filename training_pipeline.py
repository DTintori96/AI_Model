import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import FullTradingEnv
from exit_env import ExitEnv

def train_rl_pipeline(df_train, df_test, feature_columns):
    # ✅ CODIFICA RL: ALLENAMENTO AGENTE PPO SU TRADINGENV

    # NORMALIZZAZIONE
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Normalizza solo sui dati di training
    df_train[feature_columns] = scaler.fit_transform(df_train[feature_columns])
    df_test[feature_columns] = scaler.transform(df_test[feature_columns])

    # ANALISI RISULTATI BASE
    # ✅ Dopo il training, creiamo l'ambiente di test con exit_model definito
    df_trades = []

    # ✅ INIZIALIZZAZIONE EXIT MODEL PRIMA DEL CICLO
    # Necessaria per evitare NameError all'uso iniziale di exit_model
    print("🔧 Inizializzazione exit model per bootstrap")
    dummy_trade_df = pd.DataFrame([{
        'entry_date': df_train['Date'].iloc[0],
        'entry_price': df_train['Close'].iloc[0],
        'position': 1
    }])
    df_prices_train = df_train.set_index("Date")
    exit_env_bootstrap = DummyVecEnv([lambda: ExitEnv(dummy_trade_df, df_prices_train, feature_columns)])
    exit_model = PPO("MlpPolicy", exit_env_bootstrap, verbose=1)
    exit_model.learn(total_timesteps=10_000)

    # ✅ NUOVO CICLO INTERATTIVO ENTRY/EXIT SINCRONO
    trade_buffer = []
    n_iterazioni = 2
    entry_steps = 10000
    exit_steps = 2000  # meno passi ma aggiornamento frequente
    MAX_BUFFER_SIZE = 1000
    MINI_BATCH_SIZE = 50  # ogni batch trade eseguiti, aggiorna exit model

    for ciclo in range(n_iterazioni):
        print(f"🔁 Iterazione {ciclo+1}/{n_iterazioni}")

        env = DummyVecEnv([lambda: FullTradingEnv(df_train, feature_columns, exit_model)])
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=entry_steps, reset_num_timesteps=False)

        train_env_sim = FullTradingEnv(df_train.reset_index(), feature_columns=feature_columns, exit_model=exit_model)
        obs = train_env_sim.reset()
        new_trades = []

        while not train_env_sim.done:
            action, _ = model.predict(obs)
            obs, reward, done, info = train_env_sim.step(action)
            if 'return' in info:
                trade = {
                    'entry_date': train_env_sim.entry_date,
                    'entry_price': train_env_sim.entry_price,
                    **info
                }
                new_trades.append(trade)
                trade_buffer.append(trade)

                if len(trade_buffer) > MAX_BUFFER_SIZE:
                    trade_buffer = trade_buffer[-MAX_BUFFER_SIZE:]

            if len(new_trades) >= MINI_BATCH_SIZE:
                    print(f"🧠 Allenamento exit model su ultimi {MINI_BATCH_SIZE} trade.")
                    # Hedge-style decay: peso temporale soft dei trade passati
                    combined_trades = []
                    for i, trade in enumerate(reversed(trade_buffer[-MAX_BUFFER_SIZE:])):
                        decay_weight = 1 - (i / MAX_BUFFER_SIZE)  # peso da 1 (recente) a ~0 (vecchio)
                        trade_copy = trade.copy()
                        trade_copy['decay_weight'] = decay_weight  # assegnazione peso
                        combined_trades.append(trade_copy)

                    exit_env_train = DummyVecEnv([
                        lambda: ExitEnv(pd.DataFrame(combined_trades), df_train, feature_columns)
                    ])
                    exit_model.set_env(exit_env_train)
                    exit_model.learn(total_timesteps=exit_steps, reset_num_timesteps=False)
                    new_trades = []

        if new_trades:
            print(f"✅ Fine iterazione {ciclo+1}, trade generati: {len(new_trades)}")
        else:
            print("⚠️ Nessun trade generato in questa iterazione.")


    # ✅ Dopo il training, creiamo l'ambiente di test con exit_model definito
    test_env = FullTradingEnv(df_test.reset_index(), feature_columns=feature_columns, exit_model=exit_model)
    obs = test_env.reset()
    all_trades = []
    in_trade = False

    while not test_env.done:
        print(f"[TEST] Step: {test_env.current_step}, Position: {test_env.position}, Entry: {test_env.entry_date}")
        action, _ = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        print(f"[ENTRY] step={test_env.current_step}, reward={reward:.4f}, done={test_env.done}")

        if 'return' in info:
            print(f"[TRADE #{test_env.trade_count}] Entry: {info['entry_date']}, Exit: {info['exit_date']}, Return: {info['return']:.4f}")
            test_env.trade_count += 1
            all_trades.append({
                'entry_date': test_env.entry_date,
                'entry_price': test_env.entry_price,
                **info
            })
    
    trade_df = pd.DataFrame(all_trades)
    
    # Calcola equity giornaliera corretta
    all_dates = pd.date_range(start=trade_df['entry_date'].min(), end=trade_df['exit_date'].max())
    equity_series = pd.Series(1.0, index=all_dates)
    
    # Ordina trade per exit e somma i PnL
    trade_df = trade_df.sort_values("exit_date")
    for _, row in trade_df.iterrows():
        exit_date = pd.to_datetime(row['exit_date'])
        pnl = row['return'] * row['capital_allocated']
        equity_series.loc[exit_date:] += pnl
    
    true_equity = equity_series

    return trade_df, true_equity, model, exit_model

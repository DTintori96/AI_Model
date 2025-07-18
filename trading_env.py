import numpy as np
import pandas as pd
import gym
from gym import spaces

from exit_env import ExitEnv

class FullTradingEnv(gym.Env):
    # ➕ MODIFICA: nuova logica di reward basata su potenziale massimo
    def _get_max_return(self, start_step, horizon=5):
        if start_step + horizon >= len(self.df):
            return 0.0
        entry_price = self.df.iloc[start_step]['Close']
        future_prices = self.df.iloc[start_step+1:start_step+1+horizon]['Close']
        return_max = (future_prices.max() - entry_price) / entry_price
        return return_max

    def __init__(self, df, feature_columns, exit_model, max_exit_delay=20, leverage_threshold=1.0):
        super(FullTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.exit_model = exit_model
        self.max_exit_delay = max_exit_delay
        self.leverage_threshold = leverage_threshold

        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(feature_columns) + 2,), dtype=np.float32)

        self.margin_rate = 0.1
        self.active_trades = []
        self.closed_trades = []
        self.reset()
        self.trade_count = 0

    def _get_observation(self):
      if self.current_step >= len(self.df):
          return np.zeros(self.observation_space.shape, dtype=np.float32)
      obs = self.df.iloc[self.current_step][self.feature_columns].values.astype(np.float32)
      obs = np.append(obs, [self.position, self.days_in_trade])
      return obs

    def reset(self):
        self.realized_equity = 1.0
        self.pending_realized_pnl = 0.0
        self.last_closed_equity = 1.0
        self.last_exit_date = pd.Timestamp.min
        self.current_date = None
        self.current_step = 1
        self.done = False
        self.position = 0
        self.entry_price = None
        self.entry_date = None
        self.days_in_trade = 0
        self.active_trades = []
        self.equity_ledger = {}
        self.leverage_usage = {}
        self.episode_reward = 0
        return self._get_observation()

    def step(self, action):
        entry_date = self.df.iloc[self.current_step]['Date']
        current_row = self.df.iloc[self.current_step]
        reward = 0
        info = {}

        leverage = float(action[0])
        past_trades = [t for t in self.closed_trades if t["exit_date"] < entry_date]
        equity_available = 1.0 + sum(t["pnl"] for t in past_trades)
        allocated_capital = equity_available * abs(leverage)
        new_margin = allocated_capital * self.margin_rate

        self.active_trades = [t for t in self.active_trades if t["exit_date"] >= entry_date]
        total_margin_used = sum(t["margin_required"] for t in self.active_trades)

        if (
            self.position == 0 and
            abs(leverage) >= self.leverage_threshold and
            total_margin_used + new_margin <= self.realized_equity
        ):
            self.entry_price = current_row['Close']
            self.entry_date = current_row['Date']
            self.position = np.sign(leverage)
            self.trade_leverage = abs(leverage)
            self.days_in_trade = 1

            trade = pd.DataFrame([{
                'entry_date': self.entry_date,
                'entry_price': self.entry_price,
                'position': self.position,
                'leverage': self.trade_leverage
            }])
            df_prices = self.df.set_index("Date")

            exit_env = ExitEnv(trade, df_prices, self.feature_columns, max_exit_delay=self.max_exit_delay)
            obs_exit = exit_env.reset()
            done_exit = False
            while not done_exit:
                exit_action, _ = self.exit_model.predict(obs_exit, deterministic=True)
                obs_exit, _, done_exit, exit_info = exit_env.step(exit_action)

            pnl = exit_info['return'] * allocated_capital
            self.realized_equity += pnl
            reward = pnl / allocated_capital if allocated_capital > 0 else 0
            self.closed_trades.append({"exit_date": exit_info["exit_date"], "pnl": pnl})

            if self.current_step + 1 < len(self.df):
                next_day = self.df.iloc[self.current_step + 1]['Date']
                self.equity_ledger[str(next_day.date())] = self.realized_equity + self.pending_realized_pnl

            info = dict(exit_info)
            info['PnL'] = pnl
            info['equity_used_for_entry'] = equity_available
            info['capital_allocated'] = allocated_capital
            info['equity_post_trade'] = equity_available + pnl
            info['leverage'] = self.trade_leverage
            info['entry_date'] = self.entry_date
            info['entry_price'] = self.entry_price
            info['position'] = self.position

            self.active_trades.append({
                "entry_date": self.entry_date,
                "exit_date": exit_info["exit_date"],
                "allocated_capital": allocated_capital,
                "margin_required": new_margin,
                "pnl": pnl
            })

            self.position = 0
            self.entry_price = None
            self.days_in_trade = 0
            self.current_step += 1

        else:
            self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.done = True

        obs = self._get_observation()

        return obs, reward, self.done, info
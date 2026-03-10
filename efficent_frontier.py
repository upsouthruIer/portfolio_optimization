import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# -----------------------------
# 1) 설정
# -----------------------------
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]  # 원하는 5개로 변경
period = "5y"
interval = "1d"
rf = 0.03               # 무위험이자율(연)
n_portfolios = 50_000   # 무작위 포트폴리오 개수
n_frontier = 40         # frontier 선을 위한 목표수익률 개수

np.random.seed(42)

# -----------------------------
# 2) 데이터 다운로드
# -----------------------------
data = yf.download(tickers, period=period, interval=interval, auto_adjust=False, progress=False)

price_key = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"
prices = data[price_key]

prices = prices.dropna(how="all")

# -----------------------------
# 3) 일별 수익률 -> 연환산 mu, cov
# -----------------------------

# 로그 수익률 계산
returns_daily = np.log(prices / prices.shift(1)).dropna()

# 연환산 기대수익률
mu = returns_daily.mean() * 252

# 연환산 공분산
cov = returns_daily.cov() * 252

n_assets = len(tickers)

# numpy array 변환 (최적화 계산용)
mu_v = mu.values
cov_m = cov.values

# -----------------------------
# 4) 무작위 포트폴리오(점구름)
# -----------------------------
W = np.random.random((n_portfolios, n_assets))
W = W / W.sum(axis=1, keepdims=True)

port_returns = W @ mu_v
port_vars = np.einsum("ij,jk,ik->i", W, cov_m, W)
port_vols = np.sqrt(port_vars)
port_sharpes = (port_returns - rf) / port_vols

idx_max_sharpe = np.argmax(port_sharpes)
idx_min_vol = np.argmin(port_vols)

# -----------------------------
# 5) Efficient Frontier 선(최적 포트폴리오들 연결)
#    목표수익률별 최소분산 최적화
# -----------------------------
def portfolio_return(w):
    return w @ mu_v

def portfolio_var(w):
    return w @ cov_m @ w

# 제약: 합=1, (필요 시) 목표수익률=target
bounds = [(0.0, 1.0)] * n_assets
w0 = np.ones(n_assets) / n_assets

targets = np.linspace(port_returns.min(), port_returns.max(), n_frontier)

frontier_rets = []
frontier_vols = []
frontier_weights = []

for tr in targets:
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w, tr=tr: portfolio_return(w) - tr},
    )
    res = minimize(
        portfolio_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if res.success:
        w_opt = res.x
        r_opt = portfolio_return(w_opt)
        v_opt = np.sqrt(portfolio_var(w_opt))
        frontier_rets.append(r_opt)
        frontier_vols.append(v_opt)
        frontier_weights.append(w_opt)

frontier_rets = np.array(frontier_rets)
frontier_vols = np.array(frontier_vols)

# (정렬: 선이 꼬이지 않게 volatility 기준 정렬)
order = np.argsort(frontier_vols)
frontier_vols = frontier_vols[order]
frontier_rets = frontier_rets[order]

# 누적 최대 수익률만 유지 (지배당한 점 제거) 이상한 점 제거
mask = frontier_rets == np.maximum.accumulate(frontier_rets)
frontier_vols, frontier_rets = frontier_vols[mask], frontier_rets[mask]

# -----------------------------
# 6) 그래프
# -----------------------------
plt.figure(figsize=(10, 6))
sc = plt.scatter(port_vols, port_returns, c=port_sharpes, s=6)
plt.colorbar(sc, label="Sharpe Ratio")

# ✅ Efficient Frontier 선
plt.plot(frontier_vols, frontier_rets, linewidth=2.5, label="Efficient Frontier (Optimized)")

# 최대 샤프 / 최소 변동성
plt.scatter(port_vols[idx_max_sharpe], port_returns[idx_max_sharpe], marker="*", s=220, label="Max Sharpe")
plt.scatter(port_vols[idx_min_vol], port_returns[idx_min_vol], marker="X", s=160, label="Min Volatility")

plt.title("Efficient Frontier (Random Portfolios + Optimized Frontier Line)")
plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")
plt.legend()
plt.tight_layout()
plt.show()
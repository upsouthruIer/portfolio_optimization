import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# -----------------------------
# 1) 설정
# -----------------------------
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
period = "5y"
interval = "1d"
rf = 0.03
n_portfolios = 50_000
n_frontier = 40

# -----------------------------
# 추가 입력값 (holdings-based)
# -----------------------------
# 현재 보유 주식 수
shares_old = np.array([10, 8, 5, 6, 7], dtype=float)

# 개별 자산 비중 제한
weight_min = np.array([0.00, 0.00, 0.00, 0.00, 0.00], dtype=float)
weight_max = np.array([0.40, 0.40, 0.40, 0.40, 0.40], dtype=float)

# Turnover 제한
# 예: 0.20 이면 전체 포트폴리오 가치의 20%까지만 매매 가능
turnover_limit = 0.20

np.random.seed(42)

# -----------------------------
# 2) 가격 데이터 다운로드
# -----------------------------
data = yf.download(
    tickers,
    period=period,
    interval=interval,
    auto_adjust=False,
    progress=False
)

price_key = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"
prices = data[price_key]
prices = prices.dropna(how="all")

# -----------------------------
# 3) 일별 수익률 -> 연환산 mu, cov
# -----------------------------
returns_daily = np.log(prices / prices.shift(1)).dropna()

mu = returns_daily.mean() * 252
cov = returns_daily.cov() * 252

n_assets = len(tickers)

mu_v = mu.values
cov_m = cov.values

# -----------------------------
# 4) 현재 가격 / 기존 포트폴리오 정보
# -----------------------------
current_prices = prices.iloc[-1].values.astype(float)

# 기존 자산 가치
values_old = current_prices * shares_old
portfolio_value_old = values_old.sum()

# 기존 비중
weights_old = values_old / portfolio_value_old

# -----------------------------
# 5) holdings -> weights 변환 함수
# -----------------------------
def shares_to_weights(shares):
    values = current_prices * shares
    total_value = values.sum()
    return values / total_value

def portfolio_return(shares):
    w = shares_to_weights(shares)
    return w @ mu_v

def portfolio_var(shares):
    w = shares_to_weights(shares)
    return w @ cov_m @ w

def portfolio_vol(shares):
    return np.sqrt(portfolio_var(shares))

def portfolio_turnover(shares):
    traded_value = np.sum(current_prices * np.abs(shares - shares_old))
    return traded_value / portfolio_value_old

# -----------------------------
# 6) 초기값 / bounds
# -----------------------------
# 기존 보유수량을 초기값으로 사용
x0 = shares_old.copy()

# shares에 대한 단순 bounds
# 공매도 금지이므로 0 이상
# 상한은 weight_max를 바탕으로 대략 설정
# budget constraint가 있으므로 너무 엄밀할 필요는 없지만 안정성을 위해 설정
shares_upper = (weight_max * portfolio_value_old) / current_prices
bounds = [(0.0, max(1e-12, ub)) for ub in shares_upper]

# -----------------------------
# 7) 제약 조건
# -----------------------------
constraints_base = []

# 1) Budget constraint
# 총 자산가치는 기존 포트폴리오 가치와 같게 유지
constraints_base.append({
    "type": "eq",
    "fun": lambda s: np.sum(current_prices * s) - portfolio_value_old
})

# 2) Weight constraint
# weight_min <= w_i <= weight_max
for i in range(n_assets):
    constraints_base.append({
        "type": "ineq",
        "fun": lambda s, i=i: shares_to_weights(s)[i] - weight_min[i]
    })
    constraints_base.append({
        "type": "ineq",
        "fun": lambda s, i=i: weight_max[i] - shares_to_weights(s)[i]
    })

# 3) Turnover constraint
constraints_base.append({
    "type": "ineq",
    "fun": lambda s: turnover_limit - portfolio_turnover(s)
})

# -----------------------------
# 8) 무작위 포트폴리오 생성 (기존 형식 최대한 유지)
#    단, holdings 기반 제약을 만족하는 것만 저장
# -----------------------------
random_shares = []
port_returns = []
port_vars = []
port_vols = []
port_sharpes = []

n_trials = 0
max_trials = n_portfolios * 20  # feasible sample 확보용

while len(random_shares) < n_portfolios and n_trials < max_trials:
    n_trials += 1

    # 무작위 비중 생성
    w_rand = np.random.random(n_assets)
    w_rand = w_rand / w_rand.sum()

    # weight -> shares 변환
    shares_rand = (w_rand * portfolio_value_old) / current_prices

    # 제약 확인
    w_check = shares_to_weights(shares_rand)
    turnover_check = portfolio_turnover(shares_rand)

    if np.all(w_check >= weight_min - 1e-10) and np.all(w_check <= weight_max + 1e-10):
        if turnover_check <= turnover_limit + 1e-10:
            r = portfolio_return(shares_rand)
            v = portfolio_var(shares_rand)
            vol = np.sqrt(v)
            sr = (r - rf) / vol if vol > 0 else np.nan

            random_shares.append(shares_rand)
            port_returns.append(r)
            port_vars.append(v)
            port_vols.append(vol)
            port_sharpes.append(sr)

random_shares = np.array(random_shares)
port_returns = np.array(port_returns)
port_vars = np.array(port_vars)
port_vols = np.array(port_vols)
port_sharpes = np.array(port_sharpes)

if len(port_returns) == 0:
    raise ValueError("무작위 포트폴리오 중 제약을 만족하는 feasible portfolio를 찾지 못했습니다.")

idx_max_sharpe = np.nanargmax(port_sharpes)
idx_min_vol = np.argmin(port_vols)

# -----------------------------
# 9) Efficient Frontier (holdings-based)
# -----------------------------
targets = np.linspace(port_returns.min(), port_returns.max(), n_frontier)

frontier_rets = []
frontier_vols = []
frontier_shares = []
frontier_weights = []

for tr in targets:
    constraints = constraints_base.copy()
    constraints.append({
        "type": "eq",
        "fun": lambda s, tr=tr: portfolio_return(s) - tr
    })

    res = minimize(
        portfolio_var,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if res.success:
        s_opt = res.x
        w_opt = shares_to_weights(s_opt)
        r_opt = portfolio_return(s_opt)
        v_opt = np.sqrt(portfolio_var(s_opt))

        frontier_rets.append(r_opt)
        frontier_vols.append(v_opt)
        frontier_shares.append(s_opt)
        frontier_weights.append(w_opt)

frontier_rets = np.array(frontier_rets)
frontier_vols = np.array(frontier_vols)
frontier_shares = np.array(frontier_shares)
frontier_weights = np.array(frontier_weights)

if len(frontier_rets) == 0:
    raise ValueError("Efficient Frontier 최적화에 성공한 해가 없습니다. 제약조건을 완화해보세요.")

# -----------------------------
# 10) 최대 샤프 / 최소 변동성 포트폴리오
# -----------------------------
shares_max_sharpe = random_shares[idx_max_sharpe]
weights_max_sharpe = shares_to_weights(shares_max_sharpe)

shares_min_vol = random_shares[idx_min_vol]
weights_min_vol = shares_to_weights(shares_min_vol)

# -----------------------------
# 11) 결과 출력
# -----------------------------
print("\n[기존 포트폴리오]")
print("Shares:", shares_old)
print("Weights:", np.round(weights_old, 4))
print("Return:", round(portfolio_return(shares_old), 4))
print("Volatility:", round(portfolio_vol(shares_old), 4))
print("Turnover:", round(portfolio_turnover(shares_old), 4))

print("\n[Max Sharpe Portfolio]")
print("Shares:", np.round(shares_max_sharpe, 4))
print("Weights:", np.round(weights_max_sharpe, 4))
print("Return:", round(port_returns[idx_max_sharpe], 4))
print("Volatility:", round(port_vols[idx_max_sharpe], 4))
print("Sharpe:", round(port_sharpes[idx_max_sharpe], 4))
print("Turnover:", round(portfolio_turnover(shares_max_sharpe), 4))

print("\n[Min Volatility Portfolio]")
print("Shares:", np.round(shares_min_vol, 4))
print("Weights:", np.round(weights_min_vol, 4))
print("Return:", round(port_returns[idx_min_vol], 4))
print("Volatility:", round(port_vols[idx_min_vol], 4))
print("Sharpe:", round(port_sharpes[idx_min_vol], 4))
print("Turnover:", round(portfolio_turnover(shares_min_vol), 4))

# -----------------------------
# 12) 시각화
# -----------------------------
plt.figure(figsize=(10, 6))
sc = plt.scatter(port_vols, port_returns, c=port_sharpes, s=6)
plt.colorbar(sc, label="Sharpe Ratio")

plt.plot(frontier_vols, frontier_rets, linewidth=2.5, label="Efficient Frontier (Holdings-based)")
plt.scatter(port_vols[idx_max_sharpe], port_returns[idx_max_sharpe], marker="*", s=220, label="Max Sharpe")
plt.scatter(port_vols[idx_min_vol], port_returns[idx_min_vol], marker="X", s=160, label="Min Volatility")

plt.title("Efficient Frontier with Holdings-based Optimization")
plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
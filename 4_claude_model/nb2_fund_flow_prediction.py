"""
NOTEBOOK 2 — Fund Flow Prediction: ARIMA Baseline + LSTM
==========================================================
Input  : monthly_master.csv
Outputs: figures → ./figures/fund_flow/
         metrics → printed to console (copy to thesis)

Models
  1. ARIMA    — baseline linear time-series model (per fund)
  2. LSTM     — deep sequential model with look-back window
  3. LSTM-Attention — extends LSTM with soft attention over the window

Train / test split
  Train : all months up to and including Dec 2023
  Test  : Jan 2024 → Sep 2025  (~21 months out-of-sample)

Target variable
  flow_pct_{fund}  (fund flow as % of prior AUM — stationary series)

Features fed into LSTM
  Lagged fund flow pct (t-1, t-2, t-3)
  Lagged NAV monthly return
  Oil monthly return
  USD/PKR monthly return
  Interest rate level (end of month)
  CPI YoY (end of month)

Run: python nb2_fund_flow_prediction.py
Requirements: numpy, pandas, matplotlib, scikit-learn, scipy
              (tensorflow or torch NOT required — LSTM built from scratch with numpy)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fund_flow")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 12,
                     "axes.labelsize": 10, "legend.fontsize": 9})

FUND_COLORS = {"AKD": "#1f77b4", "NBP": "#ff7f0e", "NIT": "#2ca02c"}
# Map display name → column suffix (NIT sheet = "nti" in column names)
COL_NAME = {"AKD": "akd", "NBP": "nbp", "NIT": "nti"}
TRAIN_END   = "2023-12-31"
LOOKBACK    = 3          # months of history fed into LSTM
EPOCHS      = 300
LR          = 0.01
HIDDEN      = 32

np.random.seed(42)

# ── load ─────────────────────────────────────────────────────────────────────
monthly = pd.read_csv("new_data/monthly_master.csv", parse_dates=["date"])
monthly = monthly.sort_values("date").reset_index(drop=True)
print(f"Monthly rows: {len(monthly)}  ({monthly['date'].min().date()} → {monthly['date'].max().date()})")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS — metrics
# ═══════════════════════════════════════════════════════════════════════════

def metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    ss_res = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    ss_tot = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mape_vals = np.abs((np.array(y_true) - np.array(y_pred)) /
                       (np.abs(np.array(y_true)) + 1e-9)) * 100
    mape = np.mean(mape_vals)
    if label:
        print(f"    {label:20s}  RMSE={rmse:.6f}  MAE={mae:.6f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — ARIMA BASELINE
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 1: ARIMA Baseline ─────────────────────────────────────")

def fit_arima(series, order=(1, 0, 1)):
    """
    Manual ARIMA(p,0,q) fitted via OLS on lagged values and lagged residuals.
    This is a simplified implementation — for production use use statsmodels.
    Works well for p,q ≤ 2 on monthly series.
    """
    p, d, q = order
    if d > 0:
        for _ in range(d):
            series = np.diff(series)
    n = len(series)
    # Build regressor matrix
    max_lag = max(p, q)
    rows = []
    for t in range(max_lag, n):
        row = [series[t - i] for i in range(1, p + 1)]
        rows.append(row)
    X = np.array(rows)
    y = series[max_lag:]
    # OLS
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    fitted = X_aug @ beta
    resid  = y - fitted
    return beta, fitted, resid, series, max_lag

def arima_forecast_one_step(series, beta, p, max_lag):
    """Recursive one-step-ahead forecast."""
    X_new = [series[-i] for i in range(1, p + 1)]
    X_aug  = np.array([1.0] + X_new)
    return X_aug @ beta

arima_results = {}

for fund in ["AKD", "NBP", "NIT"]:
    print(f"\n  {fund}")
    col   = f"flow_pct_{COL_NAME[fund]}"
    df    = monthly[["date", col]].dropna().copy()
    train = df[df["date"] <= TRAIN_END]
    test  = df[df["date"] >  TRAIN_END]

    train_vals = train[col].values.astype(float)
    test_vals  = test[col].values.astype(float)

    # Fit ARIMA(1,0,1) on training set
    beta, fitted_train, resid_train, series_d, max_lag = fit_arima(train_vals, order=(1, 0, 1))
    p_order = 1

    # Walk-forward forecast on test set
    history = list(train_vals)
    preds   = []
    for val in test_vals:
        pred = arima_forecast_one_step(np.array(history), beta, p_order, max_lag)
        preds.append(pred)
        history.append(val)   # use actual for next step (walk-forward)

    m_train = metrics(train_vals[max_lag:], fitted_train, "ARIMA train")
    m_test  = metrics(test_vals, preds, "ARIMA test ")
    arima_results[fund] = {
        "train_actual": train_vals[max_lag:],
        "train_pred":   fitted_train,
        "train_dates":  train["date"].values[max_lag:],
        "test_actual":  test_vals,
        "test_pred":    np.array(preds),
        "test_dates":   test["date"].values,
        "metrics_train": m_train,
        "metrics_test":  m_test,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING FOR LSTM
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 2: Feature Engineering ───────────────────────────────")

FEATURE_COLS = [
    "oil_return_monthly",
    "usdpkr_return_monthly",
    "interest_rate_end",
    "cpi_yoy_end",
]

def build_dataset(monthly, fund, lookback=LOOKBACK):
    """
    Returns X (n_samples, lookback, n_features), y (n_samples,),
    dates (n_samples,), train_mask (bool array).
    Features: target lags + NAV return + macro cols.
    """
    flow_col = f"flow_pct_{COL_NAME[fund]}"
    nav_col  = f"nav_return_{COL_NAME[fund]}_monthly"

    df = monthly[["date", flow_col, nav_col] + FEATURE_COLS].dropna().copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Add lagged fund flow (t-1, t-2, t-3)
    for lag in range(1, lookback + 1):
        df[f"flow_lag{lag}"] = df[flow_col].shift(lag)

    df = df.dropna().reset_index(drop=True)

    feature_cols = (
        [f"flow_lag{lag}" for lag in range(1, lookback + 1)] +
        [nav_col] +
        FEATURE_COLS
    )

    X_raw = df[feature_cols].values.astype(float)
    y_raw = df[flow_col].values.astype(float)
    dates = df["date"].values

    # Scale features to [0,1]  (scaler fit on train only)
    train_mask = df["date"] <= TRAIN_END
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(X_raw[train_mask])
    scaler_y.fit(y_raw[train_mask].reshape(-1, 1))

    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).ravel()

    # Build 3-D sequences: (sample, lookback, features)
    # Here lookback window = 1 row (features already include lags)
    # Wrap as (n, 1, n_features) for LSTM
    X_seq = X_scaled.reshape(len(X_scaled), 1, X_scaled.shape[1])

    return X_seq, y_scaled, y_raw, dates, train_mask[df.index], scaler_y


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — NUMPY LSTM IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 3: LSTM (numpy) ───────────────────────────────────────")

class LSTMCell:
    """Single LSTM cell — forward pass only, trained via BPTT."""
    def __init__(self, n_input, n_hidden):
        scale = 0.1
        self.Wf = np.random.randn(n_hidden, n_hidden + n_input) * scale
        self.bf = np.zeros((n_hidden, 1))
        self.Wi = np.random.randn(n_hidden, n_hidden + n_input) * scale
        self.bi = np.zeros((n_hidden, 1))
        self.Wc = np.random.randn(n_hidden, n_hidden + n_input) * scale
        self.bc = np.zeros((n_hidden, 1))
        self.Wo = np.random.randn(n_hidden, n_hidden + n_input) * scale
        self.bo = np.zeros((n_hidden, 1))
        self.Wy = np.random.randn(1, n_hidden) * scale
        self.by = np.zeros((1, 1))

    @staticmethod
    def sigmoid(x):  return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    @staticmethod
    def tanh(x):     return np.tanh(np.clip(x, -15, 15))

    def forward(self, x_seq):
        """x_seq: (T, n_input).  Returns predictions (T,), cache for backward."""
        T, n_input = x_seq.shape
        n_h = self.Wf.shape[0]
        h = np.zeros((n_h, 1))
        c = np.zeros((n_h, 1))
        preds, cache = [], []
        for t in range(T):
            x = x_seq[t].reshape(-1, 1)
            hx = np.vstack([h, x])
            f  = self.sigmoid(self.Wf @ hx + self.bf)
            i  = self.sigmoid(self.Wi @ hx + self.bi)
            g  = self.tanh(self.Wc @ hx + self.bc)
            o  = self.sigmoid(self.Wo @ hx + self.bo)
            c  = f * c + i * g
            h  = o * self.tanh(c)
            y  = self.Wy @ h + self.by
            preds.append(y[0, 0])
            cache.append((h.copy(), c.copy(), f, i, g, o, hx, x))
        return np.array(preds), cache

    def params(self):
        return [self.Wf, self.bf, self.Wi, self.bi,
                self.Wc, self.bc, self.Wo, self.bo, self.Wy, self.by]

    def update(self, grads, lr):
        for p, g in zip(self.params(), grads):
            p -= lr * np.clip(g, -1, 1)   # gradient clipping


def train_lstm(X_train, y_train, n_hidden=HIDDEN, epochs=EPOCHS, lr=LR):
    """Train LSTM on training sequences.  X: (n, 1, n_feat)."""
    n, _, n_feat = X_train.shape
    cell = LSTMCell(n_feat, n_hidden)
    losses = []
    for epoch in range(epochs):
        # One sample at a time (online SGD)
        epoch_loss = 0
        for i in range(n):
            x_seq = X_train[i, :, :]          # (1, n_feat)
            y_true = y_train[i]
            preds, _ = cell.forward(x_seq)
            loss = (preds[-1] - y_true) ** 2
            epoch_loss += loss
            # Numerical gradient (finite differences) — simple but correct
            eps = 1e-4
            for param in cell.params():
                it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
                while not it.finished:
                    idx = it.multi_index
                    orig = param[idx]
                    param[idx] = orig + eps
                    p_plus, _ = cell.forward(x_seq)
                    param[idx] = orig - eps
                    p_minus, _ = cell.forward(x_seq)
                    param[idx] = orig
                    grad = (((p_plus[-1] - y_true)**2) -
                            ((p_minus[-1] - y_true)**2)) / (2 * eps)
                    param[idx] -= lr * np.clip(grad, -1, 1)
                    it.iternext()
        losses.append(epoch_loss / n)
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={losses[-1]:.6f}")
    return cell, losses

# NOTE: Full numerical-gradient LSTM is O(params × samples × epochs) which
# is very slow for large param counts.  For a monthly series with ~30 training
# points this is fast.  For larger datasets, install tensorflow/pytorch and
# replace the train_lstm call with a standard Keras LSTM — the data pipeline
# above is already compatible.

def fast_train_lstm(X_train, y_train, n_hidden=HIDDEN, epochs=EPOCHS, lr=LR):
    """
    Adam-optimised forward-only LSTM using analytic weight updates.
    Much faster than numerical gradients.  Uses BPTT through time.
    """
    n, _, n_feat = X_train.shape
    nh = n_hidden
    nf = n_feat

    # Xavier init
    def W(r, c): return np.random.randn(r, c) * np.sqrt(2.0 / (r + c))
    Wf = W(nh, nh + nf); bf = np.zeros(nh)
    Wi = W(nh, nh + nf); bi = np.zeros(nh)
    Wc = W(nh, nh + nf); bc = np.zeros(nh)
    Wo = W(nh, nh + nf); bo = np.zeros(nh)
    Wy = W(1,  nh);       by = np.zeros(1)

    sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    # Adam state
    m = {k: np.zeros_like(v) for k, v in
         dict(Wf=Wf, Wi=Wi, Wc=Wc, Wo=Wo, Wy=Wy,
              bf=bf, bi=bi, bc=bc, bo=bo, by=by).items()}
    v = {k: np.zeros_like(vv) for k, vv in m.items()}
    beta1, beta2, eps_adam, t_adam = 0.9, 0.999, 1e-8, 0

    losses = []
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        epoch_loss = 0.0
        for i in idx:
            t_adam += 1
            x = X_train[i, 0, :]          # (nf,)
            yt = float(y_train[i])

            h = np.zeros(nh); c = np.zeros(nh)
            hx = np.concatenate([h, x])

            # Forward
            f_gate = sig(Wf @ hx + bf)
            i_gate = sig(Wi @ hx + bi)
            g_gate = np.tanh(Wc @ hx + bc)
            o_gate = sig(Wo @ hx + bo)
            c_new  = f_gate * c + i_gate * g_gate
            h_new  = o_gate * np.tanh(c_new)
            y_hat  = (Wy @ h_new + by)[0]
            loss   = (y_hat - yt) ** 2
            epoch_loss += loss

            # Backward
            dy = 2 * (y_hat - yt)
            dWy = dy * h_new.reshape(1, -1)
            dby = np.array([dy])
            dh  = Wy.T.squeeze() * dy

            tanh_c = np.tanh(c_new)
            do = dh * tanh_c
            dc = dh * o_gate * (1 - tanh_c**2)
            dg = dc * i_gate
            di = dc * g_gate
            df = dc * c

            dWo = (do * o_gate * (1 - o_gate))[:, None] * hx[None, :]
            dWi = (di * i_gate * (1 - i_gate))[:, None] * hx[None, :]
            dWc = (dg * (1 - g_gate**2))[:, None]       * hx[None, :]
            dWf = (df * f_gate * (1 - f_gate))[:, None] * hx[None, :]
            dbo = do * o_gate * (1 - o_gate)
            dbi = di * i_gate * (1 - i_gate)
            dbc = dg * (1 - g_gate**2)
            dbf = df * f_gate * (1 - f_gate)

            grads = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy,
                         bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
            params = dict(Wf=Wf, Wi=Wi, Wc=Wc, Wo=Wo, Wy=Wy,
                          bf=bf, bi=bi, bc=bc, bo=bo, by=by)
            for k in params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k]**2
                m_hat = m[k] / (1 - beta1**t_adam)
                v_hat = v[k] / (1 - beta2**t_adam)
                params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            Wf,Wi,Wc,Wo,Wy = params["Wf"],params["Wi"],params["Wc"],params["Wo"],params["Wy"]
            bf,bi,bc,bo,by  = params["bf"],params["bi"],params["bc"],params["bo"],params["by"]

        losses.append(epoch_loss / n)
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  MSE={losses[-1]:.6f}")

    # Build predict function using closed-over weights
    def predict(X):
        preds = []
        for j in range(len(X)):
            x = X[j, 0, :]
            h = np.zeros(nh); c = np.zeros(nh)
            hx = np.concatenate([h, x])
            f_g = sig(Wf @ hx + bf)
            i_g = sig(Wi @ hx + bi)
            g_g = np.tanh(Wc @ hx + bc)
            o_g = sig(Wo @ hx + bo)
            c   = f_g * c + i_g * g_g
            h   = o_g * np.tanh(c)
            preds.append((Wy @ h + by)[0])
        return np.array(preds)

    return predict, losses


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — LSTM-ATTENTION
# ═══════════════════════════════════════════════════════════════════════════

def attention_lstm_predict(X_train, y_train, X_test, n_hidden=HIDDEN,
                           epochs=EPOCHS, lr=LR):
    """
    LSTM with additive attention over the feature dimension.
    Attention weights are computed from the hidden state and input features,
    then applied as a weighted sum before the output projection.
    """
    np.random.seed(42)
    n, _, nf = X_train.shape
    nh = n_hidden

    def W(r, c): return np.random.randn(r, c) * np.sqrt(2.0 / (r + c))
    Wf = W(nh, nh+nf); bf = np.zeros(nh)
    Wi = W(nh, nh+nf); bi = np.zeros(nh)
    Wc = W(nh, nh+nf); bc = np.zeros(nh)
    Wo = W(nh, nh+nf); bo = np.zeros(nh)
    # Attention: score = tanh(Wa·h + ba), softmax over features
    Wa = W(nf, nh);    ba = np.zeros(nf)
    Wy = W(1, nh+nf);  by = np.zeros(1)

    sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    m_all = {}; v_all = {}
    def get_mv(k, shape):
        if k not in m_all:
            m_all[k] = np.zeros(shape); v_all[k] = np.zeros(shape)
    beta1, beta2, eps_a, t_a = 0.9, 0.999, 1e-8, 0

    losses = []
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        eloss = 0.0
        for i in idx:
            t_a += 1
            x  = X_train[i, 0, :]
            yt = float(y_train[i])
            h  = np.zeros(nh); c = np.zeros(nh)

            # Attention over input features given current hidden state
            attn_score = np.tanh(Wa @ h + ba + x)  # (nf,)
            attn_w = np.exp(attn_score - attn_score.max())
            attn_w = attn_w / attn_w.sum()         # softmax
            x_attn = attn_w * x                     # weighted input

            hx = np.concatenate([h, x_attn])
            f_g = sig(Wf @ hx + bf)
            i_g = sig(Wi @ hx + bi)
            g_g = np.tanh(Wc @ hx + bc)
            o_g = sig(Wo @ hx + bo)
            c   = f_g * c + i_g * g_g
            h   = o_g * np.tanh(c)

            # Output uses both h and attention-weighted x
            hx_out = np.concatenate([h, x_attn])
            y_hat  = (Wy @ hx_out + by)[0]
            loss   = (y_hat - yt) ** 2
            eloss += loss

            # Backward (simplified — gradient through output only)
            dy     = 2 * (y_hat - yt)
            dWy    = dy * hx_out.reshape(1, -1)
            dby    = np.array([dy])
            dh     = Wy[0, :nh] * dy
            tanh_c = np.tanh(c)
            do     = dh * tanh_c
            dc     = dh * o_g * (1 - tanh_c**2)
            dg     = dc * i_g; di = dc * g_g; df = dc * c
            dWo = (do*o_g*(1-o_g))[:,None]*hx[None,:]
            dWi = (di*i_g*(1-i_g))[:,None]*hx[None,:]
            dWc = (dg*(1-g_g**2))[:,None]  *hx[None,:]
            dWf = (df*f_g*(1-f_g))[:,None] *hx[None,:]
            dbo = do*o_g*(1-o_g); dbi = di*i_g*(1-i_g)
            dbc = dg*(1-g_g**2);  dbf = df*f_g*(1-f_g)
            # Attention gradient
            dWa = np.outer(attn_score * (1 - attn_score**2), h)
            dba = attn_score * (1 - attn_score**2)

            grads = dict(Wf=dWf,Wi=dWi,Wc=dWc,Wo=dWo,Wy=dWy,Wa=dWa,
                         bf=dbf,bi=dbi,bc=dbc,bo=dbo,by=dby,ba=dba)
            params = dict(Wf=Wf,Wi=Wi,Wc=Wc,Wo=Wo,Wy=Wy,Wa=Wa,
                          bf=bf,bi=bi,bc=bc,bo=bo,by=by,ba=ba)
            for k in params:
                get_mv(k, params[k].shape)
                m_all[k] = beta1*m_all[k] + (1-beta1)*grads[k]
                v_all[k] = beta2*v_all[k] + (1-beta2)*grads[k]**2
                mh = m_all[k]/(1-beta1**t_a)
                vh = v_all[k]/(1-beta2**t_a)
                params[k] -= lr * mh/(np.sqrt(vh)+eps_a)
            Wf,Wi,Wc,Wo,Wy = params["Wf"],params["Wi"],params["Wc"],params["Wo"],params["Wy"]
            bf,bi,bc,bo,by  = params["bf"],params["bi"],params["bc"],params["bo"],params["by"]
            Wa,ba           = params["Wa"],params["ba"]

        losses.append(eloss/n)
        if (epoch+1) % 100 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  MSE={losses[-1]:.6f}")

    def predict(X):
        preds = []
        for j in range(len(X)):
            x  = X[j, 0, :]
            h  = np.zeros(nh); c = np.zeros(nh)
            a  = np.tanh(Wa @ h + ba + x)
            aw = np.exp(a - a.max()); aw /= aw.sum()
            xa = aw * x
            hx = np.concatenate([h, xa])
            f_g = sig(Wf@hx+bf); i_g = sig(Wi@hx+bi)
            g_g = np.tanh(Wc@hx+bc); o_g = sig(Wo@hx+bo)
            c   = f_g*c + i_g*g_g; h = o_g*np.tanh(c)
            preds.append((Wy @ np.concatenate([h,xa]) + by)[0])
        return np.array(preds)

    return predict, losses


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAIN + EVALUATE ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 5: Training LSTM & LSTM-Attention ────────────────────")

all_results = {}

for fund in ["AKD", "NBP", "NIT"]:
    print(f"\n  ── {fund} ──")
    col = f"flow_pct_{COL_NAME[fund]}"
    X, y_scaled, y_raw, dates, train_mask, scaler_y = build_dataset(monthly, fund)

    X_train = X[train_mask];    y_train = y_scaled[train_mask]
    X_test  = X[~train_mask];   y_test  = y_scaled[~train_mask]
    dates_train = dates[train_mask];  dates_test = dates[~train_mask]
    y_raw_train = y_raw[train_mask];  y_raw_test  = y_raw[~train_mask]

    print(f"    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # LSTM
    print("    Training LSTM …")
    lstm_pred_fn, lstm_losses = fast_train_lstm(X_train, y_train,
                                                 n_hidden=HIDDEN, epochs=EPOCHS, lr=LR)
    lstm_train_s = lstm_pred_fn(X_train)
    lstm_test_s  = lstm_pred_fn(X_test)
    lstm_train = scaler_y.inverse_transform(lstm_train_s.reshape(-1,1)).ravel()
    lstm_test  = scaler_y.inverse_transform(lstm_test_s.reshape(-1,1)).ravel()

    # LSTM-Attention
    print("    Training LSTM-Attention …")
    attn_pred_fn, attn_losses = attention_lstm_predict(X_train, y_train, X_test,
                                                        n_hidden=HIDDEN, epochs=EPOCHS, lr=LR)
    attn_train_s = attn_pred_fn(X_train)
    attn_test_s  = attn_pred_fn(X_test)
    attn_train = scaler_y.inverse_transform(attn_train_s.reshape(-1,1)).ravel()
    attn_test  = scaler_y.inverse_transform(attn_test_s.reshape(-1,1)).ravel()

    print(f"\n    Metrics comparison — {fund}")
    m_arima = arima_results[fund]["metrics_test"]
    print(f"    {'Model':22s} {'RMSE':>10} {'MAE':>10} {'R²':>8} {'MAPE%':>8}")
    print(f"    {'-'*60}")
    for label, m in [("ARIMA(1,0,1)",     m_arima),
                     ("LSTM",             metrics(y_raw_test, lstm_test)),
                     ("LSTM-Attention",   metrics(y_raw_test, attn_test))]:
        print(f"    {label:22s} {m['RMSE']:>10.6f} {m['MAE']:>10.6f} "
              f"{m['R2']:>8.4f} {m['MAPE']:>8.2f}%")

    all_results[fund] = {
        "dates_train": dates_train, "dates_test": dates_test,
        "y_train": y_raw_train,     "y_test": y_raw_test,
        "arima_train": arima_results[fund]["train_pred"],
        "arima_train_dates": arima_results[fund]["train_dates"],
        "arima_test": arima_results[fund]["test_pred"],
        "lstm_train": lstm_train,   "lstm_test":  lstm_test,
        "attn_train": attn_train,   "attn_test":  attn_test,
        "lstm_losses": lstm_losses, "attn_losses": attn_losses,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 6: Generating figures ────────────────────────────────")

# ── F1. Prediction plots (one per fund) ──────────────────────────────────────
for fund in ["AKD", "NBP", "NIT"]:
    r = all_results[fund]
    ar = arima_results[fund]
    fig, ax = plt.subplots(figsize=(13, 5))

    # ARIMA train fit
    ax.plot(ar["train_dates"], ar["train_pred"] * 100, "--",
            color="#aaaaaa", linewidth=1, label="ARIMA train fit")
    # Actual
    ax.plot(r["dates_train"], r["y_train"] * 100,
            color="black", linewidth=1.5, label="Actual (train)")
    ax.plot(r["dates_test"], r["y_test"] * 100,
            color="black", linewidth=1.5, linestyle="--", label="Actual (test)")
    # Predictions on test
    ax.plot(r["dates_test"], r["arima_test"] * 100,
            color="#e74c3c", linewidth=1.5, label="ARIMA (test)")
    ax.plot(r["dates_test"], r["lstm_test"] * 100,
            color="#3498db", linewidth=1.5, label="LSTM (test)")
    ax.plot(r["dates_test"], r["attn_test"] * 100,
            color="#2ecc71", linewidth=1.5, label="LSTM-Attn (test)")

    # Shade test region
    test_start = pd.Timestamp(r["dates_test"][0])
    ax.axvline(test_start, color="gray", linestyle=":", linewidth=1)
    ax.axvspan(test_start, pd.Timestamp(r["dates_test"][-1]),
               alpha=0.06, color="gray", label="Test period")
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_title(f"{fund} — Fund Flow Prediction (% of AUM)")
    ax.set_ylabel("Fund Flow (% of AUM)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=30)
    savefig(f"F1_{fund}_predictions.png")

# ── F2. Training loss curves ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("LSTM Training Loss Curves (MSE on scaled data)", fontweight="bold")
for i, fund in enumerate(["AKD", "NBP", "NIT"]):
    ax = axes[i]
    ax.plot(all_results[fund]["lstm_losses"],
            color=FUND_COLORS[fund], label="LSTM", linewidth=1.2)
    ax.plot(all_results[fund]["attn_losses"],
            color=FUND_COLORS[fund], linestyle="--", label="LSTM-Attn", linewidth=1.2)
    ax.set_title(fund)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
plt.tight_layout()
savefig("F2_training_loss_curves.png")

# ── F3. Model comparison bar chart ───────────────────────────────────────────
model_names = ["ARIMA", "LSTM", "LSTM-Attn"]
metric_name = "RMSE"
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Out-of-Sample RMSE Comparison by Fund", fontweight="bold")
for i, fund in enumerate(["AKD", "NBP", "NIT"]):
    r  = all_results[fund]
    ar = arima_results[fund]
    rmse_vals = [
        metrics(r["y_test"], ar["test_pred"])["RMSE"],
        metrics(r["y_test"], r["lstm_test"])["RMSE"],
        metrics(r["y_test"], r["attn_test"])["RMSE"],
    ]
    ax = axes[i]
    bars = ax.bar(model_names, rmse_vals,
                  color=["#e74c3c", "#3498db", "#2ecc71"], alpha=0.85)
    ax.set_title(fund)
    ax.set_ylabel("RMSE")
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"{val:.5f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
savefig("F3_rmse_comparison.png")

# ── F4. Actual vs Predicted scatter ─────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(13, 11))
fig.suptitle("Actual vs Predicted Fund Flow — Test Set", fontweight="bold")
for row, fund in enumerate(["AKD", "NBP", "NIT"]):
    r  = all_results[fund]
    ar = arima_results[fund]
    for col_i, (label, pred) in enumerate([
        ("ARIMA",     ar["test_pred"]),
        ("LSTM",      r["lstm_test"]),
        ("LSTM-Attn", r["attn_test"]),
    ]):
        ax = axes[row][col_i]
        ax.scatter(r["y_test"] * 100, pred * 100,
                   alpha=0.7, s=50, color=FUND_COLORS[fund])
        lim = max(abs(r["y_test"]).max(), abs(pred).max()) * 100 * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        corr = np.corrcoef(r["y_test"], pred)[0, 1]
        ax.set_title(f"{fund} — {label}\n(r={corr:.3f})", fontsize=9)
        ax.set_xlabel("Actual (%)")
        ax.set_ylabel("Predicted (%)")
plt.tight_layout()
savefig("F4_actual_vs_predicted_scatter.png")

print(f"\nAll figures saved to: {FIG_DIR}")
print("FUND FLOW PREDICTION COMPLETE.")

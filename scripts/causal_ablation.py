"""
Leakage audit and causal re-evaluation (IEEE HONET reviewer comment).

Reviewer: "Possible target leakage. Some derived features (e.g. sm_change,
rolling means, lagged cumulative irrigation) may include information from the
current interval or carry irrigation persistence. Please define the prediction
horizon, compute the features causally, and report results with and without
irrigation-history features."

Prediction task (now explicit):
    predict y[t] = irr_volume[t]  using ONLY information available at end of t-1
    horizon h = 1 interval = 10 minutes

Arms:
    A  non-causal   : original feature set (reproduces the submitted numbers)
    B  causal+hist  : all sensor features lagged; cum_daily_irr retained
    C  causal-hist  : all sensor features lagged; cum_daily_irr removed
"""
import json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

BASE = Path("/Users/islambekdaiyn/Desktop/Research/Smart Irrigation")
MET  = BASE/"outputs"/"metrics"; MET.mkdir(parents=True, exist_ok=True)
SEED, LOOKBACK = 42, 12

# ---------------------------------------------------------------- load + merge
def load_csv_robust(path, na_values=None):
    df = pd.read_csv(path, dtype=str)
    first = df.columns[0]
    df = df[df[first] != first].reset_index(drop=True)
    if na_values: df = df.replace(na_values, np.nan)
    for c in df.columns:
        orig = df[c].isna().sum()
        cand = pd.to_numeric(df[c], errors="coerce")
        if (cand.isna().sum() - orig) / max(len(df), 1) < 0.01:
            df[c] = cand
    return df

env  = load_csv_robust(BASE/"stuard_environmental_data.csv", ["NULL"])
soil = load_csv_robust(BASE/"stuard_soil_data.csv",          ["NULL"])
wtr  = load_csv_robust(BASE/"stuard_water_meter_data.csv",   ["NULL"])
ind  = pd.read_csv(BASE/"indicators.csv"); ind.rename(columns={ind.columns[0]:"ts_ms"}, inplace=True)

def to_bins(df):
    df = df.copy()
    df["ts_bin"] = pd.to_datetime(pd.to_numeric(df["ts_generation"], errors="coerce"),
                                  unit="ms").dt.floor("10min")
    return df

env = to_bins(env).rename(columns={"humidity":"air_humidity","temperature":"air_temp",
                                   "pressure":"air_pressure"})
env.drop_duplicates(subset="ts_bin", keep="first", inplace=True)
for c in ["co2","air_humidity","air_pressure","air_temp"]:
    env[c] = env[c].fillna(env[c].median())

soil = to_bins(soil).rename(columns={"humidity":"soil_moisture","temperature":"soil_temp",
                                     "electrical_conductivity":"soil_ec"})
soil.drop_duplicates(subset=["ts_bin","line"], keep="first", inplace=True)
for c in ["soil_ec","soil_moisture","soil_temp"]:
    soil[c] = soil[c].fillna(soil[c].median())

wtr = to_bins(wtr)
wtr.drop_duplicates(subset=["ts_bin","line"], keep="first", inplace=True)
wtr.sort_values(["line","ts_bin"], inplace=True)
wtr["delta_vol"] = wtr.groupby("line")["current_volume"].diff().clip(lower=0).fillna(0)

ind["date"] = pd.to_datetime(pd.to_numeric(ind["ts_ms"], errors="coerce"), unit="ms").dt.date
ind_cols = [c for c in ["gdd","standard_day_degree","ontario_units"] if c in ind.columns]
ind_clean = ind[["date"]+ind_cols].drop_duplicates("date")

soil_agg = soil.groupby("ts_bin").agg(soil_ec=("soil_ec","mean"),
        soil_moisture=("soil_moisture","mean"), soil_temp=("soil_temp","mean")).reset_index()
wtr_agg  = wtr.groupby("ts_bin").agg(irr_volume=("delta_vol","sum")).reset_index()

m = soil_agg.merge(wtr_agg, on="ts_bin", how="inner")
m = m.merge(env[["ts_bin","co2","air_humidity","air_pressure","air_temp"]], on="ts_bin", how="left")
m["date"] = m["ts_bin"].dt.date
m = m.merge(ind_clean, on="date", how="left")
m[ind_cols] = m[ind_cols].ffill().bfill()
m.sort_values("ts_bin", inplace=True); m.reset_index(drop=True, inplace=True)
print(f"merged: {m.shape}")

# ------------------------------------------------- calendar features (known ahead)
m["hour"] = m["ts_bin"].dt.hour; m["dow"] = m["ts_bin"].dt.dayofweek
m["month"] = m["ts_bin"].dt.month
m["is_daytime"] = ((m["hour"]>=6)&(m["hour"]<=20)).astype(int)
m["day_sin"] = np.sin(2*np.pi*m["hour"]/24); m["day_cos"] = np.cos(2*np.pi*m["hour"]/24)
CAL = ["hour","dow","month","is_daytime","day_sin","day_cos"]
RAW = ["soil_ec","soil_moisture","soil_temp","co2","air_humidity","air_pressure","air_temp"]

def vpd(T, RH): return 0.6108*np.exp(17.27*T/(T+237.3))*(1-RH/100)

# =============================== ARM A : original, NON-CAUSAL =================
a = m.copy()
a["vpd"]         = vpd(a["air_temp"], a["air_humidity"])
a["roll_sm_1h"]  = a["soil_moisture"].rolling(6, min_periods=1).mean()   # includes t
a["roll_ec_1h"]  = a["soil_ec"].rolling(6, min_periods=1).mean()         # includes t
a["roll_co2_1h"] = a["co2"].rolling(6, min_periods=1).mean()             # includes t
a["temp_diff"]   = a["soil_temp"] - a["air_temp"]                        # at t
a["sm_change"]   = a["soil_moisture"].diff().fillna(0)                   # sm[t]-sm[t-1]
a["ec_change"]   = a["soil_ec"].diff().fillna(0)                         # at t
a["date_str"]    = a["ts_bin"].dt.date.astype(str)
a["cum_daily_irr"] = a.groupby("date_str")["irr_volume"].cumsum().shift(1).fillna(0)

# =============================== ARM B/C : CAUSAL =============================
c = m.copy()
for col in RAW:                                   # sensors observed at t-1
    c[col] = c[col].shift(1)
c["vpd"]         = vpd(c["air_temp"], c["air_humidity"])      # from lagged sensors
c["temp_diff"]   = c["soil_temp"] - c["air_temp"]             # from lagged sensors
c["roll_sm_1h"]  = m["soil_moisture"].rolling(6, min_periods=1).mean().shift(1)
c["roll_ec_1h"]  = m["soil_ec"].rolling(6, min_periods=1).mean().shift(1)
c["roll_co2_1h"] = m["co2"].rolling(6, min_periods=1).mean().shift(1)
c["sm_change"]   = m["soil_moisture"].diff().shift(1)         # sm[t-1]-sm[t-2]
c["ec_change"]   = m["soil_ec"].diff().shift(1)
for col in ind_cols:                              # daily indicators -> previous day
    c[col] = c["date"].map(ind_clean.set_index("date")[col].shift(1))
# causal cumulative daily irrigation, correctly reset at each midnight
c["date_str"] = c["ts_bin"].dt.date.astype(str)
c["cum_daily_irr"] = (c.groupby("date_str")["irr_volume"]
                        .transform(lambda s: s.shift(1).fillna(0).cumsum()))

DERIV = ["vpd","roll_sm_1h","roll_ec_1h","roll_co2_1h","temp_diff","sm_change","ec_change"]
FULL  = RAW + ind_cols + CAL + DERIV + ["cum_daily_irr"]

# One row mask for ALL arms: a row is usable only if every feature in either
# arm is defined. This guarantees identical rows, identical 70/30 split point
# and identical test sets across every configuration compared below.
VALID = a[FULL+["irr_volume"]].notna().all(axis=1) & c[FULL+["irr_volume"]].notna().all(axis=1)
a = a[VALID].reset_index(drop=True); c = c[VALID].reset_index(drop=True)
print(f"shared usable rows across all arms: {len(a):,}")

def evaluate(df, cols, tag):
    X, y = df[cols].values, df["irr_volume"].values
    X, y = X[LOOKBACK:], y[LOOKBACK:]                 # same alignment as the paper
    s = int(len(X)*0.70)
    sc = StandardScaler().fit(X[:s])
    Xtr, Xte = sc.transform(X[:s]), sc.transform(X[s:])
    ytr, yte = y[:s], y[s:]
    mdl = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                            verbose=0, random_seed=SEED).fit(Xtr, ytr)
    p = mdl.predict(Xte).clip(min=0)
    r = {"config":tag, "n_features":len(cols), "n_test":int(len(yte)),
         "MAE":float(mean_absolute_error(yte,p)),
         "RMSE":float(np.sqrt(mean_squared_error(yte,p))),
         "R2":float(r2_score(yte,p)),
         "total_pred_m3":float(p.sum()), "total_actual_m3":float(yte.sum())}
    print(f"  {tag:46} nf={len(cols):2}  MAE={r['MAE']:6.3f}  "
          f"RMSE={r['RMSE']:7.3f}  R2={r['R2']:+.3f}")
    return r

res = []
print("\n=== ARM A: non-causal (as submitted) ===")
res.append(evaluate(a, FULL, "A. Non-causal, full 24 feat (as submitted)"))
res.append(evaluate(a, [x for x in FULL if x!="cum_daily_irr"], "A. Non-causal, no irrigation history"))

print("\n=== ARM B/C: causal (horizon = 1 interval = 10 min) ===")
res.append(evaluate(c, FULL, "B. Causal, with irrigation history"))
res.append(evaluate(c, [x for x in FULL if x!="cum_daily_irr"], "C. Causal, no irrigation history"))

print("\n=== sensor ladder, causal, no irrigation history ===")
for tag, cols in [("soil moisture only", ["soil_moisture"]),
                  ("soil suite (3)", ["soil_moisture","soil_ec","soil_temp"]),
                  ("all raw sensors (7)", RAW),
                  ("raw + agronomic (10)", RAW+ind_cols),
                  ("full engineered, no irr-hist", [x for x in FULL if x!="cum_daily_irr"])]:
    res.append(evaluate(c, cols, f"ladder/causal: {tag}"))

print("\n=== sensor ladder, causal, WITH irrigation history ===")
for tag, cols in [("soil moisture + irr-hist", ["soil_moisture","cum_daily_irr"]),
                  ("full engineered + irr-hist", FULL)]:
    res.append(evaluate(c, cols, f"ladder/causal: {tag}"))

json.dump(res, open(MET/"causal_ablation.json","w"), indent=2)
pd.DataFrame(res).to_csv(MET/"causal_ablation.csv", index=False)
print(f"\nsaved -> {MET/'causal_ablation.json'}")

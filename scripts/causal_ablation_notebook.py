
# ====================================================================
# CAUSAL ABLATION, run inside the notebook's own pipeline so that the
# non-causal arm reproduces the submitted numbers exactly.
# ====================================================================
import json as _j
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

d = clean.sort_values("ts_bin").reset_index(drop=True).copy()
RAW = ["soil_ec","soil_moisture","soil_temp","co2","air_humidity","air_pressure","air_temp"]
IND = [c for c in ["gdd","standard_day_degree","ontario_units"] if c in d.columns]
CAL = ["hour","dow","month","is_daytime","day_sin","day_cos"]
DER = ["vpd","roll_sm_1h","roll_ec_1h","roll_co2_1h","temp_diff","sm_change","ec_change"]
FULL = [c for c in FEATURE_COLS]                      # exactly the paper's 24

def _vpd(T,RH): return 0.6108*np.exp(17.27*T/(T+237.3))*(1-RH/100)

# ---- causal frame: every sensor-derived input uses data up to t-1 only ----
cz = d.copy()
for col in RAW:
    cz[col] = d[col].shift(1)
cz["vpd"]        = _vpd(cz["air_temp"], cz["air_humidity"])
cz["temp_diff"]  = cz["soil_temp"] - cz["air_temp"]
cz["roll_sm_1h"] = d["soil_moisture"].rolling(6, min_periods=1).mean().shift(1)
cz["roll_ec_1h"] = d["soil_ec"].rolling(6, min_periods=1).mean().shift(1)
cz["roll_co2_1h"]= d["co2"].rolling(6, min_periods=1).mean().shift(1)
cz["sm_change"]  = d["soil_moisture"].diff().shift(1)
cz["ec_change"]  = d["soil_ec"].diff().shift(1)
for col in IND:                                        # daily values -> previous day
    daily = d.groupby("date_str")[col].first()
    cz[col] = cz["date_str"].map(daily.shift(1))
# causal cumulative daily irrigation, reset at each midnight (fixes day-boundary bug)
cz["cum_daily_irr"] = d.groupby("date_str")["irr_volume"].transform(
                          lambda s: s.shift(1).fillna(0).cumsum())

VALID = d[FULL+["irr_volume"]].notna().all(axis=1) & cz[FULL+["irr_volume"]].notna().all(axis=1)
d, cz = d[VALID].reset_index(drop=True), cz[VALID].reset_index(drop=True)
print(f"\nshared rows across all arms: {len(d):,}")

def ev(df, cols, tag):
    Xa, ya = df[cols].values, df["irr_volume"].values
    # replicate make_sequences(): window X[i-lb:i] -> target y[i]
    # so the tabular row fed to the model is the feature vector at t-1
    X = Xa[LOOKBACK-1 : len(Xa)-1]
    y = ya[LOOKBACK :]
    s = int(len(X)*0.70)
    sc = StandardScaler().fit(X[:s])
    m = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                          random_seed=SEED, verbose=0).fit(sc.transform(X[:s]), y[:s])
    p = m.predict(sc.transform(X[s:])).clip(min=0); yt = y[s:]
    r = {"config":tag,"n_features":len(cols),"n_train":int(s),"n_test":int(len(yt)),
         "MAE":float(mean_absolute_error(yt,p)),
         "RMSE":float(np.sqrt(mean_squared_error(yt,p))),
         "R2":float(r2_score(yt,p)),
         "total_pred_m3":float(p.sum()),"total_actual_m3":float(yt.sum())}
    print(f"  {tag:44} nf={len(cols):2}  MAE={r['MAE']:6.3f}  RMSE={r['RMSE']:7.3f}  R2={r['R2']:+.3f}")
    return r

NOHIST = [c for c in FULL if c != "cum_daily_irr"]
res=[]
print("\n--- headline arms ---")
res.append(ev(d,  FULL,   "A. Non-causal, full (as submitted)"))
res.append(ev(cz, FULL,   "B. Causal, with irrigation history"))
res.append(ev(cz, NOHIST, "C. Causal, without irrigation history"))

print("\n--- sensor ladder, causal, WITHOUT irrigation history ---")
LAD=[("Soil moisture only",["soil_moisture"]),
     ("Soil suite (moisture+EC+temp)",["soil_moisture","soil_ec","soil_temp"]),
     ("All raw sensors",RAW),
     ("Raw + agronomic indicators",RAW+IND),
     ("Full engineered (no irr-history)",NOHIST)]
for t,cols in LAD: res.append(ev(cz,cols,f"ladder-nohist: {t}"))

print("\n--- irrigation history in isolation ---")
res.append(ev(cz,["cum_daily_irr"],               "irr-history alone (1 feature)"))
res.append(ev(cz,["soil_moisture","cum_daily_irr"],"soil moisture + irr-history (2)"))

_j.dump(res, open(MET_DIR/"causal_ablation.json","w"), indent=2)
pd.DataFrame(res).to_csv(MET_DIR/"causal_ablation.csv", index=False)
print(f"\nsaved -> {MET_DIR/'causal_ablation.json'}")

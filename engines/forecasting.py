"""Forecasting Engine — Layer E1"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class ForecastingEngine:
    HOLDOUT_RATIO = 0.20

    def forecast(self, df, date_col, value_col, periods=4):
        ts = self._prep(df, date_col, value_col)
        if ts is None or len(ts) < 8:
            raise ValueError(f"Need ≥8 data points; got {0 if ts is None else len(ts)}.")
        n_hold = max(1, int(len(ts)*self.HOLDOUT_RATIO))
        train = ts.iloc[:-n_hold]; holdout = ts.iloc[-n_hold:]
        model, poly, scaler = self._fit(train)
        # Holdout corresponds to the END of the train set. So time index should run from len(train) to len(ts)
        holdout_preds = self._pred(model, poly, scaler, len(train), len(ts))
        metrics = self._metrics(holdout.values, holdout_preds, ts)
        model_f, poly_f, scaler_f = self._fit(ts)
        freq = self._freq(ts)
        future_dates = pd.date_range(start=ts.index[-1], periods=periods+1, freq=freq)[1:]
        future_preds = self._pred(model_f, poly_f, scaler_f, len(ts), len(ts)+periods)
        res_std = float(np.std(ts.values - self._pred(model_f, poly_f, scaler_f, 0, len(ts))))
        z = 1.645
        forecast_df = pd.DataFrame({
            date_col: future_dates, "yhat": future_preds,
            "yhat_lower": future_preds - z*res_std,
            "yhat_upper": future_preds + z*res_std,
        })
        history_df = pd.DataFrame({
            date_col: ts.index, "actual": ts.values,
            "fitted": self._pred(model_f, poly_f, scaler_f, 0, len(ts)),
        })
        return forecast_df, history_df, metrics

    def _prep(self, df, date_col, value_col):
        if date_col not in df.columns or value_col not in df.columns: return None
        sub = df[[date_col, value_col]].copy()
        sub = sub.dropna(subset=[date_col, value_col])
        sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
        sub = sub.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)[value_col]
        
        if sub.empty: return None
        sub = sub.groupby(level=0).sum()
        
        if len(sub) < 2: return sub
        
        med_days = pd.Series(sub.index).diff().dropna().dt.days.median()
        if med_days <= 2: freq = "D"
        elif med_days <= 10: freq = "W-MON"
        elif med_days <= 35: freq = "ME"
        else: freq = "QE"
        
        return sub.resample(freq).sum().fillna(0)

    def _fit(self, series):
        x = np.arange(len(series)).reshape(-1,1)
        y = series.values
        # Lock degree to 1 (Linear Trend) to prevent catastrophic cubic polynomial extrapolation 
        # on sparse, zero-padded time series data.
        poly = PolynomialFeatures(degree=1, include_bias=False)
        xp = poly.fit_transform(x)
        scaler = StandardScaler()
        xps = scaler.fit_transform(xp)
        model = Ridge(alpha=10.0); model.fit(xps, y)
        return model, poly, scaler

    def _pred(self, model, poly, scaler, start, end):
        x = np.arange(start, end).reshape(-1,1)
        return model.predict(scaler.transform(poly.transform(x)))

    def _freq(self, series):
        if len(series) < 2: return "D"
        med = pd.Series(series.index).diff().dropna().dt.days.median()
        if med <= 1.5: return "D"
        if med <= 8:   return "W"
        if med <= 32:  return "MS"
        return "QS"

    def _metrics(self, actual, predicted, full):
        mae  = float(mean_absolute_error(actual, predicted))
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        
        # WAPE (Weighted Absolute Percentage Error) is completely robust to actual=0 compared to standard MAPE
        mean_actual = np.mean(np.abs(actual))
        mape = float((mae / (mean_actual + 1e-9)) * 100)
        
        r2   = float(r2_score(actual, predicted)) if len(actual)>1 else 0.0
        da   = float(np.mean(np.sign(np.diff(actual))==np.sign(np.diff(predicted)))*100) if len(actual)>1 else 0.0
        baseline = float(np.std(full.values))
        return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE (%)":round(mape,2),
                "R²":round(r2,4),"Directional Accuracy (%)":round(da,1),
                "Baseline RMSE (std)":round(baseline,4),
                "Model vs Baseline":"✅ Better" if rmse<baseline else "⚠️ Worse",
                "Hold-out Samples":len(actual)}

    def auto_detect_columns(self, df, schema_metadata):
        date_col = val_col = None
        dt_cands = df.select_dtypes(include=["datetime64"]).columns.tolist()
        for col in df.select_dtypes(include="object").columns:
            try:
                if pd.to_datetime(df[col],errors="coerce").notna().sum()/max(len(df),1)>0.8:
                    dt_cands.append(col)
            except Exception: pass
        if dt_cands: date_col = dt_cands[0]
        num_cols = [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]
        if num_cols: val_col = max(num_cols, key=lambda c: df[c].std())
        return date_col, val_col

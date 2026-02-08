"""
Step 06: Benchmark Index Comparison
=====================================
Compares our multi-factor index against single-factor and equal-weight benchmarks:
  1. Constituent overlap analysis
  2. Sector composition comparison
  3. Score distribution comparison
  4. Performance simulation (equal-weight portfolio returns)
  5. Risk-adjusted metrics (Sharpe, Sortino, max drawdown, information ratio)
  6. Multi-factor advantage analysis (diversification benefit)
  7. US vs India sub-index comparison

Input:  data/processed/indexed_data.csv
Output: reports/tables/benchmark_*.csv
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

TABLES = PROJECT_ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def load_data():
    path = PROJECT_ROOT / "data" / "processed" / "indexed_data.csv"
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} companies")
    return df


# ---------------------------------------------------------------------------
# 1. Sector Composition
# ---------------------------------------------------------------------------
def sector_composition(df):
    print("\n--- Benchmark: Sector Composition ---")
    if "sector" not in df.columns:
        return

    top30 = df.nlargest(30, "pref_balanced") if "pref_balanced" in df.columns else df.head(30)

    full_sectors = df["sector"].value_counts(normalize=True).rename("full_universe")
    top_sectors = top30["sector"].value_counts(normalize=True).rename("our_top30")

    rows = []
    for country in df["country"].unique():
        sub = df[df["country"] == country]
        top_sub = sub.nlargest(min(15, len(sub)), "pref_balanced") if "pref_balanced" in sub.columns else sub
        for sector in sub["sector"].unique():
            rows.append({
                "country": country, "sector": sector,
                "count_universe": len(sub[sub["sector"] == sector]),
                "pct_universe": len(sub[sub["sector"] == sector]) / len(sub) * 100,
                "count_top": len(top_sub[top_sub["sector"] == sector]),
                "pct_top": len(top_sub[top_sub["sector"] == sector]) / max(1, len(top_sub)) * 100,
            })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "benchmark_sector_composition.csv", index=False)

    summary = pd.DataFrame({"full_universe": full_sectors, "our_top30": top_sectors}).fillna(0)
    summary.to_csv(TABLES / "benchmark_sector_summary.csv")
    print(f"  [OK] Saved benchmark_sector_composition.csv, benchmark_sector_summary.csv")
    return result


# ---------------------------------------------------------------------------
# 2. Score Comparison: Multiple Strategies
# ---------------------------------------------------------------------------
def score_comparison(df):
    print("\n--- Benchmark: Score Comparison ---")

    rankings = {}
    if "pref_balanced" in df.columns:
        rankings["our_balanced"] = df.nlargest(20, "pref_balanced")["ticker"].tolist()
    if "pref_esg_first" in df.columns:
        rankings["our_esg_first"] = df.nlargest(20, "pref_esg_first")["ticker"].tolist()
    if "pref_financial_first" in df.columns:
        rankings["our_fin_first"] = df.nlargest(20, "pref_financial_first")["ticker"].tolist()
    if "ESG_composite" in df.columns:
        rankings["esg_only"] = df.nlargest(20, "ESG_composite")["ticker"].tolist()
    if "financial_score" in df.columns:
        rankings["financial_only"] = df.nlargest(20, "financial_score")["ticker"].tolist()
    if "risk_adjusted_score" in df.columns:
        rankings["risk_adj_only"] = df.nlargest(20, "risk_adjusted_score")["ticker"].tolist()
    if "growth_score" in df.columns:
        rankings["growth_only"] = df.nlargest(20, "growth_score")["ticker"].tolist()

    # Overlap matrix
    names = list(rankings.keys())
    overlap_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for n1 in names:
        for n2 in names:
            overlap_matrix.loc[n1, n2] = len(set(rankings[n1]) & set(rankings[n2]))

    overlap_matrix.to_csv(TABLES / "benchmark_ranking_overlap.csv")

    # Per-strategy mean scores
    score_cols = ["ESG_composite", "financial_score", "market_score", "operational_score",
                  "risk_adjusted_score", "growth_score", "value_score", "stability_score"]
    avail = [c for c in score_cols if c in df.columns]
    rows = []
    for name, tickers in rankings.items():
        sub = df[df["ticker"].isin(tickers)]
        row = {"strategy": name, "n_companies": len(sub)}
        for col in avail:
            row[f"avg_{col}"] = sub[col].mean()
        rows.append(row)

    strat_scores = pd.DataFrame(rows)
    strat_scores.to_csv(TABLES / "benchmark_strategy_scores.csv", index=False)
    print(f"  [OK] Saved benchmark_ranking_overlap.csv, benchmark_strategy_scores.csv")
    return strat_scores


# ---------------------------------------------------------------------------
# 3. Simulated Portfolio Performance (with Information Ratio)
# ---------------------------------------------------------------------------
def simulated_performance(df):
    print("\n--- Benchmark: Simulated Portfolio Performance ---")

    return_cols = ["price_momentum_1m", "price_momentum_3m", "price_momentum_6m"]
    avail_ret = [c for c in return_cols if c in df.columns and df[c].notna().sum() > 5]
    if not avail_ret:
        print("  [SKIP] No return data available")
        return

    # Build portfolios from different strategies
    portfolios = {}
    if "pref_balanced" in df.columns:
        portfolios["our_balanced_top20"] = df.nlargest(20, "pref_balanced")
        portfolios["our_balanced_top30"] = df.nlargest(30, "pref_balanced")
    if "pref_esg_first" in df.columns:
        portfolios["our_esg_first_top20"] = df.nlargest(20, "pref_esg_first")
    if "pref_financial_first" in df.columns:
        portfolios["our_fin_first_top20"] = df.nlargest(20, "pref_financial_first")
    if "ESG_composite" in df.columns:
        portfolios["esg_only_top20"] = df.nlargest(20, "ESG_composite")
    if "financial_score" in df.columns:
        portfolios["financial_only_top20"] = df.nlargest(20, "financial_score")
    if "risk_adjusted_score" in df.columns:
        portfolios["risk_adj_only_top20"] = df.nlargest(20, "risk_adjusted_score")
    if "growth_score" in df.columns:
        portfolios["growth_only_top20"] = df.nlargest(20, "growth_score")
    portfolios["full_universe"] = df

    # Compute universe benchmark returns for information ratio
    universe_returns = {}
    for rc in avail_ret:
        universe_returns[rc] = df[rc].dropna().mean()

    rows = []
    for name, port_df in portfolios.items():
        row = {"portfolio": name, "n_companies": len(port_df)}
        for rc in avail_ret:
            rets = port_df[rc].dropna()
            row[f"avg_{rc}"] = rets.mean()
            row[f"std_{rc}"] = rets.std()
            if rets.std() > 1e-10:
                row[f"sharpe_{rc}"] = rets.mean() / rets.std()
            else:
                row[f"sharpe_{rc}"] = 0
            # Sortino ratio (only downside deviation)
            downside = rets[rets < 0]
            if len(downside) > 1 and downside.std() > 1e-10:
                row[f"sortino_{rc}"] = rets.mean() / downside.std()
            else:
                row[f"sortino_{rc}"] = row[f"sharpe_{rc}"]
            row[f"max_drawdown_{rc}"] = rets.min()
            # Information ratio vs full universe
            excess = rets.mean() - universe_returns.get(rc, 0)
            tracking_error = (rets - universe_returns.get(rc, 0)).std()
            if tracking_error > 1e-10:
                row[f"info_ratio_{rc}"] = excess / tracking_error
            else:
                row[f"info_ratio_{rc}"] = 0
            # Percentage of companies with positive returns
            row[f"pct_positive_{rc}"] = (rets > 0).mean() * 100

        # Score quality of portfolio
        for sc in ["ESG_composite", "financial_score", "risk_adjusted_score"]:
            if sc in port_df.columns:
                row[f"avg_{sc}"] = port_df[sc].mean()

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "benchmark_portfolio_performance.csv", index=False)
    print(f"  [OK] Saved benchmark_portfolio_performance.csv ({len(portfolios)} strategies)")

    # Print key comparison
    for rc in avail_ret[:1]:  # Just the first return col
        print(f"\n  Portfolio returns ({rc}):")
        for _, r in result.iterrows():
            avg_key = f"avg_{rc}"
            sharpe_key = f"sharpe_{rc}"
            ir_key = f"info_ratio_{rc}"
            if avg_key in r:
                print(f"    {r['portfolio']:30s}: return={r[avg_key]:+6.2f}%, "
                      f"sharpe={r.get(sharpe_key, 0):+.3f}, IR={r.get(ir_key, 0):+.3f}")

    return result


# ---------------------------------------------------------------------------
# 4. US vs India Sub-Index
# ---------------------------------------------------------------------------
def us_vs_india(df):
    print("\n--- Benchmark: US vs India Sub-Index ---")
    if "country" not in df.columns:
        return

    score_cols = ["ESG_composite", "E_score", "S_score", "G_score",
                  "financial_score", "market_score", "operational_score",
                  "risk_adjusted_score", "growth_score", "value_score",
                  "stability_score", "pref_balanced"]
    avail = [c for c in score_cols if c in df.columns]

    rows = []
    for country in df["country"].unique():
        sub = df[df["country"] == country]
        row = {"country": country, "n_companies": len(sub)}
        for col in avail:
            row[f"mean_{col}"] = sub[col].mean()
            row[f"std_{col}"] = sub[col].std()
            row[f"median_{col}"] = sub[col].median()
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "benchmark_us_vs_india.csv", index=False)

    if "sector" in df.columns:
        sector_country = df.groupby(["country", "sector"])[avail].mean()
        sector_country.to_csv(TABLES / "benchmark_sector_by_country.csv")

    print(f"  [OK] Saved benchmark_us_vs_india.csv, benchmark_sector_by_country.csv")
    return result


# ---------------------------------------------------------------------------
# 5. Benchmark Summary
# ---------------------------------------------------------------------------
def benchmark_summary(df):
    print("\n--- Benchmark: Summary ---")

    def _build_row(name, sub, full_df):
        row = {
            "index": name,
            "n_companies": len(sub),
            "avg_ESG": sub.get("ESG_composite", pd.Series()).mean(),
            "avg_financial": sub.get("financial_score", pd.Series()).mean(),
            "avg_market": sub.get("market_score", pd.Series()).mean(),
            "avg_risk_adj": sub.get("risk_adjusted_score", pd.Series()).mean(),
            "n_sectors": sub["sector"].nunique() if "sector" in sub.columns else None,
            "us_pct": (sub["country"] == "US").mean() * 100 if "country" in sub.columns else None,
        }
        # Diversification: sector HHI (lower is more diversified)
        if "sector" in sub.columns and len(sub) > 0:
            sector_pcts = sub["sector"].value_counts(normalize=True)
            row["sector_hhi"] = (sector_pcts ** 2).sum()
            row["effective_n_sectors"] = 1 / row["sector_hhi"] if row["sector_hhi"] > 0 else 0
        return row

    rows = []
    if "pref_balanced" in df.columns:
        rows.append(_build_row("Our Multi-Factor (Top 20)", df.nlargest(20, "pref_balanced"), df))
    if "ESG_composite" in df.columns:
        rows.append(_build_row("ESG-Only (Top 20)", df.nlargest(20, "ESG_composite"), df))
    if "financial_score" in df.columns:
        rows.append(_build_row("Financial-Only (Top 20)", df.nlargest(20, "financial_score"), df))
    if "growth_score" in df.columns:
        rows.append(_build_row("Growth-Only (Top 20)", df.nlargest(20, "growth_score"), df))
    rows.append(_build_row("Full Universe", df, df))

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "benchmark_summary.csv", index=False)
    print(f"  [OK] Saved benchmark_summary.csv")
    return result


# ---------------------------------------------------------------------------
# 6. Multi-Horizon Return Comparison
# ---------------------------------------------------------------------------
def multi_horizon_comparison(df):
    """Create a clean comparison table across 1m, 3m, 6m, 12m horizons."""
    print("\n--- Benchmark: Multi-Horizon Return Comparison ---")

    horizons = {
        "1m": "price_momentum_1m", "3m": "price_momentum_3m",
        "6m": "price_momentum_6m", "12m": "price_momentum_12m",
    }
    avail_horizons = {k: v for k, v in horizons.items() if v in df.columns and df[v].notna().sum() > 5}

    if not avail_horizons:
        print("  [SKIP] No return data")
        return

    # Define strategies
    strategies = {}
    if "pref_balanced" in df.columns:
        strategies["Our_MultiF_Top20"] = df.nlargest(20, "pref_balanced")
    if "pref_esg_first" in df.columns:
        strategies["Our_ESGFirst_Top20"] = df.nlargest(20, "pref_esg_first")
    if "ESG_composite" in df.columns:
        strategies["ESG_Only_Top20"] = df.nlargest(20, "ESG_composite")
    if "financial_score" in df.columns:
        strategies["Financial_Only_Top20"] = df.nlargest(20, "financial_score")
    if "growth_score" in df.columns:
        strategies["Growth_Only_Top20"] = df.nlargest(20, "growth_score")
    strategies["Full_Universe"] = df

    rows = []
    for strat_name, sub in strategies.items():
        row = {"strategy": strat_name, "n": len(sub)}
        for horizon_label, col in avail_horizons.items():
            rets = sub[col].dropna()
            row[f"return_{horizon_label}"] = rets.mean()
            row[f"sharpe_{horizon_label}"] = rets.mean() / (rets.std() + 1e-10)
            row[f"max_dd_{horizon_label}"] = rets.min()
            row[f"pct_positive_{horizon_label}"] = (rets > 0).mean() * 100
        # Diversification
        if "sector" in sub.columns:
            sector_pcts = sub["sector"].value_counts(normalize=True)
            row["sector_hhi"] = (sector_pcts ** 2).sum()
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "benchmark_multi_horizon.csv", index=False)
    print(f"  [OK] Saved benchmark_multi_horizon.csv ({len(strategies)} strategies x {len(avail_horizons)} horizons)")

    # Print comparison
    for h_label, col in list(avail_horizons.items())[:2]:
        print(f"\n  {h_label} horizon:")
        for _, r in result.iterrows():
            ret_key = f"return_{h_label}"
            sh_key = f"sharpe_{h_label}"
            print(f"    {r['strategy']:25s}: return={r.get(ret_key, 0):+6.2f}%, sharpe={r.get(sh_key, 0):+.3f}")

    return result


# ---------------------------------------------------------------------------
# 7. Alpha/Beta Decomposition
# ---------------------------------------------------------------------------
def alpha_beta_analysis(df):
    """Decompose each strategy's return into alpha + beta * benchmark."""
    print("\n--- Benchmark: Alpha/Beta Decomposition ---")

    return_col = None
    for rc in ["price_momentum_6m", "price_momentum_3m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return_col = rc
            break
    if return_col is None:
        print("  [SKIP] No return data")
        return

    # Benchmark: full universe return
    bench_returns = df[return_col].dropna()
    bench_mean = bench_returns.mean()
    bench_std = bench_returns.std()

    strategies = {}
    if "pref_balanced" in df.columns:
        strategies["Our_MultiF_Top20"] = df.nlargest(20, "pref_balanced")
    if "pref_esg_first" in df.columns:
        strategies["Our_ESGFirst_Top20"] = df.nlargest(20, "pref_esg_first")
    if "ESG_composite" in df.columns:
        strategies["ESG_Only_Top20"] = df.nlargest(20, "ESG_composite")
    if "financial_score" in df.columns:
        strategies["Financial_Only_Top20"] = df.nlargest(20, "financial_score")
    if "growth_score" in df.columns:
        strategies["Growth_Only_Top20"] = df.nlargest(20, "growth_score")

    rows = []
    for name, sub in strategies.items():
        port_rets = sub[return_col].dropna()
        port_mean = port_rets.mean()
        port_std = port_rets.std()

        # Beta = cov(port, bench) / var(bench)
        # For cross-sectional: use company-level returns
        merged = pd.DataFrame({
            "port_ret": port_rets,
            "bench_ret": df.loc[port_rets.index, return_col],
        }).dropna()

        if len(merged) > 3 and bench_std > 1e-10:
            from numpy import polyfit
            beta_val, alpha_val = polyfit(merged["bench_ret"], merged["port_ret"], 1)
        else:
            beta_val, alpha_val = 1.0, 0.0

        # Excess return vs benchmark
        excess = port_mean - bench_mean
        tracking_error = (port_rets - bench_mean).std()
        info_ratio = excess / (tracking_error + 1e-10)

        rows.append({
            "strategy": name,
            "return": port_mean,
            "benchmark_return": bench_mean,
            "excess_return": excess,
            "alpha": alpha_val,
            "beta": beta_val,
            "sharpe": port_mean / (port_std + 1e-10),
            "information_ratio": info_ratio,
            "tracking_error": tracking_error,
            "return_col": return_col,
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "benchmark_alpha_beta.csv", index=False)
    print(f"  [OK] Saved benchmark_alpha_beta.csv")

    for _, r in result.iterrows():
        print(f"    {r['strategy']:25s}: alpha={r['alpha']:+.2f}%, beta={r['beta']:.2f}, "
              f"excess={r['excess_return']:+.2f}%, IR={r['information_ratio']:+.3f}")

    return result


def main():
    print("=" * 70)
    print("STEP 06: BENCHMARK INDEX COMPARISON")
    print("=" * 70)

    df = load_data()
    sector_composition(df)
    score_comparison(df)
    simulated_performance(df)
    us_vs_india(df)
    benchmark_summary(df)
    multi_horizon_comparison(df)
    alpha_beta_analysis(df)

    print(f"\n[DONE] Benchmark comparison complete. Results in {TABLES}/")
    print("Next: python scripts/07_visualizations.py")


if __name__ == "__main__":
    main()

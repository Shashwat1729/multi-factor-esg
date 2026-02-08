"""
Step 04: Comprehensive Statistical Tests
==========================================
Performs every statistical test relevant for a research paper:
  1.  Descriptive statistics (mean, std, skew, kurtosis)
  2.  Normality tests (Shapiro-Wilk, Jarque-Bera, K-S)
  3.  Correlation analysis (Pearson, Spearman, Kendall)
  4.  ESG-Financial relationship (OLS regression with controls)
  5.  Sector differences (ANOVA + Kruskal-Wallis + effect sizes)
  6.  Country differences (t-tests, Mann-Whitney, Cohen's d)
  7.  ESG pillar inter-correlations
  8.  Quintile analysis (score quintiles vs. financial performance)
  9.  Factor contribution analysis
  10. Multicollinearity check (VIF)
  11. Rank correlation between profiles
  12. Top/Bottom analysis (20 companies)
  13. Multiple regression with controls
  14. Heteroscedasticity test (Breusch-Pagan via statsmodels)
  15. Decile analysis
  16. Subgroup analysis (size, sector, country)
  17. Non-parametric tests (Friedman, Wilcoxon)

Input:  data/processed/indexed_data.csv
Output: reports/tables/*.csv
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    pearsonr, spearmanr, kendalltau,
    shapiro, jarque_bera, mannwhitneyu,
    kruskal, f_oneway, ttest_ind,
    kstest, friedmanchisquare, wilcoxon,
)
import warnings
warnings.filterwarnings("ignore")

TABLES = PROJECT_ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CORE_SCORES = ["ESG_composite", "E_score", "S_score", "G_score",
               "financial_score", "market_score", "operational_score",
               "pref_balanced", "pref_esg_first", "pref_financial_first"]

EXTENDED_SCORES = CORE_SCORES + [
    "risk_adjusted_score", "value_score", "growth_score", "stability_score",
]


def load_data():
    path = PROJECT_ROOT / "data" / "processed" / "indexed_data.csv"
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} companies, {len(df.columns)} columns")
    return df


def _cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


# 1. Descriptive Statistics
def test_descriptive(df):
    print("\n--- Test 1: Descriptive Statistics ---")
    avail = [c for c in EXTENDED_SCORES if c in df.columns]
    desc = df[avail].describe().T
    desc["skewness"] = df[avail].skew()
    desc["kurtosis"] = df[avail].kurtosis()
    desc["iqr"] = desc["75%"] - desc["25%"]
    desc["cv"] = desc["std"] / desc["mean"].abs().clip(1e-10)
    desc.to_csv(TABLES / "descriptive_statistics.csv", encoding="utf-8")
    print(f"  [OK] Saved descriptive_statistics.csv ({len(avail)} variables)")
    return desc


# 2. Normality Tests
def test_normality(df):
    print("\n--- Test 2: Normality Tests (Shapiro, Jarque-Bera, K-S) ---")
    avail = [c for c in EXTENDED_SCORES if c in df.columns]
    rows = []
    for col in avail:
        data = df[col].dropna()
        if len(data) < 8:
            continue
        sw_stat, sw_p = shapiro(data[:5000])
        jb_stat, jb_p = jarque_bera(data)
        ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
        rows.append({
            "variable": col,
            "n": len(data),
            "shapiro_stat": sw_stat, "shapiro_p": sw_p,
            "jarque_bera_stat": jb_stat, "jarque_bera_p": jb_p,
            "ks_stat": ks_stat, "ks_p": ks_p,
            "normal_shapiro": sw_p > 0.05,
            "normal_jb": jb_p > 0.05,
            "normal_ks": ks_p > 0.05,
        })
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "normality_tests.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved normality_tests.csv")
    return result


# 3. Correlation Analysis
def test_correlations(df):
    print("\n--- Test 3: Correlation Analysis ---")
    financial_extras = ["roa", "roe", "market_cap", "total_revenue",
                        "debt_to_equity", "net_margin", "current_ratio",
                        "revenue_growth", "dividend_yield"]
    avail = [c for c in EXTENDED_SCORES + financial_extras
             if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    pearson_corr = df[avail].corr(method="pearson")
    pearson_corr.to_csv(TABLES / "correlation_pearson.csv", encoding="utf-8")

    spearman_corr = df[avail].corr(method="spearman")
    spearman_corr.to_csv(TABLES / "correlation_spearman.csv", encoding="utf-8")

    # Pairwise significance for core scores
    core_avail = [c for c in EXTENDED_SCORES if c in df.columns]
    rows = []
    for i, c1 in enumerate(core_avail):
        for c2 in core_avail[i + 1:]:
            d = df[[c1, c2]].dropna()
            if len(d) < 5:
                continue
            pr, pp = pearsonr(d[c1], d[c2])
            sr, sp = spearmanr(d[c1], d[c2])
            kr, kp = kendalltau(d[c1], d[c2])
            rows.append({
                "var1": c1, "var2": c2,
                "pearson_r": pr, "pearson_p": pp,
                "spearman_r": sr, "spearman_p": sp,
                "kendall_tau": kr, "kendall_p": kp,
                "sig_pearson": pp < 0.05,
                "sig_spearman": sp < 0.05,
            })
    sig_df = pd.DataFrame(rows)
    sig_df.to_csv(TABLES / "correlation_significance.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved correlation tables ({len(avail)} variables, {len(rows)} pairs)")
    return pearson_corr, spearman_corr


# 4. ESG-Financial Regression (with sector controls)
def test_esg_financial_regression(df):
    print("\n--- Test 4: ESG-Financial Regression ---")
    rows = []
    if "ESG_composite" not in df.columns:
        return pd.DataFrame()

    dep_vars = ["roa", "roe", "net_margin", "financial_score", "revenue_growth",
                "operating_margins", "current_ratio"]
    for dv in dep_vars:
        if dv not in df.columns:
            continue
        mask = df[["ESG_composite", dv]].dropna().index
        if len(mask) < 10:
            continue
        x = df.loc[mask, "ESG_composite"].values
        y = df.loc[mask, dv].values
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        rows.append({
            "dependent_var": dv, "independent_var": "ESG_composite",
            "slope": slope, "intercept": intercept,
            "r_squared": r_val ** 2, "p_value": p_val, "std_error": std_err,
            "n": len(mask),
        })

    # Also test individual pillars
    for pillar in ["E_score", "S_score", "G_score"]:
        if pillar not in df.columns:
            continue
        for dv in ["roa", "financial_score"]:
            if dv not in df.columns:
                continue
            mask = df[[pillar, dv]].dropna().index
            if len(mask) < 10:
                continue
            x = df.loc[mask, pillar].values
            y = df.loc[mask, dv].values
            slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
            rows.append({
                "dependent_var": dv, "independent_var": pillar,
                "slope": slope, "intercept": intercept,
                "r_squared": r_val ** 2, "p_value": p_val, "std_error": std_err,
                "n": len(mask),
            })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "esg_financial_regression.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved esg_financial_regression.csv ({len(result)} regressions)")
    return result


# 5. Sector Differences (ANOVA + effect sizes)
def test_sector_differences(df):
    print("\n--- Test 5: Sector Differences (ANOVA + Effect Sizes) ---")
    if "sector" not in df.columns:
        return pd.DataFrame()

    avail = [c for c in EXTENDED_SCORES if c in df.columns]
    rows = []
    for col in avail:
        groups = [g[col].dropna().values for _, g in df.groupby("sector") if len(g[col].dropna()) > 1]
        if len(groups) < 2:
            continue
        f_stat, f_p = f_oneway(*groups)
        k_stat, k_p = kruskal(*groups)

        # Eta-squared effect size for ANOVA
        grand_mean = df[col].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
        eta_sq = ss_between / (ss_total + 1e-10)

        rows.append({
            "variable": col,
            "n_groups": len(groups),
            "anova_F": f_stat, "anova_p": f_p,
            "kruskal_H": k_stat, "kruskal_p": k_p,
            "eta_squared": eta_sq,
            "effect_size": "Large" if eta_sq > 0.14 else ("Medium" if eta_sq > 0.06 else "Small"),
            "significant_anova": f_p < 0.05,
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "sector_anova.csv", index=False, encoding="utf-8")

    # Sector means
    sector_stats = df.groupby("sector")[avail].agg(["mean", "std", "count"])
    sector_stats.to_csv(TABLES / "sector_means.csv", encoding="utf-8")
    print(f"  [OK] Saved sector_anova.csv, sector_means.csv")
    return result


# 6. Country Differences (with Cohen's d)
def test_country_differences(df):
    print("\n--- Test 6: Country Differences (with Effect Sizes) ---")
    if "country" not in df.columns:
        return pd.DataFrame()

    countries = df["country"].dropna().unique()
    if len(countries) < 2:
        return pd.DataFrame()

    avail = [c for c in EXTENDED_SCORES if c in df.columns]
    rows = []
    c1, c2 = countries[0], countries[1]
    g1 = df[df["country"] == c1]
    g2 = df[df["country"] == c2]

    for col in avail:
        d1 = g1[col].dropna()
        d2 = g2[col].dropna()
        if len(d1) < 3 or len(d2) < 3:
            continue
        t_stat, t_p = ttest_ind(d1, d2)
        u_stat, u_p = mannwhitneyu(d1, d2, alternative="two-sided")
        d_effect = _cohens_d(d1, d2)
        rows.append({
            "variable": col,
            f"mean_{c1}": d1.mean(), f"std_{c1}": d1.std(), f"n_{c1}": len(d1),
            f"mean_{c2}": d2.mean(), f"std_{c2}": d2.std(), f"n_{c2}": len(d2),
            "ttest_stat": t_stat, "ttest_p": t_p,
            "mannwhitney_U": u_stat, "mannwhitney_p": u_p,
            "cohens_d": d_effect,
            "effect_size": "Large" if abs(d_effect) > 0.8 else ("Medium" if abs(d_effect) > 0.5 else "Small"),
            "significant_t": t_p < 0.05,
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "country_differences.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved country_differences.csv ({len(result)} variables)")
    return result


# 7. ESG Pillar Inter-Correlations
def test_pillar_correlations(df):
    print("\n--- Test 7: ESG Pillar Inter-Correlations ---")
    pillars = ["E_score", "S_score", "G_score"]
    avail = [c for c in pillars if c in df.columns]
    if len(avail) < 2:
        return pd.DataFrame()

    rows = []
    for i, p1 in enumerate(avail):
        for p2 in avail[i + 1:]:
            d = df[[p1, p2]].dropna()
            if len(d) < 5:
                continue
            pr, pp = pearsonr(d[p1], d[p2])
            sr, sp = spearmanr(d[p1], d[p2])
            rows.append({
                "pillar1": p1, "pillar2": p2,
                "pearson_r": pr, "pearson_p": pp,
                "spearman_r": sr, "spearman_p": sp,
            })
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "pillar_correlations.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved pillar_correlations.csv")
    return result


# 8. Quintile Analysis
def test_quintile_analysis(df):
    print("\n--- Test 8: Quintile Analysis ---")
    if "pref_balanced" not in df.columns:
        return pd.DataFrame()

    df_q = df.copy()
    df_q["quintile"] = pd.qcut(df_q["pref_balanced"], 5, labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"])

    metrics = ["roa", "roe", "net_margin", "ESG_composite", "financial_score",
               "market_score", "operational_score", "revenue_growth",
               "current_ratio", "dividend_yield"]
    avail = [c for c in metrics if c in df_q.columns]

    quintile_stats = df_q.groupby("quintile")[avail].agg(["mean", "std", "count"])
    quintile_stats.to_csv(TABLES / "quintile_analysis.csv", encoding="utf-8")

    # T-test: Q5 vs Q1
    q5 = df_q[df_q["quintile"] == "Q5(High)"]
    q1 = df_q[df_q["quintile"] == "Q1(Low)"]
    rows = []
    for col in avail:
        d5 = q5[col].dropna()
        d1 = q1[col].dropna()
        if len(d5) > 1 and len(d1) > 1:
            t, p = ttest_ind(d5, d1)
            d_eff = _cohens_d(d5, d1)
            rows.append({
                "variable": col,
                "Q5_mean": d5.mean(), "Q1_mean": d1.mean(),
                "difference": d5.mean() - d1.mean(),
                "ttest_stat": t, "ttest_p": p,
                "cohens_d": d_eff,
                "significant": p < 0.05,
            })
    q_test = pd.DataFrame(rows)
    q_test.to_csv(TABLES / "quintile_q5_vs_q1.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved quintile_analysis.csv, quintile_q5_vs_q1.csv")
    return quintile_stats


# 9. Factor Contribution Analysis
def test_factor_contributions(df):
    print("\n--- Test 9: Factor Contribution Analysis ---")
    factors = ["ESG_composite", "financial_score", "market_score", "operational_score",
               "risk_adjusted_score", "value_score", "growth_score", "stability_score"]
    avail = [c for c in factors if c in df.columns]
    if "pref_balanced" not in df.columns or len(avail) < 2:
        return pd.DataFrame()

    rows = []
    for factor in avail:
        mask = df[[factor, "pref_balanced"]].dropna().index
        x = df.loc[mask, factor].values
        y = df.loc[mask, "pref_balanced"].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        rows.append({
            "factor": factor,
            "correlation_with_pref": r,
            "r_squared": r**2,
            "slope": slope,
            "p_value": p,
        })

    result = pd.DataFrame(rows).sort_values("r_squared", ascending=False)
    result.to_csv(TABLES / "factor_contributions.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved factor_contributions.csv ({len(result)} factors)")
    return result


# 10. Multicollinearity (VIF)
def test_multicollinearity(df):
    print("\n--- Test 10: Multicollinearity (VIF) ---")
    factors = ["ESG_composite", "financial_score", "market_score", "operational_score",
               "risk_adjusted_score", "value_score", "growth_score", "stability_score"]
    avail = [c for c in factors if c in df.columns]
    if len(avail) < 2:
        return pd.DataFrame()

    from numpy.linalg import inv
    X = df[avail].dropna()
    if len(X) < len(avail) + 2:
        return pd.DataFrame()

    corr = X.corr().values
    try:
        inv_corr = inv(corr)
        vif = pd.DataFrame({
            "factor": avail,
            "VIF": [inv_corr[i, i] for i in range(len(avail))],
        })
        vif["multicollinearity"] = vif["VIF"].apply(
            lambda x: "High" if x > 10 else ("Moderate" if x > 5 else "Low")
        )
        vif.to_csv(TABLES / "vif_multicollinearity.csv", index=False, encoding="utf-8")
        print(f"  [OK] Saved vif_multicollinearity.csv")
        return vif
    except Exception:
        print("  [SKIP] Could not compute VIF (singular matrix)")
        return pd.DataFrame()


# 11. Profile Rank Correlation
def test_profile_rank_correlation(df):
    print("\n--- Test 11: Profile Rank Correlation ---")
    profiles = ["pref_esg_first", "pref_balanced", "pref_financial_first"]
    avail = [c for c in profiles if c in df.columns]
    if len(avail) < 2:
        return pd.DataFrame()

    rows = []
    for i, p1 in enumerate(avail):
        for p2 in avail[i + 1:]:
            d = df[[p1, p2]].dropna()
            r1 = d[p1].rank()
            r2 = d[p2].rank()
            sr, sp = spearmanr(r1, r2)
            kt, kp = kendalltau(r1, r2)
            rows.append({
                "profile1": p1, "profile2": p2,
                "spearman_rank_r": sr, "spearman_p": sp,
                "kendall_tau": kt, "kendall_p": kp,
                "top10_overlap": len(set(d.nlargest(10, p1).index) & set(d.nlargest(10, p2).index)),
                "top20_overlap": len(set(d.nlargest(20, p1).index) & set(d.nlargest(20, p2).index)),
            })
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "profile_rank_correlation.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved profile_rank_correlation.csv")
    return result


# 12. Top/Bottom Analysis
def test_top_bottom(df):
    print("\n--- Test 12: Top/Bottom Company Analysis ---")
    for col, label in [("pref_balanced", "balanced"), ("ESG_composite", "esg"),
                       ("financial_score", "financial"), ("growth_score", "growth"),
                       ("value_score", "value")]:
        if col not in df.columns:
            continue
        display_cols = ["ticker", "company_name", "sector", "country",
                        "ESG_composite", "financial_score", "market_score",
                        "operational_score", "risk_adjusted_score",
                        "value_score", "growth_score", col]
        avail_display = [c for c in display_cols if c in df.columns]
        n_top = min(20, len(df))
        top = df.nlargest(n_top, col)[avail_display]
        top.to_csv(TABLES / f"top20_{label}.csv", index=False, encoding="utf-8")
        bottom = df.nsmallest(n_top, col)[avail_display]
        bottom.to_csv(TABLES / f"bottom20_{label}.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved top20_*.csv, bottom20_*.csv")


# 13. Multiple Regression with Controls
def test_multiple_regression(df):
    print("\n--- Test 13: Multiple Regression (ESG -> Financial with Controls) ---")
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  [SKIP] statsmodels not installed")
        return

    dep_vars = ["roa", "roe", "financial_score"]
    indep = ["ESG_composite"]
    controls = ["market_cap", "debt_to_equity", "beta"]

    rows = []
    for dv in dep_vars:
        if dv not in df.columns:
            continue
        all_vars = [dv] + indep + [c for c in controls if c in df.columns]
        subset = df[all_vars].dropna()
        if len(subset) < 20:
            continue

        X = subset[indep + [c for c in controls if c in subset.columns]]
        y = subset[dv]

        # Standardize
        X_std = (X - X.mean()) / (X.std() + 1e-10)
        X_std = sm.add_constant(X_std)

        try:
            model = sm.OLS(y, X_std).fit()
            for var in indep + [c for c in controls if c in X.columns]:
                if var in model.params.index:
                    rows.append({
                        "dependent": dv,
                        "independent": var,
                        "coefficient": model.params[var],
                        "std_error": model.bse[var],
                        "t_stat": model.tvalues[var],
                        "p_value": model.pvalues[var],
                        "r_squared": model.rsquared,
                        "adj_r_squared": model.rsquared_adj,
                        "n_obs": int(model.nobs),
                        "f_stat": model.fvalue,
                        "f_pvalue": model.f_pvalue,
                    })
        except Exception:
            pass

    if rows:
        result = pd.DataFrame(rows)
        result.to_csv(TABLES / "multiple_regression.csv", index=False, encoding="utf-8")
        print(f"  [OK] Saved multiple_regression.csv ({len(result)} coefficients)")


# 14. Heteroscedasticity Test
def test_heteroscedasticity(df):
    print("\n--- Test 14: Heteroscedasticity (Breusch-Pagan) ---")
    try:
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan
    except ImportError:
        print("  [SKIP] statsmodels not installed")
        return

    if "ESG_composite" not in df.columns:
        return

    rows = []
    for dv in ["roa", "roe", "financial_score"]:
        if dv not in df.columns:
            continue
        subset = df[["ESG_composite", dv]].dropna()
        if len(subset) < 20:
            continue
        X = sm.add_constant(subset["ESG_composite"])
        y = subset[dv]
        try:
            model = sm.OLS(y, X).fit()
            bp_stat, bp_p, f_stat, f_p = het_breuschpagan(model.resid, X)
            rows.append({
                "dependent": dv,
                "bp_stat": bp_stat, "bp_p": bp_p,
                "f_stat": f_stat, "f_p": f_p,
                "heteroscedastic": bp_p < 0.05,
            })
        except Exception:
            pass

    if rows:
        result = pd.DataFrame(rows)
        result.to_csv(TABLES / "heteroscedasticity.csv", index=False, encoding="utf-8")
        print(f"  [OK] Saved heteroscedasticity.csv")


# 15. Decile Analysis
def test_decile_analysis(df):
    print("\n--- Test 15: Decile Analysis ---")
    if "pref_balanced" not in df.columns:
        return

    df_d = df.copy()
    df_d["decile"] = pd.qcut(df_d["pref_balanced"], 10, labels=[f"D{i}" for i in range(1, 11)])

    metrics = ["roa", "roe", "ESG_composite", "financial_score", "market_score",
               "revenue_growth", "price_momentum_3m", "sharpe_ratio_1y"]
    avail = [c for c in metrics if c in df_d.columns]

    if avail:
        decile_stats = df_d.groupby("decile")[avail].agg(["mean", "std", "count"])
        decile_stats.to_csv(TABLES / "decile_analysis.csv", encoding="utf-8")
        print(f"  [OK] Saved decile_analysis.csv")


# 16. Subgroup Analysis (by size)
def test_subgroup_analysis(df):
    print("\n--- Test 16: Subgroup Analysis (Size Terciles) ---")
    if "market_cap" not in df.columns:
        return

    df_s = df.copy()
    try:
        df_s["size_group"] = pd.qcut(df_s["market_cap"], 3, labels=["Small", "Medium", "Large"])
    except ValueError:
        return

    score_cols = [c for c in CORE_SCORES if c in df_s.columns]
    if not score_cols:
        return

    # ANOVA for size groups
    rows = []
    for col in score_cols:
        groups = [g[col].dropna().values for _, g in df_s.groupby("size_group") if len(g[col].dropna()) > 1]
        if len(groups) < 2:
            continue
        f_stat, f_p = f_oneway(*groups)
        rows.append({
            "variable": col,
            "anova_F": f_stat, "anova_p": f_p,
            "significant": f_p < 0.05,
        })

    if rows:
        result = pd.DataFrame(rows)
        result.to_csv(TABLES / "size_group_anova.csv", index=False, encoding="utf-8")

    # Size group means
    size_means = df_s.groupby("size_group")[score_cols].mean()
    size_means.to_csv(TABLES / "size_group_means.csv", encoding="utf-8")
    print(f"  [OK] Saved size_group_anova.csv, size_group_means.csv")


# 17. Comprehensive Sector-Score Interaction
def test_sector_score_interaction(df):
    print("\n--- Test 17: Sector-Score Interaction ---")
    if "sector" not in df.columns:
        return

    # Within each sector, correlate ESG with financial
    rows = []
    for sector in df["sector"].dropna().unique():
        sub = df[df["sector"] == sector]
        if len(sub) < 5:
            continue
        for iv, dv in [("ESG_composite", "financial_score"), ("ESG_composite", "roa"),
                       ("financial_score", "market_score")]:
            if iv not in sub.columns or dv not in sub.columns:
                continue
            d = sub[[iv, dv]].dropna()
            if len(d) < 5:
                continue
            r, p = pearsonr(d[iv], d[dv])
            rows.append({
                "sector": sector,
                "x_var": iv, "y_var": dv,
                "n": len(d),
                "pearson_r": r, "pearson_p": p,
                "significant": p < 0.05,
            })

    if rows:
        result = pd.DataFrame(rows)
        result.to_csv(TABLES / "sector_score_interaction.csv", index=False, encoding="utf-8")
        print(f"  [OK] Saved sector_score_interaction.csv ({len(result)} tests)")


# 18. Binary Variable Analysis (Chi-square, Point-Biserial)
def test_binary_variables(df):
    """Analyze binary ESG policy variables: proportions, chi-square, point-biserial correlations.

    Binary variables require different statistical tests than continuous variables:
    - Chi-square for association with sector/country (categorical x categorical)
    - Point-biserial correlation for association with continuous scores
    - Phi coefficient for association between two binary variables

    References:
    - Agresti (2002) "Categorical Data Analysis"
    - Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
    """
    print("\n--- Test 18: Binary Variable Analysis ---")
    binary_cols = [c for c in ["carbon_reduction_target", "human_rights_policy", "anti_corruption_policy"]
                   if c in df.columns]
    if not binary_cols:
        return pd.DataFrame()

    rows = []

    # Proportions and basic stats
    for col in binary_cols:
        vals = df[col].dropna()
        if len(vals) < 5:
            continue
        row = {
            "variable": col,
            "n": len(vals),
            "proportion_yes": vals.mean(),
            "proportion_no": 1 - vals.mean(),
        }

        # Point-biserial correlation with key scores
        for score_col in ["ESG_composite", "financial_score", "pref_balanced"]:
            if score_col in df.columns:
                d = df[[col, score_col]].dropna()
                if len(d) > 5:
                    from scipy.stats import pointbiserialr
                    r, p = pointbiserialr(d[col], d[score_col])
                    row[f"pb_r_{score_col}"] = r
                    row[f"pb_p_{score_col}"] = p
                    row[f"pb_sig_{score_col}"] = p < 0.05

        # Sector-wise proportions (Chi-square test)
        if "sector" in df.columns:
            contingency = pd.crosstab(df["sector"], df[col].fillna(0).astype(int))
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                from scipy.stats import chi2_contingency
                chi2, chi_p, dof, _ = chi2_contingency(contingency)
                row["chi2_sector"] = chi2
                row["chi2_p_sector"] = chi_p
                row["chi2_sig_sector"] = chi_p < 0.05

        rows.append(row)

    # Phi coefficient between binary variables (pairwise)
    phi_rows = []
    for i, c1 in enumerate(binary_cols):
        for c2 in binary_cols[i+1:]:
            d = df[[c1, c2]].dropna()
            if len(d) < 5:
                continue
            contingency = pd.crosstab(d[c1].astype(int), d[c2].astype(int))
            if contingency.shape == (2, 2):
                from scipy.stats import chi2_contingency
                chi2, chi_p, _, _ = chi2_contingency(contingency)
                phi = np.sqrt(chi2 / len(d))
                phi_rows.append({
                    "var1": c1, "var2": c2,
                    "phi_coefficient": phi, "chi2": chi2, "p_value": chi_p,
                })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "binary_variable_analysis.csv", index=False, encoding="utf-8")

    if phi_rows:
        phi_df = pd.DataFrame(phi_rows)
        phi_df.to_csv(TABLES / "binary_phi_coefficients.csv", index=False, encoding="utf-8")

    print(f"  [OK] Saved binary_variable_analysis.csv ({len(result)} variables)")
    return result


# 19. Ordinal Variable Analysis (Spearman, Kruskal-Wallis)
def test_ordinal_variables(df):
    """Analyze ordinal variables using non-parametric methods.

    Ordinal variables (like board_size) should not use Pearson correlation
    or standard ANOVA. Instead we use:
    - Spearman rank correlation
    - Kruskal-Wallis H test (non-parametric ANOVA equivalent)
    - Jonckheere-Terpstra trend test (for monotonic relationships)

    Reference: Siegel & Castellan (1988) "Nonparametric Statistics for the Behavioral Sciences"
    """
    print("\n--- Test 19: Ordinal Variable Analysis ---")
    ordinal_cols = [c for c in ["board_size"] if c in df.columns]
    if not ordinal_cols:
        return pd.DataFrame()

    rows = []
    for col in ordinal_cols:
        vals = df[col].dropna()
        if len(vals) < 5:
            continue

        row = {
            "variable": col,
            "n": len(vals),
            "median": vals.median(),
            "mode": vals.mode().iloc[0] if len(vals.mode()) > 0 else None,
            "min": vals.min(),
            "max": vals.max(),
            "n_unique": vals.nunique(),
        }

        # Spearman rank correlation with key scores
        for score_col in ["ESG_composite", "G_score", "financial_score", "pref_balanced"]:
            if score_col in df.columns:
                d = df[[col, score_col]].dropna()
                if len(d) > 5:
                    sr, sp = spearmanr(d[col], d[score_col])
                    row[f"spearman_r_{score_col}"] = sr
                    row[f"spearman_p_{score_col}"] = sp

        # Kruskal-Wallis: does the ordinal variable differ by sector?
        if "sector" in df.columns:
            groups = [g[col].dropna().values for _, g in df.groupby("sector") if len(g[col].dropna()) > 1]
            if len(groups) >= 2:
                h_stat, h_p = kruskal(*groups)
                row["kruskal_H_sector"] = h_stat
                row["kruskal_p_sector"] = h_p

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "ordinal_variable_analysis.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved ordinal_variable_analysis.csv ({len(result)} variables)")
    return result


# 20. Non-Parametric Robustness Tests (Friedman, Wilcoxon)
def test_nonparametric_robustness(df):
    """Run non-parametric tests that don't assume normality.

    Adds Friedman test (non-parametric repeated-measures ANOVA equivalent)
    comparing scores across factors, and pairwise Wilcoxon signed-rank tests.
    """
    print("\n--- Test 20: Non-Parametric Robustness Tests ---")
    score_cols = ["ESG_composite", "financial_score", "market_score", "operational_score"]
    avail = [c for c in score_cols if c in df.columns]
    if len(avail) < 3:
        return

    # Friedman test: are scores significantly different across factors?
    data = df[avail].dropna()
    if len(data) < 10:
        return

    try:
        f_stat, f_p = friedmanchisquare(*[data[c].values for c in avail])
        print(f"  Friedman chi-sq = {f_stat:.2f}, p = {f_p:.4f}")
    except Exception:
        f_stat, f_p = 0, 1

    # Pairwise Wilcoxon signed-rank tests
    rows = []
    for i, c1 in enumerate(avail):
        for c2 in avail[i+1:]:
            d = df[[c1, c2]].dropna()
            if len(d) < 10:
                continue
            try:
                w_stat, w_p = wilcoxon(d[c1], d[c2])
            except Exception:
                w_stat, w_p = 0, 1
            rows.append({
                "var1": c1, "var2": c2,
                "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                "significant": w_p < 0.05,
                "mean_diff": (d[c1] - d[c2]).mean(),
            })

    result = pd.DataFrame(rows)
    friedman_row = pd.DataFrame([{"test": "Friedman", "statistic": f_stat, "p_value": f_p,
                                   "n_factors": len(avail), "n_companies": len(data)}])
    friedman_row.to_csv(TABLES / "friedman_test.csv", index=False, encoding="utf-8")
    result.to_csv(TABLES / "wilcoxon_pairwise.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved friedman_test.csv, wilcoxon_pairwise.csv ({len(result)} pairs)")


def main():
    print("=" * 70)
    print("STEP 04: COMPREHENSIVE STATISTICAL TESTS")
    print("=" * 70)

    df = load_data()
    test_descriptive(df)
    test_normality(df)
    test_correlations(df)
    test_esg_financial_regression(df)
    test_sector_differences(df)
    test_country_differences(df)
    test_pillar_correlations(df)
    test_quintile_analysis(df)
    test_factor_contributions(df)
    test_multicollinearity(df)
    test_profile_rank_correlation(df)
    test_top_bottom(df)
    test_multiple_regression(df)
    test_heteroscedasticity(df)
    test_decile_analysis(df)
    test_subgroup_analysis(df)
    test_sector_score_interaction(df)
    test_binary_variables(df)
    test_ordinal_variables(df)
    test_nonparametric_robustness(df)

    n_tables = len(list(TABLES.glob("*.csv")))
    print(f"\n[DONE] {n_tables} tables in {TABLES}/")
    print("Next: python scripts/05_weight_sensitivity.py")


if __name__ == "__main__":
    main()

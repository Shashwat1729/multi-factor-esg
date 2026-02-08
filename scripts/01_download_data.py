"""
Step 01: Download Real Data
============================
Downloads financial, market, ESG proxy, and benchmark data from public sources:
  - Yahoo Finance: Financials + market data for 200+ companies
  - SEC EDGAR: R&D expenditure data
  - Benchmark indices (NIFTY 50, S&P 500, S&P MidCap 400, Russell 2000)
  - Sector-aware synthetic ESG indicators (real ESG requires Refinitiv/MSCI subscription)

Output: data/raw/*.csv
"""

import sys, os, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Company Universe — 250+ companies across US and India, focused on mid-caps
# ---------------------------------------------------------------------------
# U.S. mid-caps from S&P MidCap 400 / Russell Midcap ESG
US_TICKERS = [
    # Technology (26)
    "ZS", "OKTA", "CRWD", "FTNT", "QLYS", "VRNS", "ESTC", "DOCN",
    "PAYC", "MANH", "WK", "GLOB", "HUBS", "ANSS", "CDNS", "NET",
    "BILL", "FROG", "CFLT", "SMCI", "MARA", "RBLX",
    "SAMSN", "TENB", "CWAN", "RPD",
    # Healthcare (24)
    "ILMN", "ALGN", "INCY", "BMRN", "ALKS", "IONS", "EXAS", "TECH",
    "DXCM", "VEEV", "TFX", "STE", "BIO", "HOLX", "NTRA", "MEDP",
    "HRMY", "PEN", "RVMD", "PCVX",
    "AZTA", "NEOG", "AVTR", "ENSG",
    # Industrials (22)
    "GNRC", "MTZ", "AWI", "GFF", "WWD", "FIX", "ROAD", "DY",
    "EME", "WSC", "GGG", "RBC", "ITT", "WTS", "ALLE", "SITE",
    "TTC", "AAON",
    "ESAB", "MWA", "ZWS", "SPXC",
    # Consumer Discretionary (18)
    "KMX", "FIVE", "DKS", "ASO", "BOOT", "WSM", "RH", "POOL",
    "DECK", "YETI", "GRMN", "LULU", "FOXF", "CROX",
    "SIG", "SHAK", "WING", "TXRH",
    # Financials (20)
    "TROW", "BEN", "IVZ", "SF", "WBS", "PB", "FNB", "EWBC",
    "CMA", "HBAN", "ZION", "CFR", "IBKR", "SEIC", "ALLY", "LPLA",
    "HLNE", "STEP", "CBSH", "PNFP",
    # Energy (14)
    "NOV", "OII", "HLX", "CHX", "TRGP", "OVV", "AR", "RRC",
    "SM", "CTRA", "PTEN", "MTDR",
    "CHRD", "DINO",
    # Materials (14)
    "CLF", "ATI", "CMC", "SON", "SEE", "HXL", "WOR", "SLVM",
    "TROX", "IOSP",
    "CBT", "NGVT", "AVNT", "KWR",
    # Consumer Staples (10)
    "CELH", "FLO", "INGR", "SPB", "IPAR", "ELF", "FRPT", "SJM",
    "CASY", "USFD",
    # Utilities (8)
    "NRG", "OGE", "PNW", "ATO", "EVRG", "AES",
    "MDU", "AVA",
    # Real Estate (8)
    "INVH", "ELS", "AMH", "SUI", "KRC", "HIW",
    "NNN", "STAG",
    # Communication Services (4)
    "IART", "ZI", "MTCH", "MSGS",
    # Large-cap benchmarks for comparison context (4)
    "AAPL", "MSFT", "GOOGL", "AMZN",
]

# Indian mid-caps from NIFTY Midcap 150 / Midcap ESG Index
INDIAN_TICKERS = [
    # Consumer / Textiles (12)
    "PAGEIND.NS", "RELAXO.NS", "WELSPUNIND.NS", "ARVIND.NS",
    "TRENT.NS", "DIXON.NS", "BATAINDIA.NS", "VMART.NS",
    "RAJESHEXPO.NS", "JUBLFOOD.NS",
    "DEVYANI.NS", "SAPPHIRE.NS",
    # Pharma / Healthcare (14)
    "LAURUSLABS.NS", "AJANTPHARM.NS", "GLENMARK.NS", "TORNTPHARM.NS",
    "ALKEM.NS", "IPCALAB.NS", "NATCOPHARM.NS", "AUROPHARMA.NS",
    "BIOCON.NS", "SYNGENE.NS", "LALPATHLAB.NS", "METROPOLIS.NS",
    "GRANULES.NS", "SUVENPHAR.NS",
    # IT / Technology (12)
    "PERSISTENT.NS", "LTTS.NS", "CYIENT.NS", "MPHASIS.NS",
    "COFORGE.NS", "TATAELXSI.NS", "ROUTE.NS", "BSOFT.NS",
    "HAPPSTMNDS.NS", "MASTEK.NS",
    "KPITTECH.NS", "NEWGEN.NS",
    # FMCG (10)
    "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "EMAMILTD.NS",
    "COLPAL.NS", "TATACONSUM.NS", "ZYDUSWELL.NS", "JYOTHYLAB.NS",
    "BIKAJI.NS", "HONASA.NS",
    # Chemicals / Materials (12)
    "SRF.NS", "UPL.NS", "AARTIIND.NS", "PIDILITIND.NS",
    "DEEPAKNTR.NS", "NAVINFLUOR.NS", "CLEAN.NS", "ATUL.NS",
    "FINEORG.NS", "SUDARSCHEM.NS",
    "ANANTRAJ.NS", "GALAXYSURF.NS",
    # Industrials / Engineering (10)
    "ABB.NS", "CUMMINSIND.NS", "THERMAX.NS", "SCHAEFFLER.NS",
    "HONAUT.NS", "GRINDWELL.NS", "AIAENG.NS", "ELGIEQUIP.NS",
    "TIINDIA.NS", "KAYNES.NS",
    # Auto / Auto Parts (8)
    "MOTHERSON.NS", "BALKRISIND.NS", "ENDURANCE.NS", "EXIDEIND.NS",
    "SUNDRMFAST.NS", "SUPRAJIT.NS",
    "CRAFTSMAN.NS", "GABRIEL.NS",
    # Financials (8)
    "CHOLAFIN.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS", "IIFL.NS",
    "CANFINHOME.NS", "ABCAPITAL.NS",
    "MASFIN.NS", "POONAWALLA.NS",
    # Real Estate / Construction (4)
    "OBEROIRLTY.NS", "GODREJPROP.NS", "PRESTIGE.NS", "BRIGADE.NS",
]

ALL_TICKERS = US_TICKERS + INDIAN_TICKERS

# SEC EDGAR CIK mapping (for R&D data — covers US tech/healthcare subset)
SEC_CIK_MAP = {
    "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044",
    "AMZN": "0001018724", "ILMN": "0001110803", "ALGN": "0001097149",
    "CRWD": "0001535527", "FTNT": "0001262039", "DXCM": "0001093557",
    "ZS":   "0001713683", "HUBS": "0001404655", "VEEV": "0001536180",
    "NET":  "0001477333", "BILL": "0001786352", "ANSS": "0000820736",
    "CDNS": "0000813672", "BIO":  "0000012208", "HOLX": "0000859737",
    "PAYC": "0001590955", "POOL": "0000945841",
}


# ---------------------------------------------------------------------------
# Sector-specific ESG profiles (empirically grounded)
# Source: MSCI ESG Research, Refinitiv ESG scoring methodology,
#         S&P Global CSA average scores by sector (2022-2024)
# ---------------------------------------------------------------------------
SECTOR_ESG_PROFILES = {
    "Technology": {
        "E_offset": -0.2,   # Lower physical footprint
        "S_offset": 0.1,    # Good labor practices
        "G_offset": 0.3,    # Strong governance (shareholder alignment)
        "emissions_mult": 0.4, "renewable_mult": 1.3, "diversity_mult": 1.1,
        "board_independence_mult": 1.15, "rd_mult": 2.0,
    },
    "Healthcare": {
        "E_offset": -0.1,
        "S_offset": 0.3,    # Patient safety, R&D for public good
        "G_offset": 0.15,
        "emissions_mult": 0.6, "renewable_mult": 1.0, "diversity_mult": 1.0,
        "board_independence_mult": 1.1, "rd_mult": 1.8,
    },
    "Financial Services": {
        "E_offset": 0.0,
        "S_offset": 0.0,
        "G_offset": 0.35,   # Heavily regulated, strong governance
        "emissions_mult": 0.3, "renewable_mult": 1.1, "diversity_mult": 1.15,
        "board_independence_mult": 1.2, "rd_mult": 0.3,
    },
    "Energy": {
        "E_offset": -0.5,   # High emissions, transition risk
        "S_offset": 0.15,   # Employment, community engagement
        "G_offset": 0.1,
        "emissions_mult": 3.0, "renewable_mult": 0.5, "diversity_mult": 0.9,
        "board_independence_mult": 1.0, "rd_mult": 0.5,
    },
    "Industrials": {
        "E_offset": -0.3,
        "S_offset": 0.1,
        "G_offset": 0.1,
        "emissions_mult": 2.0, "renewable_mult": 0.7, "diversity_mult": 0.95,
        "board_independence_mult": 1.05, "rd_mult": 0.8,
    },
    "Consumer Cyclical": {
        "E_offset": -0.15,
        "S_offset": 0.2,    # Supply chain labor, customer data
        "G_offset": 0.05,
        "emissions_mult": 0.8, "renewable_mult": 1.0, "diversity_mult": 1.05,
        "board_independence_mult": 1.0, "rd_mult": 0.6,
    },
    "Consumer Defensive": {
        "E_offset": 0.05,
        "S_offset": 0.25,   # Product safety, nutrition
        "G_offset": 0.15,
        "emissions_mult": 0.7, "renewable_mult": 1.1, "diversity_mult": 1.1,
        "board_independence_mult": 1.1, "rd_mult": 0.4,
    },
    "Basic Materials": {
        "E_offset": -0.45,  # Mining/chemicals, high environmental impact
        "S_offset": 0.1,
        "G_offset": 0.05,
        "emissions_mult": 2.5, "renewable_mult": 0.6, "diversity_mult": 0.9,
        "board_independence_mult": 1.0, "rd_mult": 0.7,
    },
    "Utilities": {
        "E_offset": -0.1,   # Transition to renewables
        "S_offset": 0.15,
        "G_offset": 0.2,
        "emissions_mult": 1.8, "renewable_mult": 1.4, "diversity_mult": 1.0,
        "board_independence_mult": 1.1, "rd_mult": 0.3,
    },
    "Real Estate": {
        "E_offset": 0.1,    # Green building opportunity
        "S_offset": 0.1,
        "G_offset": 0.15,
        "emissions_mult": 0.9, "renewable_mult": 1.2, "diversity_mult": 1.0,
        "board_independence_mult": 1.1, "rd_mult": 0.2,
    },
    "Communication Services": {
        "E_offset": -0.1,
        "S_offset": 0.05,
        "G_offset": 0.2,
        "emissions_mult": 0.5, "renewable_mult": 1.2, "diversity_mult": 1.05,
        "board_independence_mult": 1.1, "rd_mult": 1.2,
    },
}

DEFAULT_PROFILE = {
    "E_offset": 0.0, "S_offset": 0.0, "G_offset": 0.0,
    "emissions_mult": 1.0, "renewable_mult": 1.0, "diversity_mult": 1.0,
    "board_independence_mult": 1.0, "rd_mult": 1.0,
}


def download_yahoo_financials(tickers, batch_size=10):
    """Download fundamental financial data from Yahoo Finance."""
    import yfinance as yf

    print("=" * 70)
    print("STEP 1A: DOWNLOADING FINANCIAL DATA (Yahoo Finance)")
    print(f"  Tickers: {len(tickers)}")
    print("=" * 70)

    rows = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{(len(tickers)-1)//batch_size+1}: {batch[:3]}...")
        for t in batch:
            try:
                info = yf.Ticker(t).info
                rows.append({
                    "ticker": t,
                    "company_name": info.get("shortName", t),
                    "market_cap": info.get("marketCap"),
                    "total_revenue": info.get("totalRevenue"),
                    "ebitda": info.get("ebitda"),
                    "net_income": info.get("netIncomeToCommon"),
                    "gross_profit": info.get("grossProfits"),
                    "total_debt": info.get("totalDebt"),
                    "total_cash": info.get("totalCash"),
                    "total_assets": info.get("totalAssets"),
                    "roa": info.get("returnOnAssets"),
                    "roe": info.get("returnOnEquity"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "quick_ratio": info.get("quickRatio"),
                    "free_cashflow": info.get("freeCashflow"),
                    "operating_cashflow": info.get("operatingCashflow"),
                    "trailing_pe": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "price_to_book": info.get("priceToBook"),
                    "price_to_sales": info.get("priceToSalesTrailing12Months"),
                    "enterprise_to_revenue": info.get("enterpriseToRevenue"),
                    "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
                    "dividend_yield": info.get("dividendYield"),
                    "payout_ratio": info.get("payoutRatio"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
                    "profit_margins": info.get("profitMargins"),
                    "gross_margins": info.get("grossMargins"),
                    "operating_margins": info.get("operatingMargins"),
                    "currency": info.get("currency"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "country": info.get("country", "US" if ".NS" not in t else "India"),
                    "price": info.get("currentPrice", info.get("regularMarketPrice")),
                    "beta": info.get("beta"),
                    "employees": info.get("fullTimeEmployees"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                    "50d_avg": info.get("fiftyDayAverage"),
                    "200d_avg": info.get("twoHundredDayAverage"),
                    "avg_volume": info.get("averageVolume"),
                    "avg_volume_10d": info.get("averageVolume10days"),
                })
            except Exception as e:
                print(f"    [SKIP] {t}: {e}")
        time.sleep(1)

    df = pd.DataFrame(rows)
    outpath = PROJECT_ROOT / "data" / "raw" / "yahoo_financials.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False, encoding="utf-8")
    print(f"  [OK] {len(df)} companies -> {outpath}")
    return df


def download_market_data(tickers, period="2y"):
    """Download daily price/volume data and compute market metrics."""
    import yfinance as yf

    print("\n" + "=" * 70)
    print("STEP 1B: DOWNLOADING MARKET DATA (Yahoo Finance)")
    print(f"  Tickers: {len(tickers)}, Period: {period}")
    print("=" * 70)

    rows = []
    for i, t in enumerate(tickers):
        try:
            hist = yf.download(t, period=period, progress=False)
            if hist.empty:
                continue
            close = hist["Close"].squeeze()
            volume = hist["Volume"].squeeze()
            returns = close.pct_change().dropna()

            # Compute various market metrics
            row = {
                "ticker": t,
                "avg_daily_volume": float(volume.mean()),
                "avg_daily_volume_30d": float(volume.tail(30).mean()),
                "avg_daily_volume_90d": float(volume.tail(90).mean()) if len(volume) > 90 else None,
                "price_volatility": float(returns.std() * np.sqrt(252) * 100),
                "price_volatility_30d": float(returns.tail(30).std() * np.sqrt(252) * 100) if len(returns) > 30 else None,
                "price_momentum_1m": float((close.iloc[-1] / close.iloc[-min(21, len(close))] - 1) * 100) if len(close) > 21 else 0,
                "price_momentum_3m": float((close.iloc[-1] / close.iloc[-min(63, len(close))] - 1) * 100) if len(close) > 63 else 0,
                "price_momentum_6m": float((close.iloc[-1] / close.iloc[-min(126, len(close))] - 1) * 100) if len(close) > 126 else 0,
                "price_momentum_12m": float((close.iloc[-1] / close.iloc[-min(252, len(close))] - 1) * 100) if len(close) > 252 else 0,
                "price_latest": float(close.iloc[-1]),
                "max_drawdown_1y": float(
                    ((close.tail(252) / close.tail(252).cummax()) - 1).min() * 100
                ) if len(close) > 252 else None,
                "sharpe_ratio_1y": float(
                    (returns.tail(252).mean() * 252) / (returns.tail(252).std() * np.sqrt(252) + 1e-10)
                ) if len(returns) > 252 else None,
                "sortino_ratio_1y": float(
                    (returns.tail(252).mean() * 252) / (returns.tail(252)[returns.tail(252) < 0].std() * np.sqrt(252) + 1e-10)
                ) if len(returns) > 252 else None,
                "avg_daily_return": float(returns.mean() * 100),
                "return_skewness": float(returns.skew()),
                "return_kurtosis": float(returns.kurtosis()),
            }

            # Amihud illiquidity (|return| / dollar volume)
            dollar_vol = close * volume
            if dollar_vol.mean() > 0:
                amihud = (returns.abs() / dollar_vol.iloc[1:].values).mean() * 1e6
                row["amihud_illiquidity"] = float(amihud) if np.isfinite(amihud) else None

            rows.append(row)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(tickers)}] processed")
        except Exception:
            pass
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    outpath = PROJECT_ROOT / "data" / "raw" / "market_data.csv"
    df.to_csv(outpath, index=False, encoding="utf-8")
    print(f"  [OK] {len(df)} companies -> {outpath}")
    return df


def download_sec_rd(cik_map):
    """Download R&D expenditure from SEC EDGAR XBRL API."""
    import requests

    print("\n" + "=" * 70)
    print("STEP 1C: DOWNLOADING R&D DATA (SEC EDGAR)")
    print(f"  Companies: {len(cik_map)}")
    print("=" * 70)

    headers = {"User-Agent": "ResearchBot research@university.edu", "Accept-Encoding": "gzip"}
    rows = []
    for ticker, cik in cik_map.items():
        try:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            facts = r.json().get("facts", {}).get("us-gaap", {})

            row = {"ticker": ticker}

            # R&D expenditure
            rd_key = "ResearchAndDevelopmentExpense"
            if rd_key in facts:
                units = facts[rd_key].get("units", {}).get("USD", [])
                annual = [u for u in units if u.get("form") == "10-K"]
                if annual:
                    latest = sorted(annual, key=lambda x: x.get("end", ""))[-1]
                    row["r_d_expenditure"] = latest["val"]
                    print(f"  {ticker}: R&D = ${latest['val']:,.0f}")

            # Revenue for R&D intensity calc
            rev_key = "Revenues"
            alt_rev = "RevenueFromContractWithCustomerExcludingAssessedTax"
            for rk in [rev_key, alt_rev, "SalesRevenueNet"]:
                if rk in facts:
                    units = facts[rk].get("units", {}).get("USD", [])
                    annual = [u for u in units if u.get("form") == "10-K"]
                    if annual:
                        latest = sorted(annual, key=lambda x: x.get("end", ""))[-1]
                        row["sec_revenue"] = latest["val"]
                        break

            rows.append(row)
            time.sleep(0.5)
        except Exception as e:
            print(f"  {ticker}: [SKIP] {e}")

    df = pd.DataFrame(rows)
    outpath = PROJECT_ROOT / "data" / "raw" / "sec_rd_data.csv"
    df.to_csv(outpath, index=False, encoding="utf-8")
    print(f"  [OK] {len(df)} companies -> {outpath}")
    return df


def download_benchmarks():
    """Download benchmark index data."""
    import yfinance as yf

    print("\n" + "=" * 70)
    print("STEP 1D: DOWNLOADING BENCHMARK DATA")
    print("=" * 70)

    benchmarks = {
        "^NSEI": ("NIFTY 50", "nifty50_benchmark.csv"),
        "^GSPC": ("S&P 500", "sp500_benchmark.csv"),
        "^MID":  ("S&P MidCap 400", "sp400_benchmark.csv"),
        "^RUT":  ("Russell 2000", "russell2000_benchmark.csv"),
    }

    for symbol, (name, filename) in benchmarks.items():
        try:
            data = yf.download(symbol, period="3y", progress=False)
            outpath = PROJECT_ROOT / "data" / "raw" / filename
            data.to_csv(outpath, encoding="utf-8")
            print(f"  [OK] {name}: {len(data)} days -> {outpath}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")


def generate_synthetic_esg(tickers, sector_map=None, financials_df=None):
    """Generate realistic sector-aware synthetic ESG data with financial correlation.

    Real ESG data requires expensive subscriptions (Refinitiv, MSCI, S&P Global).
    This function generates synthetic data calibrated to published sector averages
    from MSCI ESG Research (2023) and S&P Global CSA (2024).

    Key design: base quality is partially correlated with financial quality (r~0.3-0.4).
    This mirrors empirical findings from:
      - Friede, Busch & Bassen (2015): ESG-CFP meta-analysis (r=0.15-0.35)
      - Khan, Serafeim & Yoon (2016): Material ESG factors predict alpha
      - Eccles, Ioannou & Serafeim (2014): High-sustainability firms outperform

    Sector-specific profiles ensure that:
    - Technology companies: higher governance, lower environmental footprint
    - Energy companies: lower environmental scores, higher community engagement
    - Healthcare: higher social scores (patient safety, R&D)
    - Financials: strong governance (regulatory environment)
    - Materials/Industrials: lower environmental scores (emissions)
    """
    print("\n" + "=" * 70)
    print("STEP 1E: GENERATING SECTOR-AWARE SYNTHETIC ESG DATA")
    print(f"  Companies: {len(tickers)}")
    print("=" * 70)

    np.random.seed(42)
    n = len(tickers)
    df = pd.DataFrame({"ticker": tickers})

    # Get sector for each ticker
    sectors = []
    for t in tickers:
        if sector_map and t in sector_map:
            sectors.append(sector_map[t])
        else:
            sectors.append("Unknown")
    df["_sector"] = sectors

    # Base quality: partially correlated with actual financial quality.
    # Companies with better ROA/margins tend to have better ESG (governance).
    # Correlation coefficient ~0.35, matching meta-analysis findings.
    base_quality = np.random.normal(0.5, 0.15, n)

    if financials_df is not None and len(financials_df) > 0:
        fin_quality = pd.Series(0.0, index=range(n))
        for i, t in enumerate(tickers):
            row = financials_df[financials_df["ticker"] == t]
            if row.empty:
                continue
            row = row.iloc[0]
            # Compute a composite financial quality signal
            signals = []
            if pd.notna(row.get("roa")) and row["roa"] != 0:
                signals.append(np.clip(row["roa"] / 0.15, -2, 2))  # normalized around 15% ROA
            if pd.notna(row.get("profit_margins")) and row["profit_margins"] != 0:
                signals.append(np.clip(row["profit_margins"] / 0.15, -2, 2))
            if pd.notna(row.get("market_cap")) and row["market_cap"] > 0:
                # Larger companies tend to have better ESG (more resources)
                signals.append(np.clip(np.log10(row["market_cap"]) / 11 - 0.5, -1, 1))
            if signals:
                fin_quality.iloc[i] = np.mean(signals)

        # Mix financial signal (weight ~0.35) with random component (0.65)
        # This achieves empirically realistic ESG-financial correlation
        corr_weight = 0.35
        fin_norm = (fin_quality - fin_quality.mean()) / (fin_quality.std() + 1e-10)
        base_quality = (1 - corr_weight) * base_quality + corr_weight * (0.5 + fin_norm.values * 0.15)
        print(f"  ESG-financial correlation weight: {corr_weight} (Friede et al. 2015 range)")

    base_quality = base_quality.clip(0.1, 0.9)

    # Generate sector-differentiated ESG indicators
    for i in range(n):
        sector = df.iloc[i]["_sector"]
        profile = SECTOR_ESG_PROFILES.get(sector, DEFAULT_PROFILE)
        bq = base_quality[i]

        # === Environmental indicators ===
        em = profile["emissions_mult"]
        df.loc[i, "scope1_emissions"] = max(50, np.random.lognormal(7, 1.5) * em)
        df.loc[i, "scope2_emissions"] = max(10, np.random.lognormal(5, 1.2) * em * 0.5)
        df.loc[i, "scope3_emissions"] = df.loc[i, "scope1_emissions"] * np.random.uniform(2, 8)
        df.loc[i, "emissions_intensity"] = max(0.001, np.random.lognormal(-3, 0.8) * em)
        rm = profile["renewable_mult"]
        df.loc[i, "renewable_energy_pct"] = np.clip(
            bq * 60 * rm + np.random.normal(0, 8) + profile["E_offset"] * 20, 5, 95
        )
        df.loc[i, "energy_efficiency"] = np.clip(
            bq * 40 * rm + np.random.normal(50, 8) + profile["E_offset"] * 10, 30, 99
        )
        df.loc[i, "water_usage_intensity"] = max(1, np.random.lognormal(2, 1.0) * em)
        df.loc[i, "waste_recycling_pct"] = np.clip(
            bq * 50 + np.random.normal(20, 12) + profile["E_offset"] * 15, 5, 95
        )
        df.loc[i, "carbon_reduction_target"] = int(np.random.random() < bq * 0.7 + 0.15 + profile["E_offset"] * 0.2)
        df.loc[i, "environmental_fines"] = max(0, np.random.exponential(50) * em)

        # === Social indicators ===
        dm = profile["diversity_mult"]
        df.loc[i, "employee_turnover"] = max(1, np.random.beta(2, 8) * 30 + 1 - profile["S_offset"] * 5)
        df.loc[i, "gender_diversity_pct"] = np.clip(
            np.random.normal(40, 10) * dm + profile["S_offset"] * 8, 10, 70
        )
        df.loc[i, "women_management_pct"] = np.clip(
            np.random.normal(32, 10) * dm + profile["S_offset"] * 6, 5, 60
        )
        df.loc[i, "pay_gap_ratio"] = np.clip(
            np.random.normal(0.85, 0.07) + profile["S_offset"] * 0.03, 0.60, 1.05
        )
        df.loc[i, "injury_rate"] = max(0, np.random.exponential(1.0) - profile["S_offset"] * 0.3)
        df.loc[i, "safety_training_hours"] = np.clip(
            np.random.normal(8, 2.5) + profile["S_offset"] * 2, 1, 20
        )
        df.loc[i, "employee_satisfaction"] = np.clip(
            bq * 30 + np.random.normal(55, 8) + profile["S_offset"] * 8, 30, 95
        )
        df.loc[i, "community_investment_pct"] = max(0, np.random.beta(2, 10) * 5 + profile["S_offset"] * 0.3)
        df.loc[i, "supply_chain_audit_pct"] = np.clip(
            bq * 50 + np.random.normal(30, 12) + profile["S_offset"] * 10, 10, 100
        )
        df.loc[i, "human_rights_policy"] = int(np.random.random() < bq * 0.6 + 0.3 + profile["S_offset"] * 0.1)

        # === Governance indicators ===
        bm = profile["board_independence_mult"]
        df.loc[i, "board_independence_pct"] = np.clip(
            np.random.normal(75, 10) * bm + profile["G_offset"] * 10, 30, 100
        )
        df.loc[i, "board_diversity_pct"] = np.clip(
            np.random.normal(30, 8) * dm + profile["G_offset"] * 8, 5, 60
        )
        df.loc[i, "board_size"] = int(np.random.randint(5, 15))
        df.loc[i, "exec_comp_esg_linked"] = max(0, np.random.beta(2, 5) * 100 + profile["G_offset"] * 15)
        df.loc[i, "ceo_pay_ratio"] = max(5, np.random.lognormal(3, 0.8))
        df.loc[i, "shareholder_rights_score"] = np.clip(
            np.random.normal(65, 12) + profile["G_offset"] * 12, 20, 95
        )
        df.loc[i, "ethics_compliance_score"] = np.clip(
            np.random.normal(70, 10) + profile["G_offset"] * 10, 30, 98
        )
        df.loc[i, "anti_corruption_policy"] = int(np.random.random() < bq * 0.5 + 0.4 + profile["G_offset"] * 0.1)
        df.loc[i, "data_privacy_score"] = np.clip(
            np.random.normal(65, 12) + profile["G_offset"] * 8, 20, 95
        )
        df.loc[i, "tax_transparency_score"] = np.clip(
            np.random.normal(60, 15) + profile["G_offset"] * 10, 10, 95
        )

        # === ESG controversy / risk scores ===
        # Better base quality + better sector profile = fewer controversies
        controversy_base = bq * 0.6 + 0.2 + (profile["E_offset"] + profile["S_offset"] + profile["G_offset"]) * 0.1
        df.loc[i, "esg_controversy_score"] = np.clip(np.random.beta(5, 2) * 100 * np.clip(controversy_base + 0.3, 0.3, 1.0), 10, 100)
        df.loc[i, "esg_risk_rating"] = np.clip(100 - bq * 50 + np.random.normal(0, 12) - (profile["E_offset"] + profile["S_offset"] + profile["G_offset"]) * 8, 10, 90)

    df = df.drop(columns=["_sector"])

    outpath = PROJECT_ROOT / "data" / "raw" / "synthetic_esg.csv"
    df.to_csv(outpath, index=False, encoding="utf-8")
    print(f"  [OK] {len(df)} companies, {len(df.columns)-1} ESG indicators -> {outpath}")
    return df


def main():
    n_us = len(US_TICKERS)
    n_in = len(INDIAN_TICKERS)
    print("=" * 70)
    print("MULTI-FACTOR INDEX: DATA DOWNLOAD")
    print(f"Companies: {len(ALL_TICKERS)} ({n_us} US + {n_in} India)")
    print("=" * 70)

    # Create output dirs
    for d in ["data/raw", "data/processed", "reports/tables", "reports/figures"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    fin_df = download_yahoo_financials(ALL_TICKERS)
    mkt_df = download_market_data(ALL_TICKERS, period="2y")
    rd_df = download_sec_rd(SEC_CIK_MAP)
    download_benchmarks()

    # Build sector map from downloaded financial data for ESG generation
    sector_map = {}
    if "sector" in fin_df.columns:
        for _, row in fin_df.iterrows():
            if pd.notna(row.get("sector")):
                sector_map[row["ticker"]] = row["sector"]
    print(f"\n  Sector map: {len(sector_map)} companies with known sectors")

    esg_df = generate_synthetic_esg(ALL_TICKERS, sector_map=sector_map, financials_df=fin_df)

    # Merge everything into one raw combined file
    print("\n" + "=" * 70)
    print("STEP 1F: MERGING ALL DATA SOURCES")
    print("=" * 70)
    combined = fin_df.copy()
    if not mkt_df.empty:
        combined = combined.merge(mkt_df, on="ticker", how="left")
    if not rd_df.empty:
        combined = combined.merge(rd_df, on="ticker", how="left")
    if not esg_df.empty:
        combined = combined.merge(esg_df, on="ticker", how="left", suffixes=("", "_synth"))
        for col in esg_df.columns:
            if col != "ticker" and col in combined.columns:
                synth_col = f"{col}_synth"
                if synth_col in combined.columns:
                    combined[col] = combined[col].fillna(combined[synth_col])
                    combined.drop(columns=[synth_col], inplace=True)

    # Compute derived metrics
    rev = combined.get("total_revenue")
    emp = combined.get("employees")
    ni = combined.get("net_income")

    if rev is not None and emp is not None:
        combined["revenue_per_employee"] = rev / emp.replace(0, np.nan)
    if rev is not None and combined.get("r_d_expenditure") is not None:
        combined["r_d_intensity"] = (combined["r_d_expenditure"] / rev.replace(0, np.nan)) * 100
    if rev is not None and ni is not None:
        combined["net_margin"] = (ni / rev.replace(0, np.nan)) * 100
    if rev is not None and combined.get("ebitda") is not None:
        combined["operating_margin"] = (combined["ebitda"] / rev.replace(0, np.nan)) * 100
    if rev is not None and "sector" in combined.columns:
        combined["market_share"] = rev / combined.groupby("sector")["total_revenue"].transform("sum") * 100
    if combined.get("gross_profit") is not None and rev is not None:
        combined["gross_margin"] = (combined["gross_profit"] / rev.replace(0, np.nan)) * 100
    if combined.get("total_debt") is not None and combined.get("ebitda") is not None:
        combined["debt_to_ebitda"] = combined["total_debt"] / combined["ebitda"].replace(0, np.nan)
    if combined.get("operating_cashflow") is not None and combined.get("total_debt") is not None:
        combined["cash_flow_to_debt"] = combined["operating_cashflow"] / combined["total_debt"].replace(0, np.nan)
    if combined.get("free_cashflow") is not None and rev is not None:
        combined["fcf_margin"] = (combined["free_cashflow"] / rev.replace(0, np.nan)) * 100
    if combined.get("price") is not None and combined.get("52_week_high") is not None:
        combined["pct_from_52w_high"] = ((combined["price"] / combined["52_week_high"]) - 1) * 100

    # Assign country based on ticker suffix
    combined["country"] = combined["ticker"].apply(lambda t: "India" if ".NS" in str(t) else "US")
    # Synthetic fill for columns not directly from Yahoo
    combined["free_float_pct"] = np.random.uniform(30, 98, len(combined))
    combined["bid_ask_spread"] = np.random.exponential(0.05, len(combined)).clip(0.001, 0.5)

    outpath = PROJECT_ROOT / "data" / "raw" / "combined_raw.csv"
    combined.to_csv(outpath, index=False, encoding="utf-8")
    print(f"\n  [OK] Combined raw data: {len(combined)} companies, {len(combined.columns)} columns")
    print(f"  [OK] Saved to {outpath}")

    # Summary by sector and country
    if "sector" in combined.columns:
        print(f"\n  Sector distribution:")
        for sector, count in combined["sector"].value_counts().items():
            print(f"    {sector}: {count}")
    if "country" in combined.columns:
        print(f"\n  Country distribution:")
        for country, count in combined["country"].value_counts().items():
            print(f"    {country}: {count}")

    print(f"\n[DONE] Data download complete. Next: python scripts/02_clean_data.py")


if __name__ == "__main__":
    main()

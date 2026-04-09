import streamlit as st
import pandas as pd

MARKET_OPTIONS = {
    "Min Watchlist": None,
    "Oslo Bors": "oslo",
    "USA (Nasdaq 100)": "nasdaq",
    "USA (S&P 500)": "sp500",
    "USA (Alle Aksjer)": "us_all",
    "Stockholm (Large)": "stockholm",
    "Frankfurt (DAX)": "frankfurt",
    "Paris (CAC)": "paris",
    "London (FTSE)": "london",
    "Egen liste": None,
}


def get_tickers_for_market(market_name: str) -> list:
    key = MARKET_OPTIONS.get(market_name)
    if key is None:
        return []
    fetchers = {
        "oslo": get_oslo_tickers,
        "nasdaq": get_nasdaq_tickers,
        "sp500": get_sp500_tickers,
        "us_all": get_all_us_tickers,
        "stockholm": get_stockholm_tickers,
        "frankfurt": get_frankfurt_tickers,
        "paris": get_paris_tickers,
        "london": get_london_tickers,
    }
    return fetchers[key]()


@st.cache_data(ttl=24 * 3600)
def get_sp500_tickers():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].tolist()
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]


@st.cache_data(ttl=24 * 3600)
def get_nasdaq_tickers():
    return [
        "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP",
        "COST", "ADBE", "CSCO", "NFLX", "AMD", "TMUS", "CMCSA", "TXN", "HON", "AMGN",
        "INTC", "INTU", "QCOM", "SBUX", "GILD", "AMAT", "ISRG", "MDLZ", "BKNG", "ADI",
        "ADP", "VRTX", "REGN", "PYPL", "FISV", "LRCX", "MU", "CSX", "MELI", "MNST",
        "PANW", "SNPS", "ASML", "KLAC", "CDNS", "CHTR", "MAR", "ORLY", "CTAS", "FTNT",
        "DXCM", "ABNB", "KDP", "NXPI", "AEP", "ADSK", "KHC", "MCHP", "IDXX", "PAYX",
        "EXC", "LULU", "PCAR", "AZN", "ROST", "ODFL", "MRVL", "BIIB", "WBA", "BKR",
        "CPRT", "FAST", "XEL", "EA", "DLTR", "VRSK", "CTSH", "CSGP", "CRWD", "ANSS",
        "TEAM", "ALGN", "ILMN", "EBAY", "CEG", "DDOG", "ZM", "PDD", "WDAY", "ZS", "RIVN",
    ]


@st.cache_data(ttl=24 * 3600)
def get_all_us_tickers():
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        df = pd.read_csv(url, header=None, dtype=str).dropna()
        tickers = df[0].tolist()
        return [str(t) for t in tickers if isinstance(t, str) and "^" not in t]
    except Exception:
        return ["AAPL", "MSFT", "TSLA", "NVDA"]


def get_oslo_tickers():
    return [
        "EQNR.OL", "DNB.OL", "NHY.OL", "TEL.OL", "ORK.OL", "MOWI.OL", "YAR.OL",
        "TOM.OL", "GJF.OL", "STB.OL", "SALM.OL", "AKRBP.OL", "SUBC.OL", "KOG.OL",
        "NAS.OL", "FRO.OL", "MPCC.OL", "VAR.OL", "PGS.OL", "TGS.OL", "LSG.OL",
        "BAKKA.OL", "ENTRA.OL", "SCHA.OL", "SCHB.OL", "AFG.OL", "ATEA.OL", "BON.OL",
        "BWLPG.OL", "DNO.OL", "ELK.OL", "EPR.OL", "FLNG.OL", "GOGL.OL", "HAFNI.OL",
        "HEX.OL", "IDEX.OL", "KIT.OL", "NOD.OL", "OTOVO.OL", "PHEL.OL", "RECSI.OL",
        "SRBNK.OL", "VEI.OL", "VOW.OL", "XXL.OL", "ADE.OL", "BOUV.OL", "AMS.OL",
        "ASTK.OL", "AUSS.OL", "BEL.OL", "BGBIO.OL", "BWE.OL", "CADLR.OL", "CLOUD.OL",
        "DSRT.OL", "ELMRA.OL", "EMGS.OL", "GIG.OL", "GOLF.OL", "HAVI.OL", "HUNT.OL",
        "KAHOT.OL", "LINK.OL", "MSEIS.OL", "NORBT.OL", "NANOV.OL", "OET.OL", "OKEA.OL",
        "PCIB.OL", "PHO.OL", "PNOR.OL", "PRS.OL", "PSI.OL", "RANA.OL", "SATS.OL",
        "SDSD.OL", "SIOFF.OL", "SKUE.OL", "SOON.OL", "SPOL.OL", "TRE.OL", "VOLUE.OL",
        "WAWI.OL", "ZAP.OL", "AUTO.OL", "VEND.OL",
    ]


@st.cache_data(ttl=24 * 3600)
def get_stockholm_tickers():
    return [
        "ABB.ST", "ALFA.ST", "ASSAB.ST", "AZN.ST", "ATCO-A.ST", "ATCO-B.ST", "BOL.ST",
        "ELUX-B.ST", "ERIC-B.ST", "ESSITY-B.ST", "EVO.ST", "GETI-B.ST", "HEXA-B.ST",
        "HM-B.ST", "HUSQ-B.ST", "INVE-B.ST", "KINV-B.ST", "NDA-SE.ST", "NIBE-B.ST",
        "SAAB-B.ST", "SAND.ST", "SCA-B.ST", "SEB-A.ST", "SINCH.ST", "SKF-B.ST",
        "SSAB-A.ST", "SSAB-B.ST", "SWED-A.ST", "SHB-A.ST", "TEL2-B.ST", "TELIA.ST",
        "VOLV-B.ST", "BALD-B.ST", "CAST.ST", "LATO-B.ST", "INDT.ST", "LUND-B.ST",
        "LUMI.ST", "AXFO.ST", "BILI-A.ST", "VITR.ST", "THULE.ST", "AAK.ST",
    ]


@st.cache_data(ttl=24 * 3600)
def get_frankfurt_tickers():
    return [
        "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE",
        "CBK.DE", "CON.DE", "1COV.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE",
        "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "MBG.DE",
        "MRK.DE", "MTX.DE", "MUV2.DE", "PUM.DE", "QIA.DE", "RWE.DE", "SAP.DE",
        "SRT3.DE", "SIE.DE", "ENR.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE",
        "LHA.DE", "BOSS.DE", "P911.DE",
    ]


@st.cache_data(ttl=24 * 3600)
def get_paris_tickers():
    return [
        "AI.PA", "AIR.PA", "ALO.PA", "MT.PA", "CS.PA", "BNP.PA", "EN.PA", "CAP.PA",
        "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "EDEN.PA", "EL.PA", "ERF.PA", "RMS.PA",
        "KER.PA", "LR.PA", "OR.PA", "MC.PA", "ML.PA", "ORA.PA", "RI.PA", "PUB.PA",
        "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA", "SU.PA", "GLE.PA", "STLAM.PA",
        "STMPA.PA", "TEP.PA", "HO.PA", "TTE.PA", "URW.PA", "VIE.PA", "DG.PA",
        "VIV.PA", "WLN.PA",
    ]


@st.cache_data(ttl=24 * 3600)
def get_london_tickers():
    return [
        "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "DGE.L", "RIO.L",
        "REL.L", "GLEN.L", "BATS.L", "LSEG.L", "CNA.L", "NG.L", "CPG.L", "BA.L",
        "LLOY.L", "PRU.L", "BARC.L", "VOD.L", "RR.L", "AAL.L", "NWG.L", "EXPN.L",
        "TSCO.L", "STAN.L", "WPP.L", "AV.L", "LGEN.L", "SSE.L", "SMT.L", "MKS.L",
        "SBRY.L", "KGF.L", "IMB.L", "TW.L", "BDEV.L", "JD.L", "RMV.L", "AUTO.L",
    ]

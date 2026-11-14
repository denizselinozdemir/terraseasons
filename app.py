"""TerraSeasons: which season is changing fastest in Turkey? (NASA MERRA-2, 1980-2025)"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
TR2EN = {"Kış": "Winter", "İlkbahar": "Spring", "Yaz": "Summer", "Sonbahar": "Autumn"}
AREA_EN = {"Türkiye": "Turkey", "Akdeniz": "Mediterranean", "Doğu Anadolu": "Eastern Anatolia",
           "Ege": "Aegean", "Güneydoğu Anadolu": "Southeastern Anatolia", "Karadeniz": "Black Sea",
           "Marmara": "Marmara", "İç Anadolu": "Central Anatolia"}
VARS = {"Temperature": ("sicaklik", "°C", "Seasonal mean 2 m air temperature (°C)", "°C per decade"),
        "Precipitation": ("yagis", "mm", "Seasonal total precipitation (mm)", "mm per decade")}

st.set_page_config(page_title="TerraSeasons", page_icon="🌍", layout="wide")


@st.cache_data
def load(name):
    p = DATA / name
    if not p.exists():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


series = load("mevsim_serileri.csv")
trends = load("mevsim_egilimleri.csv")
if series is None or trends is None:
    st.error("Data files not found. Run `python mevsim_analiz.py` first.")
    st.stop()
for df in (series, trends):
    df["season"] = df["mevsim"].map(TR2EN)
    df["area"] = df["bolge"].map(AREA_EN)

st.title("🌍 TerraSeasons")
st.caption("Which season is changing fastest in Turkey? · NASA MERRA-2 · 1980–2025")
st.markdown(
    "Annual averages hide a lot. **TerraSeasons** splits Turkey's climate record into winter, spring, "
    "summer and autumn and shows how fast **temperature** and **precipitation** are changing in each "
    "season, for Turkey as a whole and for its seven geographic regions. Every trend comes with a "
    "confidence interval and a significance test."
)

# ---- headline findings (computed from the data, not typed in) ----
tt = trends[(trends.degisken == "sicaklik") & (trends.area == "Turkey")].set_index("season")
tp = trends[(trends.degisken == "yagis") & (trends.area == "Turkey")].set_index("season")
fast, slow = tt["hac_egim"].idxmax(), tt["hac_egim"].idxmin()
c1, c2, c3 = st.columns(3)
c1.metric("Fastest-warming season", fast, f"{tt.loc[fast, 'hac_egim']:+.2f} °C / decade")
c2.metric("Slowest-warming season", slow, f"{tt.loc[slow, 'hac_egim']:+.2f} °C / decade")
sig_p = [s for s in SEASONS if s in tp.index and tp.loc[s, "hac_p"] < 0.05]
c3.metric("Seasons with a significant national rain trend", f"{len(sig_p)} of 4",
          ", ".join(sig_p) if sig_p else "none")

st.divider()
ca, cb = st.columns(2)
var_label = ca.radio("Variable", list(VARS), horizontal=True)
areas = ["Turkey"] + sorted(a for a in trends["area"].unique() if a != "Turkey")
area = cb.selectbox("Area", areas)
vkey, unit, ylabel, slope_unit = VARS[var_label]

sel = trends[(trends.degisken == vkey) & (trends.area == area)].set_index("season").reindex(SEASONS)
sser = series[(series.degisken == vkey) & (series.area == area)]

st.subheader(f"{var_label} trend by season: {area}")
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(4)
vals = sel["hac_egim"].to_numpy(float)
lo = vals - sel["hac_ga_alt"].to_numpy(float)
hi = sel["hac_ga_ust"].to_numpy(float) - vals
ax.bar(x, vals, yerr=[lo, hi], capsize=6, alpha=0.8)
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(SEASONS)
ax.set_ylabel(slope_unit)
ax.set_title("Trend per decade with 95% confidence interval (HAC)")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("A bar whose interval crosses zero means the data cannot confidently tell the trend from no change.")

tab = sel.reset_index()[["season", "ortalama", "hac_egim", "hac_ga_alt", "hac_ga_ust", "hac_p", "kendall_p"]]
tab.columns = ["Season", f"Mean ({unit})", f"Trend ({slope_unit})", "CI low", "CI high", "HAC p", "Mann-Kendall p"]
st.dataframe(tab.round(3), hide_index=True)

st.subheader("Look at one season in detail")
season = st.radio("Season", SEASONS, horizontal=True)
one = sser[sser.season == season].sort_values("yil")
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(one["yil"], one["deger"], marker="o", markersize=3, linewidth=1.3, label=f"{season} value")
if len(one) > 2:
    ax.plot(one["yil"], np.polyval(np.polyfit(one["yil"], one["deger"], 1), one["yil"]),
            linestyle="--", linewidth=2.2, label="Linear trend")
ax.set_xlabel("Year")
ax.set_ylabel(ylabel)
ax.set_title(f"{area}: {season.lower()} {var_label.lower()}, 1980–2025")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Winter of year Y = December (Y-1), January and February of Y, so winters start in 1981.")

st.subheader(f"Where is it changing? {var_label} trend, all regions × seasons")
reg = trends[(trends.degisken == vkey) & (trends.area != "Turkey")]
piv = reg.pivot(index="area", columns="season", values="hac_egim").reindex(columns=SEASONS)
sig = reg.pivot(index="area", columns="season", values="anlamli").reindex(columns=SEASONS)
fig, ax = plt.subplots(figsize=(8, 4.6))
if vkey == "sicaklik":
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
else:
    lim = float(np.nanmax(np.abs(piv.values)))
    im = ax.imshow(piv.values, aspect="auto", cmap="BrBG", vmin=-lim, vmax=lim)
ax.set_xticks(range(4))
ax.set_xticklabels(SEASONS)
ax.set_yticks(range(len(piv)))
ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        s = "*" if bool(sig.values[i, j]) else ""
        ax.text(j, i, f"{piv.values[i, j]:+.2f}{s}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, label=slope_unit)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("* = significant after Benjamini-Hochberg correction for 28 regional tests (28 = 7 regions × 4 seasons).")

with st.expander("ℹ️ How to read these numbers"):
    st.markdown(
        "- **Trend per decade** is the average change every 10 years, estimated by linear regression.\n"
        "- **95% confidence interval (CI)**: if it does not include zero, the trend is statistically significant.\n"
        "- **HAC (Newey-West)** standard errors account for neighbouring years being similar (autocorrelation).\n"
        "- **Mann-Kendall** and **Theil-Sen** are rank-based checks that do not assume a normal distribution.\n"
        "- **Benjamini-Hochberg** correction controls false positives when many regional tests are run together.\n"
        "- Precipitation varies a lot from year to year, so its trends are much harder to confirm than temperature trends."
    )
with st.expander("⚠️ Limitations"):
    st.markdown(
        "- MERRA-2 is a **reanalysis** (model + observations), not station data. Its grid is coarse (about 55 km), "
        "so small regions contain few grid cells.\n"
        "- The record is 46 years; natural multi-decade variability can influence a trend.\n"
        "- Only seasonal averages and totals are analysed, not extremes, droughts or rainfall intensity.\n"
        "- Only one dataset (NASA MERRA-2) is used here. A significant trend does not identify its cause.\n"
        "- Seasons follow meteorological definitions (DJF, MAM, JJA, SON)."
    )
with st.expander("📚 Data, methods and AI-assistance disclosure"):
    st.markdown(
        "**NASA data:** MERRA-2 (GMAO, NASA GES DISC): `M2TMNXSLV` (T2M, 2 m air temperature) and `M2TMNXFLX` "
        "(PRECTOTCORR, bias-corrected precipitation), monthly, 1980–2025, Turkey box 35.5–42.5°N, 25–45°E.\n\n"
        "**Method:** cosine-latitude weighted area means over grid cells inside each region polygon; seasonal means "
        "(temperature, day-weighted) or totals (precipitation); OLS with HAC standard errors, Mann-Kendall, Theil-Sen; "
        "Benjamini-Hochberg for regional tests.\n\n"
        "**Timeline:** the MERRA-2 monthly files were downloaded before the hackathon as general preparation. "
        "The seasonal question, the analysis code and this app were written during the hackathon.\n\n"
        "**AI assistance:** AI assistants (ChatGPT, Claude) helped write and debug the code and draft the text. "
        "The question, data choice, methods and interpretation were decided and checked by the author. "
        "No AI-generated images are used; no NASA logos are used."
    )
st.caption("Data: NASA GMAO MERRA-2 via NASA GES DISC. Code: see the project repository.")
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
VARS = {"Temperature": dict(key="sicaklik", unit="°C", ylabel="Seasonal mean 2 m air temperature (°C)",
                            slope="°C per decade", cmap="YlOrRd", stripes="RdBu_r"),
        "Precipitation": dict(key="yagis", unit="mm", ylabel="Seasonal total precipitation (mm)",
                              slope="mm per decade", cmap="BrBG", stripes="BrBG")}
EARLY = (1980, 1999)
LATE = (2006, 2025)

st.set_page_config(page_title="TerraSeasons", page_icon="🌍", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
                                 padding: 14px 16px;}
    h1 {letter-spacing: -0.5px;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def trend_row(vkey, area, season):
    r = trends[(trends.degisken == vkey) & (trends.area == area) & (trends.season == season)]
    return r.iloc[0] if len(r) else None


def get_series(vkey, area, season=None):
    d = series[(series.degisken == vkey) & (series.area == area)]
    if season is not None:
        d = d[d.season == season]
    return d.sort_values("yil")


def interpret(r, vkey, season):
    """Plain-language sentence for one trend row (all numbers come from the data)."""
    tr, lo, hi = float(r["hac_egim"]), float(r["hac_ga_alt"]), float(r["hac_ga_ust"])
    n = int(r["yil_sayisi"])
    span = tr * (n - 1) / 10.0
    sig = bool(r["anlamli"])
    if vkey == "sicaklik":
        word = "warming" if tr > 0 else "cooling"
        if sig:
            return (f"**{season}:** {word} of {tr:+.2f} °C per decade (95% CI {lo:+.2f} to {hi:+.2f}), roughly "
                    f"{span:+.1f} °C over {n} years. The trend is statistically significant.")
        return (f"**{season}:** a {tr:+.2f} °C per decade tendency (95% CI {lo:+.2f} to {hi:+.2f}), but the interval "
                f"includes zero, so it cannot be told apart from natural variability.")
    pct = float(r["yuzde_on_yil"])
    word = "wetter" if tr > 0 else "drier"
    if sig:
        return (f"**{season}:** getting {word} by {tr:+.1f} mm per decade ({pct:+.1f}% of the seasonal average "
                f"per decade, 95% CI {lo:+.1f} to {hi:+.1f} mm). The trend is statistically significant.")
    return (f"**{season}:** {tr:+.1f} mm per decade ({pct:+.1f}% per decade), but the 95% CI ({lo:+.1f} to {hi:+.1f} mm) "
            f"includes zero, so there is no clear trend; year-to-year variability dominates.")


def sig_text(p):
    return "statistically significant" if p < 0.05 else "not statistically significant"


# ------------------------------------------------------------------ header + sidebar
st.title("🌍 TerraSeasons")
st.caption("Which season is changing fastest in Turkey? · NASA MERRA-2 reanalysis · 1980–2025")

with st.sidebar:
    st.header("Controls")
    var_label = st.radio("Variable", list(VARS))
    areas = ["Turkey"] + sorted(a for a in trends["area"].unique() if a != "Turkey")
    area = st.selectbox("Area", areas)
    st.caption("These choices apply to the *Explore* and *Then vs now* tabs.")
    st.divider()
    st.markdown("**Data:** NASA MERRA-2 (GES DISC)\n\n**Period:** 1980–2025\n\n**Resolution:** about 55 km")
V = VARS[var_label]
vkey, unit, slope_unit = V["key"], V["unit"], V["slope"]

tab_over, tab_exp, tab_reg, tab_cmp, tab_then, tab_rec, tab_about = st.tabs(
    ["Overview", "Explore", "Regions", "Compare", "Then vs now", "Records", "About"])

# ------------------------------------------------------------------ Overview
with tab_over:
    st.markdown(
        "Annual averages hide a lot. **TerraSeasons** splits Turkey's climate record into winter, spring, "
        "summer and autumn and shows how fast **temperature** and **precipitation** are changing in each "
        "season, for Turkey and its seven geographic regions, with confidence intervals and significance tests."
    )
    tt = trends[(trends.degisken == "sicaklik") & (trends.area == "Turkey")].set_index("season").reindex(SEASONS)
    tp = trends[(trends.degisken == "yagis") & (trends.area == "Turkey")].set_index("season").reindex(SEASONS)
    fast, slow = tt["hac_egim"].idxmax(), tt["hac_egim"].idxmin()
    sig_p = [s for s in SEASONS if tp.loc[s, "hac_p"] < 0.05]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fastest-warming season", fast, f"{tt.loc[fast, 'hac_egim']:+.2f} °C / decade")
    m2.metric("Slowest-warming season", slow, f"{tt.loc[slow, 'hac_egim']:+.2f} °C / decade")
    m3.metric("Warming seasons (significant)", f"{int((tt['hac_p'] < 0.05).sum())} of 4")
    m4.metric("Seasons with a rain trend", f"{len(sig_p)} of 4", ", ".join(sig_p) if sig_p else "none")

    st.subheader("Key findings")
    reg_t = trends[(trends.degisken == "sicaklik") & (trends.area != "Turkey")]
    top = reg_t.loc[reg_t["hac_egim"].idxmax()]
    n_sig = int(reg_t["anlamli"].sum())
    bullets = [
        f"**{fast}** is warming fastest in Turkey ({tt.loc[fast, 'hac_egim']:+.2f} °C per decade, 95% CI "
        f"{tt.loc[fast, 'hac_ga_alt']:+.2f} to {tt.loc[fast, 'hac_ga_ust']:+.2f}); **{slow}** is warming most "
        f"slowly ({tt.loc[slow, 'hac_egim']:+.2f} °C per decade, {sig_text(tt.loc[slow, 'hac_p'])}).",
        f"The single fastest region-season warming is **{top['area']} in {top['season'].lower()}** "
        f"({top['hac_egim']:+.2f} °C per decade).",
        f"{n_sig} of 28 regional temperature trends (7 regions × 4 seasons) are significant after multiple-test correction.",
    ]
    if sig_p:
        s0 = sig_p[0]
        bullets.append(f"National precipitation has a significant trend only in **{', '.join(sig_p).lower()}** "
                       f"({tp.loc[s0, 'hac_egim']:+.1f} mm per decade in {s0.lower()}); other seasons are not distinguishable from no change.")
    else:
        bullets.append("National precipitation trends are not statistically significant in any season.")
    for b in bullets:
        st.markdown("- " + b)

    st.subheader("What this means")
    n_warm = int((tt["hac_p"] < 0.05).sum())
    spread = float(tt.loc[fast, "hac_egim"] - tt.loc[slow, "hac_egim"])
    meaning = [
        f"**Warming is not confined to one season.** {n_warm} of 4 seasons show a statistically significant "
        f"national warming trend, so the annual warming seen in yearly averages is built from the whole year.",
        f"**The pace differs by season.** {fast} warms about {spread:.2f} °C per decade faster than {slow}. "
        "Different seasons affect different things (summer heat for health, energy demand and water use; winter "
        "temperature for snow, soil moisture and the start of the growing season).",
        "**Rainfall is the harder signal.** Precipitation varies strongly from year to year, so most seasonal rain "
        "trends cannot be separated from natural variability in a 46-year record. That does not mean nothing is "
        "changing; it means this dataset cannot confirm it.",
        "**Caution.** These are reanalysis results from one dataset. They describe what changed, not why, and they "
        "should be checked against station observations before local decisions are made.",
    ]
    for m in meaning:
        st.markdown("- " + m)

    st.subheader("Turkey at a glance")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (lab, v, tdf) in zip(axes, (("Temperature", VARS["Temperature"], tt), ("Precipitation", VARS["Precipitation"], tp))):
        vals = tdf["hac_egim"].to_numpy(float)
        err = [vals - tdf["hac_ga_alt"].to_numpy(float), tdf["hac_ga_ust"].to_numpy(float) - vals]
        ax.bar(range(4), vals, yerr=err, capsize=5, alpha=0.85)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(SEASONS)
        ax.set_title(f"{lab}: trend per decade", fontsize=11)
        ax.set_ylabel(v["slope"])
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Bars show the trend per decade with 95% confidence intervals. An interval that crosses zero "
               "means the data cannot confidently tell the trend from no change.")
    allt = trends[["degisken", "area", "season", "ortalama", "hac_egim", "hac_ga_alt", "hac_ga_ust", "hac_p",
                   "BH_p", "anlamli", "kendall_p", "sen_egim"]].rename(
        columns={"degisken": "variable", "ortalama": "mean", "hac_egim": "trend_per_decade",
                 "hac_ga_alt": "ci_low", "hac_ga_ust": "ci_high", "hac_p": "hac_p",
                 "BH_p": "bh_adjusted_p", "anlamli": "significant", "kendall_p": "mann_kendall_p",
                 "sen_egim": "theil_sen_per_decade"})
    allt["variable"] = allt["variable"].map({"sicaklik": "temperature_C", "yagis": "precipitation_mm"})
    report = "# TerraSeasons summary (NASA MERRA-2, 1980-2025)\n\n" + "\n".join("- " + b.replace("**", "") for b in bullets)
    d1, d2 = st.columns(2)
    d1.download_button("Download all trend results (CSV)", allt.round(4).to_csv(index=False).encode("utf-8"),
                       file_name="terraseasons_all_trends.csv", mime="text/csv")
    d2.download_button("Download key findings (text)", report.encode("utf-8"),
                       file_name="terraseasons_key_findings.md", mime="text/markdown")

# ------------------------------------------------------------------ Explore
with tab_exp:
    st.subheader(f"{var_label} by season: {area}")
    sel = trends[(trends.degisken == vkey) & (trends.area == area)].set_index("season").reindex(SEASONS)
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = sel["hac_egim"].to_numpy(float)
    err = [vals - sel["hac_ga_alt"].to_numpy(float), sel["hac_ga_ust"].to_numpy(float) - vals]
    ax.bar(range(4), vals, yerr=err, capsize=6, alpha=0.85)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel(slope_unit)
    ax.set_title("Trend per decade with 95% confidence interval (HAC)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    tab = sel.reset_index()[["season", "ortalama", "hac_egim", "hac_ga_alt", "hac_ga_ust", "hac_p", "kendall_p", "sen_egim"]]
    tab.columns = ["Season", f"Mean ({unit})", f"Trend ({slope_unit})", "CI low", "CI high", "HAC p",
                   "Mann-Kendall p", f"Theil-Sen ({slope_unit})"]
    st.dataframe(tab.round(3), hide_index=True)
    st.download_button("Download this table (CSV)", tab.round(4).to_csv(index=False).encode("utf-8"),
                       file_name=f"terraseasons_{vkey}_{area.replace(' ', '_')}.csv", mime="text/csv")

    st.markdown(f"#### What the data say for {area}")
    for ss in SEASONS:
        rr = sel.loc[ss].copy()
        st.markdown("- " + interpret(rr, vkey, ss))
    if area != "Turkey":
        st.caption("For regions, 'significant' uses the Benjamini-Hochberg-adjusted p-value (28 tests).")

    st.subheader("One season in detail")
    c1, c2 = st.columns([1, 2])
    season = c1.radio("Season", SEASONS)
    one = get_series(vkey, area, season)
    y0, y1 = int(one["yil"].min()), int(one["yil"].max())
    yr = c2.slider("Year range", min_value=y0, max_value=y1, value=(y0, y1))
    show_roll = c2.checkbox("Show 10-year moving average", value=True)
    part = one[one["yil"].between(*yr)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(part["yil"], part["deger"], marker="o", markersize=3, linewidth=1.3, label=f"{season} value")
    if show_roll and len(part) >= 10:
        ax.plot(part["yil"], part["deger"].rolling(10, center=True, min_periods=6).mean(),
                linewidth=2.4, alpha=0.9, label="10-year moving average")
    if len(part) > 2:
        slope, icpt = np.polyfit(part["yil"], part["deger"], 1)
        ax.plot(part["yil"], slope * part["yil"] + icpt, linestyle="--", linewidth=2.2, label="Linear trend")
        st.markdown(f"**Descriptive slope for {yr[0]}–{yr[1]}:** {slope * 10:+.2f} {slope_unit} "
                    f"({len(part)} seasons)")
    ax.set_xlabel("Year")
    ax.set_ylabel(V["ylabel"])
    ax.set_title(f"{area}: {season.lower()} {var_label.lower()}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The slope for a custom year range is descriptive only (no significance test); short periods are "
               "dominated by natural year-to-year variability. Winter of year Y = Dec (Y-1), Jan and Feb of Y.")

# ------------------------------------------------------------------ Regions
with tab_reg:
    st.subheader(f"Where is it changing? {var_label} trend, regions × seasons")
    reg = trends[(trends.degisken == vkey) & (trends.area != "Turkey")]
    piv = reg.pivot(index="area", columns="season", values="hac_egim").reindex(columns=SEASONS)
    sig = reg.pivot(index="area", columns="season", values="anlamli").reindex(columns=SEASONS)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    if vkey == "sicaklik":
        im = ax.imshow(piv.values, aspect="auto", cmap=V["cmap"])
    else:
        lim = float(np.nanmax(np.abs(piv.values)))
        im = ax.imshow(piv.values, aspect="auto", cmap=V["cmap"], vmin=-lim, vmax=lim)
    ax.set_xticks(range(4))
    ax.set_xticklabels(SEASONS)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.2f}{'*' if bool(sig.values[i, j]) else ''}",
                    ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label=slope_unit)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("* = significant after Benjamini-Hochberg correction for 28 regional tests (7 regions × 4 seasons).")

    st.markdown("#### Reading the map")
    for ss in SEASONS:
        col = reg[reg.season == ss]
        k = int(col["anlamli"].sum())
        best = col.loc[col["hac_egim"].idxmax()] if vkey == "sicaklik" else col.loc[col["hac_egim"].abs().idxmax()]
        if vkey == "sicaklik":
            st.markdown(f"- **{ss}:** {k} of 7 regions show significant warming; the strongest is "
                        f"**{best['area']}** ({best['hac_egim']:+.2f} °C per decade).")
        elif k:
            st.markdown(f"- **{ss}:** {k} of 7 regions show a significant rain change; the largest change is in "
                        f"**{best['area']}** ({best['hac_egim']:+.1f} mm per decade).")
        else:
            st.markdown(f"- **{ss}:** no region shows a rain change that survives multiple-test correction.")
    st.caption("Regions are compared by trend size and significance only. MERRA-2's grid is coarse, so small regions "
               "rest on few grid cells.")

    st.subheader("Ranking of region-season trends")
    rk = reg[["area", "season", "hac_egim", "hac_ga_alt", "hac_ga_ust", "BH_p", "anlamli"]].copy()
    rk = rk.sort_values("hac_egim", ascending=False)
    rk.columns = ["Region", "Season", f"Trend ({slope_unit})", "CI low", "CI high", "BH-adjusted p", "Significant"]
    st.dataframe(rk.round(3), hide_index=True)

# ------------------------------------------------------------------ Then vs now
with tab_then:
    st.subheader(f"Then vs now: {var_label.lower()}, {area}")
    st.markdown(f"Average of **{EARLY[0]}–{EARLY[1]}** compared with **{LATE[0]}–{LATE[1]}** for each season.")
    d = get_series(vkey, area)
    early = d[d["yil"].between(*EARLY)].groupby("season")["deger"].mean().reindex(SEASONS)
    late = d[d["yil"].between(*LATE)].groupby("season")["deger"].mean().reindex(SEASONS)
    diff = late - early
    fig, ax = plt.subplots(figsize=(8.5, 4))
    w = 0.38
    xx = np.arange(4)
    ax.bar(xx - w / 2, early.values, w, label=f"{EARLY[0]}–{EARLY[1]}", alpha=0.85)
    ax.bar(xx + w / 2, late.values, w, label=f"{LATE[0]}–{LATE[1]}", alpha=0.85)
    ax.set_xticks(xx)
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel(V["ylabel"])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    cols = st.columns(4)
    for c, s in zip(cols, SEASONS):
        c.metric(s, f"{late[s]:.1f} {unit}", f"{diff[s]:+.2f} {unit} vs {EARLY[0]}–{EARLY[1]}")
    big = diff.abs().idxmax()
    st.markdown(f"**Reading it:** the largest change between the two periods is in **{big.lower()}** "
                f"({diff[big]:+.2f} {unit}). " + (
                    "Every season is warmer in the recent period." if (vkey == "sicaklik" and (diff > 0).all()) else
                    "The direction of change differs between seasons." if vkey == "yagis" else ""))
    st.caption("The difference between two 20-year averages is a simple, intuitive summary; it is not a significance test.")

    st.subheader(f"Seasonal stripes: {area}")
    fig, axes = plt.subplots(4, 1, figsize=(10, 4.2), sharex=True)
    for ax, s in zip(axes, SEASONS):
        ds = get_series(vkey, area, s)
        a = (ds["deger"] - ds["deger"].mean()).to_numpy(float)
        lim = max(float(np.abs(a).max()), 1e-9)
        ax.imshow(a[None, :], aspect="auto", cmap=V["stripes"], vmin=-lim, vmax=lim,
                  extent=[ds["yil"].min() - 0.5, ds["yil"].max() + 0.5, 0, 1])
        ax.set_yticks([])
        ax.set_ylabel(s, rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Year")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    cap = "Each stripe is one year's season relative to that season's 1980–2025 average (red = warmer than average)."
    if vkey == "yagis":
        cap = "Each stripe is one year's season relative to that season's 1980–2025 average (brown = drier, green = wetter)."
    st.caption(cap + " Each season has its own colour scale.")

# ------------------------------------------------------------------ Compare
with tab_cmp:
    st.subheader(f"Compare two areas: {var_label.lower()} trend per decade")
    cc1, cc2 = st.columns(2)
    a_name = cc1.selectbox("Area A", areas, key="cmp_a")
    b_name = cc2.selectbox("Area B", areas, index=min(1, len(areas) - 1), key="cmp_b")
    ra = trends[(trends.degisken == vkey) & (trends.area == a_name)].set_index("season").reindex(SEASONS)
    rb = trends[(trends.degisken == vkey) & (trends.area == b_name)].set_index("season").reindex(SEASONS)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.38
    xx = np.arange(4)
    for off, r, nm in ((-w / 2, ra, a_name), (w / 2, rb, b_name)):
        v = r["hac_egim"].to_numpy(float)
        ax.bar(xx + off, v, w, yerr=[v - r["hac_ga_alt"].to_numpy(float), r["hac_ga_ust"].to_numpy(float) - v],
               capsize=4, label=nm, alpha=0.85)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel(slope_unit)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    va, vb = ra["hac_egim"].to_numpy(float), rb["hac_egim"].to_numpy(float)
    cmp_tab = pd.DataFrame({"Season": SEASONS, f"A: {a_name}": va, f"B: {b_name}": vb, "Difference (A - B)": va - vb})
    st.markdown(f"Trend per decade ({slope_unit}):")
    st.dataframe(cmp_tab.round(3), hide_index=True)
    st.markdown("#### Reading the comparison")
    for i, ss in enumerate(SEASONS):
        overlap = not (ra["hac_ga_alt"].iloc[i] > rb["hac_ga_ust"].iloc[i] or rb["hac_ga_alt"].iloc[i] > ra["hac_ga_ust"].iloc[i])
        if a_name == b_name:
            break
        who = a_name if va[i] > vb[i] else b_name
        if overlap:
            st.markdown(f"- **{ss}:** the confidence intervals overlap, so the difference between {a_name} and "
                        f"{b_name} is not clear from this analysis (higher estimate: {who}).")
        else:
            st.markdown(f"- **{ss}:** the intervals do not overlap; {who} has the larger trend, which points to a "
                        f"real difference (informal check, not a formal test).")
    st.caption("Error bars are 95% confidence intervals. If two intervals overlap strongly, the difference "
               "between the areas is not clear from this analysis; no formal test of the difference is done.")

# ------------------------------------------------------------------ Records
with tab_rec:
    st.subheader(f"Record seasons: {area}")
    if vkey == "sicaklik":
        hi_lab, lo_lab = "Warmest", "Coldest"
    else:
        hi_lab, lo_lab = "Wettest", "Driest"
    season_r = st.radio("Season", SEASONS, key="rec_season", horizontal=True)
    rs = get_series(vkey, area, season_r)
    top_hi = rs.nlargest(5, "deger")[["yil", "deger"]].rename(columns={"yil": "Year", "deger": unit})
    top_lo = rs.nsmallest(5, "deger")[["yil", "deger"]].rename(columns={"yil": "Year", "deger": unit})
    r1, r2 = st.columns(2)
    r1.markdown(f"**{hi_lab} 5 {season_r.lower()}s**")
    r1.dataframe(top_hi.round(2), hide_index=True)
    r2.markdown(f"**{lo_lab} 5 {season_r.lower()}s**")
    r2.dataframe(top_lo.round(2), hide_index=True)
    ten_hi = rs.nlargest(10, "deger")
    recent = int((ten_hi["yil"] >= 2006).sum())
    st.markdown(f"Of the 10 {hi_lab.lower()} {season_r.lower()}s in the record, **{recent}** happened in 2006–2025 "
                f"(that period holds 20 of the {len(rs)} seasons).")
    st.caption("Records come from a model-based reanalysis and a 46-year window; they describe this dataset, not "
               "all-time extremes.")

# ------------------------------------------------------------------ About
with tab_about:
    st.subheader("How to read these numbers")
    st.markdown(
        "- **Trend per decade** is the average change every 10 years, estimated by linear regression.\n"
        "- **95% confidence interval (CI):** if it does not include zero, the trend is statistically significant.\n"
        "- **HAC (Newey-West)** standard errors account for neighbouring years being similar (autocorrelation).\n"
        "- **Mann-Kendall** and **Theil-Sen** are rank-based checks that do not assume a normal distribution.\n"
        "- **Benjamini-Hochberg** correction controls false positives when many regional tests are run together.\n"
        "- Precipitation varies a lot from year to year, so its trends are much harder to confirm than temperature trends."
    )
    st.subheader("Limitations")
    st.markdown(
        "- MERRA-2 is a **reanalysis** (model + observations), not station data. Its grid is coarse (about 55 km), "
        "so small regions contain few grid cells.\n"
        "- The record is 46 years; natural multi-decade variability can influence a trend.\n"
        "- Only seasonal averages and totals are analysed, not extremes, droughts or rainfall intensity.\n"
        "- One dataset (NASA MERRA-2) is used. A significant trend does not identify its cause.\n"
        "- Seasons follow meteorological definitions (DJF, MAM, JJA, SON)."
    )
    st.subheader("Data and methods")
    st.markdown(
        "**NASA data:** MERRA-2 (GMAO, NASA GES DISC): `M2TMNXSLV` (T2M, 2 m air temperature) and `M2TMNXFLX` "
        "(PRECTOTCORR, bias-corrected precipitation), monthly, 1980–2025, Turkey box 35.5–42.5°N, 25–45°E.\n\n"
        "**Method:** cosine-latitude weighted area means over grid cells inside each region polygon; seasonal means "
        "(temperature, day-weighted) or totals (precipitation); OLS with HAC standard errors, Mann-Kendall, Theil-Sen; "
        "Benjamini-Hochberg for regional tests."
    )
    st.subheader("Timeline and AI-assistance disclosure")
    st.markdown(
        "The MERRA-2 monthly files were downloaded before the hackathon as general preparation. "
        "The seasonal question, the analysis code and this app were written during the hackathon.\n\n"
        "AI assistants (ChatGPT, Claude) helped write and debug the code and draft the text. "
        "The question, data choice, methods and interpretation were decided and checked by the author. "
        "No AI-generated images are used; no NASA logos are used."
    )
    st.caption("Data: NASA GMAO MERRA-2 via NASA GES DISC. Code: see the project repository.")
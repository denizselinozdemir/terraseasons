"""TerraSeasons: NASA MERRA-2 ile Türkiye ve 7 bölge için MEVSİMSEL sıcaklık ve yağış eğilimleri.

Girdiler (data/ klasöründe):
    merra2_t2m_turkiye.npz            (aylık T2M, Kelvin)
    merra2_prectotcorr_turkiye.npz    (aylık PRECTOTCORR, kg m-2 s-1)
    turkiye-regions.geojson           (7 bölge poligonu, 'name' sütunu)
Çıktılar (data/ klasörüne): mevsim_serileri.csv, mevsim_egilimleri.csv

Mevsimler: DJF (kış), MAM (ilkbahar), JJA (yaz), SON (sonbahar). Kış, Ocak-Şubat'ın yılına yazılır
(Aralık bir önceki yıldandır), bu yüzden kış 1981'den başlar.
Çalıştır:  python mevsim_analiz.py
Gerekli: numpy pandas scipy statsmodels geopandas shapely
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
MEVSIM_AY = {12: "Kış", 1: "Kış", 2: "Kış", 3: "İlkbahar", 4: "İlkbahar", 5: "İlkbahar",
             6: "Yaz", 7: "Yaz", 8: "Yaz", 9: "Sonbahar", 10: "Sonbahar", 11: "Sonbahar"}
MEVSIM_SIRA = ["Kış", "İlkbahar", "Yaz", "Sonbahar"]
START, END = 1980, 2025


def npz_yukle(ad, anahtar):
    for p in (DATA / ad, Path.home() / ad):
        if p.exists():
            with open(p, "rb") as f:
                z = np.load(f)
                return (np.asarray(z[anahtar], float), pd.to_datetime(z["time"]),
                        np.asarray(z["lat"], float), np.asarray(z["lon"], float))
    raise SystemExit(f"{ad} bulunamadı. data klasörüne koy.")


def maskeler(lat, lon):
    import geopandas as gpd
    import shapely
    gdf = gpd.read_file(DATA / "turkiye-regions.geojson").to_crs(4326)
    lon2, lat2 = np.meshgrid(lon, lat)
    sonuc = {}
    for ad, grp in gdf.groupby("name"):
        geom = grp.geometry.union_all() if hasattr(grp.geometry, "union_all") else grp.geometry.unary_union
        sonuc[str(ad)] = shapely.contains_xy(geom, lon2, lat2)
    ulusal = np.zeros_like(next(iter(sonuc.values())))
    for m in sonuc.values():
        ulusal |= m
    return ulusal, sonuc


def alan_ortalamasi(vals, lat, maske):
    """(zaman, enlem, boylam) -> cos(enlem) ağırlıklı alan ortalaması (zaman,)."""
    w = np.cos(np.deg2rad(lat))[:, None] * maske
    ok = np.isfinite(vals)
    wt = ok * w[None]
    return (np.where(ok, vals, 0) * wt).sum(axis=(1, 2)) / wt.sum(axis=(1, 2))


def mevsimlik(aylik, zaman, tur):
    """Aylık seri -> mevsimlik seri. tur='ortalama' (gün ağırlıklı) ya da 'toplam' (mm/gün aylık ort. -> mm)."""
    idx = pd.DatetimeIndex(zaman)
    df = pd.DataFrame({"v": np.asarray(aylik, float), "g": idx.days_in_month.values.astype(float),
                       "ay": idx.month, "yil": idx.year})
    df["mevsim"] = df["ay"].map(MEVSIM_AY)
    df["myil"] = np.where(df["ay"] == 12, df["yil"] + 1, df["yil"])   # Aralık sonraki yılın kışına
    out = []
    for (my, ms), g in df.groupby(["myil", "mevsim"]):
        if len(g) != 3 or g["v"].isna().any():
            continue   # eksik mevsim
        if tur == "ortalama":
            val = float(np.average(g["v"], weights=g["g"]))
        else:
            val = float((g["v"] * g["g"]).sum())
        out.append((my, ms, val))
    s = pd.DataFrame(out, columns=["yil", "mevsim", "deger"])
    return s[(s["yil"] >= START) & (s["yil"] <= END)]


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    r = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(r, 1.0); return out


def egilim(y, x):
    import statsmodels.api as sm
    from scipy import stats
    y = np.asarray(y, float); x = np.asarray(x, float)
    xs = (x - x.mean()) / 10.0
    m = sm.OLS(y, sm.add_constant(xs)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    ci = m.conf_int()
    tau, kp = stats.kendalltau(x, y)
    sen, _, lo, hi = stats.theilslopes(y, x, 0.95)
    return {"ortalama": float(y.mean()), "hac_egim": float(m.params[1]), "hac_ga_alt": float(ci[1, 0]),
            "hac_ga_ust": float(ci[1, 1]), "hac_p": float(m.pvalues[1]), "kendall_tau": float(tau),
            "kendall_p": float(kp), "sen_egim": float(sen) * 10, "yil_sayisi": len(y)}


def main():
    seri_satirlari = []
    for deg, dosya, anahtar, tur in (("sicaklik", "merra2_t2m_turkiye.npz", "T2M", "ortalama"),
                                     ("yagis", "merra2_prectotcorr_turkiye.npz", "PRECTOTCORR", "toplam")):
        vals, zaman, lat, lon = npz_yukle(dosya, anahtar)
        if deg == "sicaklik":
            vals = vals - 273.15                 # K -> °C
        else:
            vals = vals * 86400.0                # kg m-2 s-1 -> mm/gün (mevsimlikte ay uzunluğuyla çarpılır)
        ulusal_m, bolgesel = maskeler(lat, lon)
        alanlar = {"Türkiye": ulusal_m, **bolgesel}
        print(f"[{deg}] alanlar: {list(alanlar)}")
        for ad, m in alanlar.items():
            s = mevsimlik(alan_ortalamasi(vals, lat, m), zaman, tur)
            s.insert(0, "bolge", ad); s.insert(0, "degisken", deg)
            seri_satirlari.append(s)
    seri = pd.concat(seri_satirlari, ignore_index=True)

    # sağlık kontrolleri
    t = seri[(seri.degisken == "sicaklik") & (seri.bolge == "Türkiye")].groupby("mevsim")["deger"].mean()
    y = seri[(seri.degisken == "yagis") & (seri.bolge == "Türkiye")].groupby("mevsim")["deger"].mean()
    print("Ulusal mevsim ort. sıcaklık (°C):", t.round(1).to_dict())
    print("Ulusal mevsim ort. yağış (mm):", y.round(0).to_dict(), "| toplam:", round(y.sum()))
    if not (t["Yaz"] > t["Kış"] + 8) or not (350 < y.sum() < 900):
        raise SystemExit("DURDUM: mevsim değerleri mantıksız (yaz/kış farkı ya da yıllık yağış). Çıktıyı bana yapıştır.")

    satirlar = []
    for (deg, bolge, mevsim), g in seri.groupby(["degisken", "bolge", "mevsim"]):
        g = g.sort_values("yil")
        e = egilim(g["deger"], g["yil"])
        e["yuzde_on_yil"] = 100 * e["hac_egim"] / e["ortalama"] if deg == "yagis" else np.nan
        satirlar.append({"degisken": deg, "bolge": bolge, "mevsim": mevsim, **e})
    eg = pd.DataFrame(satirlar)
    eg["BH_p"] = np.nan
    for deg in eg["degisken"].unique():      # düzeltme: bölgesel 7 x 4 = 28 test, her değişken ayrı
        k = (eg.degisken == deg) & (eg.bolge != "Türkiye")
        eg.loc[k, "BH_p"] = bh(eg.loc[k, "hac_p"].values)
    eg["anlamli"] = np.where(eg["bolge"] == "Türkiye", eg["hac_p"] < 0.05, eg["BH_p"] < 0.05)
    eg["mevsim"] = pd.Categorical(eg["mevsim"], MEVSIM_SIRA, ordered=True)
    eg = eg.sort_values(["degisken", "bolge", "mevsim"])

    DATA.mkdir(exist_ok=True)
    seri.to_csv(DATA / "mevsim_serileri.csv", index=False, encoding="utf-8-sig")
    eg.to_csv(DATA / "mevsim_egilimleri.csv", index=False, encoding="utf-8-sig")
    u = eg[eg.bolge == "Türkiye"]
    print("\n=== TÜRKİYE MEVSİMSEL EĞİLİMLER ===")
    print(u[["degisken", "mevsim", "ortalama", "hac_egim", "hac_ga_alt", "hac_ga_ust", "hac_p"]].round(3).to_string(index=False))
    print("\nBölgesel anlamlı bulgular (BH):")
    print(eg[(eg.bolge != "Türkiye") & eg.anlamli][["degisken", "bolge", "mevsim", "hac_egim", "BH_p"]].round(3).to_string(index=False))
    print("\nKaydedildi: data/mevsim_serileri.csv, data/mevsim_egilimleri.csv")


if __name__ == "__main__":
    main()
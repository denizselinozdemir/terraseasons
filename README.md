# TerraSeasons: which season is changing fastest in Turkey?

NASA Space Apps Challenge 2026, challenge "Be an Earth System Trend Detective".

**Live app:** _add the Streamlit link here_

TerraSeasons splits Turkey's 1980–2025 climate record into winter, spring, summer and autumn and shows
how fast **temperature** and **precipitation** are changing in each season, for Turkey and its seven
geographic regions, with confidence intervals and significance tests.

## Data (NASA)

- NASA MERRA-2 `M2TMNXSLV`, variable `T2M` (2 m air temperature), monthly, 1980–2025 (GMAO, NASA GES DISC).
- NASA MERRA-2 `M2TMNXFLX`, variable `PRECTOTCORR` (bias-corrected precipitation), monthly, 1980–2025.
- Turkey region polygons (GeoJSON, 7 regions).

## Methods

Cosine-latitude weighted area means over grid cells inside each region; meteorological seasons
(DJF, MAM, JJA, SON; winter assigned to the year of January/February); temperature as day-weighted
seasonal mean, precipitation as seasonal total (mm). Trends: OLS with Newey-West (HAC) standard errors,
Mann-Kendall and Theil-Sen checks, Benjamini-Hochberg correction across 28 regional tests per variable.

## Timeline and AI disclosure

The MERRA-2 monthly files were downloaded before the hackathon as general preparation. The seasonal
question, the analysis code (`mevsim_analiz.py`) and the app (`app.py`) were written during the hackathon.
AI assistants (ChatGPT, Claude) helped write and debug code and draft text; the question, data choice,
methods and interpretation were decided and checked by the author.

## Limitations

Reanalysis, not station data; coarse grid (about 55 km); 46-year record; single dataset; seasonal
aggregates only (no extremes); significance does not identify a cause.

## Run

```
pip install -r requirements.txt
python mevsim_analiz.py   # needs numpy pandas scipy statsmodels geopandas shapely and the files in data/
streamlit run app.py
```
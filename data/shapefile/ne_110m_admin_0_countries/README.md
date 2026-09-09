# data/shapefile/ne_110m_admin_0_countries

Natural Earth **110m Admin 0 – Countries**, version **5.1.1**, the basemap behind every
choropleth in this project (`P.WORLD_SHP`).

| | |
|---|---|
| source | <https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip> |
| downloaded | 2026-08-26 |
| licence | public domain (Natural Earth terms of use) — no key, no attribution required |
| rows × cols | 177 × 169 |

Columns the analysis code depends on: `ADMIN` (country name — the join key in
`collab_helpers.plot_country_map`), `ISO_A3` and `ADM0_A3` (the ISO-3 join in
`04_non_academic_01_clinical_trials.ipynb`, which falls back to `ADM0_A3` wherever
`ISO_A3` is the `-99` sentinel).

Everything here except this file is gitignored (`data/**`). To restore the directory on a
new machine, re-download the URL above and unzip it in place.

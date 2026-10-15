---
orphan: true
---

# Motorcycle data provenance

`mcycle.csv` contains all 133 rows and both numeric columns of `MASS::mcycle`,
exported without changing values or row order from MASS 7.3-65:

```r
write.csv(MASS::mcycle, "mcycle.csv", row.names = FALSE)
```

- `times`: time after impact, in milliseconds.
- `accel`: head acceleration, in g.
- Source: Silverman, B. W. (1985). Some aspects of the spline smoothing approach
  to non-parametric curve fitting. *Journal of the Royal Statistical Society,
  Series B*, **47**, 1–52.
- Dataset documentation: [MASS reference manual](https://stat.ethz.ch/CRAN/web/packages/MASS/refman/MASS.html#mcycle).
- Upstream package: [MASS on CRAN](https://CRAN.R-project.org/package=MASS).
- Exported on 26 September 2026 using the installed MASS 7.3-65 dataset.
- SHA-256: `1303710411a874f7fe90e588a67e3fc2f098b903b0ea96c1ab9c719dafd8d068`.

MASS's DESCRIPTION licenses the package under `GPL-2 | GPL-3` and identifies
Brian Ripley and Bill Venables as copyright holders. This bundled dataset copy
is redistributed under GPL version 3; see {download}`MASS-COPYING.txt <MASS-COPYING.txt>`
for the full license. This notice applies to the bundled MASS data, independently
of Liesel's own license. MASS credits the data to Silverman (1985); its manual
also cites Venables and Ripley (2002), *Modern Applied Statistics with S*,
fourth edition, Springer.

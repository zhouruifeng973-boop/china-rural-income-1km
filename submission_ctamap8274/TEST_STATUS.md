# Test status and remaining work

Candidate date: 2026-11-21. This branch is not a fully reproduced final release.

Passed: package hashes and 8,274-label cohort; five-year input grids and counts;
first 4,096 valid training pixels/year feature/weight/code equivalence; a two-epoch
partial-bag test via the new launcher; four parser configurations; cropped
calibration, unbounded changes and GeoTIFF readback; cropped county-uniform
baseline (maximum persisted county-mean difference below 0.001 yuan); cropped
bounded changes; five-year cropped quality/risk generation (26 + 12 GeoTIFF).
Four-variant sensitivity input preflight passed. No full variant was rerun.

Not completed: independent fresh environment, all full stages through the new
launcher, full national reproduction and sensitivity execution. Source feature
and population preprocessing is still required externally. See README Inputs.
Global pip check reported face-recognition requires dlib. These packages are not
used by this workflow; the conflict was not repaired by changing the user's
environment. The observed dependency file is not a complete environment lock.

No test metrics from partial county bags or cropped maps belong in the paper.
The existing scientific results, manuscript and Zenodo ZIP were not modified.

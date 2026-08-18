---
layout: page
title: Motor Thermal Modeling with Deep Learning
description: Recurrent (LSTM, GRU) and feed-forward virtual sensors for online motor temperature estimation on the public Paderborn PMSM dataset. Methodology generalizes my industrial work on rare-earth-free EESM motors to an open benchmark.
img: assets/img/12.jpg
importance: 1
category: automotive
related_publications: true
---

## Overview

A clean, reproducible Python implementation of three deep-learning model families for predicting hot-spot temperatures inside a running motor from its electrical and thermal sensor stream:

- **GRU** — single-layer, last-output regression head
- **LSTM** — single-layer, last-output regression head
- **MLP** — feed-forward baseline with flattened windowed input

All three are trained and evaluated with **leave-one-profile-out cross-validation** on the public Paderborn dataset, using train-only target standardization and early stopping on validation MAE.

## Headline result

Cross-validation on the `pm` (permanent-magnet rotor temperature) target — 10 profiles × 5 random seeds = 50 runs per model.

| Model | RMSE [°C] | MAE [°C] | MaxAE [°C] | Latency [ms/batch] |
|---|---|---|---|---|
| **MLP (baseline)** | **10.14 ± 4.82** | **8.48 ± 4.47** | **24.98 ± 9.69** | **~0.1** |
| LSTM | 10.61 ± 5.55 | 9.02 ± 5.45 | 27.18 ± 10.10 | ~30.7 |
| GRU | 11.45 ± 6.13 | 9.97 ± 6.05 | 26.68 ± 10.28 | ~24.9 |

## Related publications

- **F. Tatari, M. M. Aligoudarzi.** *Deep Learning-Based Rotor Temperature Estimation for Rare-Earth-Free Motors.* NDIA GVSETS 2026.
- **F. Tatari et al.** *Data-driven Thermal Modeling for Electrically Excited Synchronous Motors — A Supervised Machine Learning Approach.* IEEE ITEC 2024.

## Repository

[github.com/FarzanehTatari/motor-thermal-deeplearning](https://github.com/FarzanehTatari/motor-thermal-deeplearning)

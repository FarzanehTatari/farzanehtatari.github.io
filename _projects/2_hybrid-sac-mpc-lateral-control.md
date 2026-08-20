---
layout: page
title: Hybrid SAC + MPC for Vehicle Lateral Control
description: A closed-form blend of a Soft Actor-Critic policy and a constrained linear MPC for connected-and-automated-vehicle lateral control. Combines the tracking quality of deep RL with the actuator envelope and interpretability of MPC.
img: assets/img/hybrid-sac-mpc.png
importance: 1
category: research
related_publications: true
---

## Overview

Deep reinforcement learning gives excellent tracking performance on vehicle lateral control but offers no guarantees about what the controller will command. Model predictive control gives constraint satisfaction and interpretability but is limited by model fidelity. This project blends the two through a **single monotone coefficient λ**, so a practitioner can dial continuously between pure MPC and pure RL.

## Headline result

The hybrid **matches stand-alone SAC's tracking quality** while staying inside the MPC's actuator envelope and preserving a deterministic model-based contribution at every step.

| Controller | RMSE(eᵧ) [m] | max \|eᵧ\| [m] | mean \|δ\| [rad] |
| --- | --- | --- | --- |
| PID | 0.0319 | 0.250 | 2.79 × 10⁻³ |
| Linear MPC | 0.0283 | 0.2375 | 1.55 × 10⁻³ |
| SAC (deep RL) | 0.0167 ± 0.0006 | 0.2375 | 3.02 × 10⁻³ |
| **Hybrid (λ = 12)** | **0.0171 ± 0.0006** | **0.2375** | **2.90 × 10⁻³** |

SAC and Hybrid figures are mean ± std over five training seeds; PID and MPC are deterministic.

## What the hybrid does and does not guarantee

**Guaranteed by construction:** per-step actuator magnitude bound, and a non-zero deterministic model-based contribution to every steering command.

**Not guaranteed:** the input-rate constraint over the blended horizon, recursive feasibility, terminal invariance, or prevention of corner-case divergence when the SAC action saturates in the wrong direction.

## Stack

Python · PyTorch · stable-baselines3 (SAC) · OSQP (QP solver for the MPC) · gymnasium

## Paper

[arXiv:2608.17258](https://arxiv.org/abs/2608.17258)

## Repository

[github.com/FarzanehTatari/hybrid-sac-mpc-lateral-control](https://github.com/FarzanehTatari/hybrid-sac-mpc-lateral-control)

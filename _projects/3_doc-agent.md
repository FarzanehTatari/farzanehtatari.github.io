---
layout: page
title: doc-agent — AI Documentation for Simulink Control Models
description: An LLM agent that reads Simulink models and writes their documentation. Extracts model structure through MATLAB, grounds generation in tool calls against the real model, and exports to HTML, DOCX, and PDF.
img: assets/img/doc-agent.png
importance: 3
category: tools
---

## The problem

Control software documentation is written by hand, goes stale the moment the model changes, and is the first thing cut when a program runs late. The information is already in the model — subsystem hierarchy, signal types and ranges, calibration defaults, Stateflow logic — but getting it into a reviewable document is manual work nobody wants to do twice.

## Approach

doc-agent treats documentation as a **grounded generation** problem rather than a summarisation one. The model is extracted into a canonical JSON schema first, and the language model is then given *tools* to query that structure rather than the whole thing as context.

```
.slx / .sldd  →  MATLAB extraction  →  canonical JSON  →  agent loop  →  Markdown  →  HTML / DOCX / PDF
```

The agent has ten tools — `list_subsystems`, `get_subsystem`, `trace_signal`, `lookup_calibration`, `list_stateflow_charts`, `search_rag`, and others — and iterates: it calls a tool, receives the real answer from the model, and continues until it can write a complete section. A calibration name appears in the output only because `lookup_calibration` returned it, which is what keeps the tool from inventing plausible-sounding signal names.

## What it produces

Three deliverable kinds, generated per subsystem or across an entire model bottom-up:

| Kind | Output |
| --- | --- |
| `autodoc` | Narrative description — purpose, block-by-block behaviour in execution order, referenced calibrations with units and ranges |
| `sysreq` | Numbered system requirements as shall-statements, traceable to specific blocks and signals |
| `unitreq` | Unit-level requirements targeting individual blocks with explicit input/output relations |

Each includes an *Engineering Observations* section flagging what the model noticed but couldn't state as a requirement — unconnected ports, suspicious calibration defaults, out-of-range placeholder values.

## Retrieval

Project-specific context — naming conventions, domain rules, prior specs — is indexed into a local vector store and retrieved per generation. A `facts.md` file holds authoritative project rules in priority order, so constraints like *"signal names use camelCase with a unit suffix"* are enforced rather than guessed.

## Interfaces

A Typer CLI for scripted and batch use, and a Streamlit web UI covering upload → extract → generate → export for people who would rather not touch a terminal. Deployment scaffolding for Databricks Apps is included, so the tool can run inside an organisation's own workspace with models and secrets never leaving their tenant.

## Engineering

21 test files, 171 tests, no API key required to run the suite. Optional dependencies degrade gracefully — PDF export detects a missing weasyprint and disables itself with an explanation rather than failing. MIT licensed.

## Stack

Python · Anthropic API · MATLAB/Simulink · ChromaDB · Typer · Streamlit · pytest

## Repository

[github.com/FarzanehTatari/doc-agent](https://github.com/FarzanehTatari/doc-agent)

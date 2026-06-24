# OncoPulse — Visual Redesign Phase Implementation Plan

This implementation plan covers the visual overhaul of OncoPulse to align with the requested **Linear + Vercel + clinical instrument** aesthetic: a dark near-black theme (`#0a0a0b`), cold clinical teal accent (`#00b8a0`), Satoshi typography, zero emojis, and high information-density status readouts.

The data pipelines, modeling, and `predict_tab` have already been successfully migrated. This plan details the modifications required for the remaining tabs: **EXPLAIN**, **TARGETS**, **NETWORK**, and **METHOD** (formerly ABOUT).

---

## User Review Required

> [!IMPORTANT]
> **Complete Emoji Elimination**: All emojis in tab names, section headers, card labels, and tooltips will be removed to maintain maximum scientific and clinical credibility.
>
> **Monochrome / Dual-Color Coding**: We will replace blue colors (`#3B82F6`, `#38BDF8`, `#4299E1`) globally with:
> - Cold clinical teal (`#00b8a0`) for interactive highlights, target genes, agonists/activators, and accents.
> - Soft green (`#22c55e`) for risk-reducing/favorable genes and low-risk tiers.
> - Soft red (`#ef4444` / `#f56565`) for risk-increasing/unfavorable genes and high-risk tiers.

---

## Proposed Changes

### 1. Tab Modules Visual Overhaul

#### [MODIFY] [explain_tab.py](file:///Users/praneeth/Desktop/Antigravity%20Projects/QBIHack/oncopulse/src/ui/explain_tab.py)
* **Remove Emojis**: Strip `🔍`, `📖`, `📊`, `📋` from all headers.
* **Section Labels**: Replace standard markdown `#` or `##` subheaders with:
  ```python
  st.markdown('<div class="section-label">AI CLINICAL TRIAGE NARRATIVE</div><div class="section-divider"></div>', unsafe_allow_html=True)
  ```
* **Narrative Card**: Restyle the container to a premium dark status card:
  ```html
  <div class="onco-card" style="border-left: 4px solid var(--primary-color); background-color: #18181c; border-top-left-radius: 0; border-bottom-left-radius: 0;">
  ```
* **Plotly Driver Chart**:
  * Replace the blue color `#3B82F6` (risk-reducing contribution) with soft green `#22c55e`.
  * Update axes fonts and layout tick colors to use `#e8e8ea` (text) and `#6b6b72` (muted text) instead of `#94A3B8`.
* **Breakdown Table**:
  * Update CSS classes and cell styling.
  * Map "DECREASED RISK" and "FAVORABLE" labels to green `#22c55e` instead of blue `#3B82F6`.

#### [MODIFY] [targets_tab.py](file:///Users/praneeth/Desktop/Antigravity%20Projects/QBIHack/oncopulse/src/ui/targets_tab.py)
* **Remove Emojis**: Strip `💊`, `⚠️`, `📋`, `🔍` from headers.
* **Section Labels**: Restyle headers using `section-label` and `section-divider`.
* **Scientific Disclaimer Block**: Restyle to fit the dark surface palette:
  ```html
  <div style="background-color: #111113; border: 1px solid rgba(255,255,255,0.07); border-left: 4px solid #ef4444; padding: 15px; border-radius: 4px; margin-bottom: 25px;">
  ```
* **Ranked Table**:
  * Use clinical teal `#00b8a0` for gene labels instead of blue `#38BDF8`.
  * Map agonist/activator modulations to teal `#00b8a0` instead of blue `#60A5FA`.
* **Target Detail Panel**:
  * Update card styling to use `#18181c` dark surface.
  * Use teal `#00b8a0` for headings instead of blue `#38BDF8`.
  * Change "decreased risk" text color from blue `#3B82F6` to green `#22c55e`.

#### [MODIFY] [network_tab.py](file:///Users/praneeth/Desktop/Antigravity%20Projects/QBIHack/oncopulse/src/ui/network_tab.py)
* **Remove Emojis**: Strip `🕸️`, `📖` from headings.
* **Section Labels**: Implement `section-label` and `section-divider`.
* **Visual Code Legend**:
  * Restyle the indicator dots to align with the new palette.
  * Replace blue node indicators (underexpressed/risk-reducing) with soft green `#22c55e`.
  * Remove generic slate background tags.

#### [MODIFY] [about_tab.py](file:///Users/praneeth/Desktop/Antigravity%20Projects/QBIHack/oncopulse/src/ui/about_tab.py)
* **Remove Emojis**: Strip `📖`, `📊`, `🧬` from headers.
* **Section Labels**: Implement `section-label` and `section-divider`.
* **Type Safety**: Import `Any` from `typing` to resolve potential syntax issues.
* **Metrics Cards**:
  * Replace sky blue headers (`#38BDF8`) with clinical teal (`#00b8a0`).
  * Style the metrics panel using `onco-card` with discrete left status borders (`border-left: 4px solid #00b8a0`).

---

### 2. Plotly Visual Configurations

#### [MODIFY] [network.py](file:///Users/praneeth/Desktop/Antigravity%20Projects/QBIHack/oncopulse/src/bio/network.py)
* **Node Color Mapping**:
  * Change soft blue `#4299E1` (risk-reducing) to soft green `#22c55e`.
  * Change soft red `#F56565` (risk-increasing) to soft red `#ef4444`.
  * Change default text color to `#e8e8ea` and label fonts to match Satoshi.

---

## Verification Plan

### Automated Tests
Execute the pytest suite:
```bash
PYTHONPATH=. pytest tests/
```

### Manual Verification
1. Open the local Streamlit environment: `http://localhost:8501`.
2. Inspect the **PREDICT**, **EXPLAIN**, **TARGETS**, **NETWORK**, and **METHOD** tabs.
3. Validate that:
   - There are zero emojis visible anywhere.
   - All section labels use uniform tracked uppercase headers and dividers.
   - The electric blue color is completely removed.
   - The layout feels like a clean, high-density medical readout.

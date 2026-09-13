# Session Improvements & Fixes (2026-09-13)

This document details the architectural, UI, pricing, and observability improvements implemented during the 2026-09-13 development session.

---

## 1. Frontend UI Hardening & Layout Containment

### Issue
When an appraisal encountered a refusal or VLM interruption, the dark inspection stage (`.stage-main` / `.stage-foot`) stuck to the viewport and visibly clipped across the white error card (`#refusal` inside `#result`), obscuring the error message and action buttons.

### Root Causes
1. **Screen Mutex Violation**: In `app/web/app.js`, the `error` callback unhid `#intake` (`$('intake').hidden = false`) and unhid `#result` while `#run` was still active. This mounted the 1,700px intake shelf and gallery above `#run`, pushing down content and breaking single-screen focus.
2. **Sticky Stage Overflow**: `.stage-main` inside `#run` was styled with `position: sticky; top: 6rem; max-height: calc(100vh - 7.25rem);` but had no overflow containment. Its flex children (`.frame`, `.frame-intel`, `.strip`, `.stage-foot`) had a combined minimum height exceeding 740px on compact viewports (such as 768px laptop screens), causing `.stage-foot` to poke out of `.stage-main` by ~110px and stick over `#result`.

### Fixes
- **`app/web/styles/run.css`**:
  - Added `overflow: hidden; contain: paint layout;` to `.stage-main`.
  - Added responsive scaling to `.frame` (`flex: 1 1 auto;`) and `.frame-stage` (`max-height: min(460px, calc(100vh - var(--run-head-room) - 18rem)); min-height: 11rem; margin: 0 auto;`).
  - Constrained `.frame-intel` (`min-height: 4.5rem; max-height: clamp(6rem, 25vh, 16rem);`) so the frame, briefing, thumbnail strip, and footer fit within the viewport height across all standard breakpoints.
- **`app/web/styles/result.css`**:
  - Added `scroll-margin-top: var(--s-5);` to `.result` for comfortable top breathing room when scrolled into view.
- **`app/web/app.js`**:
  - Removed `$('intake').hidden = false` from the `error` handler. The intake shelf remains hidden during run/result inspection; clicking `[Start over]` or `[Appraise another truck]` is the single clean reset path.
  - Added smooth auto-scroll to `$('result')` upon refusal/error.
- **`app/web/js/run.js`**:
  - In `onError()`, unhid `$('view-result')` (`View result ↓`) so users examining previous photos can jump directly to error details.

---

## 2. Structured Logging & Observability

### Additions
- **`app/log.py`**:
  - Introduced `get_logger(name)` with formatted, colored console logging.
  - Added `ExecutionContext` tracking with unique identifiers (`run_<timestamp>_<uuid>`), capturing start timestamps, elapsed time, stage transitions, and error logs.
- **Pipeline & Server Integration**:
  - `app/pipeline.py`: Added structured stage markers (`gate`, `evidence`, `reconcile`, `price`) with item counts and execution summaries.
  - `app/server.py`: Added logging on `/api/appraise` requests, parameters, uploads, and stream lifecycles.
  - `app/evidence/stage.py` & `app/vlm/cursor_backend.py`: Added token usage, timing, and error logs.
- **Tests**:
  - Added `tests/test_logging.py` validating run IDs, context management, log formats, and error capture.

---

## 3. Backend Pricing & Valuation Enhancements

### Additions
- **`app/pricing/model.py`**:
  - Expanded brand support with calibrated multipliers for major Turkish commercial truck brands (`MAN`, `DAF`, `IVECO`, `RENAULT`).
  - Added robust vehicle age and mileage clamping to avoid invalid extrapolation.
  - Added asking price comparison metrics: `vs_comparables_pct` and `inside_comparable_band` to ground seller expectations against market ranges.
- **Tests**:
  - Added `tests/test_pricing_enhancements.py` covering multi-brand valuations, edge-case bounds, and price delta classifications.

---

## 4. Vision Model (VLM) Error Diagnostics

### Additions
- **`app/vlm/cursor_backend.py`**:
  - Enhanced error handling when Cursor runs terminate with non-finished status by extracting `result.result` or `result.error`.
  - Surfaced actionable backend errors (e.g. rate limits, credit exhaustion, model availability) rather than generic status strings.
  - Added timeout and token usage accounting in logs.

---

## 5. Verification & Test Coverage

- **Unit Test Suite**: Ran `python -m unittest discover -s tests` — **845 tests passing**.
- **Browser Regression**: Ran `scripts/check_run_screen.py` testing responsive viewports `1366x768`, `1024x600`, `1440x900`, `390x844` — **PASS**.
- **Impeccable Scan**: Ran `impeccable detect --scope layout` on modified styles — **0 findings**.
- **Graphify**: Knowledge graph updated via `graphify update .`.

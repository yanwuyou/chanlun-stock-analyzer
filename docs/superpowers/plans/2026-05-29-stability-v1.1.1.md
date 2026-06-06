# v1.1.1 Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the skill and EXE delivery paths reliable for date-bounded analysis, multi-timeframe defaults, and data-fetch fallback.

**Architecture:** Keep the existing module boundaries. Add date-range handling in `chan_lib.data`, pass `end_date` through `chan_analyzer` and `gui_app`, and verify with standard-library `unittest` tests.

**Tech Stack:** Python standard library `unittest`, pandas, existing Baostock data layer.

---

### Task 1: Date-Bounded Data Fetch

**Files:**
- Modify: `chan_lib/data.py`
- Test: `tests/test_stability_v111.py`

- [ ] Add tests showing `fetch_kline(..., end_date="20200103")` returns no bars after that date and retries transient failures.
- [ ] Implement minimal date normalization, cache coverage checks, retry loop, and date filtering.
- [ ] Run `python -m unittest tests.test_stability_v111`.

### Task 2: Propagate End Date

**Files:**
- Modify: `chan_analyzer.py`
- Modify: `gui_app.py`
- Test: `tests/test_stability_v111.py`

- [ ] Add tests showing `analyze_stock` forwards `end_date` and default multi-timeframe periods include monthly.
- [ ] Add `end_date` parameters to analysis functions and CLI/GUI calls.
- [ ] Run `python -m unittest tests.test_stability_v111`.

### Task 3: Delivery Artifacts

**Files:**
- Modify: `缠论股析.md`
- Create or update: `dist/缠论股析.skill`

- [ ] Update docs to match `--end` and default monthly behavior.
- [ ] Repack `.skill` so bundled scripts match source files.
- [ ] Run compile/import smoke checks.

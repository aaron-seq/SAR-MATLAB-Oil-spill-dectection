# Verification evidence

Checked 2026-11-04 on Windows, Python 3.13.7. This is bounded local verification
and same-system agent review, not human approval, user validation, or a full
security audit. [Current delivery state](product/PROJECT-STATUS.md) records CI.

## Revisions and method

- Original `main`: [`1bb8eb0`](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/commit/1bb8eb0f0d542ce6aab75dc26684e08c01b4a55c), initially clean.
- Existing [PR #12](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/pull/12):
  `fed7a1c33f19b82d9112a4ea162e6095a07cd822`. Reused as an unchanged predecessor;
  contributor commits and attribution remain intact. Its eight successful
  [CI jobs](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/actions/runs/39046702401)
  were inspected; they do not validate later commits.
- New delivery branch: `codex/17-batch-job-reliability`, stacked on that PR.
  No merge, release, deployment, or repository-security-setting change.

Agents reviewed scripts before execution. Tests use synthetic files, disposable
source exports, redirected home/cache directories, and child environments with
credential variables excluded. This is process/environment isolation, not an OS
filesystem sandbox. No deployed systems or third-party targets were tested.

## Claims and gaps

| ID | Expected versus observed | Disposition |
|---|---|---|
| CLM-001 | Clean-clone CLI demo: original main exits 1, missing `sar_oil_spill.data`; the source directory is absent from Git. | Existing #6 / PR #12 restores the package and anchors ignores. Installed demo passes on the stacked branch. |
| CLM-002 | Original test command fails before collection (exit 4); Ruff 0.16 reports six import-order errors. | Predecessor rechecked in a clean venv: 182 passed, two optional-extra skips; lint passes. |
| CLM-003 | Python 3.10 support: main imports Python 3.11's `datetime.UTC`. | PR #12 uses `timezone.utc`; prior exact-PR CI covers 3.10–3.13. Local runtime was 3.13 only. |
| CLM-004 | README passes its mask to `compare_methods`'s `methods` parameter, raising ambiguous-array `ValueError`. | #25 uses `ground_truth=`; execute the full corrected example. |
| CLM-005 | Storm, no-false-alarm, and thickness/concentration wording exceeds this implementation's evidence. | #25 narrows interpretation and explains contrast confidence. Synthetic table is historical measured evidence, not real-scene validation. |
| GAP-001 | Docker starts two processes while batch state is process-local. | Existing #17; one-worker default, real HTTP regression and bounded comparison. Restart loss, FIFO eviction and replicas remain explicit limits. |
| GAP-002 | Native-depth image handling differs across entry points. | Existing #18 remains a characterization spike; no scaling rule changed here. |
| GAP-003 | Real-data accuracy, georeferencing, trained weights and cloud deployment are unverified. | Reuse #13–#16 / #20; [product decisions](product/PRODUCT.md) record prerequisites. |

The baseline and security agents proposed a possible empty-mask JSON failure.
Both observed HTTP 200 with Hausdorff `null` on tested FastAPI versions; the
failure hypothesis was rejected. A number/null schema question remains
unconfirmed and was not represented as a fixed defect.

## Design, threat model and test oracles

Two agents designed independently; a fresh skeptic reviewed the same #17/code
evidence and challenged throughput and durability claims. Selected: extend the
existing production command to one worker. Rejected alternative: shared queue
and storage, requiring atomic transitions, ownership, retention and recovery
without an evidenced distributed workload. This reversible configuration
change needs no new endpoint, dependency, setting, storage format or service.

Initial budget: four implementation/CI files, one missing real-server test seam
(`scripts/verify_batch_runtime.py`, at most 180 lines), at most 40 added CI
lines; documentation in existing guides plus one combined product brief,
this evidence appendix and one current checkpoint. No algorithm changes.
The first container run exposed a readiness connection reset. A justified
fifth file, `tests/test_runtime_check.py`, adds eight startup/deadline/error
regressions; the production code and public surface remain unchanged.

| Boundary / asset | Control or oracle | Coverage and limits |
|---|---|---|
| Anonymous upload → image parser / server memory | Existing byte/header pixel checks; benign and rejection tests from PR #12. | Code review and API tests; no hostile load or deployed parser assessment. |
| HTTP request → worker-local job state | Accepted ID must complete with one successful image; 100 fresh status polls cannot return 404; unknown UUID must still return 404. | Red on two local Uvicorn processes, green on one. Actual shipped Gunicorn/container needs CI. |
| Process lifetime → retained jobs | One process/replica; oldest-first cap of 100. | Limits documented; restart persistence, eviction recovery and distributed ownership excluded. |
| Contributor code → CI credentials | Push/PR workflow reviewed; no privileged PR trigger or new credential use. | No Actions scanner or permissions-enforcement claim. |
| Raster → interpretation | Ground-truth keyword; finite output ranges; synthetic-only accuracy. | No real-image, oil-identity, thickness, or calibrated-probability claim. |

Identity/tenancy are not implemented. Authentication and concurrency proposals
remain #19. Browser accessibility, deployment TLS, recovery, history secrets,
container/IaC scanning and MATLAB execution were not verified. No new
confidential vulnerability was confirmed in this bounded review.

## Commands and results

The primary agent personally installed `.[api,dev]` in a new `.venv`. An initial
`--no-build-isolation` attempt lacked `editables`; the normal isolated pip build
succeeded. Dependencies are unpinned in the existing manifest, so this is a
record of resolved versions, not a reproducible lockfile guarantee.

```text
python -m venv .venv
python -m pip install -e '.[api,dev]'
pytest --cov=src/sar_oil_spill --cov=api --cov-report=term-missing
ruff check .
mypy src/sar_oil_spill api app.py scripts
sar-oil-spill demo --size 256 --output-dir results/demo
pip-audit --progress-spinner off
ruff check src api app.py --select S
```

Primary-agent predecessor results: tests **182 passed, 2 skipped**, 84% combined
coverage; lint and mypy (19 files) pass; installed demo writes both figures.
The optional Streamlit/PyTorch suites skip in this lean environment.
Implementer ran the integrated code with 182 passes/two skips, mypy 20 files,
lint, and a tracked-source export (180 passes/four skips: two Git-only guards
also skip outside a checkout). Current-revision results are in the checkpoint.

Resolved lean environment: NumPy 2.5.4, SciPy 1.18.1, OpenCV 5.0.0.93,
scikit-image 0.26.0, scikit-learn 1.9.1, Pillow 13.0.0, FastAPI 0.162.0,
Starlette 1.10.0, httpx2 2.16.0, Uvicorn 0.54.0, pytest 9.2.0,
Ruff 0.17.3 and mypy 2.4.1. Neither the app nor dl extra was installed locally.

`pip-audit` 2.10.1 initially reported 12 rows (six distinct advisory IDs), all
against bootstrap **pip 25.2**, not application packages. Updating only the
disposable venv's pip to 26.3 produced exit 0 with no known advisories in that
resolved environment. This does not audit every supported resolution/image.
Extra Ruff security rules reported two inherited `assert` uses (Streamlit type
narrowing and a model invariant), not demonstrated input/security controls;
normal configured lint passes. No rule was disabled or suppressed.

Docker client is installed but its Linux-engine pipe is absent. Private
reporting is disabled; no SECURITY.md route is established. Alert inventory:
Dependabot disabled (403), code analysis absent (404), secret scanning disabled
(404). No settings were changed or token scope expanded. Missing scanner and
private-reporting work is [#26](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/26).

Primary-agent logs, resolved versions and fresh-verifier/implementer runtime
logs remain in local TEMP directories named `sar-delivery-20261104`,
`sar-baseline-1bb8eb0` and `sar-17-runtime`. These are local, non-durable evidence;
CI and this safe summary preserve the reproducible commands and limits.

## CI recovery, 2026-11-05

At `6695667`, [run 41879472016](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/actions/runs/41879472016)
passed all seven non-container jobs. The actual shipped Gunicorn command also
passed valid-batch completion and 100 fresh status polls. The separate worker
comparison then failed during startup with a direct `ConnectionResetError`;
the UI step was skipped. This was a harness failure, not a failed batch check.

Readiness now catches `ConnectionError` within its existing 180-second bound.
No request or accepted-job retry was added. Implementer-observed Red: four
focused failures/four passes; Green: eight passes. Full local tests: **190
passed, two app/dl skips**; configured lint and mypy pass, also personally
rerun by the primary agent on delivery revision `2e736c9`. A fresh checker
reviewed the one-line change and the startup/deadline/post-acceptance oracles.
Current-head hosted results and matched image measurements are recorded on
[PR #27](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/pull/27).

Published [run 41971504417](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/actions/runs/41971504417)
passed all eight jobs on `2ac5140`: Python 3.10–3.13, optional extras, base
smoke, dependency audit and Docker. The optional-extra suite reported 223 passes
and one skip. Docker verified the shipped command, a completed batch and 100
fresh status polls with zero missing statuses, the matched worker comparison,
restoration and UI health. The comparison output is also printed to CI logs
for API-readable evidence; CI adds 28 lines in total, within its 40-line budget.
Inspect PR #27 for the newest head after documentation-only updates.

## Review provenance

- `baseline_claims`: immutable-main baseline, API/CLI claims and design A.
- `code_security`: source/PR review, threat boundaries and blind design B.
- `product_research`: primary-source comparison and provisional jobs.
- `skeptic_evidence`: fresh source audit, opposing opening and rebuttal.
- `batch_implementer`: isolated branch, local real-HTTP Red/Green and comparison.
- `docs_product`: documentation/product brief; no claimed test execution.
- `fresh_verifier`: independently reproduced Red/Green, rejected malformed
  results/failures/timeouts, and completed the corrected README examples using
  an installed wheel from a tracked export. No remaining material findings.
- `publication_review`: checked changed docs, local links, source claims,
  issue/Project read-backs and PR text. Corrections applied before publication.
- `ci_recovery_review`: verified the hosted failure, reviewed the readiness
  fix, and ran eight focused tests plus malformed-health/accepted-job failure
  checks. These transport mocks do not substitute for the hosted container run.

Primary agent verified the missing Git files, predecessor diff/CI, clean install,
tests/lint/types/demo, scanner results, design evidence and source references.
Local worker timings vary between runs: do not infer throughput improvement or
regression. CI records a matched two/one-worker batch microbenchmark including
upload, detection and polling; it is not a production capacity gate.

# Project status and continuation

Last verified: 2026-11-05 (Asia/Kolkata, UTC+05:30).

[Product/roadmap/decisions](PRODUCT.md) · [Evidence](../engineering-evidence.md) ·
[Private Project 14](https://github.com/users/aaron-seq/projects/14) ·
[Open milestone 1](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/milestone/1).

## Current state

Delivery [draft PR #27](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/pull/27),
branch `codex/17-batch-job-reliability`, is stacked on existing
[PR #12](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/pull/12),
whose head is `fed7a1c33f19b82d9112a4ea162e6095a07cd822`;
`main` remains `1bb8eb0`. Runtime changes start at `d4684c6`, followed by a
bounded-readiness recovery and its regression tests. Use PR #27's head for the
current code/documentation revision and inspect its checks rather than reusing
PR #12's CI. The [evidence appendix](../engineering-evidence.md#ci-recovery-2026-11-05)
records the first run's startup failure and correction.
The prerequisite's source restoration and existing fixes were reused with
their attribution. Neither PR is merged; no release or deployment was made.

New scope is #17's single-worker default and real-server verification, plus
[#25](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/25)'s
working comparison example and honest scientific wording. Review and current
head CI remain delivery gates. Issues and milestone stay open.

Implementation: four original files plus one focused readiness regression file;
no dependency/API/config/storage/service additions. Helper is 166 lines and
CI adds 28 lines (budgets 180/40). Jobs remain
ephemeral, oldest-first capped at 100; multiple workers/replicas unsupported.
Both blind designs and the skeptic chose this bounded repair over shared state.
Aaron's product acceptance remains pending. Agent reviews are not human approval.

## Validation and gaps

- Primary-agent clean install and installed demo pass. Original delivery code
  passed 182 tests/two app/dl skips (84% combined coverage), lint and mypy over
  20 files. On readiness revision `2e736c9`, the primary agent personally reran
  190 tests/two skips, lint and types successfully. Fresh review also verified
  startup recovery, its deadline and propagation of accepted-job failures.
- Implementer and fresh verifier reproduced two-worker false 404s over actual
  local Uvicorn HTTP, then one-worker completion and 100 fresh polls without 404.
  Windows Uvicorn is not the shipped Linux Gunicorn/container command.
- Bounded one/two-worker local comparisons completed; variable timings support
  no capacity claim. The new container CI runs the shipped command unchanged
  and a separately labeled matched worker comparison.
- Initial hosted run passed seven jobs and the shipped-container batch check;
  a connection reset during comparison-server startup failed the Docker job.
  Only the bounded readiness probe was changed. Current-head results and
  worker measurements are linked on PR #27 and the Project status update.
- Published [run 41971504417](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/actions/runs/41971504417)
  passed all eight jobs at `2ac5140`, including shipped Gunicorn batch polling,
  the matched image comparison and UI health. Optional extras: 223 passes/one
  skip. The following evidence-only change prints existing measurements into
  CI logs; inspect PR #27 for the newest head/checks before integration.
- A fresh installed-wheel walkthrough completed the corrected Python comparison
  and declared console demo, producing both PNGs. Fresh malformed-result,
  image/job-failure and simulated-timeout checks rejected invalid completion.
- Dependency audit passed after repairing disposable venv bootstrap pip;
  Streamlit/PyTorch, MATLAB, live hosts and real imagery were not tested locally.
- Missing private-reporting/scanner baseline is tracked in
  [#26](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/26).
  No security settings were changed. Local Docker is blocked by absent engine.

## Tracking and decisions

Project is private; linked repository issues remain public. Status is manual:
Todo = captured; In Progress = implementation; In review / Retest = reviewable
but unaccepted/unmerged; Done requires demonstrated acceptance and integration.
Default auto-close/status workflows were removed from this new Project before
setting final states. No due dates, assignments, schedules or notifications
were configured. Priority P0–P3 means critical/high/normal/low; Horizon expresses
sequence, not promised delivery. Native labels/milestone supply kind/scope;
Evidence and Decision hold graded research and candidate dispositions.

Saved views: Overview/triage, Execution (`Horizon:Now`), Review/Retest,
Roadmap, Research/spikes. View layouts/filters and field values are read back;
grouping/Insights and automatic transitions are not behavior-tested.

Next spikes reuse #18 (input units/scaling) and #13 (lawful annotated corpus).
#14–#16 await usable real data and a consuming job; #19/#20 need a defined
deployment. Chatbots/alerts are rejected for this scope, not claimed unnecessary
for all users. #22's branch deletion was not authorized or performed.

## Resume here

1. Verify `git status --short`, branch/HEAD, `git worktree list`, both PRs,
   Project and milestone before editing. Do not recreate tracking records.
2. Read #17/#25, [product decisions](PRODUCT.md) and
   [verification limits](../engineering-evidence.md).
3. Inspect current-head CI, especially the production-container batch check,
   worker comparison and optional-extra jobs. Fix introduced failures first.
4. Run `pytest`, `ruff check .`, `mypy src/sar_oil_spill api app.py scripts`
   and the installed demo. Test server: `uvicorn api.main:app --host 127.0.0.1`;
   in another terminal run `python scripts/verify_batch_runtime.py`.
5. After separately authorized integration, land prerequisite #12 first,
   retarget the stacked PR to `main`, and retest the combined revision.
   Do not merge, deploy, close the milestone or claim operational accuracy here.

Local source/agent worktrees and TEMP evidence are retained. Worktree
`sar-17` / branch `codex/17-readiness-check` contains the latest implementer commit;
`codex/17-batch-worker-fix` preserves its earlier commits;
the delivery branch contains their integrated equivalents. No WIP was discarded.

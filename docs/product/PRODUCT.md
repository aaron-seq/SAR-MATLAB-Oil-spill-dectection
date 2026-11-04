# Product brief and decisions

Updated 2026-11-04. **Recommendation pending Aaron's acceptance.** Branch and
draft-PR work is reviewable implementation, not product approval or deployment.
Delivery state and validation: [PROJECT-STATUS.md](PROJECT-STATUS.md).
Tracking: [private Project](https://github.com/users/aaron-seq/projects/14) ·
[initial milestone](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/milestone/1).

## Purpose, users, and evidence

For learners and researchers who need to compare dark-region segmentation,
this project provides a synthetic SAR comparison lab with CLI, Python, API,
and browser entry points. Its bundled generator supplies reference masks;
it has no demonstrated accuracy on operational oil-spill monitoring.

| Segment and job | Evidence and current pain | Good enough / uncertainty |
|---|---|---|
| Maintainer or API integrator: submit a batch and retrieve its result | **Moderate:** detailed defect report [#17](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/17); per-process state breaks multi-worker polling. Single-worker operation is the workaround. | An accepted, retained batch remains pollable to completion in one live process. No observed external adoption or demand volume. |
| Learner or researcher: run a reproducible comparison and interpret its limits | **Weak, provisional:** inferred from the generator, metrics, and README. Broken example and overstated claims tracked in [#25](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/25). | A clean-clone demo and working comparison example with explicit synthetic scope. No interviews or observed user workflow. |
| Contributor evaluating real SAR inputs | **Moderate pain evidence:** detailed loader failures in [#18](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/18); preprocessing differs across entry points. | Native-depth and unit behavior characterized before implementation. Broader demand remains unknown. |

Maintainers support the interfaces; Python clients and HTTP clients consume
their outputs. Operational responders are a possible future audience, not a
validated segment. There is no user research, adoption measurement, or product
owner acceptance in this record. Defect evidence justifies fixing existing
contracts; the provisional researcher segment alone does not justify features.

## Scope and requirements

Keep the synthetic generator, four methods, optional PyTorch, and MATLAB
historical record. Prefer existing entry points and defaults over new services.
No new dependency, storage system, configuration surface, or public endpoint is
needed for this increment. Python 3.10–3.13 is the existing CI matrix; actual
platform coverage and current results belong in the linked delivery evidence.

| ID | Outcome and acceptance | Evidence / measurement |
|---|---|---|
| R1 | A fresh checkout installs and completes `sar-oil-spill demo`; preserve the data-packaging and compatibility repairs in predecessor PR #12. | [#6](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/6), [PR #12](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/pull/12); fresh exported checkout, documented installation, demo exit status and outputs. |
| R2 | An accepted, retained batch in one live server process completes; repeated polls over new HTTP connections return status, never 404. Unknown IDs still return 404. | [#17](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/17); integration check starts the actual shipped command. Compare bounded one/two-worker one-image batch throughput. |
| R3 | The documented Python comparison runs with `ground_truth=`; Streamlit instructions install its extra. Claims distinguish synthetic scores, contrast confidence, and real-data gaps. | [#25](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/25); execute corrected examples and review README, background, and deployment guidance. |

Success means these jobs complete under the stated checks, not issue counts or
unmeasured adoption. Baseline: the reported multi-worker polling defect and
the broken comparison call; current reproduced results are in the checkpoint.
Timing measurements characterize the trade-off, not a universal throughput SLA.
R2 excludes restart recovery, eviction after 100 retained jobs, and replicas.
Header-before-decode upload limits and optional dependency behavior must remain.

## Opportunity map and roadmap

| When | Opportunity → smallest candidate | Decision, gate, and cheapest test |
|---|---|---|
| Now | Finish setup → retain predecessor repairs (R1); trustworthy batch result → one worker (R2); understandable examples → documentation repair (R3) | **Solve differently:** fix existing behavior and claims. Initial milestone covers #2–#8 through predecessor PR #12, plus #17/#25. Exit requires retest, docs walkthrough, and reviewable draft. |
| Next | Preserve real input information → characterize loaders and units | **Spike first**, [#18](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/18): compare the same fixture in uint8, uint16 and float32 across entry points; decide scaling only after units and expected results are explicit. |
| Next | Learn whether real evaluation is feasible → corpus feasibility | **Spike first**, [#13](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/13): verify lawful access, annotation quality, calibration and split feasibility for one small corpus before a downloader or accuracy claim. |
| Later | Separate look-alikes; provide geographic output; train U-Net | **Later**, [#14](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/14), [#15](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/15), [#16](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/16). Reopen when a usable annotated corpus and a specific consumer job exist; test a held-out fixture before widening scope. |
| Later | Operate a public or larger service → deployment hardening | **Later**, [#19](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/19). Reopen for a defined deployment and workload; measure concurrency, memory and failure behavior before adding infrastructure. |
| Rejected | Add a chatbot, automatic alerts, or a new dashboard | No evidenced job or consumer. Existing interfaces serve the comparison workflow; reconsider only with specific evidence and a cheaper-alternative check. |

Sequence is proposed; no dates or capacity commitments are inferred. If lawful
data or annotations are unavailable, stop the real-data expansion and retain
the synthetic lab scope. If units remain ambiguous, do not silently stretch
real radiometry into the current 8-bit pipeline.

## Lightweight tribunal and decision log

**D1 / #17 — single worker.** Blind same-system reviews included
a researcher advocate, evidence skeptic, and two architect perspectives. They
favored the existing one-worker mechanism over shared queue/state infrastructure.
These are agent reviews, not independent human review or owner approval.
The evidence pack is the reported #17 failure, the in-process registry in
[`api/main.py`](../../api/main.py), and shipped [`Dockerfile`](../../Dockerfile).

The strongest objection is potential throughput or fault-isolation loss. Local
comparisons varied; production capacity remains unknown. Correct batch polling is an
existing contract; a distributed queue would add atomic state, ownership,
recovery and operations without demonstrated demand. The smallest reversible
change is one worker plus a real-server regression check and matching docs.
No ADR is needed for this reversible default.

Pre-mortem: later scaling advice restores multiple workers and the 404 defect.
Mitigation: document one process/replica and test new connections in CI. Kill
the increment if an accepted, retained batch returns 404 or fails to finish in
the bounded check. Reopen architecture when measured workload or recovery needs
outgrow one process. Rollback must stop batch acceptance before restoring the
old worker count. Job loss on restart and FIFO eviction remain explicit limits.

**D2 / #25 — repair claims and examples.** A documentation correction solves the
observed mismatch without algorithm changes. Remove unconditional storm and
false-alarm claims; describe confidence as normalized contrast. No evidence
supports oil thickness, concentration, or calibrated oil probabilities here.

## Comparable workflows and open evidence

Primary sources checked 2026-11-04; this is a scope comparison, not a benchmark.

| Reference | Verified workflow | Implication for this project |
|---|---|---|
| [ESA SNAP 9 operator](https://step.esa.int/main/wp-content/help/versions/9.0.0/snap-toolboxes/org.esa.s1tbx.s1tbx.op.feature.extraction.ui/operators/OilSpillDetectionOp.html) | Calibrated, speckle-filtered inputs; adaptive dark-spot detection and size filtering. | Real preprocessing is a prerequisite to evaluate, not an accuracy claim to copy. |
| [pyroSAR 0.39 processing](https://pyrosar.readthedocs.io/en/latest/general/processing.html) | Metadata handling and SNAP geocoding, scaling, and GeoTIFF output. | Consider an adjacent preprocessing tool before implementing a full satellite-product stack. |
| [EMSA CleanSeaNet](https://www.emsa.europa.eu/csn-menu.html) | Trained operators combine SAR with weather, oceanographic and vessel information. | Operational interpretation is broader than this lab's dark-region mask. |

Demand, corpus suitability, operational accuracy, and deployment targets remain
unverified. No external contacts, interviews, telemetry, release, or deployment
are part of this increment. Keep detailed validation and continuation in the
checkpoint rather than treating these recommendations as accepted delivery.

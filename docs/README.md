# Documentation

| Document | For |
|---|---|
| [../README.md](../README.md) | Start here — what it does, how to run it, measured results |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the code is put together and why |
| [CONFIGURATION.md](CONFIGURATION.md) | Every config key, with tuning guidance |
| [MATLAB.md](MATLAB.md) | The original MATLAB code and how the port diverges |
| [sar-background.md](sar-background.md) | Why SAR suits oil spill monitoring |
| [dispatch-prompt.md](dispatch-prompt.md) | Ready-to-paste prompt for a Claude Code session |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Development workflow and house rules |
| [../CLAUDE.md](../CLAUDE.md) | Repository guide for AI coding agents |
| [../deployment/README.md](../deployment/README.md) | Deployment targets and sizing |

## Generated assets

| File | Produced by |
|---|---|
| [images/](images/) | `python scripts/generate_docs_images.py` |
| [benchmark-results.json](benchmark-results.json) | the same script |

Both are regenerated from real runs. If you change an algorithm, re-run the
script and update the results table in the README to match — the numbers there
are measured, and a stale table is worse than none.

## Archive

[code-explanation-technical-implementation.pdf](code-explanation-technical-implementation.pdf)
is the original project write-up, describing the MATLAB implementation. It
predates the Python port and is kept for historical reference; where it
disagrees with the code, the code is correct. [MATLAB.md](MATLAB.md) records the
specific divergences.

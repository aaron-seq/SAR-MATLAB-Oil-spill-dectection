# Deployment

| Target | Config | Notes |
|---|---|---|
| **Docker** | `Dockerfile`, `docker-compose.yml` | Recommended. Full control, works anywhere. |
| **Render** | `deployment/render.yaml` | Free plan sleeps when idle; expect a cold start. |
| **Railway** | `deployment/railway.yml` | Usage-based; no forced sleep. |

## Why there is no Vercel config

Vercel's Python runtime is serverless, with a 250 MB uncompressed bundle limit
and a short execution ceiling. The dependency set here (OpenCV, SciPy,
scikit-image, scikit-learn) exceeds that limit on its own before any
application code is added, and the deployment fails at build time. Earlier
revisions of this repository shipped a `vercel.json` that could not work; it
has been removed rather than left as a trap.

Use Docker, Render or Railway instead, all of which run a long-lived process.

## Sizing

The service is CPU-bound and holds no per-request state beyond the in-memory
batch registry. One 512x512 scene takes roughly 150-500 ms depending on the
method (see the benchmark table in the README), so a single 1-vCPU instance
handles a few requests per second.

Two caveats before scaling to multiple replicas:

- **Batch jobs are in-process.** `/api/v1/batch-status/{id}` only knows about
  jobs created by the worker that received the request, and jobs are lost on
  restart. Put Redis or a real queue behind it before running more than one
  replica.
- **CORS is wide open** (`allow_origins=["*"]`) so the demo works from
  anywhere. Restrict it in `api/main.py` before exposing the service publicly.

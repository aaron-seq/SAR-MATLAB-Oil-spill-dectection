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

Memory scales with image size, because despeckling runs at full resolution
before the scene is resized. Measured through the API: a 4-megapixel upload
adds 131 MB over idle and takes 1.0 s; one at the 25-megapixel limit adds
790 MB and takes 10.3 s. Each worker can run several requests at once, so
size memory for workers x concurrent requests x 0.8 GB, or lower
`MAX_IMAGE_PIXELS` in `api/main.py`.

Two caveats before exposing or scaling the service:

- **Batch jobs are per-process.** `/api/v1/batch-status/{id}` only knows about
  jobs created by the worker process that handled the upload, and jobs are lost
  on restart. This bites inside a single container: the production image runs
  two gunicorn workers, and 56 of 100 status polls returned 404 in a measured
  run. Run one worker if you rely on batches, or put a shared store behind
  them ([#17](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/17)).
- **CORS is wide open** (`allow_origins=["*"]`) so the demo works from
  anywhere. Restrict it in `api/main.py` before exposing the service publicly.

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

## Batch reliability and scaling

The production command runs **one Gunicorn worker** because batch jobs live in
an in-process registry. Keep **one process and one replica** when using
`/api/v1/batch-process` and `/api/v1/batch-status/{id}`. Adding workers or replicas
can route a valid poll to a process that does not know the job, returning 404
([#17](https://github.com/aaron-seq/SAR-MATLAB-Oil-spill-dectection/issues/17)).
Single-image requests still run detection on worker threads.

Jobs are lost on restart. The registry retains at most 100 entries and evicts
the oldest first, including an active entry if the registry fills. This is a
small demonstration service, with no durable queue or recovery guarantee.
Horizontal scaling requires a separate design for shared job state, atomic
updates, ownership and recovery; a shared dictionary alone does not provide it.

Validate the shipped command using repeated status requests over new HTTP
connections until an accepted batch completes, and check that an unknown job
still returns 404. See the [product plan](../docs/product/PRODUCT.md) for the
decision and acceptance gates. Reverting to multiple workers reintroduces the
batch-status defect; stop accepting batches before such a rollback.

## Public exposure

CORS permits all origins (`allow_origins=["*"]`) for the demo. Restrict it in
`api/main.py` for a chosen deployment. There is no authentication, durable job
storage, or application-level concurrency limit; a public service needs a
separately scoped deployment review.

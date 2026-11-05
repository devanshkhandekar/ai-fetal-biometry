# Phase 2 + Phase 3 — Real model integration and production-parity hardening

## Phase 2 goal

Keep the Phase-1 React/FastAPI/PostgreSQL/RustFS/ElasticMQ/Keycloak/report stack unchanged. Stop only the mock inference consumer. Run a lightweight host orchestration worker that consumes the existing `inference-jobs` queue and invokes the already-working unified model Python environment as a subprocess.

The model subprocess receives the input path, expected anatomy, physical spacing, ROI and work directory. It must return real JSON from the existing pipeline. There is no fallback from real mode to mock mode.

## Safe stop commands

Stop only the Phase-1 mock inference worker while preserving the rest of the platform:

```bash
sudo docker compose --profile mock-worker stop inference-worker-mock
```

Confirm it is stopped:

```bash
sudo docker compose --profile mock-worker ps -a
```

To stop the entire Docker platform but KEEP PostgreSQL/RustFS data:

```bash
sudo docker compose --profile mock-worker down
```

Never use `down -v` if you want to keep Phase-1/2 data.

Restart the platform later:

```bash
sudo docker compose --profile mock-worker up -d
sudo docker compose --profile mock-worker stop inference-worker-mock
```

The real host worker is a foreground process. Stop it with `Ctrl+C` before stopping Docker.

## Install this patch

Back up the files replaced by the patch before copying them. Copy the patch contents into the root of the existing `fetal-biometry-local` project, preserving paths. Restart API/report/web after the copy so the new code is loaded:

```bash
sudo docker compose --profile mock-worker up -d --build api report-worker web
```

The local API calls `Base.metadata.create_all()` on startup, so the new local-only `study_aggregates` table is created automatically. Before AWS/staging, create and use a normal Alembic migration rather than relying on `create_all()`.

## Configure Phase 2

```bash
cp scripts/phase2.env.example scripts/phase2.env
nano scripts/phase2.env
```

Set absolute paths for:

- `PIPELINE_PYTHON`: Python from the already-working unified model environment.
- `PIPELINE_BRIDGE_SCRIPT`: `<this full-stack repo>/ml/bridge_cli.py`.
- `MODEL_REPO_PATH`: the existing working fetal-biometry ML repo.
- `HEAD_CHECKPOINT_PATH`: current Residual U-Net checkpoint.
- `HRNET_PYTHON`: current HRNet environment interpreter.
- `REAL_SCAN_MODULE` / `REAL_SCAN_FUNCTION`: strongly preferred if the latest code already exposes one integrated Head/Abdomen/Femur callable.

If `REAL_SCAN_MODULE/FUNCTION` are blank, the supplied adapter can prove the known HEAD path using `src.pipeline_devansh.run_case_devansh`, because that is the callable used by the current Streamlit head app. Abdomen/Femur deliberately fail until the integrated callable is configured; the patch does not invent your current HRNet wrapper signature.

Run preflight:

```bash
./scripts/phase2-preflight.sh
```

It creates `.venv-worker` for queue/DB/S3 orchestration only. PyTorch/FetalCLIP/HRNet remain in their current environments.

## Start the real worker

Make sure the Docker mock worker is stopped:

```bash
sudo docker compose --profile mock-worker stop inference-worker-mock
```

Then run:

```bash
./scripts/run-phase2-real-worker.sh
```

You should see:

```text
Inference worker started. PIPELINE_MODE=subprocess
```

Never run the mock and real inference consumers on the same `inference-jobs` queue at the same time.

## Phase 2 test order

Create NEW studies for real-model tests. Do not reuse Phase-1 mock studies because completed scans are intentionally not re-queued.

1. HEAD only first: upload a known-good image/DICOM already validated in Streamlit, with the same calibration. Run analysis. Compare HC/BPD/OFD/Head Area/GA, quality/fusion, plane/route and overlay against the frozen Streamlit output.
2. Configure/use the integrated multi-anatomy callable. Test one known-good ABDOMEN case. Confirm real AC, routing, quality/fusion and landmark overlay. For early abdomen tests retain the current ROI/calibration limitations; do not fabricate scale.
3. Test one known-good FEMUR case. Confirm real FL, endpoints/overlay, calibration and quality/fusion.
4. Create one fresh study containing all three known-good scans and run all three through the queue.
5. Run `./scripts/phase2-acceptance.sh <study-id>`.

Phase 2 is complete only when the script reports PASS and the real web UI shows real (not mock) values/artifacts for all three scans.

## Phase 3 goal

Phase 3 adds two production-parity pieces without changing clinical formulas:

1. deterministic study-level aggregation using YOUR existing validated HC + AC + FL -> EFW function;
2. saved provenance/checksums for every real inference run.

The platform never chooses or re-implements a Hadlock variant. `REAL_EFW_MODULE/FUNCTION` must point to your existing function.

Configure in `scripts/phase2.env`:

```dotenv
ENABLE_STUDY_AGGREGATION=true
REAL_EFW_MODULE=your.module
REAL_EFW_FUNCTION=your_existing_efw_function
EFW_FORMULA_VERSION=your-existing-method-version
```

Restart only the host worker after changing this file:

```text
Ctrl+C
./scripts/run-phase2-real-worker.sh
```

Create another NEW three-scan study. All three scan-level fusion statuses must be `ACCEPT`. When the last scan completes, the worker automatically selects the only HEAD/ABDOMEN/FEMUR runs, invokes the existing EFW function, persists `study_aggregates`, and exposes `efw_g` through `/v1/studies/{id}/results` and the React Overall Study Measurements card.

If there are multiple scans for an anatomy, aggregation intentionally becomes REVIEW until an explicit run-selection workflow is added. If any scan is REVIEW/REJECT, EFW is not generated.

Run:

```bash
./scripts/phase3-acceptance.sh <study-id>
```

It requires:

- `study_aggregate.status == COMPLETE`;
- real `efw_g > 0`;
- all three scan fusion statuses `ACCEPT`;
- `input_sha256`, `output_sha256`, `runtime_seconds`, and release metadata on every scan result.

After PASS, generate a Doctor PDF again and verify EFW appears in the same deterministic report JSON used to render the PDF.

## Phase 3 GPU-container parity (after host parity only)

Do not rebuild the ML environments before host parity. Once Phase 2 and Phase 3 pass on the host, export the exact existing environments:

```bash
conda env export -n fetal_biometry_unified --no-builds > ml/env-unified.yml
conda env export -n fetalbiometry_hrnet --no-builds > ml/env-hrnet.yml
```

Remove only host-specific `prefix:` lines. Preserve Python/Torch/CUDA/model preprocessing versions. The production inference image should contain two separate environments, `/opt/envs/unified` and `/opt/envs/hrnet`, and start `app.worker_inference` from the unified environment. The existing HRNet bridge should invoke `/opt/envs/hrnet/bin/python` exactly as it invokes the current host HRNet interpreter.

Do not claim container parity until the same frozen HEAD/ABDOMEN/FEMUR cases match the host outputs within the tolerances you already use.

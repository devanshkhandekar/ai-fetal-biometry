Phase-3 corrective application patch

Fixes only application-layer issues:
1. Measurement status inherits final fusion.status (fixes ACCEPT Head measurements shown as REVIEW).
2. /results exposes original_url and clinician-facing de-duplicated warnings while preserving raw warning_codes.
3. React shows Original / AI overlay / Predicted mask when available.
4. Doctor PDF embeds original/overlay/mask through internal object-storage downloads; it does not depend on browser localhost URLs.
5. No model threshold, routing, quality or EFW formula is weakened or changed.

Copy the files over the same relative paths in fetal-biometry-local, then:

  sudo docker compose build api report-worker web
  sudo docker compose up -d api report-worker web

Restart the HOST real inference worker separately so new runs use the status-normalization code path:

  # stop old worker with Ctrl+C
  ./scripts/run-phase2-real-worker.sh

IMPORTANT: existing InferenceRun.result_json rows are immutable historical results and remain as originally stored.
Create a NEW Phase-3 study after applying this patch.

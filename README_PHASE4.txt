Phase 4 helper bundle for the current fetal-biometry-local project.

Purpose: package the already-working fetal_biometry_unified environment with conda-pack,
build a GPU worker image, and run the existing model repository in a container-compatible
runtime copy. This is for LOCAL GPU PARITY first, not AWS deployment.

Expected prerequisites:
- Phase 2 real scan parity already passing on host
- Phase 3 EFW aggregation passing on host
- scripts/phase2.env configured
- Docker base stack running without mock worker
- NVIDIA driver + NVIDIA Container Toolkit

Copy these files into the project root, preserving phase4/, scripts/, and docker-compose.phase4.yml.

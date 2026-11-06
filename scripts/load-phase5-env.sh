#!/usr/bin/env bash

set -a

source .env

if [ -f scripts/phase2.env ]; then
    source scripts/phase2.env
fi

if [ -f scripts/phase5.env ]; then
    source scripts/phase5.env
fi

set +a

export MODEL_REPO_PATH="/media/devanshot/Storage/ubuntu_universe/Devansh_Research/AI_Fetal_Ultrasound_Biometry/mvp_tested_versions/fetal_biometry_devansh_mvp_ellipse_ga_robust_complete/fetal_biometry_devansh_mvp_ellipse_ga_robust"

export PIPELINE_PYTHON="/home/devanshot/miniconda3/envs/fetal_biometry_unified/bin/python"

export HRNET_PYTHON="$PIPELINE_PYTHON"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"

echo "Phase 5 local environment loaded."
echo "MODEL_REPO_PATH=$MODEL_REPO_PATH"
echo "OPENAI_MODEL=${OPENAI_MODEL:-unset}"
echo "ENABLE_OPENAI=${ENABLE_OPENAI:-unset}"

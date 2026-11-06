#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)"
streamlit run app/streamlit_app.py

#!/bin/zsh
set -e
cd "${0:A:h}"
python -m streamlit run app.py --server.fileWatcherType none

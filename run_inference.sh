#!/bin/bash

# Script pentru activarea mediului virtual si rularea inferentei

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

# Verificăm dacă există mediul virtual
if [ ! -d "$VENV_DIR" ]; then
  echo "Eroare: mediul virtual .venv nu există în proiect."
  echo "Creează-l cu: python3 -m venv .venv"
  exit 1
fi

# Activare venv
source "$VENV_DIR/bin/activate"

# Rulare inference
python3 "$PROJECT_DIR/ml/src/inference.py"
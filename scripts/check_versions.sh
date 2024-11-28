#!/bin/bash
# Add to .git/hooks/pre-commit

grep -r "torch==2.0.1" . && {
    echo "Error: Found outdated torch version reference"
    exit 1
} 
#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [covr|all]"
    exit 1
fi

case "$1" in
    covr)
        bash tools/scripts/download_annotation_covr.sh
        ;;
    all)
        bash tools/scripts/download_annotation_covr.sh
        ;;
    *)
        echo "Invalid argument. Usage: $0 [covr|cirr|fiq|all]"
        exit 1
        ;;
esac

#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier

cd /tests
pytest test_verify.py --ctrf=/logs/verifier/ctrf.json -q
status=$?

if [ "$status" -eq 0 ]; then
    printf '1' > /logs/verifier/reward.txt
else
    printf '0' > /logs/verifier/reward.txt
fi

exit 0

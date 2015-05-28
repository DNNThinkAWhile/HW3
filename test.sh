#!/bin/bash -e
cat Holmes_Training_Data/test/ori/*.txt \
    | sed 's/\.//g' \
    | sed 's/,//g' \
    | sed "s/  */ /g" \
    | sed 's/^/\-start\- /g' \
    | sed 's/$/ \-end\-/g' >testing/testing_clean.txt

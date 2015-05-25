#!/bin/bash -e
cat training/*.TXT \
    | LC_ALL=C sed 's/_/ /g' \
    | LC_ALL=C tr '\n' ' ' \
    | LC_ALL=C sed "s/Mr\./Mr /g" \
    | LC_ALL=C sed "s/Mrs\./Mrs /g" \
    | LC_ALL=C sed "s/Ms\./Ms /g" \
    | LC_ALL=C sed 's/\./\'$'\n/g'\
    | LC_ALL=C sed "s/[^a-zA-Z0-9]/ /g" \
    | LC_ALL=C sed "s/^ll//g" \
    | LC_ALL=C sed "s/^[ \t]*//g" \
    | LC_ALL=C sed "s/^[0-9]*//g" \
    | LC_ALL=C sed "s///g" > tmp.txt

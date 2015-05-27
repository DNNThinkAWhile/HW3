#!/bin/bash -e
cat Holmes_Training_Data/testing/testing_data.txt \
                     | LC_ALL=C sed "s/[^a-zA-z0-9_]/ /g" \
                     | LC_ALL=C sed "s/^[0-9]*//g" \
                     | LC_ALL=C sed "s/^[a-z]//g" \
                     | LC_ALL=C sed "s/^[ \t]*//g" \
                     | LC_ALL=C sed "s/\[//g" \
                     | LC_ALL=C sed "s/\]//g" \
                     | LC_ALL=C sed "s/  */ /g" > pure_test.txt

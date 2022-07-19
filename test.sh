#!/bin/bash -e
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

prepare_default() {
  ./build.sh
}

run_default() {
  ./build/fpnge "$@"
}

prepare_fast() {
  ./build.sh -DFPNGE_FIXED_PREDICTOR=4
}

run_fast() {
  ./build/fpnge "$@"
}

prepare_superfast() {
  ./build.sh -DFPNGE_FIXED_PREDICTOR=2
}

run_superfast() {
  ./build/fpnge "$@"
}

FAILURES=0
CASES=0

run_testcase() {
  run_$1 $2 /tmp/fast.png
  compare -metric pae $FILE /tmp/fast.png null: 2> /tmp/pae && status=0 || status=1 
  if [ $status == 0 ]
  then
    echo -e "\033[0;32m$1 $2\033[;m"
  else
    echo -e "\033[1;31m$1 $2\033[;m: $(cat /tmp/pae)"
    FAILURES=$((FAILURES+1))
  fi
  CASES=$((CASES+1))
}


for SETTING in default fast superfast
do
  prepare_$SETTING
  for FILE in testdata/terminal.png $(find jxl_testdata -name '*.png')
  do
    run_testcase $SETTING $FILE
  done
done


echo $CASES cases, $FAILURES failures

if [ $FAILURES > 0 ]
then
  exit 1
fi

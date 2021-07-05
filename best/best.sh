#!/bin/bash

BEST=$(cat ../best/results.txt)
NEW=$(cat ../data/model/results.txt)

echo "\nBest score: ${BEST}"
echo "New score: ${NEW}\n"

if (( $(echo "${NEW} > ${BEST}" |bc -l) ))
then
  echo "NEW RECORD!"
  # move the new best files
  cp -rf ../data/model/results.txt ../best/results.txt
  cp -rf ../data/train/* ../best/train/
  cp -rf ../data/val/* ../best/val/
  cp -rf ../data/model/* ../best/model/
fi
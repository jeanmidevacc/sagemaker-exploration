#!/bin/bash

cd /home/ubuntu/Development/github/sagemaker-exploration/0_poc
for i in {1..100}
do
    echo $i
    python exploration.py
done






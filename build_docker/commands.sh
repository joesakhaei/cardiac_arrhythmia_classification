#! /bin/sh

spark-submit --master local --name app_train --executor-memory 5g --driver-memory 5g /usr/local/train.py

spark-submit --master local --name app_inference --executor-memory 4g --driver-memory 4g /usr/local/inference.py

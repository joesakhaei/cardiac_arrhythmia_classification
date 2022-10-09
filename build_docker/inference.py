import os
import warnings
warnings.simplefilter(action='ignore')

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

import pyspark.sql.functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, VectorIndexer, Imputer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, RandomForestClassificationModel, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


print("\n*** Inferencing Started ***")


## Ingesting whole data for inference testing
colnames = ["col" + str(c) for c in range(1, 280)] + ["target"]
schema = " DOUBLE, ".join(colnames) + " INT"

df0 = spark.read.csv("/usr/local/arrhythmia.data", schema=schema, inferSchema=False, header=False, nullValue="?")



## Loading Preprocessing Pipeline from /tmp and applying on raw data
prep_pipe = PipelineModel.load('/tmp/prep_pipe.model')
df1 = prep_pipe.transform(df0)



## Loading Pre-trained Random Forest Classifier from /tmp
rf_model = RandomForestClassificationModel.load('/tmp/rf.model')

predictions = rf_model.transform(df1)
evaluator = MulticlassClassificationEvaluator(
    labelCol="target", predictionCol="prediction", metricName="accuracy")
print("\nRandom Forest Accuracy on Whole dataset: %.2f" % (evaluator.evaluate(predictions)))
y_true = predictions.select(['target']).toPandas().values
y_pred = predictions.select(['prediction']).toPandas().values
print("Classification Report:\n", classification_report(y_true, y_pred))



## Loading Pre-trained Logistic Regressor from /tmp
lr_model = LogisticRegressionModel.load('/tmp/lr.model')

predictions = lr_model.transform(df1)
evaluator = MulticlassClassificationEvaluator(
    labelCol="target", predictionCol="prediction", metricName="accuracy")
print("\nLogit Regression Accuracy on Whole dataset: %.2f" % (evaluator.evaluate(predictions)))
y_true = predictions.select(['target']).toPandas().values
y_pred = predictions.select(['prediction']).toPandas().values
print("Classification Report:\n", classification_report(y_true, y_pred))
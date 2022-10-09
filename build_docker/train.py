import os
import warnings
warnings.simplefilter(action='ignore')

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

import pyspark.sql.functions as F

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, VectorIndexer, Imputer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

print("\n\n\n*** Pre-processing Started ***")
print("Ingesting Data ... naming columns col1 ~ col279 & target")


## Ingesting CSV

colnames = ["col" + str(c) for c in range(1, 280)] + ["target"]
schema = " DOUBLE, ".join(colnames) + " INT"


df0 = spark.read.csv("/usr/local/arrhythmia.data", schema=schema, inferSchema=False, header=False, nullValue="?")
df0.cache()



## Count of Nulls
null_counts = df0.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df0.columns
]).toPandas().values.ravel()

#print("\nMissing records:")
#for i in range(len(colnames)):
    #if (null_counts[i] > 0):
        #print(colnames[i], " Null Count: ", null_counts[i])



## Count of Distinct Values
print("\nCounting unique values in columns ...")
unique_counts = df0.toPandas().nunique(axis=0).values
monotonous_cols = [("col"+str(i+1)) for i,c in enumerate(unique_counts) if c < 2]
print("Columns to Drop due to lack of any variation:")
print(monotonous_cols)
 
        
##
print("\nImputing missing records by `mean`: Missing records are among linear columns")
toImpute_cols = ['col11', 'col12', 'col13', 'col15'] # col14 dropped due to majority missing
imputer = Imputer(strategy='mean', inputCols=toImpute_cols, outputCols=toImpute_cols)


print("\nAssembling and indexing features as a Spark vector")
input_cols = list(set(colnames) -set(['col14', 'target']) -set(monotonous_cols))
assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
indexer = VectorIndexer(inputCol=assembler.getOutputCol(), outputCol='indexed_features', maxCategories=2)

prep_stages = [imputer, assembler, indexer]
prep_pipe = Pipeline().setStages(prep_stages).fit(df0)


prep_pipe.write().overwrite().save('/tmp/prep_pipe.model')
print("\nPre-processing Pipeline `prep_pipe.model` Successfully Saved")


print("\n\n*** Training started ***")

## Train/Test Split
(trainingData, testData) = prep_pipe.transform(df0).randomSplit([0.85, 0.15], seed=123456)

rf = RandomForestClassifier(labelCol="target", featuresCol='indexed_features')
# pipeline = Pipeline(stages=[rf])

# Define Pram Grid
paramGrid = ParamGridBuilder()\
    .addGrid(rf.numTrees, [20, 30, 40])\
    .addGrid(rf.maxDepth, [5, 10, 15, 20])\
    .addGrid(rf.seed, [12345, 123456])\
    .build()

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy"),
                          numFolds=3)

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(trainingData)

print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])

predictions = cvModel.bestModel.transform(testData)
evaluator = MulticlassClassificationEvaluator(
    labelCol="target", predictionCol="prediction", metricName="accuracy"
)
print("\nRandom Forest Accuracy on Test data is: %.2f" % (evaluator.evaluate(predictions)))
y_true = predictions.select(['target']).toPandas().values
y_pred = predictions.select(['prediction']).toPandas().values
print("Classification Report:\n", classification_report(y_true, y_pred))


cvModel.bestModel.write().overwrite().save('/tmp/rf.model')
print("\nRandom Forest `rf.model` Successfully Saved")




lr = LogisticRegression(labelCol="target", featuresCol='indexed_features')

# Define Pram Grid
paramGrid = ParamGridBuilder()\
    .addGrid(lr.maxIter, [80, 100, 120])\
    .addGrid(lr.regParam, [.05, .1, .15])\
    .addGrid(lr.elasticNetParam, [0, .9])\
    .addGrid(lr.fitIntercept, [True, False])\
    .build()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy"),
                          numFolds=3)

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(trainingData)

print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])

predictions = cvModel.bestModel.transform(testData)
evaluator = MulticlassClassificationEvaluator(
    labelCol="target", predictionCol="prediction", metricName="accuracy"
)
print("\nLogit Regression Accuracy on Test data is: %.2f" % (evaluator.evaluate(predictions)))
y_true = predictions.select(['target']).toPandas().values
y_pred = predictions.select(['prediction']).toPandas().values
print("Classification Report:\n", classification_report(y_true, y_pred))

cvModel.bestModel.write().overwrite().save('/tmp/lr.model')
print("\nLogit Regression `lr.model` Successfully Saved")

from pyspark.sql import SparkSession, streaming
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
# from textblob import TextBlob
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline


def preprocessing(lines):
    words = lines.select(explode(split(lines.value, "t_end")).alias("word"))
    words = words.na.replace('', None)
    words = words.na.drop()
    words = words.withColumn('word', F.regexp_replace('word', r'http\S+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '@\w+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '#', ''))
    words = words.withColumn('word', F.regexp_replace('word', 'RT', ''))
    words = words.withColumn('word', F.regexp_replace('word', ':', ''))
    return words

# text classification


# def polarity_detection(text):
#     return TextBlob(text).sentiment.polarity


# def subjectivity_detection(text):
#     return TextBlob(text).sentiment.subjectivity


# def text_classification(words):
#     # polarity detection
#     polarity_detection_udf = udf(polarity_detection, StringType())
#     words = words.withColumn("polarity", polarity_detection_udf("word"))
#     # subjectivity detection
#     subjectivity_detection_udf = udf(subjectivity_detection, StringType())
#     words = words.withColumn(
#         "subjectivity", subjectivity_detection_udf("word"))
#     return words


if __name__ == "__main__":
    # create Spark session
    spark = SparkSession.builder.appName(
        "TwitterSentimentAnalysis")\
        .master('local[*]') \
        .config("spark.driver.memory", "15g") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.2.3").getOrCreate()

    document_assembler = DocumentAssembler() \
        .setInputCol("word") \
        .setOutputCol("document")

    use = UniversalSentenceEncoder.pretrained('tfhub_use', lang="en") \
        .setInputCols(["document"])\
        .setOutputCol("sentence_embeddings")

    classifier = SentimentDLModel().pretrained('sentimentdl_use_twitter')\
        .setInputCols(["sentence_embeddings"])\
        .setOutputCol("sentiment")

    nlpPipeline = Pipeline(
        stages=[
            document_assembler,
            use,
            classifier
        ])

    empty_df = spark.createDataFrame([['']]).toDF("word")

    pipelineModel = nlpPipeline.fit(empty_df)

    # read the tweet data from socket
    lines = spark.readStream.format("socket").option(
        "host", "0.0.0.0").option("port", 5555).load()
    # Preprocess the data
    words = preprocessing(lines)

    result = pipelineModel.transform(words)

    words = result.select(F.explode(F.arrays_zip('document.result', 'sentiment.result')).alias("cols"))\
        .select(F.expr("cols['0']").alias("document"),
                F.expr("cols['1']").alias("sentiment"))

    words = words.repartition(1)

    query = words.writeStream.queryName("all_tweets")\
        .outputMode("append").format("csv")\
        .option("path", "./parc")\
        .option("checkpointLocation", "./check")\
        .trigger(processingTime='60 seconds').start()
    query.awaitTermination()

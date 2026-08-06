from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, current_timestamp, window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType
import json
import os

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

def create_spark_session():
    spark = SparkSession.builder \
        .appName("ProphecyAI-Processor") \
        .master(os.getenv("SPARK_MASTER_URL", "local[*]")) \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

schema = StructType([
    StructField("id", StringType(), True),
    StructField("area", StringType(), True),
    StructField("property_type", StringType(), True),
    StructField("bedrooms", IntegerType(), True),
    StructField("bathrooms", IntegerType(), True),
    StructField("sqft", IntegerType(), True),
    StructField("floor", IntegerType(), True),
    StructField("rent_price_aed", IntegerType(), True),
    StructField("sale_price_aed", IntegerType(), True),
    StructField("days_on_market", IntegerType(), True),
    StructField("amenities", ArrayType(StringType()), True),
    StructField("listing_date", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("seasonal_factor", FloatType(), True)
])

# feature engineering UDFs
@udf("float")
def price_per_sqft(price, sqft):
    if sqft and sqft > 0:
        return float(price) / float(sqft)
    return 0.0

@udf("integer")
def amenity_score(amenities):
    if not amenities:
        return 0
    weights = {"pool": 3, "gym": 2, "parking": 2, "security": 2, 
               "balcony": 1, "sea_view": 4, "metro": 3, "mall": 2}
    return sum(weights.get(a, 1) for a in amenities)

@udf("float")
def rent_demand_score(days_on_market, seasonal_factor, area):
    # simple heuristic - lower days = higher demand
    # normalized roughly 0-100
    base = max(0, 100 - days_on_market * 1.5)
    base *= seasonal_factor
    # area boost
    area_boost = {"Downtown Dubai": 1.1, "Dubai Marina": 1.08, "Palm Jumeirah": 1.05,
                  "Bluewaters": 1.03, "JLT": 0.98, "Arabian Ranches": 0.95}
    base *= area_boost.get(area, 1.0)
    return round(min(100, base), 2)

def main():
    print("[spark] initializing...")
    spark = create_spark_session()

    # read from kafka
    raw_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_SERVERS) \
        .option("subscribe", "raw-properties") \
        .option("startingOffsets", "latest") \
        .load()

    parsed = raw_df.select(
        from_json(col("value").cast("string"), schema).alias("data"),
        col("timestamp").alias("kafka_timestamp")
    ).select("data.*", "kafka_timestamp")

    # feature engineering
    featured = parsed \
        .withColumn("rent_per_sqft", price_per_sqft(col("rent_price_aed"), col("sqft"))) \
        .withColumn("sale_per_sqft", price_per_sqft(col("sale_price_aed"), col("sqft"))) \
        .withColumn("amenity_score", amenity_score(col("amenities"))) \
        .withColumn("rent_demand_idx", rent_demand_score(col("days_on_market"), col("seasonal_factor"), col("area"))) \
        .withColumn("processed_at", current_timestamp())

    # write to console for debugging (in prod this goes to feature store/db)
    query = featured.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .trigger(processingTime="10 seconds") \
        .start()

    # also write to parquet for batch training
    query2 = featured.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", "/app/data/processed") \
        .option("checkpointLocation", "/tmp/checkpoint_parquet") \
        .trigger(processingTime="60 seconds") \
        .start()

    print("[spark] streaming started, waiting for data...")
    query.awaitTermination()

if __name__ == "__main__":
    main()

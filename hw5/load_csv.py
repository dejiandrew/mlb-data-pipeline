from pyspark.sql import SparkSession

# Initialize Spark Session
spark = (
    SparkSession.builder
    .appName("CSVLoader")
    .master("local[*]")
    .getOrCreate()
)

# Load CSV into Spark DataFrame
df = (
    spark.read
    .option("header", True)  # Read header row
    .option("inferSchema", True)  # Infer data types
    .csv("gl2024.csv")  # Replace with actual CSV path
)

df = df.repartition(4)

# Show Data
df.show(5)
#df.printSchema()

# Stop Spark Session
spark.stop()

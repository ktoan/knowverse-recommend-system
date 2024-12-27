# batch_processor.py

from realtime_service import RATINGS_PATH
from model_training import train_and_save_model

def process_batch(batch_df, batch_id, spark):
    # Append new ratings to the dataset
    # If RATINGS_PATH is a Delta table, you could do:
    # batch_df.write.format("delta").mode("append").save(RATINGS_PATH)
    # For simplicity, using Parquet here:
    batch_df.write.mode("append").parquet(RATINGS_PATH)

    # Retrain the model after appending new data
    train_and_save_model(spark)

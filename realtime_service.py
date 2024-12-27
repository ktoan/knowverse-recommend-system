# app.py
from flask_cors import CORS, cross_origin
import psycopg2
import select
import torch
import numpy as np
from flask import Flask, request, jsonify
import threading
import os
import pickle
import logging
import json
import time
from typing import List, Tuple

from initial_load import load_data_from_db, build_mappings, load_all_users
from model_training import train_neumf_model, NeuMF

app = Flask(__name__)
CORS(app, resources={r"/recommend": {"origins": "http://localhost:5173"}}, supports_credentials=True)

MODEL_FILE = r"C:\Users\akdie\PycharmProjects\RecommendationModel\current_model.pt"
MODEL_LOCK = threading.Lock()
model = None
user_to_index = {}
item_to_index = {}
index_to_item = {}
num_users = 0
num_items = 0

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Constants for batching retraining
NEW_RATINGS = []
BATCH_SIZE = 10  # Number of new ratings to accumulate before retraining
BATCH_TIME = 300  # Time in seconds to wait before retraining if batch size not reached

user_rated_items = {}
top_items = []  # Global variable to store top 10 items based on ratings


def save_mappings(user_to_index: dict, item_to_index: dict, index_to_item: dict):
    """
    Saves the user and item mappings to disk using pickle.

    Args:
        user_to_index: Dict mapping user_id to index.
        item_to_index: Dict mapping item_id to index.
        index_to_item: Dict mapping index to item_id.
    """
    try:
        with open("user_to_index.pkl", "wb") as f:
            pickle.dump(user_to_index, f)
        with open("item_to_index.pkl", "wb") as f:
            pickle.dump(item_to_index, f)
        with open("index_to_item.pkl", "wb") as f:
            pickle.dump(index_to_item, f)
        logging.info("User and item mappings saved successfully.")
    except Exception as e:
        logging.error(f"Error saving mappings: {e}")


def load_mappings_from_file() -> tuple:
    """
    Loads the user and item mappings from disk.

    Returns:
        Tuple containing:
            - user_to_index
            - item_to_index
            - index_to_item
    """
    try:
        with open("user_to_index.pkl", "rb") as f:
            user_to_index_loaded = pickle.load(f)
        with open("item_to_index.pkl", "rb") as f:
            item_to_index_loaded = pickle.load(f)
        with open("index_to_item.pkl", "rb") as f:
            index_to_item_loaded = pickle.load(f)
        logging.info("User and item mappings loaded successfully.")
        return user_to_index_loaded, item_to_index_loaded, index_to_item_loaded
    except Exception as e:
        logging.error(f"Error loading mappings: {e}")
        return {}, {}, {}


def save_user_rated_items(user_rated_items: dict):
    """
    Saves the user-rated items mapping to disk using pickle.

    Args:
        user_rated_items: Dict mapping user_id to set of course_ids they've rated.
    """
    try:
        with open("user_rated_items.pkl", "wb") as f:
            pickle.dump(user_rated_items, f)
        logging.info("User-rated items mappings saved successfully.")
    except Exception as e:
        logging.error(f"Error saving user-rated items mappings: {e}")


def load_user_rated_items_from_file() -> dict:
    """
    Loads the user-rated items mapping from disk.

    Returns:
        Dict mapping user_id to set of course_ids they've rated.
    """
    try:
        with open("user_rated_items.pkl", "rb") as f:
            user_rated_items_loaded = pickle.load(f)
        logging.info("User-rated items mappings loaded successfully.")
        return user_rated_items_loaded
    except Exception as e:
        logging.error(f"Error loading user-rated items mappings: {e}")
        return {}


def get_user_rated_items(data: list) -> dict:
    """
    Builds a dictionary mapping each user to a set of items they've rated.

    Args:
        data: List of tuples containing (user_id, course_id, rating).

    Returns:
        Dict mapping user_id to a set of course_ids they've rated.
    """
    user_rated = {}
    for user_id, course_id, _ in data:
        if user_id not in user_rated:
            user_rated[user_id] = set()
        user_rated[user_id].add(course_id)
    logging.info("Built user-rated items mapping.")
    return user_rated


def compute_top_items(data: List[Tuple[int, int, float]], n: int = 10) -> List[int]:
    """
    Computes the top-N items based on average ratings.

    Args:
        data: List of tuples containing (user_id, course_id, rating).
        n: Number of top items to return.

    Returns:
        List of top-N course_ids based on average ratings.
    """
    from collections import defaultdict

    rating_sum = defaultdict(float)
    rating_count = defaultdict(int)

    for _, course_id, rating in data:
        rating_sum[course_id] += rating
        rating_count[course_id] += 1

    average_ratings = {course_id: rating_sum[course_id] / rating_count[course_id] for course_id in rating_sum}

    # Sort items by average rating in descending order
    sorted_items = sorted(average_ratings.items(), key=lambda x: x[1], reverse=True)

    top_items_list = [course_id for course_id, _ in sorted_items[:n]]
    return top_items_list


def retrain_model():
    """
    Retrains the NeuMF model using the latest data from the database.
    Updates the global model, mappings, and top_items.
    """
    global model, user_to_index, item_to_index, index_to_item, num_users, num_items, user_rated_items, top_items
    with MODEL_LOCK:
        logging.info("Starting model retraining...")
        print("Starting model retraining...")
        data = load_data_from_db()
        all_users = load_all_users()
        user_to_index, item_to_index, index_to_item = build_mappings(data, all_users)
        user_rated_items = get_user_rated_items(data)  # Get user-rated items
        num_users = len(user_to_index)
        num_items = len(item_to_index)

        # Remap data to zero-based indices
        remapped_data = [
            (user_to_index[u], item_to_index[i], r)
            for (u, i, r) in data
            if u in user_to_index and i in item_to_index
        ]

        if num_users == 0 or num_items == 0:
            logging.warning("No data found to train the model.")
            print("No data found to train the model.")
            return

        # Train the NeuMF model
        model = train_neumf_model(
            remapped_data,
            num_users,
            num_items,
            epochs=20,
            lr=0.001,
            neg_per_pos=4,
            batch_size=256,
            dropout=0.2
        )
        torch.save(model.state_dict(), MODEL_FILE)
        logging.info("Model retrained and saved.")
        print("Model retrained and saved.")

        # Save mappings
        save_mappings(user_to_index, item_to_index, index_to_item)

        # Save user-rated items
        save_user_rated_items(user_rated_items)

        # Compute and store top 10 items
        top_items = compute_top_items(data, n=10)
        logging.info(f"Top 10 items based on average ratings: {top_items}")
        print(f"Top 10 items based on average ratings: {top_items}")


def load_existing_model():
    """
    Loads the existing model and mappings from disk if available.
    Otherwise, trains a new model.
    """
    global model, user_to_index, item_to_index, index_to_item, num_users, num_items, user_rated_items, top_items
    if os.path.exists(MODEL_FILE) and os.path.exists("user_to_index.pkl") \
       and os.path.exists("item_to_index.pkl") and os.path.exists("index_to_item.pkl") \
       and os.path.exists("user_rated_items.pkl"):
        logging.info("Loading existing model and mappings...")
        print("Loading existing model and mappings...")
        user_to_index, item_to_index, index_to_item = load_mappings_from_file()
        user_rated_items = load_user_rated_items_from_file()
        num_users = len(user_to_index)
        num_items = len(item_to_index)
        model = NeuMF(num_users, num_items)
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()
        logging.info("Model loaded successfully.")
        print("Model loaded successfully.")

        # Load and compute top items
        data = load_data_from_db()
        top_items = compute_top_items(data, n=10)
        logging.info(f"Top 10 items based on average ratings: {top_items}")
        print(f"Top 10 items based on average ratings: {top_items}")
    else:
        logging.info("No existing model found. Starting initial training...")
        print("No existing model found. Starting initial training...")
        retrain_model()


def get_rated_items(user_id: int) -> set:
    """
    Retrieves the set of item indices that the user has already rated.

    Args:
        user_id: The ID of the user.

    Returns:
        Set of item indices the user has rated.
    """
    rated_courses = user_rated_items.get(user_id, set())
    rated_indices = set()
    for course_id in rated_courses:
        if course_id in item_to_index:
            rated_indices.add(item_to_index[course_id])
    return rated_indices


@app.route("/recommend", methods=["GET"])
def recommend():
    """
    Flask route to provide top-N recommendations for a given user.
    If user_id is not found, returns top 10 popular items based on average ratings.
    """
    user_id = request.args.get("user_id", type=int)
    n = request.args.get("n", default=10, type=int)  # Default to top 10 recommendations

    MAX_RECOMMENDATIONS = 50  # Define a maximum limit

    if n <= 0:
        return jsonify({"error": "Number of recommendations 'n' must be a positive integer"}), 400

    if n > MAX_RECOMMENDATIONS:
        return jsonify({"error": f"Number of recommendations 'n' cannot exceed {MAX_RECOMMENDATIONS}"}), 400

    with MODEL_LOCK:
        if user_id in user_to_index:
            u_idx = user_to_index[user_id]
            item_indices = torch.arange(num_items, dtype=torch.long)
            user_indices = torch.tensor([u_idx] * num_items, dtype=torch.long)
            with torch.no_grad():
                scores = model(user_indices, item_indices).squeeze().numpy()

            # Exclude items already rated by the user
            rated_items = get_rated_items(user_id)
            if rated_items:
                scores[list(rated_items)] = -np.inf  # Assign a very low score to exclude

            top_n = n
            top_items_idx = np.argsort(-scores)[:top_n]
            recommended_items = [index_to_item[i] for i in top_items_idx]
        else:
            # User ID not found, recommend top 10 popular items
            logging.info(f"User ID {user_id} not found. Recommending top {n} popular items.")
            print(f"User ID {user_id} not found. Recommending top {n} popular items.")
            recommended_items = top_items[:n]

    return jsonify({"user_id": user_id, "recommendations": recommended_items})


def listen_for_changes(conn):
    """
    Listens for PostgreSQL notifications on the 'new_rating' channel.
    Retrains the model when a new rating is detected.
    """
    try:
        logging.info("Listener thread started, waiting for notifications...")
        print("Listener thread started, waiting for notifications...")
        while True:
            # Wait for at least one notification with a timeout
            if select.select([conn], [], [], 5) == ([], [], []):
                continue
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                if notify.channel == 'new_rating':
                    payload = json.loads(notify.payload)
                    logging.info(f"New rating detected: {payload}")
                    print(f"New rating detected: {payload}")
                    NEW_RATINGS.append(payload)
                    if len(NEW_RATINGS) >= BATCH_SIZE:
                        retrain_model()
                        NEW_RATINGS.clear()
    except Exception as e:
        logging.error(f"Error in listen_for_changes: {e}")
        print(f"Error in listen_for_changes: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")
            print("Database connection closed.")


def periodic_retrain():
    """
    Periodically checks for accumulated ratings and retrains the model if necessary.
    """
    try:
        while True:
            time.sleep(BATCH_TIME)  # Wait for BATCH_TIME seconds
            with MODEL_LOCK:
                if NEW_RATINGS:
                    logging.info(f"Periodic retraining triggered with {len(NEW_RATINGS)} new ratings.")
                    print(f"Periodic retraining triggered with {len(NEW_RATINGS)} new ratings.")
                    retrain_model()
                    NEW_RATINGS.clear()
    except Exception as e:
        logging.error(f"Error in periodic_retrain: {e}")
        print(f"Error in periodic_retrain: {e}")


if __name__ == "__main__":
    # Load existing model or train a new one
    load_existing_model()

    # Connect to PostgreSQL and set up listener
    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="knowverse",
            user="postgres",
            password="123123",
            port=5432  # Ensure correct port
        )
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute("LISTEN new_rating;")
        logging.info("Listening for new ratings on 'new_rating' channel...")
        print("Listening for new ratings...")
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        print(f"Failed to connect to database: {e}")
        exit(1)

    # Start the listener thread
    listener_thread = threading.Thread(target=listen_for_changes, args=(conn,), daemon=True)
    listener_thread.start()

    # Start the periodic retraining thread
    retrain_thread = threading.Thread(target=periodic_retrain, daemon=True)
    retrain_thread.start()

    # Start Flask app
    app.run(host="0.0.0.0", port=5000)

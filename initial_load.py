# initial_load.py

import psycopg2
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(
    filename='initial_load.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def load_data_from_db() -> List[Tuple[int, int, float]]:
    """
    Connects to the PostgreSQL database and retrieves all ratings from tbl_review.

    Returns:
        List of tuples containing (user_id, course_id, rating).
    """
    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="knowverse",
            user="postgres",
            password="123123",
            port=5432  # Ensure correct port
        )
        cur = conn.cursor()
        cur.execute("SELECT user_id, course_id, rating FROM tbl_review;")
        data = cur.fetchall()
        cur.close()
        conn.close()
        logging.info(f"Loaded {len(data)} ratings from the database.")
        return data
    except Exception as e:
        logging.error(f"Error loading data from DB: {e}")
        return []


def load_all_users() -> List[int]:
    """
    Connects to the PostgreSQL database and retrieves all user IDs from tbl_user.

    Returns:
        List of user_ids.
    """
    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="knowverse",
            user="postgres",
            password="123123",
            port=5432
        )
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM tbl_user;")
        users = cur.fetchall()
        cur.close()
        conn.close()
        user_ids = [user[0] for user in users]
        logging.info(f"Loaded {len(user_ids)} users from the database.")
        return user_ids
    except Exception as e:
        logging.error(f"Error loading users from DB: {e}")
        return []


def build_mappings(data: List[Tuple[int, int, float]], all_users: List[int]) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Builds mappings from user_id and course_id to indices and vice versa.

    Args:
        data: List of tuples containing (user_id, course_id, rating).
        all_users: List of all user_ids.

    Returns:
        Tuple containing:
            - user_to_index: Dict mapping user_id to unique index.
            - item_to_index: Dict mapping course_id to unique index.
            - index_to_item: Dict mapping unique index to course_id.
    """
    user_set = set(all_users)  # Include all users
    item_set = set()
    for user_id, course_id, _ in data:
        user_set.add(user_id)
        item_set.add(course_id)

    user_to_index = {user_id: idx for idx, user_id in enumerate(sorted(user_set))}
    item_to_index = {course_id: idx for idx, course_id in enumerate(sorted(item_set))}
    index_to_item = {idx: course_id for course_id, idx in item_to_index.items()}

    logging.info(f"Built mappings: {len(user_to_index)} users, {len(item_to_index)} items.")

    return user_to_index, item_to_index, index_to_item

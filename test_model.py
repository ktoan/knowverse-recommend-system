from initial_load import load_data_from_db, build_mappings
from model_training import train_neumf_model, NeuMF
import torch
import numpy as np
from math import sqrt


def evaluate_model(model, test_data, user_to_index, item_to_index):
    # Filter test_data to include only users and items known to the model
    filtered_test = [(u, i, r) for (u, i, r) in test_data if u in user_to_index and i in item_to_index]
    if len(filtered_test) == 0:
        raise ValueError("No test examples match the training set's users/items.")

    users = [user_to_index[u] for (u, i, r) in filtered_test]
    items = [item_to_index[i] for (u, i, r) in filtered_test]
    ratings = [r for (u, i, r) in filtered_test]

    users_t = torch.tensor(users, dtype=torch.long)
    items_t = torch.tensor(items, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        preds = model(users_t, items_t).squeeze().numpy()

    # Compute MSE and RMSE
    actual = np.array(ratings)
    mse = ((preds - actual) ** 2).mean()
    rmse = sqrt(mse)
    return mse, rmse


if __name__ == "__main__":
    # 1. Load all data from the database
    data = load_data_from_db()  # returns list of (user_id, course_id, rating)
    if len(data) == 0:
        raise ValueError("No data found in tbl_review. Cannot test model.")

    # 2. Shuffle and split the data 50% train, 50% test
    np.random.shuffle(data)
    half_point = len(data) // 2
    train_data = data[:half_point]
    test_data = data[half_point:]

    # 3. Build mappings from train_data
    user_to_index, item_to_index, index_to_item = build_mappings(train_data)
    num_users = len(user_to_index)
    num_items = len(item_to_index)

    if num_users == 0 or num_items == 0:
        raise ValueError("No users or items could be mapped. Check your training data.")

    # Remap train_data to zero-based indices
    remapped_train = [(user_to_index[u], item_to_index[i], r)
                      for (u, i, r) in train_data
                      if u in user_to_index and i in item_to_index]

    # 4. Train the NeuMF model using the remapped train_data
    print("Training the NeuMF model on database training data (50% of dataset)...")
    model = train_neumf_model(remapped_train, num_users=num_users, num_items=num_items,
                              epochs=100, lr=0.001, neg_per_pos=2)

    # Optional: Print predictions for a test user
    model.eval()
    with torch.no_grad():
        # Pick a user from the training set
        if len(user_to_index) > 0:
            test_user_id = next(iter(user_to_index.keys()))
            u_idx = user_to_index[test_user_id]

            # Generate scores for all items for this user
            item_indices = torch.arange(num_items)
            user_indices = torch.tensor([u_idx] * num_items, dtype=torch.long)
            scores = model(user_indices, item_indices).squeeze().numpy()

            print(f"\nPredicted scores for user {test_user_id} (internal idx {u_idx}):")
            for idx, score in enumerate(scores):
                original_item_id = index_to_item[idx]
                print(f"Item {original_item_id} (idx {idx}): {score:.4f}")

            # Top 5 recommendations
            top_n = 5
            top_items_idx = np.argsort(-scores)[:top_n]
            recommended_items = [index_to_item[i] for i in top_items_idx]
            print(f"\nTop {top_n} recommended items for user {test_user_id}: {recommended_items}")
        else:
            print("No users found in training set. Cannot show recommendations.")

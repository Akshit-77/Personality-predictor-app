import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle, os

def fit_lightgbm_and_save(X_train, y_train, X_test, y_test):
    """
    Improved LightGBM with optimized hyperparameters
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Optimized hyperparameters for better accuracy
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",      # Changed from "rf" to "gbdt" for better performance
        "num_leaves": 31,             # Increased from 5 for more model complexity
        "learning_rate": 0.05,        # Reduced from 0.5 for better generalization
        "feature_fraction": 0.9,      # Increased from 0.8
        "bagging_fraction": 0.8,
        "bagging_freq": 5,            # Added bagging frequency
        "metric": "binary_logloss",
        "min_data_in_leaf": 20,       # Prevent overfitting
        "lambda_l1": 0.1,             # L1 regularization
        "lambda_l2": 0.1,             # L2 regularization
        "min_gain_to_split": 0.02,    # Minimum gain to make split
        "max_depth": -1,              # No limit on depth
        "is_unbalance": True,         # Handle class imbalance if present
        "force_row_wise": True,
        "verbosity": -1               # Reduce output verbosity
    }

    # Increased number of rounds with early stopping
    num_round = 1000
    
    # Train with early stopping to prevent overfitting
    bst = lgb.train(
        params, 
        train_data, 
        num_round, 
        valid_sets=[train_data, test_data],
        valid_names=['train', 'eval'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Make predictions using best iteration
    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary) * 100
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1score = f1_score(y_test, y_pred_binary)
    
    print(f"Accuracy of LightGBM is: {accuracy: .2f}%")
    print(f"Precision of LightGBM is: {precision:.4f}")
    print(f"Recall of LightGBM is: {recall:.4f}")
    print(f"F1score of LightGBM is: {f1score:.4f}")
    print(f"Best iteration: {bst.best_iteration}")
    
    # Save the optimized model
    path_to_save = os.path.join(os.getcwd(), 'models', 'LightGBM.pkl')
    with open(path_to_save, 'wb') as file:
        pickle.dump(bst, file)
    
    return bst  # Return model if needed
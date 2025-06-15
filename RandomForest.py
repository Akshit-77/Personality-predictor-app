from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle, os

def fit_rf_classifier_and_save(X_train, y_train, X_test, y_test, n_estimators=100):
    """
    Optimized Random Forest with better hyperparameters
    """
    # Optimized Random Forest parameters
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,        # Keep user-defined or default 100
        max_depth=10,                     # Prevent overfitting
        min_samples_split=5,              # Minimum samples to split
        min_samples_leaf=2,               # Minimum samples in leaf
        max_features='sqrt',              # Number of features for best split
        bootstrap=True,                   # Use bootstrap sampling
        oob_score=True,                   # Out-of-bag score for validation
        n_jobs=-1,                        # Use all processors
        random_state=42,                  # For reproducibility
        class_weight='balanced',          # Handle class imbalance
        criterion='gini',                 # Split criterion
        max_samples=0.8                   # Bootstrap sample size
    )
    
    # Train the model
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    
    # Print results
    print(f"Accuracy of Random forest with {n_estimators} estimators is: {accuracy: .2f}%")
    print(f"Precision of Random Forest is: {precision:.4f}")
    print(f"Recall of Random Forest is: {recall:.4f}")
    print(f"F1score of Random Forest is: {f1score:.4f}")
    print(f"OOB Score: {rf_classifier.oob_score_:.4f}")
    
    # Save the optimized model
    path_to_save = os.path.join(os.getcwd(), 'models', 'RandomForest.pkl')
    with open(path_to_save, 'wb') as file:
        pickle.dump(rf_classifier, file)
    
    return rf_classifier  # Return model if needed
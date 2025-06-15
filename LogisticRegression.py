from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle, os

def fit_lr_classifier_and_save(X_train, y_train, X_test, y_test):
    """
    Optimized Logistic Regression without scaling
    """
    # Optimized Logistic Regression with regularization
    lr = LogisticRegression(
        penalty='l2',              # L2 regularization (Ridge)
        C=1.0,                     # Regularization strength (inverse)
        solver='lbfgs',            # Efficient solver for L2 penalty
        max_iter=10000,            # Keep high iteration limit
        random_state=42,           # For reproducibility
        class_weight='balanced',   # Handle class imbalance
        fit_intercept=True,        # Fit intercept term
        tol=1e-6,                  # Tolerance for stopping criteria
        n_jobs=-1                  # Use all processors
    )
    
    # Train the model
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    
    # Print results
    print(f"Accuracy of Logistic Regression is: {accuracy: .2f}%")
    print(f"Precision of Logistic Regression is: {precision:.4f}")
    print(f"Recall of Logistic Regression is: {recall:.4f}")
    print(f"F1score of Logistic Regression is: {f1score:.4f}")
    print(f"Number of iterations: {lr.n_iter_[0]}")
    
    # Save the optimized model
    path_to_save = os.path.join(os.getcwd(), 'models', 'LogisticRegression.pkl')
    with open(path_to_save, 'wb') as file:
        pickle.dump(lr, file)
    
    return lr  # Return model if needed
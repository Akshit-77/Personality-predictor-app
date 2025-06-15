from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle, os

def fit_gnb_classifier_and_save(X_train, y_train, X_test, y_test):
    """
    Optimized Gaussian Naive Bayes without scaling
    """
    # Optimized Gaussian Naive Bayes with smoothing
    gnb = GaussianNB(
        var_smoothing=1e-8,  # Reduced smoothing for better performance on small datasets
        priors=None          # Let the model learn class priors from data
    )
    
    # Train the model
    gnb.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gnb.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100  # Fixed: was y_pred, y_test
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    
    # Print results
    print(f"Accuracy of Naive Bayes classifier is: {accuracy: .2f}%")
    print(f"Precision of Naive Bayes classifier is: {precision:.4f}")
    print(f"Recall of Naive Bayes classifier is: {recall:.4f}")
    print(f"F1score of Naive Bayes classifier is: {f1score:.4f}")
    
    # Save the optimized model
    path_to_save = os.path.join(os.getcwd(), 'models', 'NaiveBayes.pkl')
    with open(path_to_save, 'wb') as file:
        pickle.dump(gnb, file)
    
    return gnb  # Return model if needed
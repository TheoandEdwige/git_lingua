import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def encode_and_split_data(df, text_column, target_column, test_size=0.2, random_state=42):
    """
    Encode the target variable and split the data into training and validation sets.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        text_column (str): The name of the text column.
        target_column (str): The name of the target column.
        test_size (float, optional): The proportion of data to include in the validation set (default is 0.2).
        random_state (int, optional): Seed for random number generation (default is 42).

    Returns:
        tuple: Four elements (X_train, X_val, y_train, y_val) representing the training and validation data.
    """
    # Create a label encoder
    label_encoder = LabelEncoder()

    # Encode the target variable
    df['encoded_target'] = label_encoder.fit_transform(df[target_column])

    # Split the data into X (text) and y (encoded target)
    X = df[text_column]
    y = df['encoded_target']

    # Split the data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_val, y_train, y_val


def tfidf_vectorization(X_train, X_val, max_features=5000):
    """
    Perform TF-IDF vectorization on the training and validation data.

    Args:
        X_train (pandas.Series): Training data containing text.
        X_val (pandas.Series): Validation data containing text.
        max_features (int, optional): The maximum number of features to consider (default is 5000).

    Returns:
        tuple: Two elements (X_train_tfidf, X_val_tfidf) representing the TF-IDF transformed data.
    """
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit and transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Transform the validation data using the same vectorizer
    X_val_tfidf = tfidf_vectorizer.transform(X_val)

    return X_train_tfidf, X_val_tfidf


def train_decision_tree(X_train, y_train, X_val, y_val):
    """
    Train a Decision Tree classifier and evaluate its performance.

    Args:
        X_train (scipy.sparse.csr.csr_matrix): Training data in TF-IDF format.
        y_train (pandas.Series): Training target variable.
        X_val (scipy.sparse.csr.csr_matrix): Validation data in TF-IDF format.
        y_val (pandas.Series): Validation target variable.

    Returns:
        tuple: Two elements (dt_classifier, accuracy) representing the trained model and its accuracy.
    """
    # Initialize the Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    
    # Train the model
    dt_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = dt_classifier.score(X_val, y_val)
    return dt_classifier, accuracy


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train a Random Forest classifier and evaluate its performance.

    Args:
        X_train (scipy.sparse.csr.csr_matrix): Training data in TF-IDF format.
        y_train (pandas.Series): Training target variable.
        X_val (scipy.sparse.csr.csr_matrix): Validation data in TF-IDF format.
        y_val (pandas.Series): Validation target variable.

    Returns:
        tuple: Two elements (rf_classifier, accuracy) representing the trained model and its accuracy.
    """
    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(min_samples_leaf=2, min_samples_split=5,
                       n_estimators=200, random_state=42)
    
    # Train the model
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = rf_classifier.score(X_val, y_val)
    return rf_classifier, accuracy


def train_knn(X_train, y_train, X_val, y_val, n_neighbors=5):
    """
    Train a K-Nearest Neighbors classifier and evaluate its performance.

    Args:
        X_train (scipy.sparse.csr.csr_matrix): Training data in TF-IDF format.
        y_train (pandas.Series): Training target variable.
        X_val (scipy.sparse.csr.csr_matrix): Validation data in TF-IDF format.
        y_val (pandas.Series): Validation target variable.
        n_neighbors (int, optional): Number of neighbors to use (default is 5).

    Returns:
        tuple: Two elements (knn_classifier, accuracy) representing the trained model and its accuracy.
    """
    # Initialize the K-Nearest Neighbors Classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model
    knn_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = knn_classifier.score(X_val, y_val)
    return knn_classifier, accuracy


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train a Logistic Regression classifier and evaluate its performance.

    Args:
        X_train (scipy.sparse.csr.csr_matrix): Training data in TF-IDF format.
        y_train (pandas.Series): Training target variable.
        X_val (scipy.sparse.csr.csr_matrix): Validation data in TF-IDF format.
        y_val (pandas.Series): Validation target variable.

    Returns:
        tuple: Two elements (lr_classifier, accuracy) representing the trained model and its accuracy.
    """
    # Initialize the Logistic Regression Classifier
    lr_classifier = LogisticRegression(random_state=42)
    
    # Train the model
    lr_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = lr_classifier.score(X_val, y_val)
    return lr_classifier, accuracy


def evaluate_and_compare_models(X_train_tfidf, y_train, X_val_tfidf, y_val):
    """
    Evaluate and compare multiple classification models.

    Args:
        X_train_tfidf (scipy.sparse.csr_matrix): TF-IDF transformed training data.
        y_train (pandas.Series): Training target labels.
        X_val_tfidf (scipy.sparse.csr_matrix): TF-IDF transformed validation data.
        y_val (pandas.Series): Validation target labels.

    Returns:
        pandas.DataFrame: A DataFrame with model performance metrics (Accuracy, Precision, Recall, and F1-Score).
    """
    # Create an empty DataFrame to store model performance
    model_comparison = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    # Define the classifiers and their names
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42)
    }

    # Evaluate and compare each model
    for name, classifier in classifiers.items():
        # Train the model (assuming you already have a function for this)
        classifier.fit(X_train_tfidf, y_train)  # Fit the model before making predictions

        # Evaluate the model
        y_pred = classifier.predict(X_val_tfidf)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        # Calculate and store the results in the DataFrame
        model_comparison = pd.concat([model_comparison, pd.DataFrame({
            'Model': [name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1-Score': [f1]
        })], ignore_index=True)

    return model_comparison


def evaluate_final_model(X_train, y_train, X_val, y_val, random_state=42):
    """
    Evaluate the final model using a test set.

    Args:
        X_train (pandas.Series): Training data containing text.
        y_train (pandas.Series): Training target labels.
        X_val (pandas.Series): Validation data containing text.
        y_val (pandas.Series): Validation target labels.
        random_state (int, optional): Random state for reproducibility (default is 42).

    Returns:
        float: Test set accuracy of the final model.
    """

    # Further split the validation set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=random_state)
    
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit and transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Define your model (e.g., RandomForestClassifier) with appropriate hyperparameters
    model = RandomForestClassifier(min_samples_leaf=2, min_samples_split=5, n_estimators=200, random_state=random_state)
    
    # Fit the model on the training dataset
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions on the test set
    y_pred_test = model.predict(X_test_tfidf)
    
    # Calculate accuracy on the test set
    test_set_accuracy = accuracy_score(y_test, y_pred_test)
    
    return test_set_accuracy



def baseline(X_train, y_train, X_val, y_val):
    """
    Evaluate a baseline model that predicts the most frequent class.

    Args:
        X_train (pandas.Series): Training data containing text.
        y_train (pandas.Series): Training target labels.
        X_val (pandas.Series): Validation data containing text.
        y_val (pandas.Series): Validation target labels.
    """
    # Calculate the most frequent class in the training data
    most_frequent_class = y_train.value_counts().idxmax()
    
    # Create a DummyClassifier that always predicts the most frequent class
    baseline_classifier = DummyClassifier(strategy="most_frequent")
    baseline_classifier.fit(X_train, y_train)
    
    # Predict the most frequent class for all instances in the validation data
    y_pred_baseline = baseline_classifier.predict(X_val)
    
    # Calculate the accuracy of the baseline model
    accuracy = accuracy_score(y_val, y_pred_baseline)
    
    print(f"Baseline Accuracy: {accuracy:.4f}")



def generate_and_save_predictions(X_train, y_train, X_test, y_test):
    """
    Generate predictions for a test set and save them to a CSV file.

    Args:
        X_train (pandas.Series): Training data containing text.
        y_train (pandas.Series): Training target labels.
        X_test (pandas.Series): Test data containing text.
        y_test (pandas.Series): Test target labels.

    Returns:
        pandas.DataFrame: A DataFrame with the top 10 predictions and actual language names.
    """
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit and transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Define your model (e.g., RandomForestClassifier) with appropriate hyperparameters
    model = RandomForestClassifier(min_samples_leaf=2, min_samples_split=5, n_estimators=200, random_state=42)
    
    # Fit the model on the training dataset
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions on X_test
    y_pred = model.predict(X_test_tfidf)
    
    # Create a label encoder instance
    label_encoder = LabelEncoder()
    
    # Fit the label encoder on the target variable
    label_encoder.fit(y_train)
    
    # Inverse transform the predictions to get actual language names
    predicted_languages = label_encoder.inverse_transform(y_pred)
    
    # Create a DataFrame to store the predictions and actual language names
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predicted_languages})
    
    # Save the predictions to a CSV file
    predictions_df.to_csv('predictions.csv', index=False)
    
    # Return the top 5 predictions
    top_5_predictions = predictions_df.head(10)
    
    return top_5_predictions



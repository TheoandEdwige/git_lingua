import pandas as pd

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
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit and transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Transform the validation data using the same vectorizer
    X_val_tfidf = tfidf_vectorizer.transform(X_val)

    return X_train_tfidf, X_val_tfidf


def train_decision_tree(X_train, y_train, X_val, y_val):
    # Initialize the Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    
    # Train the model
    dt_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = dt_classifier.score(X_val, y_val)
    return dt_classifier, accuracy


def train_random_forest(X_train, y_train, X_val, y_val):
    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(min_samples_leaf=2, min_samples_split=5,
                       n_estimators=200, random_state=42)
    
    # Train the model
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = rf_classifier.score(X_val, y_val)
    return rf_classifier, accuracy


def train_knn(X_train, y_train, X_val, y_val, n_neighbors=5):
    # Initialize the K-Nearest Neighbors Classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model
    knn_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = knn_classifier.score(X_val, y_val)
    return knn_classifier, accuracy


def train_logistic_regression(X_train, y_train, X_val, y_val):
    # Initialize the Logistic Regression Classifier
    lr_classifier = LogisticRegression(random_state=42)
    
    # Train the model
    lr_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = lr_classifier.score(X_val, y_val)
    return lr_classifier, accuracy


def evaluate_and_compare_models(X_train_tfidf, y_train, X_val_tfidf, y_val):
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

import streamlit as st
import pandas as pd
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)


# Load models and vectorizer
with open('models/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/top_features.pkl', 'rb') as f:
    top_features = pickle.load(f)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, labels, title="Performance"):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    st.pyplot()

# Function to display most used words
def display_most_used_words(person_label):
    if person_label in top_features:
        st.subheader(f'Most Used Words for {person_label}:')
        for word, score in top_features[person_label]:
            st.write(f'{word}: {score:.4f}')

# Streamlit UI
st.title('Person Classification App')

# Sidebar navigation
tab = st.sidebar.radio('Navigation', ['Predict', 'Model Metrics'])

# Main content based on selected tab
if tab == 'Predict':
    st.header('Prediction')
    
    # Randomly select a question
    question = "Enter a response to one of the 36 Questions to Love"
    st.write(question)
    
    # Text input for user response
    response = st.text_area('Your response:')
    
    # Model selection
    model_type = st.selectbox('Select model for prediction:', ['Random Forest', 'Naive Bayes'])
    
    # Prediction based on selected model
    if st.button('Predict'):
        if model_type == 'Random Forest':
            prediction = rf_model.predict(vectorizer.transform([response]))[0]
        elif model_type == 'Naive Bayes':
            prediction = nb_model.predict(vectorizer.transform([response]))[0]
        
        st.write(f'Predicted Person: {prediction}')
        display_most_used_words(prediction)

elif tab == 'Model Metrics':
    st.header('Model Test Metrics')
    
    # File uploader for custom test data
    st.subheader('Upload Custom Test Data:')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        custom_test_data = pd.read_csv(uploaded_file)
    else:
        custom_test_data = None

    # Display evaluation button
    if st.button('Evaluate Models'):
        # Load test data (default or custom)
        if custom_test_data is not None:
            test_data = custom_test_data
        else:
            test_data = pd.read_csv('test_data.csv')  # Default test data

        # Feature extraction for test data
        X_test = vectorizer.transform(test_data['response'])
        y_test = test_data['person']

        # Predictions on test data
        y_pred_rf = rf_model.predict(X_test)
        y_pred_nb = nb_model.predict(X_test)

        # Evaluation metrics
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        rf_classification_report = classification_report(y_test, y_pred_rf)
        rf_conf_matrix = confusion_matrix(y_test, y_pred_rf, labels=['PersonA', 'PersonB'])

        nb_accuracy = accuracy_score(y_test, y_pred_nb)
        nb_classification_report = classification_report(y_test, y_pred_nb)
        nb_conf_matrix = confusion_matrix(y_test, y_pred_nb, labels=['PersonA', 'PersonB'])

        # Display metrics
        st.subheader('Random Forest Metrics:')
        st.write(f'Accuracy: {rf_accuracy:.4f}')
        st.write('Classification Report:')
        st.text(rf_classification_report)
        st.write('Confusion Matrix:')
        plot_confusion_matrix(rf_conf_matrix, labels=['PersonA', 'PersonB'])

        st.subheader('Naive Bayes Metrics:')
        st.write(f'Accuracy: {nb_accuracy:.4f}')
        st.write('Classification Report:')
        st.text(nb_classification_report)
        st.write('Confusion Matrix:')
        plot_confusion_matrix(nb_conf_matrix, labels=['PersonA', 'PersonB'])

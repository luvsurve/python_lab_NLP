import streamlit as st
import pandas as pd
import pickle
import base64
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

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string.decode()});
            background-size: cover;
        }}
        .main {{
            background: rgba(255, 255, 255, 0.5);
            padding: 20px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

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

# Function to convert classification report to dataframe

def classification_report_to_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:  # Skip the header and footer lines
        row_data = line.split()
        if len(row_data) >= 5:  # Ensure there are enough elements to parse
            row = {}
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1-score'] = float(row_data[3])
            row['support'] = int(row_data[4])
            report_data.append(row)
    df = pd.DataFrame.from_dict(report_data)
    return df


# Set the background image
set_background("image.jpg")
custom_css = """
<style>
h1, h2, h3, p {
    color: #000000; 
}

.sidebar .sidebar-content {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
}

.sidebar .sidebar-content a {
    color: #007bff;
    font-weight: bold;
}

.sidebar .sidebar-content a:hover {
    color: #0056b3;
    text-decoration: none;
}

.sidebar .sidebar-content .st-radio {
    margin-bottom: 15px;
}

.sidebar .sidebar-content .st-radio > div {
    background-color: #e0e4eb;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

.sidebar .sidebar-content .st-radio > div:hover {
    background-color: #d0d4db;
}

.sidebar .sidebar-content .st-radio > div[data-baseweb='radio']:checked {
    background-color: #c0c4cb;
}

/* Hide the deploy button */
footer {
    visibility: hidden;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit UI
st.title('36 Questions of Love')

# Sidebar navigation
tab = st.sidebar.radio('Navigation', ['Predict', 'Model Metrics'])


# Main content based on selected tab
with st.container():
    if tab == 'Predict':
        st.header('Prediction')
        
        # Display prompt for user response
        st.write("Enter a response to one of the 36 Questions to Love")
        
        # Text input for user response
        response = st.text_area('Your response:')
        
        # Model selection
        model_type = st.selectbox('Select model for prediction:', ['Naive Bayes','Random Forest'])
        
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
            rf_conf_matrix = confusion_matrix(y_test, y_pred_rf, labels=['PersonA', 'PersonB', 'PersonC', 'PersonD'])

            nb_accuracy = accuracy_score(y_test, y_pred_nb)
            nb_classification_report = classification_report(y_test, y_pred_nb)
            nb_conf_matrix = confusion_matrix(y_test, y_pred_nb, labels=['PersonA', 'PersonB', 'PersonC', 'PersonD'])

            # Display metrics
            st.subheader('Random Forest Metrics:')
            st.write(f'Accuracy: {rf_accuracy*100:.2f}%')
            st.write('Classification Report:')
            rf_report_df = classification_report_to_df(rf_classification_report)
            st.dataframe(rf_report_df)
            st.write('Confusion Matrix:')
            plot_confusion_matrix(rf_conf_matrix, labels=['PersonA', 'PersonB', 'PersonC', 'PersonD'])

            st.subheader('Naive Bayes Metrics:')
            st.write(f'Accuracy: {nb_accuracy*100:.2f}%')
            st.write('Classification Report:')
            nb_report_df = classification_report_to_df(nb_classification_report)
            st.dataframe(nb_report_df)
            st.write('Confusion Matrix:')
            plot_confusion_matrix(nb_conf_matrix, labels=['PersonA', 'PersonB', 'PersonC', 'PersonD'])

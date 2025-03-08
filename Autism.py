import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="ASD Detection with Machine Learning", layout="wide")

st.title("Detection of Autism Spectrum Disorder")

# Upload dataset
uploaded_file = st.file_uploader("Upload the Autism Dataset (CSV file)", type="csv")

if uploaded_file:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    
    # Display the total number of rows and the full dataset preview
    total_rows = data.shape[0]
    st.write(f"### Dataset Preview (Total Rows: {total_rows})")
    st.dataframe(data)

    # Data preprocessing
    data = data.drop(columns=['Case_No'])
    data['Traits '] = data['Traits '].apply(lambda x: 1 if x.strip().lower() == 'yes' else 0)

    X = data.drop('Traits ', axis=1)
    y = data['Traits ']

    # Encoding categorical features
    categorical_features = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test']
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define classifiers
    classifiers = {
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    # Metrics storage
    conf_matrices = {}
    roc_values = {}
    model_accuracies = {}

    for model_name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        conf_matrices[model_name] = confusion_matrix(y_test, y_pred)
        model_accuracies[model_name] = accuracy_score(y_test, y_pred)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_values[model_name] = (fpr, tpr, roc_auc)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    numeric_data = data.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Confusion Matrix Heatmaps
    st.write("### Confusion Matrix Heatmaps")
    fig, axes = plt.subplots(1, len(classifiers), figsize=(20, 5))
    for ax, (model_name, cm) in zip(axes, conf_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", cbar=False, ax=ax)
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curves
    st.write("### ROC Curves")
    fig, ax = plt.subplots(figsize=(10, 8))
    for model_name, (fpr, tpr, roc_auc) in roc_values.items():
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc="lower right")
    st.pyplot(fig)

   
    # Feature Importance for Random Forest
    st.write("### Feature Importance (Random Forest)")
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    feature_importance = rf_model.feature_importances_

    # Plot feature importance with correct hue assignment
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=X.columns, palette="viridis", ax=ax, hue=X.columns, legend=False)  # Use 'X.columns' as hue
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

    # Model Accuracy Comparison
    st.write("### Model Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use model names as hue
    sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()), palette="viridis", ax=ax, hue=model_accuracies.keys(), legend=False)  # Use 'model_accuracies.keys()' as hue
    ax.set_title("Model Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.9, 1.05)
    st.pyplot(fig)

    #model performance for each algorithm
    print("\nModel Performance Metrics\n")
    for model_name, model in classifiers.items():
        y_pred = model.predict(X_test)
        accuracy = model_accuracies[model_name]
        conf_matrix = conf_matrices[model_name]
        class_report = classification_report(y_test, y_pred)

        print(f"{model_name} Performance Metrics")
        print("-" * 30)
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)
        print("\n")

    # Display Final Model Metrics
    st.write("### Final Model Metrics")
    for model_name, accuracy in model_accuracies.items():
        st.write(f"{model_name}:** Accuracy = {accuracy:.2f}")

    
    # User Input for Prediction
    st.write("### ASD Prediction")
    user_data = {}
    for col in X.columns:
        if col in categorical_features:
            unique_values = data[col].unique()
            user_data[col] = st.selectbox(f"{col}", unique_values)
        else:
            user_data[col] = st.number_input(f"{col}", min_value=float(X[col].min()), max_value=float(X[col].max()))

    if st.button("Predict ASD"):
        user_df = pd.DataFrame([user_data])

        # Encode and scale user data
        for col in categorical_features:
            le = LabelEncoder()
            le.fit(data[col])
            user_df[col] = le.transform(user_df[col])

        user_df_scaled = scaler.transform(user_df)

        #predictions
        prediction = rf_model.predict(user_df_scaled)
        result = "has Autism" if prediction[0] == 1 else "does not have Autism"
        st.write(f"### The model predicts that the person {result}.")
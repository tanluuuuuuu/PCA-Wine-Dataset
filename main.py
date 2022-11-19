import streamlit as st
import sklearn
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np


st.set_page_config(page_title='Wine PCA')

if __name__=='__main__':
    if 'model' not in st.session_state:
        st.session_state['model'] = None
        
    st.title("Using Principal Component Analysis to reduce feature dimensionality on Wine Dataset")
    
    data = load_wine()
    
    st.header("Dataset overview")
    # st.write(data)
    df = pd.DataFrame(data=data['data'], columns=data['feature_names'])
    df2 = pd.DataFrame({'label': data['target']})
    df = pd.concat([df, df2], axis=1)
    st.write(df)

    st.text(f"Number of rows: {data['data'].shape[0]}")
    st.text(f"Number of columns (features): {data['data'].shape[1]}")
    
    st.header("Feature Dimensionality Reduction")
    n_comp = st.slider("Desire number of features", 1, min(data['data'].shape[0], data['data'].shape[1]))
    pca = PCA(n_components=int(n_comp))
    data_train = pca.fit_transform(data['data'])
    
    st.text("Data train overview after reducing feature dimensionality")
    st.write(data_train)
    
    train_size = st.slider("Train size", min_value=1, max_value=90)
    
    X_train, X_test, y_train, y_test = train_test_split(data_train, data['target'], train_size=float(train_size)/100, random_state=7)
    st.text(f"Train samples: {X_train.shape[0]}")
    st.text(f"Test samples: {X_test.shape[0]}")
    
    if (st.button("Train model")):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        st.session_state['model'] = model
        prediction = model.predict(X_test)

        st.write(f'Model training score: {model.score(X_test, y_test)}')
        st.write(f'Model precision score: {precision_score(y_test, prediction, average="micro")}')
        st.write(f'Model recall score: {recall_score(y_test, prediction, average="micro")}')
        st.write(f'Model f1 score: {f1_score(y_test, prediction, average="micro")}')
        st.text("Confusion matrix")
        st.write(confusion_matrix(y_test, prediction))

    if st.session_state['model'] is None:
        st.stop()
    
    st.header("Prediction space")
    cols = st.columns(3)
    predict_data = {}
    for (index, feature) in enumerate(data['feature_names']):            
        number = cols[index % 3].number_input(feature)
        predict_data[feature] = number
        
    if (st.button("Predict")):
        predict = []
        for feature in data['feature_names']:
            predict.append(predict_data[feature])
        matrix = pca.transform(np.array([predict]))
        res = st.session_state['model'].predict(matrix)
        st.write(f"Type of wine: class {res[0]}")
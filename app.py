import pandas as pd
import numpy as np
import gensim
import gensim.downloader as api
import tensorflow
import joblib
import keras

wv= api.load('word2vec-google-news-300')
model_FFNN = keras.models.load_model('spam_email_classifier_FFNN_model.h5')
def vectorize_review(tokens, model):
    word_vectors = []
    for token in tokens:
        if token in model.key_to_index:  # Check if the token exists in the Word2Vec model's vocabulary
            word_vectors.append(model[token])

    if len(word_vectors) == 0:
        # If no valid word vectors were found, return a zero vector
        return np.zeros(model.vector_size)

    # Average the word vectors to get a single vector for the entire review
    review_vector = np.mean(word_vectors, axis=0)
    return review_vector

def predict_spam(email: str):
    # Preprocess the email - You may need to apply the same preprocessing steps as your training data

    pre_processed_email = gensim.utils.simple_preprocess(email)

    # Transform the email using the vectorizer

    email_vec = vectorize_review(pre_processed_email, wv)
    email_vec = np.reshape(email_vec, (1, -1))

    # Predict using the FFNN model
    prediction = model_FFNN.predict(email_vec)

    # Return whether it's spam (1) or not spam (0)
    if prediction > 0.5:
        return "Spam"
    else:
        return "Not Spam"



import streamlit as st

# Streamlit UI elements
st.title("Know Your Email is Spam or NOT")

email_input = st.text_area("Enter the email content:")

if email_input:
    result = predict_spam(email_input)
    st.write(f"The email is: {result}")






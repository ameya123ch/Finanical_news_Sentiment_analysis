import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


st.title("Financial News Sentiment Analysis")
st.write("This model will tell you if a financial news is positive or negative by using Natural Language Processing")
with st.spinner("Loading Model...."):
    new_model = tf.keras.models.load_model("best_model.h5",custom_objects={"KerasLayer": hub.KerasLayer})



pred_news_text = st.text_input("Enter your Financial News")
pred_prob = new_model.predict([pred_news_text])
pred_label = tf.squeeze(tf.round(pred_prob)).numpy()


if st.button("Check"):
    st.subheader("AI thinks that ...")
    if pred_label > 0:

        st.write(
            f"It's a Positive News. Confidence Level is {np.round(pred_prob, 3)}%  It's real news, you can trust it.")
    else:
        st.write(
            f"It's a Negative News. Confidence Level is {np.round(100 - pred_prob, 3)}%  Beware!! It's a Fake News.")








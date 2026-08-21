import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("model.h5")
tokenizer = joblib.load("tokenizer.pkl")


# Prediction function
def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200)

    prediction = model.predict(padded_sequence, verbose=0)

    probability = float(prediction[0][0])

    if probability > 0.5:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😞"

    return sentiment, probability


# Streamlit UI
st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and the LSTM model will predict its sentiment.")

review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was fantastic and amazing!"
)

if st.button("Analyze Sentiment"):
    if review.strip():
        sentiment, probability = predictive_system(review)

        st.subheader("Prediction")
        st.success(sentiment)

        st.write(f"Prediction probability: **{probability:.2%}**")
    else:
        st.warning("Please enter a movie review.")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# Safe file loading
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cleaned_df = pd.read_csv(os.path.join(BASE_DIR, "cleaned_data.csv"))
encoded_df = pd.read_csv(os.path.join(BASE_DIR, "encoded_data.csv"))

cleaned_df['cost'] = (
    cleaned_df['cost']
    .astype(str)
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)

cleaned_df['cost'] = pd.to_numeric(cleaned_df['cost'], errors='coerce')
cleaned_df['cost'] = cleaned_df['cost'].fillna(0)

with open(os.path.join(BASE_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

# Final NaN safety
encoded_df = encoded_df.apply(pd.to_numeric, errors='coerce').fillna(0)

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.set_page_config(page_title="Restaurant Recommendation System", layout="wide")
st.title("🍽️ Restaurant Recommendation System")

menu = st.sidebar.radio(
    "Navigation",
    ["Cleaned Data", "Encoded Data", "Recommend Restaurants"]
)

# ----------------------------------
# Show Cleaned Data
# ----------------------------------
if menu == "Cleaned Data":
    st.subheader("🧹 Cleaned Dataset")
    st.dataframe(cleaned_df.head(50), use_container_width=True)
    st.info(f"Total records: {cleaned_df.shape[0]}")

# ----------------------------------
# Show Encoded Data
# ----------------------------------
elif menu == "Encoded Data":
    st.subheader("🔢 Encoded Dataset")
    st.dataframe(encoded_df.head(50), use_container_width=True)
    st.info(f"Total features: {encoded_df.shape[1]}")

# ----------------------------------
# Recommendation System
# ----------------------------------
else:
    city = st.selectbox("Select City", sorted(cleaned_df['city'].unique()))
    cuisine = st.selectbox("Select Cuisine", sorted(cleaned_df['cuisine'].unique()))
    rating = st.slider("Minimum Rating", 1.0, 5.0, 3.5)
    cost = st.slider(
        "Maximum Cost",
        int(cleaned_df['cost'].min()),
        int(cleaned_df['cost'].max()),
        int(cleaned_df['cost'].median())
    )

    def recommend_restaurants(city, cuisine, rating, cost, top_n=5):
        input_cat = pd.DataFrame([[city, cuisine]], columns=['city', 'cuisine'])
        input_cat_encoded = encoder.transform(input_cat)

        input_num = np.array([[rating, 0, cost]], dtype=float)

        input_vector = np.concatenate([input_num, input_cat_encoded], axis=1)
        input_vector = np.nan_to_num(input_vector)

        similarity = cosine_similarity(input_vector, encoded_df.values)[0]
        top_idx = similarity.argsort()[::-1][1:top_n+1]

        return cleaned_df.iloc[top_idx]

    if st.button("🔍 Get Recommendations"):
        results = recommend_restaurants(city, cuisine, rating, cost)
        st.subheader("✨ Recommended Restaurants")
        st.dataframe(
            results[['name', 'city', 'cuisine', 'rating', 'cost', 'address']],
            use_container_width=True
        )


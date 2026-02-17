🎬 Movie Recommendation System with Neural Network
📌 Project Overview

This project implements a content-based Movie Recommendation System using a two-tower neural network architecture built with TensorFlow. The model leverages user-specific genre preferences and structured movie features to predict ratings for unseen movies and generate personalized recommendations.

The system learns latent representations (embeddings) for users and movies, enabling it to estimate how likely a user is to enjoy a particular film.

🚀 Features

Data Loading & Exploration

Reads movies.csv and ratings.csv

Performs exploratory analysis

Feature Engineering

One-hot encoding of movie genres

Extraction of movie release year

Calculation of average movie ratings

Construction of user genre preference vectors

Data Preprocessing

Feature scaling using StandardScaler

Target normalization using MinMaxScaler

Neural Network Architecture

Two-tower model (User & Movie networks)

Dense layers: 128 → 64 units

32-dimensional embeddings

Dot-product interaction for rating prediction

Optimized using Adam with Mean Squared Error loss

Personalized Recommendations

Generates Top-10 recommendations for new users based on genre preferences

📂 Dataset

This project uses the MovieLens dataset provided by GroupLens Research for research and educational purposes.

Files used:

movies.csv — Movie IDs, titles, genres

ratings.csv — User ratings with timestamps

🛠 Setup & Usage
Environment

Designed to run in Google Colab.

Dependencies

pandas

numpy

tensorflow

scikit-learn

Running the Notebook

Upload movies.csv and ratings.csv

Run all notebook cells sequentially

Modify new_user_dict to test different user preferences

Re-run inference cells to generate recommendations

🧠 Model Architecture

The system uses a two-tower neural network:

User Network → Processes user genre preferences and average rating

Movie Network → Processes movie features (genres, year, average rating)

Prediction → Dot product of normalized embeddings

This design enables scalable and flexible recommendation generation.

🔮 Example Output

For a user with high preference for Thriller and Sci-Fi, the model generates ranked movie recommendations with predicted ratings.

(Results may vary slightly depending on retraining.)

🔧 Future Improvements

Hybrid recommendation (content + collaborative filtering)

Temporal modeling of user preferences

Cold-start handling

Advanced architectures (RNNs / Transformers)

Recommendation explainability

This project demonstrates an end-to-end ML pipeline, from feature engineering and preprocessing to neural network modeling and personalized inference.

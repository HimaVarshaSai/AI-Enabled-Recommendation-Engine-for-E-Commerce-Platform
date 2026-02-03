from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load PKL files
with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)

with open("product_details.pkl", "rb") as f:
    product_details = pickle.load(f)

# Recommendation function


def recommend_products(customer_id, n=5):
    if customer_id not in user_item_matrix.index:
        return []

    # Get predicted scores
    user_ratings = user_item_matrix.loc[customer_id].values.reshape(1, -1)
    predictions = model.inverse_transform(model.transform(user_ratings))[0]
    scores = pd.Series(predictions, index=user_item_matrix.columns)

    # Exclude already purchased products
    purchased = user_item_matrix.loc[customer_id] > 0
    scores[purchased] = -np.inf

    # Get top-N products
    top_products = scores.sort_values(ascending=False).head(n).index

    # Prepare final recommendations
    recommendations = product_details[
        product_details['StockCode'].isin(top_products)
    ].drop_duplicates(subset='StockCode').head(n).to_dict(orient='records')

    return recommendations


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    error = None

    if request.method == "POST":
        try:
            customer_id = int(request.form["customer_id"])
            recommendations = recommend_products(customer_id)

            if not recommendations:
                error = "Customer ID not found or no recommendations available."

        except ValueError:
            error = "Please enter a valid numeric Customer ID."

    return render_template(
        "index.html",
        recommendations=recommendations,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)

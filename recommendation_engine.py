import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class RecommendationEngine:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def generate_recommendations(self, query, num_recommendations=5):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['description'].values.astype('U'))

        query_vector = tfidf_vectorizer.transform([query])

        cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
        related_indices = cosine_similarities.argsort()[::-1]

        recommendations = []
        for i in range(num_recommendations):
            index = related_indices[i]
            recommendations.append(self.data['product'][index])

        return recommendations

# Example usage:
data_path = 'path/to/your/data.csv'
query = 'comfortable running shoes'

recommender = RecommendationEngine(data_path)
recommendations = recommender.generate_recommendations(query)

print("Recommendations:")
for product in recommendations:
    print(product)

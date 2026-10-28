from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit
def solver(ans, c_ans):
    vet = TfidfVectorizer()
    vet_matrix = vet.fit_transform([ans, c_ans])
    similaraty = expit(cosine_similarity(vet_matrix[0], vet_matrix[1]))
    return similaraty[0][0]
        
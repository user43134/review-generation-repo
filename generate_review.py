from main import generate_review_from_faiss

query = "уютное кафе с вкусной едой"
category = "Кафе"
rating = 5
keywords = ["вкусная еда", "уютная атмосфера"]

new_review = generate_review_from_faiss(query, category, rating, keywords)
print(f"Сгенерированный отзыв:\n{new_review}")
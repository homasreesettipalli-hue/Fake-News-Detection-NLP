import pickle

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

news = input("Enter News: ")

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)

if prediction[0] == 1:
    print("\nREAL NEWS")
else:
    print("\nFAKE NEWS")
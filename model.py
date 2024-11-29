import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import numpy as np
import nltk


nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Đọc dữ liệu 
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Gán nhãn 'ham' = 0, 'spam' = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Chia dữ liệu thành train/test
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Các từ phổ biến
custom_stopwords = stopwords.words('english') + ['hi', 'hello', 'hell']
2
# Tạo và huấn luyện TfidfVectorizer với N-grams (1,3)
vectorizer = TfidfVectorizer(stop_words=custom_stopwords, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# huấn luyện mô hình
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# # Dự đoán và đánh giá trên tập test (Hold-out)
# y_pred = model.predict(X_test_tfidf)
# print(f"Độ chính xác trên dữ liệu thử nghiệm: {accuracy_score(y_test, y_pred)}")
# print("Báo cáo phân loại:\n", classification_report(y_test, y_pred))
# print("Ma trận nhầm lẫn:\n", confusion_matrix(y_test, y_pred))

print(nltk.data.path)


with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("Naive Bayes model and vectorizer have been saved successfully.")


# Learning Curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Điểm huấn luyện")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Điểm xác thực chéo")

    plt.legend(loc="best")
    return plt

# Vẽ Learning Curve cho mô hình Naive Bayes
plot_learning_curve(model, "Learning Curve for Naive Bayes", X_train_tfidf, y_train, cv=5)
plt.show()

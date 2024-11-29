import streamlit as st 
import pickle
import nltk 
import string 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

nltk.download('punkt_tab')
nltk.download('stopwords')

image = Image.open('mail.png')
st.image(image, caption='EMAIL')


model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb')) 


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

st.title('Lọc thư rác')

input_sms = st.text_input('Nhập tin nhắn ')

option = st.selectbox("Bạn nhận tin nhắn từ :", ["Email", "SMS", "khác"])

if st.button('Nhấp để dự đoán'):
    transform_sms = transform_text(input_sms)
    vector_input = vectorizer.transform([transform_sms])  
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')


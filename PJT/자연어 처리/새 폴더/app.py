import requests
from dotenv import load_dotenv
import os

# from flask import Flask, render_template, request, redirect, url_for, flash

# app = Flask(__name__)
# app.secret_key = 'your_secret'

load_dotenv()

client_id = os.getenv("3MLkPVZ7rfc6zebQWeXu")
client_secret = os.getenv("a8sFriCQST")

url = 'http://openapi.naver.com/v1/search/blog.json'

query = '강남피부'

headers = {
    'X-Naver-Client-Id': client_id,
    'X-Naver-Client-Secret': client_secret
}

response = requests.get(url, headers = headers, params = { 'query' : query, 'display': 20 })

if response.status_code == 200:
    data = response.json()
    for item in data['items']:
      print(f"제목: {item['title']}")
      print(f"링크: {item['link']}" )
      print(f"요약: {item['description']}")


# response = requests.get(url, headers=headers, params={'query': query})

# if response.status_code == 200:
#     data = response.json()
#     print(data)
#     for item in data['items']:
#       print(f"제목: {item['title']}")
#       print(f"링크: {item['link']}" )
#       print(f"요약: {item['description']}")

# if response.status_code == 200:
#     data = response.json()
#     print(data)
#     for item in data['items']:
#       print(f"제목: {item['title']}")
#       print(f"링크: {item['link']}" )
#       print(f"요약: {item['description']}")


# @app.route('/', method=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         keyword = request.form['keyword']


# if __name__ == '__main__':
#     app.run(debug=True)


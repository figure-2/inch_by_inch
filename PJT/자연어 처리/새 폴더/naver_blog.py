import requests
from dotenv import load_dotenv
import os


# client_id = os.getenv('3MLkPVZ7rfc6zebQWeXu')
# client_secret = os.getenv('Vn93sKlXZs')

client_id = os.getenv('3MLkPVZ7rfc6zebQWeXu')
client_secret = os.getenv('a8sFriCQST')


url = 'https://openapi.naver.com/v1/search/blog.json'

query = '강남역_피부과_추천'

headers = {
    'X-Naver-Client-Id' : client_id,
    'X-Naver-Client-Secret' : client_secret,
}

response = requests.get(url, headers=headers, params={'query': query})

if response.status_code == 200:
    data = response.json()
    print(data)
    for item in data['items']:
      print(f"제목: {item['title']}")
      print(f"링크: {item['link']}" )
      print(f"요약: {item['description']}")
      

# > GET /v1/search/blog.xml?query=%EB%A6%AC%EB%B7%B0&display=10&start=1&sort=sim HTTP/1.1
# > Host: openapi.naver.com
# > User-Agent: curl/7.49.1
# > Accept: */*
# > X-Naver-Client-Id: {애플리케이션 등록 시 발급받은 클라이언트 아이디 값}
# > X-Naver-Client-Secret: {애플리케이션 등록 시 발급받은 클라이언트 시크릿 값}

# curl  "https://openapi.naver.com/v1/search/blog.xml?query=%EB%A6%AC%EB%B7%B0&display=10&start=1&sort=sim" \
#     -H "X-Naver-Client-Id: {애플리케이션 등록 시 발급받은 클라이언트 아이디 값}" \
#     -H "X-Naver-Client-Secret: {애플리케이션 등록 시 발급받은 클라이언트 시크릿 값}" -v



# from selenium import webdriver
# import time

# driver = webdriver.Chrome()

# driver.get(url)

# time.sleep(30)
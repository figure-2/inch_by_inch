import requests
from bs4 import BeautifulSoup
import pyautogui

keyword = pyautogui.prompt("네이버 api에서 플레이스 하고 싶어")
# keyword = input("검색어를 입력하세요>>>")
lastpage = pyautogui.prompt("5")
pageNum = 1
for i in range(1, int(lastpage) * 10, 10):
    print(f"{pageNum}페이지 입니다. ======================================")
    response = requests.get(f"https://search.naver.com/search.naver?where=news&sm=tab_jum&query={keyword}&start={i}") # 검색어가 프로그램 실행될 때마다 자동으로 바뀌게 됨
    html = response.text

    soup = BeautifulSoup(html, 'html.parser') # 번역
    links = soup.select(".news_tit")  # CSS 선택자 = 가져오고 싶은 태그를 선택하는 것    # 결과는 리스트
    # print(links)

    for link in links:
        title = link.text          # 태그 안에 텍스트 요소를 가져온다
        url = link.attrs['href']   # href의 속성값을 가져온다
        print(title, url)
    
    pageNum = pageNum + 1
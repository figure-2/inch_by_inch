-- SQL 기본 정리

SELECT 문

SELECT [DISTINCT] 열 이름 [ OR 별칭 (alias)]
FROM 테이블 이름
[WHERE 조건식]
[ORDER BY 열 이름 [ASC or DESC]];

-- SELECT, FROM => 예약어
-- 열 이름, 테이블 => 필수 입력
-- [] 사이에 있는 것 => 선택사항
-- ; => SQL문이 끝났음을 의미

전체 데이터 조회하기

SELECT ~ FROM => 테이블의 모든 정보 출력

SELECT *
FROM A;
-- * 출력하는 열 (column), *은 모든 열을 출력해서 보고 싶을때 사용
-- FROM => ~ 테이블로 부터

원하는 열만 조회하고 정렬하기


  
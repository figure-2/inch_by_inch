SQL 기본 정리

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

    ex) A테이블에서 ㄱ,ㄴ,ㄷ만 출력

    SELECT ㄱ,ㄴ,ㄷ -- ㄱ,ㄴ,ㄷ 출력하려는 열
    FROM A; -- A 참조하는 테이블

    ORDER BY 명령문을 사용하면 정렬 순서를 변경 할 수 있다.
    => ORDER BY 열 이름 [ASC or  DESC] -- ASC 오름차순 (기본 정렬 방법), DESC 내림차순 정렬

    ex) 위에 SQL 문에서 출력되는것을 ㄱ기준으로 내림차순으로 정렬하기

    SELECT ㄱ,ㄴ,ㄷ
    FROM A
    ORDER BY ㄱ DESC;

  중복된 출력 값 제거하기

    => DISTINCT : 중복된 행을 제거한 후 출력, 중복된 행을 제거하고 싶은 열 앞에 작성

    ex) A 테이블에서 중복 값이 생기지 않도록 1를 출력하세요

    SELECT DISTINCT 1
    FROM A;

  SQL문의 효율성을 높이기 위해 열 이름을 변경하고 싶을 때(별칭) 사용

    => AS : 열 이름을 변경 하고 싶을때 사용
      SELECT 열 이름 AS 별칭 -- 별칭은 변경할려는 이름

      ex) A테이블에서 ㄱ은 '가'로 ㄴ은 '나'로 ㄷ은 '다'로 변경해서 출력

      SELECT ㄱ AS 가, ㄴ AS 나, ㄷ AS 다
      FROM A;

  데이터 값 연결 하기

    => 연결 연산자 ||

      SELECT 컬럼1, 컬럼2,..., 컬럼N
      FROM 테이블명
      WHERE 조건식1 OR 조건식2;

      SELECT 컬럼1, 컬럼2,..., 컬럼N
      FROM 테이블명
      WHERE 조건식1 || 조건식2;
      
      MYSQL 에서는 아래처럼 해야됨 CONCAT 사용
      SELECT CONCAT(컬럼1, 컬럼2)
      FROM 테이블명;

      SELECT CONCAT(컬럼1, ' ', 컬럼2)
      FROM 테이블명;

  산술 처리하기 : 데이터 값끼리 계산
  
    => 산술 연산자 : + , - , *, /

      ex) A 테이블에서 ㄱ, ㄴ,ㄷ에 500을 더한 값, 100을 뺀 값, 10%를 추가해서 2로 나눈 값을 출력

      SELECT ㄱ, ㄴ, ㄷ+500, ㄷ-100, (ㄷ*1.1)/2
      FROM A;
    
    
        
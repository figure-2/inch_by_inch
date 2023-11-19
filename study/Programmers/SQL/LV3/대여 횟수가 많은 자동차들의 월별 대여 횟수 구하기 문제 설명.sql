# SELECT
#     DATE_FORMAT(START_DATE, '%c') MONTH,
#     CAR_ID,
#     COUNT(HISTORY_ID) AS RECORDS
# FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY
#     WHERE(DATE_FORMAT(START_DATE, '%Y-%m') BETWEEN '2022-08' AND '2022-10')

# GROUP BY MONTH, CAR_ID;

#---------------------------------------------------

# SELECT 
#     DATE_FORMAT(START_DATE, '%c') MONTH, 
#     CAR_ID, 
#     COUNT(HISTORY_ID) AS RECORDS

# FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY
#     WHERE (DATE_FORMAT(START_DATE, '%Y-%m') BETWEEN '2022-08' AND '2022-10')
    
# GROUP BY CAR_ID
# HAVING COUNT(HISTORY_ID) >= 5;

#---------------------------------------------------

SELECT
    MONTH(START_DATE) AS MONTH,
    CAR_ID,
    COUNT(HISTORY_ID) AS RECORDS
FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY
    WHERE CAR_ID IN (
        SELECT
            CAR_ID
        
        FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY
            WHERE (DATE_FORMAT(START_DATE, '%Y-%m') BETWEEN '2022-08' AND '2022-10')
        
        GROUP BY CAR_ID
        HAVING COUNT(HISTORY_ID) >= 5
    ) AND (DATE_FORMAT(START_DATE, '%Y-%m') BETWEEN '2022-08' AND '2022-10')

GROUP BY MONTH, CAR_ID
HAVING RECORDS > 0
ORDER BY MONTH ASC, CAR_ID DESC;



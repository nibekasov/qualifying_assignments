--1.

SELECT 
	num
FROM
	Dep
/*Прописываю status != 'сlose', так как новые вклады тоже активны
В случае если они пустые - в ответ не попадут, как и закрытые 
там по логике нулевые суммы*/
WHERE
	status != 'close' AND  
	balance >= 10000 AND 
	(currency = 'EUR' OR
	(currency = 'USD' and balance >= 11000));




--2.

--Включаем FEMALE для более быстрой работы так как нет мужчин с именем Алина
SELECT
	client_id
FROM 
	Contact
WHERE 
	gender = 'FEMALE' AND
	client LIKE '%Алина %'
ORDER BY 
	age DESC;

--3
SELECT 
	AVG(age) as avg_age
FROM 
	Contact;

/* Если бы мы хотели посмотреть только средний возраст активных клиентов

SELECT AVG(active_users.age)
FROM (SELECT  distinct c.client
FROM Contact as c 
LEFT JOIN Dep as d
	ON d.client = c.client_id
WHERE status = 'active') as active_users;
 
 */

--4
SELECT distinct
	split_part(c.client,' ',2) as Name
	/* Можно использовать для решения задачи про Алину
	так как в банке нужно ФИО( в случае отсутвия отчества проблемы не будет)
	*/
FROM 
	Dep as d 
LEFT JOIN 
	Contact AS c 
	ON d.client = c.client_id	

WHERE 
	age > 25 AND
	balance > 18000;

--5


/*
 Если бы у нас была таблица с обменными курсами и тикетом валюты можно было бы получить из нее
 поджойнив с основным запросом
 пусть обменные курсы 70,77,87 для доллара, евро, фунта
 По скринам видно только 4 вида вкладов - если их больше
 можно было бы учесть самые торгуемые валюты на биржах
 Так же можно создать свою временную базу данных для этого запроса, если есть права 
 */
SELECT 
	--c.client AS FullName, имя может не быть уникально, как у Алины - стоит группировать по id клиента
	split_part(c.client,' ',2) AS Name,
	SUM(CASE 
			WHEN currency = 'UDS' THEN balance*70
			WHEN currency = 'EUR' THEN balance*77
			WHEN currency = 'GBP' THEN balance*87
		    ELSE balance 
	END) as balance_RUR
FROM 
	Dep as d 
LEFT JOIN 
	Contact AS c 
	ON d.client = c.client_id	
	
GROUP BY 1,c.client_id;

--6

SELECT d.client as unique_id
FROM 
	Dep as d 
left join 
	Contact AS c 
	ON d.client = c.client_id
WHERE c.client_id is Null

UNION 

SELECT c.client_id as unique_id
FROM
	Contact AS c
left join 
	Dep AS d 
	ON d.client = c.client_id
WHERE d.client is Null;	

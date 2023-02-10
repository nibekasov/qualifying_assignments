SELECT distinct p.id,p.contest_id,p.code

FROM problems as p
LEFT JOIN submissions as s 
	ON s.problem_id = p.id
WHERE s.success = True
GROUP BY p.id,p.contest_id,p.code
HAVING count(distinct user_id) >=2
	
ORDER BY p.id asc;


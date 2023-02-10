-- Решение второй 
SELECT participated.id,
	   participated.name,
	   CASE 
	   		when solved_at_least_one_contest_count is null then 0
	   		else solved_at_least_one_contest_count 
	   END as solved_at_least_one_contest_count,
	   take_part_contest_count

FROM(
Select u.id,u.name,count(distinct contest_id) as solved_at_least_one_contest_count


FROM problems as p
LEFT JOIN submissions as s 
	ON s.problem_id = p.id
/*LEFT JOIN contest as c
	ON p.contest_id = c.id*/
LEFT JOIN users as u
	ON u.id = s.user_id
WHERE s.success = True
GROUP BY u.id) as solved


RIGHT JOIN (SELECT u.id,u.name,count(distinct contest_id) as take_part_contest_count
FROM problems as p
LEFT JOIN submissions as s 
	ON s.problem_id = p.id
/*LEFT JOIN contest as c
	ON p.contest_id = c.id*/
RIGHT JOIN users as u
	ON u.id = s.user_id
GROUP BY u.id) as participated
	ON  participated.id = solved.id
ORDER BY 3 desc,4 desc,1 asc;




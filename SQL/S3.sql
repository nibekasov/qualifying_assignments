SELECT rank() over(order by problem_count desc, latest_successful_submitted_at asc),
	   fin.*
FROM(SELECT 
	   ls.user_id,
	   u.name as user_name,
	   ls.problem_count,
	   max(case 
			when success is True then latest_successful_submitted_at
			else Null
	       end) as latest_successful_submitted_at

FROM (Select 
	count(case when success is True then problem_id end) over (partition BY user_id) as problem_count,
	user_id,
	success,
	min(case 
		when success is True then s.submitted_at
		else Null
	end) as latest_successful_submitted_at

FROM problems as p
LEFT JOIN submissions as s 
	ON s.problem_id = p.id
LEFT JOIN users as u
	ON u.id = s.user_id
WHERE contest_id = (SELECT distinct contest_id
		    FROM problems
		    order by contest_id desc
		    group by problem_id,2,3) as ls
LEFT JOIN users  as u
	ON u.id = ls.user_id 
WHERE (latest_successful_submitted_at is not null and problem_count != 0) or problem_count = 0

GROUP BY 1,2,3) as fin
ORDER by 1,user_id 
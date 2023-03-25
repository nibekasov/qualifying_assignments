--№1
SELECT 
    name,
    COUNT(trip) AS count
    
FROM Passenger 
JOIN Pass_in_trip 
  ON Passenger.id=Pass_in_trip.passenger
  
GROUP BY passenger

HAVING COUNT(trip) > 0

ORDER BY COUNT(trip) DESC,
         name ASC ;
--№2

SELECT 
    DISTINCT TIMEDIFF( (SELECT 
                            end_pair 
                        FROM Timepair 
                        WHERE id=4), 
                        (SELECT 
                             start_pair 
                         FROM Timepair 
                         WHERE id=2) ) AS time 
FROM Timepair;

--№3

SELECT 
    Rooms.* 
FROM Rooms 
JOIN Reservations 
  ON Rooms.id=Reservations.room_id 
WHERE WEEK(Reservations.start_date, 1) = 12 
  AND YEAR(Reservations.start_date)=2020;

--№4

SELECT 
    classroom 
FROM Schedule 
GROUP BY classroom 
HAVING COUNT(classroom) >= ALL( (SELECT 
                                     COUNT(classroom) 
                                 FROM Schedule 
                                 GROUP BY classroom
                                 ORDER BY COUNT(classroom) DESC ) );
--№5
WITH distinct_dates as (SELECT 
                            distinct date
                        FROM income_o ) ,
     intervals as ( SELECT 
                        date dt1,
                        lead(date) over(order by date) as dt2 
                    FROM distinct_dates ) 
SELECT 
    coalesce(sum(out),0),
    dt1,
    dt2 
FROM intervals 
LEFT JOIN Outcome_o 
  on Outcome_o.date > dt1 and Outcome_o.date <= dt2 
WHERE dt2 is not null 
GROUP BY dt1, dt2;

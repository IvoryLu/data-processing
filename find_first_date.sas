proc sql;
select max(Admission_Date_notime) as maxdate format = date9., min(Admission_Date_notime) as mindate format = date9.
from C.Afcohort_total;
quit;

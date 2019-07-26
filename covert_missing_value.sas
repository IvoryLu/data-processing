data yourdata;
   set yourdata;
   array change _numeric_;
        do over change;
            if change=. then change=0;
        end;
 run ;

 data morb_af;
 set morb_af;
 if af=. then af=0;
 end;
 run;

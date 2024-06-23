set more off
set matsize 5000
program define winsor
sum `1', d
replace `1'=r(p1) if `1'<r(p1) & `1'!=.
replace `1'=r(p99) if `1'>r(p99) &`1'!=. 
end

/*
This do-file uses as inputs:

1. Firm-level information from Compustat North America Fundamentals Annual and Quarterly databases - "compustat" and "compustat_quarterly.dta"
- This database is available through WRDS

2. Information on major natural disaster hitting the US territory come from SHELDUS database - "DisastersRawData.dta"
- DisastersRawData.dta includes the 41 major disasters included in the sample (see Section III.C. and Table I for a description)
- The unit of observation is an individual event hitting a given county (with start date and end date) - counties are identified with fips codes
- zipcodecensus.dta provides the bridge between zipcodes (which identifies location in Compustat and Infogroup) and counties

3. Supplier-customer links are from Compustat Segments - "cslinks.dta"
- cslinks include the gvkey identifier for both the supplier and the customer

4. Additional databases used below
- patents.dta include for each gvkey and year the number of patents issued over the last three years
- hdq.dta includes zipcode location of firms' headquarters from Infogroup database
- birth_date.dta gives entry year in compustat
- sich.dta gives historical sic from compustat
- rauch_sic_compustat.dta
- cpiadj.dta adjusts quarterly sales for inflation 
*/
******************************************************************************************************************************************************

*Construct List of Fips-quarter observation affected by a disaster
use ../data/DisastersRawData, clear
keep fips_code qn monthb yearb
bysort qn fips_code: keep if _n==1
save affectedqn.dta, replace

*Retrieve from Compustat Annual files R&D (used as a measure of specificity), size and roa (used as controls)
use "../data/compustat.dta", clear
destring(gvkey), replace
tsset gvkey fyear
by gvkey: gen l_aty=at[_n-3] if fyear==fyear[_n-3]+3
by gvkey: gen l_roay=oibdp[_n-3]/at[_n-3] if fyear==fyear[_n-3]+3
sort gvkey fyear
rename fyear fyearq
gen rd=xrd/sale
by gvkey: gen l_rd=L2.rd
keep gvkey l_aty l_roay fyearq l_rd 
duplicates drop
save controls, replace

**********************************************************************************
*Quarterly variables
**********************************************************************************

use "../data/compustat_quarterly", clear
destring gvkey, replace
sort gvkey datadate saleq
by gvkey datadate: keep if _n==1

gen year=year(datadate)
gen month=month(datadate)
gen monthn=12*(year-1970)+month
joinby monthn using "../data/cpiadj"
replace month=3 if month==1|month==2
replace month=6 if month==4|month==5
replace month=9 if month==7|month==8
replace month=12 if month==10|month==11
replace monthn=12*(year-1970)+month
gen qn=int(monthn/3)
sort gvkey qn
by gvkey qn: keep if _n==1

*Import historical sic from compustat
joinby gvkey fyearq using "../data/sich", unm(b)
drop if _merge==2
drop _merge

*birth_date gives entry year in compustat (used to compute firm age)
joinby gvkey using  "../data/birth_date", unm(b)
drop if _merge==2
drop _merge
gen age=fyearq-year(ipodate)
by gvkey: egen minage=min(age)
replace age=. if minage<-1
replace age=fyearq-birth if age==.


joinby gvkey fyearq using "../data/patents", unm(b)
drop if _merge==2
drop _merge
gen sic3=floor(sich/10)

joinby gvkey fyearq using controls, unm(b)
drop if _merge==2
drop _merge

by gvkey: gen l_age=age[_n-12] if qn[_n-12]==qn-12
by gvkey: gen yd_ca2=(cpiadj*saleq)/saleq[_n-4]-1 if qn[_n-4]==qn-4
by gvkey: gen yd_co2=(cogsq*cpiadj)/cogsq[_n-4]-1 if qn[_n-4]==qn-4


*Sample selection 
******************
keep if year(datadate)>=1977&year(datadate)<=2013
*Exclude Financials
destring sic, replace
replace sich=sic if missing(sich)
gen  s=sich>=6000&sich<=6999
egen maxsic=max(s), by(gvkey)
drop if maxsic==1
drop s maxsic

*Retrieve information on Rauch classification
gen sic2=floor(sich/100)
joinby sich year using "../data/rauch_sic_compustat", unm(b)
drop if _merge==2
drop _merge


**Use headquarters zipcode from Infogroup; if missing, use Compustat
*Clean zipcode info in Compustat
destring gvkey, replace
gen addzip2=addzip
gen a = substr(addzip,1,strpos(addzip,"-")-1)
replace addzip=a if strpos(addzip,"-")>0
gen byte notnumeric = real(addzip)==.
replace addzip="" if notnumeric==1
drop notnumeric
destring addzip, replace
drop a
rename addzip zipcode

joinby gvkey year using "../data/hdq", unm(b)
drop if _merge==2
drop _merge
replace zipcode=zipusa if zipusa~=zipcode&!missing(zipusa)
drop state stateusa zipusa
sort gvkey qn
by gvkey: replace zipcode=zipcode[_n-1] if zipcode[_n-1]<. & zipcode==.&gvkey[_n-1]==gvkey
forvalues i=1/100 {
by gvkey: replace zipcode=zipcode[_n+1] if zipcode[_n+1]<. & zipcode==.&gvkey[_n+1]==gvkey
}

*Import fips code corresponding to each zipcode
joinby zipcode using "../data/zipcodecensus", unm(b)
keep if _merge==3
drop _merge
rename fips fips_code
by gvkey qn: keep if _n==1
tsset gvkey qn

*Create duma0, a dummy which equals 1 if a given county hit by a major natural disaster in quarter an
joinby qn fips using affectedqn, unm(b)
drop if _merge==2
gen duma0=_merge==3
drop _merge
by gvkey qn: keep if _n==1

*Create Industry dummies
do "../prog/famafrench"

gen temp=year(datadate) if fqtr==4
sort gvkey fyearq
by gvkey fyearq: egen yearq=max(temp)
drop temp 
sort gvkey qn
drop cpiadj cpi2013 cpi zip_class poname 
compress
save hurricanes, replace

**********************************************************************************
*Base de couples: CUST -> SUP
**********************************************************************************

use "../data/cslinks.dta", clear
gen year=year(srcdate)
gen month=month(srcdate)
replace month=3 if month==1|month==2
replace month=6 if month==4|month==5
replace month=9 if month==7|month==8
replace month=12 if month==10|month==11
gen monthn=12*(year-1970)+month
gen qn=int(monthn/3)
drop if cust==sup
bysort cust_gvkey sup_gvkey year: gen a=_N
keep if a==1
drop a
egen couple=group(cust_gvkey sup_gvkey)
egen minqn_C=min(qn), by(couple)
egen maxqn_C=max(qn), by(couple)
replace minqn_C=minqn_C+1
replace maxqn_C=maxqn_C+4
drop if maxqn_C<minqn_C
keep couple minqn_C maxqn_C sup_gvkey cust_gvkey 
duplicates drop
save tempcouple, replace

use tempcouple, clear
ren cust_gvkey gvkey
joinby gvkey using hurricanes
keep gvkey qn minqn_C maxqn_C sup_gvkey
foreach var of varlist gvkey{
ren `var' cust_`var'
}
keep if qn>=minqn_C & qn<=maxqn_C
rename sup_gvkey gvkey
save tempc, replace

use hurricanes, clear
joinby gvkey qn using tempc, unm(b)
drop if _m==2
drop _m
*Define sup_min and sup_max as the first and last quarter a firm reports another firm as a customer in the Customer Segments files
gen rel=1 if cust_gvkey<. & gvkey<.
egen srel=sum(rel), by(gvkey)
drop if srel==0
drop srel
egen min=min(qn) if rel==1, by(gvkey)
egen sup_min=min(min), by(gvkey)
egen max=max(qn) if rel==1, by(gvkey)
egen sup_max=max(max), by(gvkey)
drop min max rel
bysor gvkey qn: keep if _n==1
save temp_collapse2, replace

*Define specific and non specific suppliers along RD and PATENT dimensions
egen PATENT=xtile(nbiss), by(qn) nq(2)
replace PATENT=PATENT-1
egen RD=xtile(l_rd), by(qn) nq(2)
replace RD=RD-1
keep gvkey qn RD PATENT
rename gvkey sup_gvkey
save supsplit, replace


use tempcouple, clear
ren sup_gvkey gvkey
joinby gvkey using hurricanes
foreach var of varlist fips_code gvkey state sich duma0 latitude longitude con_n lib_n sale{
ren `var' sup_`var'
}
keep if qn>=minqn_C & qn<=maxqn_C
rename cust_gvkey gvkey
sort gvkey sup_gvkey qn
save temps, replace

use hurricanes, clear
joinby gvkey qn using temps, unm(b)
drop if _merge==2
drop _merge

vincenty latitude longitude sup_latitude sup_longitude, hav(dis) 

*Define cust_min and cust_max as the first and last quarter a firm appears as a customer in the Customer Segments files
gen rel=1 if sup_gvkey<. & gvkey<.
egen srel=sum(rel), by(gvkey)
drop if srel==0
drop srel
egen min=min(qn) if rel==1, by(gvkey)
egen cust_min=min(min), by(gvkey)
egen max=max(qn) if rel==1, by(gvkey)
egen cust_max=max(max), by(gvkey)
drop min max rel

*Exclude link if within 300 miles
replace sup_gvkey =. if dis<=300
replace dis=. if missing(sup_gvkey)

foreach var of varlist sup_* couple minqn_C maxqn_C{
replace `var'=. if missing(sup_gvkey)
}
bysort gvkey sup_gvkey qn: keep if _n==1

*Import info on specific and non-specific suppliers along RD and PATENT dimensions
joinby sup_gvkey qn using supsplit, unm(b)
drop if _merge==2
drop _m

*Define specific and non-specific suppliers along RAUCH goods classification dimension
egen M=median(sup_con_n), by(qn)
gen RAUCH=1 if sup_con_n>=M & sup_con_n<. &sup_gvkey<.
replace RAUCH=0 if sup_con_n<M


gen dep=sup_gvkey<.
egen m_dep=max(dep), by(gvkey qn)
egen nbsupplier=sum(dep), by(gvkey qn)

*Create dummy for whether at least one of the firm's supplier is hit by a natural disaster in year-quarter qn
*Compute the number of suppliers in year-quarter qn
egen m_sup_duma0=max(sup_duma0), by(gvkey qn)
replace m_sup_duma0=0 if m_sup_duma0==.
egen nb_sup_duma0=sum(sup_duma0), by(gvkey qn)
replace nb_sup_duma0=0 if nb_sup_duma0==.

*Create dummies for whether at least one of the firm's supplier is hit by a natural disaster in year-quarter qn, conditional on whether the supplier is defined as "specific"
foreach X in PATENT RAUCH RD{
egen `X'_sup_duma0=max(sup_duma0) if `X'==1 , by(gvkey qn)
replace `X'_sup_duma0=0 if `X'_sup_duma0==. 
egen `X'0_sup_duma0=max(sup_duma0) if `X'==0, by(gvkey qn)
replace `X'0_sup_duma0=0 if `X'0_sup_duma0==. 
}

sort gvkey sup_gvkey qn
save tempreg_co, replace

foreach X in PATENT RAUCH RD{
egen `X'_sup_duma0_temp=max(`X'_sup_duma0), by(gvkey qn)
replace `X'_sup_duma0=`X'_sup_duma0_temp 
drop `X'_sup_duma0_temp
egen `X'_sup_duma0_temp=max(`X'0_sup_duma0), by(gvkey qn)
replace `X'0_sup_duma0=`X'_sup_duma0_temp 
drop `X'_sup_duma0_temp
}

sort gvkey qn
by gvkey qn: keep if _n==1
save temp_collapse, replace

**********************************************************************************
*Spillovers: SUP -> CUST -> SUP
**********************************************************************************

use tempcouple, clear
ren sup_gvkey gvkey
joinby gvkey using hurricanes
keep if qn>=minqn_C & qn<=maxqn_C
gen n=1
bysort gvkey qn: egen nbc=sum(n)
drop n
ren gvkey sup2
keep sup2 cust_gvkey couple qn state latitude longitude fips fqtr datadate duma0 famafrench nbc sich
foreach var of varlist latitude longitude state fips fqtr datadate duma0 nbc sich{
ren `var' sup2_`var'
}
rename cust_gvkey gvkey
compress
save temps2, replace

use tempreg_co, clear
*keep customers
keep if qn>=cust_min & qn<=cust_max
*
keep sich latitude longitude sup_sich sup_latitude sup_longitude sup_gvkey year state gvkey sup_fips_code qn sup_state sup_duma0 duma0 PATENT RAUCH RD 

joinby gvkey qn using temps2, unm(b)
drop if _m==1
drop _m
drop if sup2==sup_gvkey

*drop if sup1 sup2 or cust sup2 same state 
vincenty latitude longitude sup2_latitude sup2_longitude, hav(dis) 
vincenty sup2_latitude sup2_longitude sup_latitude sup_longitude, hav(dis2) 

*Exclude links (if within 300 miles)
drop if dis2<=300& sup_gvkey<.&!missing(dis2)
drop if dis<=300&gvkey<.&!missing(dis)

*at least one customer with another supplier
gen yes=gvkey<. & sup_gvkey<.
egen S_CS=max(yes), by(sup2 qn)
drop yes

*at least one affected customer
egen S_CA=max(duma0), by(sup2 qn)

*at least one customer with another supplier affected
gen yes=gvkey<. & sup_gvkey<. & sup_duma0==1
egen S_CSA=max(yes), by(sup2 qn)
drop yes
save temp_spillover, replace

use temp_spillover, clear
keep sup2 gvkey
duplicates drop
save temp, replace

use temp_spillover, clear
keep if sup_duma0==1
keep gvkey qn  
duplicates drop
joinby gvkey using temp
duplicates drop
keep sup2 qn
duplicates drop
joinby sup2 qn using temp_spillover, unm(b)
drop if _m==1
drop _m

foreach X in PATENT RAUCH RD {
*at least one customer with another X supplier
gen yes=gvkey<. & sup_gvkey<. & `X'==1
egen S_CS_`X'=max(yes), by(sup2 qn)
drop yes
*at least one customer with another X-specific supplier affected
gen yes=gvkey<. & sup_gvkey<. & sup_duma0==1 & `X'==1
egen S_CSA_`X'=max(yes), by(sup2 qn)
drop yes
**at least one customer with another X-non-specific supplier affected
gen yes=gvkey<. & sup_gvkey<.  & sup_duma0==1 & `X'==0
egen S_CSA_`X'0=max(yes), by(sup2 qn)
drop yes
}


bysort sup2 sup_gvkey qn: keep if _n==1
gen dep=sup_gvkey<.
egen nbothersupplier=sum(dep), by(sup2 qn)
sort sup2 qn
by sup2 qn: keep if _n==1
drop gvkey
ren sup2 gvkey
keep gvkey S_* qn nbothersupplier
save spillscs, replace

****************************************************************************/
********************MERGE DATASETS
****************************************************************************

use hurricanes, clear
joinby gvkey qn using temp_collapse, unm(b)
drop if _merge==2
drop _merge
joinby gvkey qn using temp_collapse2, unm(b)
drop if _merge==2
drop _merge
joinby gvkey qn using spillscs, unm(b)
drop if _merge==2
drop _merge

sort gvkey qn
forvalues i=1(1)5{
by gvkey: gen duma`i'=1 if duma0[_n-`i']==1&qn[_n-`i']==qn-`i'
by gvkey: replace duma`i'=0 if duma0[_n-`i']~=1&qn[_n-`i']==qn-`i'
}

sort gvkey qn
forvalues i=1/5 {
foreach X in PATENT RAUCH RD{
by gvkey: gen `X'_sup_duma`i'=`X'_sup_duma0[_n-`i'] if qn[_n-`i']==qn-`i'
by gvkey: gen `X'0_sup_duma`i'=`X'0_sup_duma0[_n-`i'] if qn[_n-`i']==qn-`i'
}
}

sort gvkey qn
forvalues i=1(1)5{
by gvkey: gen m_sup_duma`i'=1 if m_sup_duma0[_n-`i']==1&qn[_n-`i']==qn-`i'
by gvkey: replace m_sup_duma`i'=0 if m_sup_duma0[_n-`i']~=1&qn[_n-`i']==qn-`i'
by gvkey: gen nb_sup_duma`i'=nb_sup_duma0[_n-`i'] if qn[_n-`i']==qn-`i'
by gvkey: replace nb_sup_duma`i'=0 if nb_sup_duma`i'==.&qn[_n-`i']==qn-`i'
}

foreach var of varlist S_*{
forvalues i=0(1)5{
by gvkey: gen `var'_a`i'=1 if `var'[_n-`i']==1 &qn[_n-`i']==qn-`i'
by gvkey:  replace  `var'_a`i'=0 if qn[_n-`i']==qn-`i'&`var'[_n-`i']~=1
}
}

*Filters
gen MM=month(datadate)
keep if MM==3 | MM==6 | MM==9 | MM==12
keep if year(datadate)>=1978 & year(datadate)<=2013
keep if ceqq>=0 & saleq>=0

label var duma0 "Disaster hits firm (t)"
label var duma1 "Disaster hits firm (t-1)"
label var duma2 "Disaster hits firm (t-2)"
label var duma3 "Disaster hits firm (t-3)"
label var duma4 "Disaster hits firm (t-4)"
label var duma5 "Disaster hits firm (t-5)"
label var m_sup_duma0 "Disaster hits one supplier (t)"
label var m_sup_duma1 "Disaster hits one supplier (t-1)"
label var m_sup_duma2 "Disaster hits one supplier (t-2)"
label var m_sup_duma3 "Disaster hits one supplier (t-3)"
label var m_sup_duma4 "Disaster hits one supplier (t-4)"
label var m_sup_duma5 "Disaster hits one supplier (t-5)"

*Create industry-year and state-year dummies
gen Y=year(datadate)
egen indqn=group(famafrench Y)
egen stateqn=group(state Y)
save sample, replace

 

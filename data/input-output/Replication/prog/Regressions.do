

clear matrix
clear mata
set maxvar 10000
set matsize 10000
set more off
cap prog drop _all
program define winsor
sum `1', d
replace `1'=r(p1) if `1'<r(p1) & `1'!=.
replace `1'=r(p99) if `1'>r(p99) &`1'!=. 
end

use sample, clear

*A firm is included in the supplier sample for each quarter between three years before the first year and three years after the last year it reports another firm as a customer in the Compustat Segment files
keep if qn>=sup_min-12& qn<=sup_max+12

sort gvkey qn
bysort gvkey: gen l2_nbsup=nbothersupplier[_n-12] if qn==qn[_n-12]+12
replace l2_nbsup=0 if missing(l2_nbsup)

*Windsorize sales growth at the 1st and 99th percentales
winsor yd_ca2

*Keep if non-missing info on size, age, ROA, and number of customers' other suppliers
keep if l_aty<. & l_roay<. & l_age<.&l2_nbsup<.
keep if duma5<.&duma4<.&duma3<.&duma2<.&duma1<.&duma0<.

*Create dummies for firm level characteristics*Year quarter fixed effects
egen supQ3=xtile(l2_nbsup), by(qn) nq(3)
egen assetQ=xtile(l_aty), by(qn) nq(3)
egen roaQ=xtile(l_roay), by(qn) nq(3)
egen ageQ=xtile(l_age), by(qn) nq(3)
xi I.qn*I.assetQ, prefix(Qa_)
xi I.qn*I.roaQ, prefix(Qr_)
xi I.qn*I.ageQ, prefix(Qg_)
xi I.fqtr, prefix(FQ)

gen duma1a4=max(duma1,duma2,duma3,duma4)
 gen S_CSA_X_a1a4=.
gen S_CSA_X0_a1a4=.
foreach X in S_CA S_CSA S_CSA_RAUCH S_CSA_RAUCH0 S_CSA_RD S_CSA_RD0 S_CSA_PATENT S_CSA_PATENT0{
gen `X'_a1a4=max(`X'_a1,`X'_a2,`X'_a3,`X'_a4)
}

label var duma1a4 "Disaster hits firm (t-4,t-1)"
label var S_CA_a1a4 "Disaster hits one customer (t-4,t-1)"
label var S_CSA_a1a4 "Disaster hits one customer's supplier (t-4,t-1)"
label var S_CSA_X_a1a4 "Disaster hits one customer's specific supplier (t-4,t-1)"
label var S_CSA_X0_a1a4 "Disaster hits one customer's non-specific supplier (t-4,t-1)"

*Table III
eststo clear  
eststo: xi: areg yd_ca2 i.fqtr i.qn duma0 duma1 duma2 duma3 duma4 duma5, a(gvkey) cl(gvkey)
eststo: xi: areg yd_ca2 i.fqtr Qa_* Qr_* Qg_* 	duma0 duma1 duma2 duma3 duma4 duma5, a(gvkey) cl(gvkey)
eststo:reg2hdfe  yd_ca2  FQ* Qa_* Qr_* Qg_* duma0 duma1 duma2 duma3 duma4 duma5, id1(gvkey) id2(stateqn) maxiter(10) cluster(gvkey)
eststo:reg2hdfe  yd_ca2  FQ* Qa_* Qr_* Qg_* i.stateqn duma0 duma1 duma2 duma3 duma4 duma5, id1(gvkey) id2(indqn) maxiter(10) cluster(gvkey)
esttab using "../prod/table3.tex", se r2 nolines nogaps nomtitles noconst label b(%5.3f) se(%5.3f) replace keep(duma0 duma1 duma2 duma3 duma4 duma5) ///
prehead(\begin{table}\caption{\bf Natural Disaster Disruptions -- Supplier Sales Growth}\label{suppliers}\begin{footnotesize} ///
This table presents estimates from panel regressions of firms' sales growth relative to the same quarter in the previous year on a dummy indicated whether the firm is hit by a major disaster in the current and each of the previous five quarters.  /// 
All regressions include fiscal-quarter, year-quarter, and firm fixed effects. ///
In columns (2) to (4), we also control for firm-level characteristics (dummies indicating terciles of size, age, and ROA respectively) interacted with year-quarter dummies. In column (3), we include 48 Fama-French industry dummies interacted with year dummies. In column (4) we include state dummies interacted with year dummies. ///
Standard errors presented in parentheses are clustered at the firm-level. Regressions contain all firm-quarters of our supplier sample (described in Table II, Panel B) between 1978 and 2013. *, **, and *** denote significance at the 10\%, 5\%, and 1\%, respectively. ///
\begin{center}\begin{tabular}{l*{4}{c}}\hline\hline&&&&\\&\multicolumn{4}{c}{Sales Growth $(t-4,t)$ }\\\cmidrule(lr){2-5} &&&&\\) ///
prefoot(&&&& \\ "Firm FE & Yes & Yes & Yes & Yes  \\" "Year-Quarter FE & Yes & Yes & Yes & Yes\\" "Industry-Year FE & No & No & Yes & No\\" "State-Year FE & No & No & No & Yes\\" "Size, Age, ROA $\times$ Year-Quarter FE &No &Yes & Yes & Yes \\" &&&&\\) ///
postfoot(&&&& \\ \hline \hline "\end{tabular} \end{center} \end{footnotesize} \end{table}")starlevels("*" 0.10 "**" 0.05 "***" 0.01)alignment(D{.}{.}{-1}) page(dcolumn) nonumber

*Table XI
eststo clear
eststo: xi:  areg yd_ca2 i.qn i.supQ3 i.fqtr Qa_* Qr_* Qg_*  duma1a4 S_CA_a1a4 S_CSA_a1a4, a(gvkey) cl(gvkey)
foreach X in RAUCH RD PATENT{
display "`X'"
replace S_CSA_X_a1a4=S_CSA_`X'_a1a4
replace S_CSA_X0_a1a4=S_CSA_`X'0_a1a4
eststo: xi: areg yd_ca2 i.qn i.supQ3 i.fqtr Qa_* Qr_* Qg_* duma1a4 S_CA_a1a4 S_CSA_X_a1a4 S_CSA_X0_a1a4, a(gvkey) cl(gvkey)
}
esttab using "../prod/table11.tex", se r2 nolines nogaps nomtitles noconst label b(%5.3f) se(%5.3f) replace order(duma1a4 S_CSA_a1_a4 S_CA_a1a4 S_CSA_X_a1a4 S_CSA_X0_a1a4) keep(duma1a4 S_CSA_a1_a4 S_CA_a1a4 S_CSA_X_a1a4 S_CSA_X0_a1a4) ///
prehead(\begin{table} \caption{\bf Horizontal Propagation -- Related Suppliers' Sales Growth}\label{spillovers}\begin{footnotesize} ///
This table presents estimated coefficients from panel regressions of firms' sales growth relative to the same quarter in the previous year on one dummy indicating whether one of the firm's customers' other suppliers is hit by a major disaster in the previous four quarters.  /// 
Columns (2) to (4) split customers' other suppliers into specific and non-specific suppliers. ///
All regressions include two dummies indicating whether the firm itself is hit in the previous four quarters and whether one of the firm's customer is hit in the previous four quarters. ///
All regressions also control for the number of customers' suppliers (dummies indicating terciles of the number of customers' suppliers). ///
All regressions include fiscal-quarter, year-quarter, and firm fixed effects as well as firm-level characteristics (dummies indicating terciles of size, age, and ROA respectively) interacted with year-quarter dummies. ///
Standard errors presented in parentheses are clustered at the firm-level. Regressions contain all firm-quarters of our supplier sample (described in Table II, Panel B) between 1978 and 2013. *, **, and *** denote significance at the 10\%, 5\%, and 1\%, respectively. ///
\begin{center}\begin{tabular}{l*{4}{c}}\hline\hline&&&& \\&\multicolumn{4}{c}{Sales Growth $(t-4,t)$}\\\cmidrule(lr){2-5} &&&& \\Supplier specificity:& &\multicolumn{1}{c}{DIFF.}&\multicolumn{1}{c}{R\&D}&\multicolumn{1}{c}{PATENT}\\\cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5}&&&& \\)prefoot(&&&& \\ "Number of Customers' Suppliers & Yes & Yes & Yes & Yes  \\" "Firm FE & Yes & Yes & Yes & Yes  \\" "Year-Quarter FE & Yes & Yes & Yes & Yes\\" "Size, Age, ROA $\times$ Year-Quarter FE & Yes &Yes & Yes & Yes \\" &&&&\\) ///
postfoot(&&&& \\ \hline \hline "\end{tabular} \end{center} \end{footnotesize} \end{table}")starlevels("*" 0.10 "**" 0.05 "***" 0.01)alignment(D{.}{.}{-1}) page(dcolumn) nonumber


use sample, clear
drop S_* 
*A firm is included in the customer sample for each quarter between three years before the first year and three years after the last year it appears as a customer in the Compustat Segment files
keep if qn>=cust_min-12 & qn<=cust_max+12

bysort gvkey: gen l2_nbsup=nbsupplier[_n-12] if qn==qn[_n-12]+12
replace l2_nbsup=0 if missing(l2_nbsup)

*Windsorize sales growth and COGS growth at the 1st and 99th percentales
winsor yd_co2
winsor yd_ca2

*Keep if non-missing info on size, age, ROA, and number of customers
keep if l_aty<. & l_roay<. & l_age<.&l2_nbsup<.
keep if m_sup_duma5<.&m_sup_duma4<.&m_sup_duma3<.&m_sup_duma2<.&m_sup_duma1<.&m_sup_duma0<.

*Create dummies for firm level characteristics*Year quarter fixed effects
egen assetQ=xtile(l_aty), by(qn) nq(3)
egen roaQ=xtile(l_roay), by(qn) nq(3)
egen ageQ=xtile(l_age), by(qn) nq(3)
xi I.qn*I.assetQ, prefix(Qa_)
xi I.qn*I.roaQ, prefix(Qr_)
xi I.qn*I.ageQ, prefix(Qg_)
xi I.fqtr, prefix(FQ)

*Create dummies for terciles of number of customers
egen supQ3=xtile(l2_nbsup), by(qn) nq(3)
xi I.qn*I.supQ3, prefix(Q3s_)


*Table V
eststo clear  
eststo:xi: quietly    areg yd_ca2 i.fqtr i.qn i.supQ3 m_sup_duma4 duma4 , a(gvkey) cl(gvkey) 
eststo:xi: quietly    areg yd_ca2 i.fqtr i.supQ3 Qa_* Qr_* Qg_* m_sup_duma4  duma4, a(gvkey) cl(gvkey)
eststo: xi: quietly  reg2hdfe  yd_ca2 i.fqtr i.supQ3 Qa_* Qr_* Qg_* m_sup_duma4  duma4 , id1(gvkey) id2(stateqn) maxiter(10) cluster(gvkey)
eststo: xi: quietly  reg2hdfe  yd_ca2 i.fqtr i.stateqn i.supQ3 Qa_* Qr_* Qg_* m_sup_duma4  duma4, id1(gvkey) id2(indqn) maxiter(10) cluster(gvkey)
esttab using "../prod/table5.tex", se r2 nolines nogaps nomtitles noconst label b(%5.3f) se(%5.3f) replace keep(m_sup_duma4 duma4) ///
prehead(\begin{table} \caption{\bf Downstream Propagation -- Baseline }~\label{baseline}\begin{footnotesize} ///
This table presents estimates from panel regressions of firms' sales growth (Panel A) or cost of goods sold growth (Panel B) relative to the same quarter in the previous year on a dummy indicating whether (at least) one of their suppliers is hit by a major disaster in the same quarter of the previous year. ///
All regressions include a dummy indicating whether the firm itself is hit by a major disaster in the same quarter of the previous year as well as fiscal-quarter, year-quarter, and firm fixed effects. ///
All regressions also control for the number of suppliers (dummies indicating terciles of the number of suppliers). ///
In columns (2) to (4), we control for firm-level characteristics (dummies indicating terciles of size, age, and ROA respectively) interacted with year-quarter dummies. In column (3), we include 48 Fama-French industry dummies interacted with year dummies. In column (4), we include state dummies interacted with year dummies. /// 
Regressions contain all firm-quarters of our customer sample (described in Table II, Panel A) between 1978 and 2013. ///
Standard errors presented in parentheses are clustered at the firm-level. *, **, and *** denote significance at the 10\%, 5\%, and 1\%, respectively. ///
\begin{center}\begin{tabular}{l*{4}{c}}\hline\hline&&&&\\Panel A:&\multicolumn{4}{c}{Sales Growth $(t-4,t)$ }\\\cmidrule(lr){1-5} &&&&\\)prefoot(&&&& \\ "Number of Suppliers & Yes & Yes & Yes & Yes  \\" "Firm FE & Yes & Yes & Yes & Yes  \\" "Year-Quarter FE & Yes & Yes & Yes & Yes\\" "Industry-Year FE & No & No & Yes & No\\" "State-Year FE & No & No & No & Yes\\" "Size, Age, ROA $\times$ Year-Quarter FE &No &Yes & Yes & Yes \\" &&&&\\) ///
starlevels("*" 0.10 "**" 0.05 "***" 0.01)alignment(D{.}{.}{-1}) page(dcolumn) nonumber postfoot("")
eststo clear  
eststo:xi: quietly    areg yd_co2 i.fqtr i.qn i.supQ3 m_sup_duma4 duma4 , a(gvkey) cl(gvkey) 
eststo:xi: quietly    areg yd_co2 i.fqtr i.supQ3 Qa_* Qr_* Qg_* m_sup_duma4  duma4, a(gvkey) cl(gvkey)
eststo: xi: quietly  reg2hdfe  yd_co2 i.fqtr i.supQ3 Qa_* Qr_* Qg_* m_sup_duma4  duma4 , id1(gvkey) id2(stateqn) maxiter(10) cluster(gvkey)
eststo: xi: quietly  reg2hdfe  yd_co2 i.fqtr i.stateqn i.supQ3 Qa_* Qr_* Qg_* m_sup_duma4  duma4, id1(gvkey) id2(indqn) maxiter(10) cluster(gvkey)
esttab using "../Prod/table5.tex", se r2 nolines nogaps nomtitles noconst label b(%5.3f) se(%5.3f) append keep(m_sup_duma4 duma4) prehead(\\Panel B:&\multicolumn{4}{c}{Cost of Goods Sold Growth $(t-4,t)$ }\\\cmidrule(lr){1-5} &&&&\\)prefoot(&&&& \\ "Number of Suppliers & Yes & Yes & Yes & Yes  \\" "Firm FE & Yes & Yes & Yes & Yes  \\" "Year-Quarter FE & Yes & Yes & Yes & Yes\\" "Industry-Year FE & No & No & Yes & No\\" "State-Year FE & No & No & No & Yes\\" "Size, Age, ROA $\times$ Year-Quarter FE &No &Yes & Yes & Yes \\" &&&&\\) ///
postfoot(&&&& \\ \hline \hline "\end{tabular} \end{center} \end{footnotesize} \end{table}")starlevels("*" 0.10 "**" 0.05 "***" 0.01)alignment(D{.}{.}{-1}) page(dcolumn) nonumber

*Table VI
eststo clear  
eststo:xi: quietly areg yd_ca2 i.fqtr 		   i.qn i.supQ3 				m_sup_duma0 m_sup_duma1 m_sup_duma2 m_sup_duma3 m_sup_duma4 m_sup_duma5 duma0 duma1 duma2 duma3 duma4 duma5, a(gvkey) cl(gvkey) 
eststo:xi:quietly areg yd_ca2 i.fqtr Qa_* Qr_* Qg_* i.supQ3 				m_sup_duma0 m_sup_duma1 m_sup_duma2 m_sup_duma3 m_sup_duma4 m_sup_duma5 duma0 duma1 duma2 duma3 duma4 duma5, a(gvkey) cl(gvkey)
eststo: xi: quietly  reg2hdfe  yd_ca2 i.fqtr 		i.supQ3 Qa_* Qr_* Qg_*  m_sup_duma0 m_sup_duma1 m_sup_duma2 m_sup_duma3 m_sup_duma4 m_sup_duma5 duma0 duma1 duma2 duma3 duma4 duma5, id1(gvkey) id2(stateqn) maxiter(10) cluster(gvkey)
eststo: xi: quietly  reg2hdfe  yd_ca2 i.fqtr 		i.supQ3 Qa_* Qr_* Qg_*  m_sup_duma0 m_sup_duma1 m_sup_duma2 m_sup_duma3 m_sup_duma4 m_sup_duma5 duma0 duma1 duma2 duma3 duma4 duma5 i.stateqn, id1(gvkey) id2(indqn) maxiter(10) cluster(gvkey)
esttab using "../prod/table6.tex", se r2 nolines nogaps nomtitles noconst label b(%5.3f) se(%5.3f) replace keep(m_sup_duma0 m_sup_duma1 m_sup_duma2 m_sup_duma3 m_sup_duma4 m_sup_duma5 duma0 duma1 duma2 duma3 duma4 duma5) ///
prehead(\begin{table}\caption{\bf Downstream Propagation -- Sales Growth Dynamics}\label{dynamics1}\begin{footnotesize} ///
This table presents estimated coefficients from panel regressions of firms' sales growth relative to the same quarter in the previous year on dummies indicating whether (at least) one of their suppliers is hit by a major disaster in the current and each of the previous five quarters. ///
All regressions include dummies indicating whether the firm itself is hit by a major disaster in the current and each of the previous five quarters, as well as fiscal-quarter, year-quarter and firm fixed effects.  ///
All regressions also control for the number of suppliers (dummies indicating terciles of the number of suppliers). ///
In columns (2) to (4), we control for firm-level characteristics (dummies indicating terciles of size, age, and ROA respectively) interacted with year-quarter dummies. In column (3), we include 48 Fama-French industry dummies interacted with year dummies. In column (4), we include state dummies interacted with year dummies. /// 
Regressions contain all firm-quarters of our customer sample (described in Table II, Panel A) between 1978 and 2013. ///
Standard errors presented in parentheses are clustered at the firm-level. *, **, and *** denote significance at the 10\%, 5\%, and 1\%, respectively. ///
\begin{center}\begin{tabular}{l*{4}{c}}\hline\hline&&&&\\&\multicolumn{4}{c}{Sales Growth $(t-4,t)$ }\\\cmidrule(lr){2-5} &&&&\\) ///
prefoot(&&&& \\ "Number of Suppliers & Yes & Yes & Yes & Yes  \\" "Firm FE & Yes & Yes & Yes & Yes  \\" "Year-Quarter FE & Yes & Yes & Yes & Yes\\" "Industry-Year FE & No & No & Yes & No\\" "State-Year FE & No & No & No & Yes\\" "Size, Age, ROA $\times$ Year-Quarter FE &No &Yes & Yes & Yes \\" &&&&\\) ///
postfoot(&&&& \\ \hline \hline "\end{tabular} \end{center} \end{footnotesize} \end{table}")starlevels("*" 0.10 "**" 0.05 "***" 0.01)alignment(D{.}{.}{-1}) page(dcolumn) nonumber


*Table VIII
gen Xm_duma4=.
gen Xm0_duma4=.
label var Xm_duma4 "Disaster hits one specific supplier (t-4)"
label var Xm0_duma4 "Disaster hits one non-specific supplier (t-4)"
eststo clear 
foreach X in RAUCH RD PATENT{
replace Xm_duma4=`X'_sup_duma4
replace Xm0_duma4=`X'0_sup_duma4
display "`X'"
eststo:xi: quietly    areg yd_ca2 i.qn i.supQ3 i.fqtr Xm0_duma4 Xm_duma4 duma4  , a(gvkey) cl(gvkey)
eststo:xi: quietly    areg yd_ca2 i.qn i.supQ3 i.fqtr Qa_* Qr_* Qg_* Xm0_duma4 Xm_duma4 duma4, a(gvkey) cl(gvkey)
}
esttab using "../Prod/table8.tex", se r2 nolines nogaps nomtitles noconst label b(%5.3f) se(%5.3f) replace keep(Xm0_duma4 Xm_duma4 duma4) ///
prehead(\begin{table}\caption{\bf Downstream Propagation -- Input Specificity}\label{specificity}\begin{footnotesize} ///
This table presents estimates from panel regressions of firms' sales growth relative to the same quarter in the previous year on two dummies indicating whether (at least) one specific supplier and whether (at least) one non-specific supplier is hit by a major disaster in the same quarter of the previous year. ///
In columns (1) and (2), a supplier is considered as specific if its industry lies above the median of the share of differentiated goods according to the classification provided by Rauch (1999). In columns (3) and (4), a supplier is considered specific if its ratio of R\&D expenses over sales is above the median in the two years prior to any given quarter. In columns (5) and (6), a supplier is considered as specific if the number of patents it issued in the previous three years is above the median. ///
All regressions include a dummy indicating whether the firm itself is hit by a major disaster in the same quarter in the previous year as well as fiscal-quarter, year-quarter, and firm fixed effects. ///
All regressions also control for the number of suppliers (dummies indicating terciles of the number of suppliers). ///
In columns (2), (4), and (6) we control for firm-level characteristics (dummies indicating terciles of size, age, and ROA respectively) interacted with year-quarter dummies. /// 
Regressions contain all firm-quarters of our customer sample (described in Table II, Panel A) between 1978 and 2013. ///
Standard errors presented in parentheses are clustered at the firm-level. *, **, and *** denote significance at the 10\%, 5\%, and 1\%, respectively. ///
\begin{center}\begin{tabular}{l*{6}{c}}\hline\hline&&&&&&\\&\multicolumn{6}{c}{ Sales Growth $(t-4,t)$}\\\cmidrule(lr){2-7} &&&&&&\\Supplier specificity: &\multicolumn{2}{c}{DIFF.}&\multicolumn{2}{c}{R\&D}&\multicolumn{2}{c}{PATENT}\\ \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} &&&&&&\\) ///
prefoot(&&&&&& \\ "Number of Suppliers & Yes & Yes & Yes & Yes & Yes & Yes \\" "Firm FE & Yes & Yes & Yes & Yes & Yes & Yes \\" "Year-Quarter FE & Yes & Yes & Yes & Yes& Yes & Yes\\" "Size, Age, ROA $\times$ Year-Quarter FE &No &Yes & No &Yes &No &Yes  \\" &&&&&&\\) ///
postfoot(&&&&&& \\ \hline \hline "\end{tabular} \end{center} \end{footnotesize} \end{table}")starlevels("*" 0.10 "**" 0.05 "***" 0.01)alignment(D{.}{.}{-1}) page(dcolumn) nonumber






FORMULA_DATA = [
    {   
        "Code" : "Global_Absence_Plan_Duration",
        "FORMULA_TYPE_NAME": "Global Absence Plan Duration",
        "Input Values": "IV_ABS_START_DATE,IV_ABS_END_DATE,IV_ABS_START_DURATION,IV_ABS_END_DURATION,IV_START_DATE,IV_END_DATE,IV_START_DURATION,IV_END_DURATION,IV_START_TIME,IV_END_TIME,IV_UOM",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "DURATION"
    },
    {
        "Code" :  "Global_Absence_Entry_Validation",
        "FORMULA_TYPE_NAME": "Global Absence Entry Validation",
        "Input Values": "IV_START_DATE,IV_END_DATE,IV_TOTALDURATION,IV_START_DURATION,IV_END_DURATION,IV_START_TIME,IV_END_TIME,IV_ACTUALCHILDBIRTHDATE,IV_ACTUALSTARTDATE,IV_ACTUALENDDATE,IV_EXPECTEDCHILDBIRTHDATE,IV_PLANNEDSTARTDATE,IV_PLANNEDENDDATE,IV_ABSENCE_REASON,IV_ATTRIBUTE_CATEGORY,IV_ATTRIBUTE_1,IV_ATTRIBUTE_NUMBER1,IV_ATTRIBUTE_DATE1,IV_ATTRIBUTE_ARR,IV_ATTRIBUTE_NUMBER_ARR,IV_ATTRIBUTE_DATE_ARR,IV_INFORMATION_CATEGORY,IV_INFORMATION_1,IV_INFORMATION_NUMBER1,IV_INFORMATION_DATE1,IV_INFORMATION_ARR,IV_INFORMATION_NUMBER_ARR,IV_INFORMATION_DATE_ARR,IV_PAYMENT_DTL_BAND,IV_NOTIFICATION_DATE,IV_MATCHING_DATE",
        "Context": "ABSENCE_AGREEMENT_ID,ABSENCE_CERTIFICATION_ID,ABSENCE_ENTRY_ID,ABSENCE_REASON_ID,ABSENCE_TYPE_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_AGREEMENT_ID,PERSON_ID,START_DATE",
        "Return Variables": "VALID,ERROR_MESSAGE,ERROR_CODE,TOKEN_NAME,TOKEN_VALUE"
    },
    {
        "Code" : "Enrollment_Certification_Required",
        "FORMULA_TYPE_NAME": "Enrollment Certification Required",
        "Input Values": "",
        "Context" : "BUSINESS_GROUP_ID ( ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PGM_ID,PL_ID,PER_IN_LER_ID,ER_ID,BENEFIT_RELATION_ID,PL_TYP_ID,OPT_ID,ORGANIZATION_ID,ELIG_PER_ELCTBL_CHC_ID,ENRT_CTFN_TYP_CD",
        "Return Variables": "L_OUTPUT"
    },
    {
        "Code" : "Global_Absence_Type_Duration",
        "FORMULA_TYPE_NAME": "Global Absence Type Duration",
        "Input Values": "IV_START_DATE,IV_END_DATE,IV_START_DURATION,IV_END_DURATION,IV_START_TIME,IV_END_TIME,IV_UOM",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "DURATION"
    },
    {
        "Code" :"Global_Absence_Band_Entitlement",
        "FORMULA_TYPE_NAME": "Global Absence Band Entitlement",
        "Input Values": "IV_START_DATE,IV_END_DATE,IV_TOTALDURATION,IV_START_DURATION,IV_END_DURATION,IV_START_TIME,IV_END_TIME,IV_ACTUALCHILDBIRTHDATE,IV_ACTUALSTARTDATE,IV_ACTUALENDDATE,IV_EXPECTEDCHILDBIRTHDATE,IV_PLANNEDSTARTDATE,IV_PLANNEDENDDATE,IV_ABSENCE_REASON,IV_ATTRIBUTE_CATEGORY,IV_ATTRIBUTE_1,IV_ATTRIBUTE_NUMBER1,IV_ATTRIBUTE_DATE1,IV_ATTRIBUTE_ARR,IV_ATTRIBUTE_NUMBER_ARR,IV_ATTRIBUTE_DATE_ARR,IV_INFORMATION_CATEGORY,IV_INFORMATION_1,IV_INFORMATION_NUMBER1,IV_INFORMATION_DATE1,IV_INFORMATION_ARR,IV_INFORMATION_NUMBER_ARR,IV_INFORMATION_DATE_ARR,IV_PAYMENT_DTL_BAND,IV_NOTIFICATION_DATE,IV_MATCHING_DATE",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "BANDPAYFACTOR,BANDENTITLEMENT,DEBUG_MESSAGE"
    },
    {
        "Code" :  "Global_Absence_Accrual_Matrix",
        "FORMULA_TYPE_NAME": "Global Absence Accrual Matrix",
        "Input Values": "IV_ACCRUAL,IV_CARRYOVER,IV_CEILING,IV_ACCRUALPERIODSTARTDATE,IV_ACCRUALPERIODENDDATE,IV_CALEDARSTARTDATE,IV_CALEDARENDDATE,IV_PLANENROLLMENTSTARTDATE,IV_PLANENROLLMENTENDDATE,IV_BAND_CHG_DT1,IV_BAND_CHG_BEFVAL1,IV_BAND_CHG_AFTVAL1,IV_EVENT_DATES,IV_ACCRUAL_VALUES,IV_ACCRUAL_CEILING,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "accrual,ceiling,carryover,adjustmentvalues,adjustmentdates,adjustmenttypes,absvalues,absdates,accrualCeiling"
    },
    {
        "Code" : "Global_Absence_Discretionary_Disbursement_Rule",
        "FORMULA_TYPE_NAME": "Global Absence Discretionary Disbursement Rule",
        "Input Values":"",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "MIN,MAX,INCREMENT"
    },
    {
        "Code" : "Global_Absence_Carryover",
        "FORMULA_TYPE_NAME": "Global Absence Carryover",
        "Input Values": "IV_ACCRUAL,IV_CARRYOVER,IV_CEILING,IV_ACCRUALPERIODSTARTDATE,IV_ACCRUALPERIODENDDATE,IV_CALEDARSTARTDATE,IV_CALEDARENDDATE,IV_PLANENROLLMENTSTARTDATE,IV_PLANENROLLMENTENDDATE,IV_ACCRUAL_CEILING",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "CARRYOVER"
    },
    {
        "Code" :  "Global_Absence_Plan_Enrollment_End",
        "FORMULA_TYPE_NAME": "Global Absence Plan Enrollment End",
        "Input Values": "",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "enrollmentEndDate"
    },
    {
        "Code" : "Global_Absence_Plan_Entitlement",
        "FORMULA_TYPE_NAME": "Global Absence Plan Entitlement",
        "Input Values": "IV_START_DATE,IV_END_DATE,IV_TOTALDURATION,IV_START_DURATION,IV_END_DURATION,IV_START_TIME,IV_END_TIME,IV_ACTUALCHILDBIRTHDATE,IV_ACTUALSTARTDATE,IV_ACTUALENDDATE,IV_EXPECTEDCHILDBIRTHDATE,IV_PLANNEDSTARTDATE,IV_PLANNEDENDDATE,IV_ABSENCE_REASON,IV_ATTRIBUTE_CATEGORY,IV_ATTRIBUTE_1,IV_ATTRIBUTE_NUMBER1,IV_ATTRIBUTE_DATE1,IV_ATTRIBUTE_ARR,IV_ATTRIBUTE_NUMBER_ARR,IV_ATTRIBUTE_DATE_ARR,IV_INFORMATION_CATEGORY,IV_INFORMATION_1,IV_INFORMATION_NUMBER1,IV_INFORMATION_DATE1,IV_INFORMATION_ARR,IV_INFORMATION_NUMBER_ARR,IV_INFORMATION_DATE_ARR,IV_PAYMENT_DTL_BAND,IV_NOTIFICATION_DATE,IV_MATCHING_DATE,IV_LNKG_ABS_ID,IV_LNKG_LNKD_ABS_ID,IV_LNKG_LNKD_ABS_START,IV_LNKG_LNKD_ABS_END,IV_LNKG_REASON,IV_LNKG_REASON_ID,IV_LNKG_CHAIN_ID,IV_UI_PER_CERT_ID,IV_UI_ABS_CERT_ID,IV_UI_CERT_TYPE,IV_UI_CERT_REVPAYSTART_DATE,IV_UI_CERT_REVPAYEND_DATE,IV_UI_CERT_REVPAY_FACTOR,IV_UI_CERT_CREATION_TYPE",
        "Context": "ABSENCE_CATEGORY_ID,ABSENCE_ENTRY_ID,ABSENCE_MATERNITY_ID,ABSENCE_REASON_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID",
        "Return Variables": "BAND1PAYFACTOR,BAND1ENTITLEMENT,BAND1RATEDEFINITION,BAND1USEDENTITLEMENT,BAND2PAYFACTOR,BAND2ENTITLEMENT,BAND2RATEDEFINITION,BAND2USEDENTITLEMENT,BAND3PAYFACTOR,BAND3ENTITLEMENT,BAND3RATEDEFINITION,BAND3USEDENTITLEMENT,BAND4PAYFACTOR,BAND4ENTITLEMENT,BAND4RATEDEFINITION,BAND4USEDENTITLEMENT,BAND5PAYFACTOR,BAND5ENTITLEMENT,BAND5RATEDEFINITION,BAND5USEDENTITLEMENT,CERT_NAMES,CERT_START_DATES,CERT_END_DATES,CERT_COMMENTS,DEBUG_MESSAGE"
    },
    {
        "Code" : "Global_Absence_Plan_Roll_Forward_Start",
        "FORMULA_TYPE_NAME": "Global Absence Plan Roll Forward Start",
        "Input Values": "IV_START_DATE,IV_END_DATE,IV_TOTALDURATION,IV_START_DURATION,IV_END_DURATION,IV_START_TIME,IV_END_TIME,IV_ACTUALCHILDBIRTHDATE,IV_ACTUALSTARTDATE,IV_ACTUALENDDATE,IV_EXPECTEDCHILDBIRTHDATE,IV_PLANNEDSTARTDATE,IV_PLANNEDENDDATE,IV_ABSENCE_REASON,IV_ATTRIBUTE_CATEGORY,IV_ATTRIBUTE_1,IV_ATTRIBUTE_NUMBER1,IV_ATTRIBUTE_DATE1,IV_ATTRIBUTE_ARR,IV_ATTRIBUTE_NUMBER_ARR,IV_ATTRIBUTE_DATE_ARR,IV_INFORMATION_CATEGORY,IV_INFORMATION_1,IV_INFORMATION_NUMBER1,IV_INFORMATION_DATE1,IV_INFORMATION_ARR,IV_INFORMATION_NUMBER_ARR,IV_INFORMATION_DATE_ARR,IV_PAYMENT_DTL_BAND,IV_NOTIFICATION_DATE",
        "Context": "ABSENCE_CATEGORY_ID,ABSENCE_ENTRY_ID,ABSENCE_MATERNITY_ID,ABSENCE_REASON_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "REFERENCEDATE"
    },
    {
        "Code" :"Global_Absence_Partial_Period_Accrual_Rate",
        "FORMULA_TYPE_NAME": "Global Absence Partial Period Accrual Rate",
        "Input Values": "IV_ACCRUAL,IV_CARRYOVER,IV_CEILING,IV_ACCRUALPERIODSTARTDATE,IV_ACCRUALPERIODENDDATE,IV_CALEDARSTARTDATE,IV_CALEDARENDDATE,IV_PLANENROLLMENTSTARTDATE,IV_PLANENROLLMENTENDDATE,IV_BAND_CHG_DT1,IV_BAND_CHG_BEFVAL1,IV_BAND_CHG_AFTVAL1,IV_ACCRUAL_CEILING",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "accrual"
    },
    {
        "Code" :  "Global_Absence_Plan_Enrollment_Start",
        "FORMULA_TYPE_NAME": "Global Absence Plan Enrollment Start",
        "Input Values": "",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "enrollmentStartDate"
    },
    {
        "Code" : "Global_Absence_Proration",
        "FORMULA_TYPE_NAME": "Global Absence Proration",
        "Input Values": "IV_ACCRUAL,IV_CARRYOVER,IV_CEILING,IV_ACCRUALPERIODSTARTDATE,IV_ACCRUALPERIODENDDATE,IV_CALEDARSTARTDATE,IV_CALEDARENDDATE,IV_PLANENROLLMENTSTARTDATE,IV_PLANENROLLMENTENDDATE,IV_ACCRUAL_CEILING",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "prorationFactor"
    },
    {
        "Code" :"Global_Absence_Plan_Period_Anniversary_Event_Date",
        "FORMULA_TYPE_NAME": "Global Absence Plan Period Anniversary Event Date",
        "Input Values": "",
        "Context": "ABSENCE_ENTRY_ID,ABSENCE_TYPE_ID,ACCRUAL_PLAN_ID,DATE_EARNED,EFFECTIVE_DATE,END_DATE,ENTERPRISE_ID,HR_ASSIGNMENT_ID,HR_RELATIONSHIP_ID,HR_TERM_ID,JOB_ID,LEGAL_EMPLOYER_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID,START_DATE",
        "Return Variables": "anniversaryDate"
    },
    {
        "Code": "Compensation_Currency_Selection",
        "FORMULA_TYPE_NAME": "Compensation Currency Selection",
        "Input Values": "CMP_IV_PLAN_ID,CMP_IV_ASSIGNMENT_ID,CMP_IV_PERIOD_ID,CMP_IV_COMPONENT_ID,CMP_IV_PLAN_START_DATE,CMP_IV_PLAN_END_DATE,CMP_IV_PLAN_EXTRACTION_DATECMP_IV_PLAN_ELIG_DATE,CMP_IV_PERFORMANCE_EFF_DATE,CMP_IV_PROMOTION_EFF_DATE,CMP_IV_XCHG_RATE_DATE,CMP_IV_ASSIGNMENT_ID,CMP_IV_PERSON_ID",
        "Context": "DATE_EARNED,EFFECTIVE_DATE,END_DATE,START_DATE,HR_ASSIGNMENT_ID,HR_TERM_ID,JOB_ID,LEGISLATIVE_DATA_GROUP_ID,COMPENSATION_RECORD_TYPE,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID",
        "Return Variables": "L_CURR_CODE"
    },
    {
        "Code": "Compensation_Default_and_Override",
        "FORMULA_TYPE_NAME": "Compensation Default and Override",
        "Input Values": "CMP_IV_PLAN_ID,CMP_IV_PERIOD_ID,CMP_IV_COMPONENT_ID,CMP_IV_ITEM_NAME,CMP_IV_PERSON_ID,CMP_IV_PLAN_START_DATE,CMP_IV_PLAN_END_DATE,CMP_IV_PLAN_ELIG_DATE,CMP_IV_PERFORMANCE_EFF_DATE,CMP_IV_PROMOTION_EFF_DATE,CMP_IV_XCHG_RATE_DATE,CMP_IV_ASSIGNMENT_ID",
        "Context": "DATE_EARNED,EFFECTIVE_DATE,END_DATE,START_DATE,HR_ASSIGNMENT_ID,HR_TERM_ID,JOB_ID,LEGISLATIVE_DATA_GROUP_ID,COMPENSATION_RECORD_TYPE,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID",
        "Return Variables": "L_DEFAULT_VALUE,L_DATA_TYPE"
    },
    {
        "Code": "Compensation_Hierarchy_Determination",
        "FORMULA_TYPE_NAME": "Compensation Hierarchy Determination",
        "Input Values": "CMP_IV_ASSIGNMENT_ID,CMP_IV_PLAN_ID,CMP_IV_PERIOD_ID,CMP_IV_COMPONENT_ID,CMP_IV_PERSON_ID,CMP_IV_PLAN_START_DATE,CMP_IV_PLAN_END_DATE,CMP_IV_PLAN_EXTRACTION_DATE,CMP_IV_PLAN_ELIG_DATE,CMP_IV_PERFORMANCE_EFF_DATE,CMP_IV_PROMOTION_EFF_DATE,CMP_IV_XCHG_RATE_DATE",
        "Context": "DATE_EARNED,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,END_DATE,START_DATE,HR_TERM_ID,JOB_ID,LEGISLATIVE_DATA_GROUP_ID,COMPENSATION_RECORD_TYPE,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID",
        "Return Variables": "L_PERSON_ID,L_ASSIGNMENT_IDL_PERSON_NUMBER"
    },
    {
        "Code": "Compensation_Person_Selection",
        "FORMULA_TYPE_NAME": "Compensation Person Selection",
        "Input Values": "CMP_IV_PLAN_ID,CMP_IV_PERIOD_ID,CMP_IV_PLAN_START_DATE,CMP_IV_PLAN_END_DATE,CMP_IV_PLAN_ELIG_DATE,CMP_IV_PERFORMANCE_EFF_DATE,CMP_IV_PROMOTION_EFF_DATE,CMP_IV_XCHG_RATE_DATE,CMP_IV_ASSIGNMENT_ID,CMP_IV_PERSON_ID",
        "Context": "DATE_EARNED,EFFECTIVE_DATE,END_DATE,START_DATE,HR_ASSIGNMENT_ID,HR_TERM_ID,JOB_ID,LEGISLATIVE_DATA_GROUP_ID,COMPENSATION_RECORD_TYPE,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID",
        "Return Variables": "L_SELECTED"
    },
    {
        "Code": "Total_Compensation_Item",
        "FORMULA_TYPE_NAME": "Total Compensation Item",
        "Input Values": "CMP_IV_PERIOD_ID,CMP_IV_PERIOD_START_DATE,CMP_IV_PERIOD_END_DATE",
        "Context": "DATE_EARNED,EFFECTIVE_DATE,END_DATE,START_DATE,HR_ASSIGNMENT_ID,HR_TERM_ID,JOB_ID,LEGISLATIVE_DATA_GROUP_ID,COMPENSATION_RECORD_TYPE,ORGANIZATION_ID,PAYROLL_ASSIGNMENT_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,PERSON_ID",
        "Return Variables": "COMPENSATION_DATES,VALUES,ASSIGNMENTS,LEGALEMPLOYERS,UNIT,COMPENSATION_DATES1,VALUES1,ASSIGNMENTS1,LEGALEMPLOYERS1,UNIT1,COMPENSATION_DATES2,VALUES2,ASSIGNMENTS2,LEGALEMPLOYERS2,UNIT2,COMPENSATION_DATES3,VALUES3,ASSIGNMENTS3,LEGALEMPLOYERS3,UNIT3"
    },
    {
        "Code": "Person_Change_Causes_Life_Event",
        "FORMULA_TYPE_NAME": "Person Change Causes Life Event",
        "Input Values": "BEN_SAL_IN_PERSON_ID,BEN_SAL_IO_PERSON_ID,BEN_SAL_IN_SALARY_AMOUNT,BEN_SAL_IN_ACTION_REASON_ID,BEN_SAL_IO_ACTION_REASON_ID,BEN_SAL_IN_SALARY_BASIS_ID,BEN_SAL_IN_DATE_FROM,BEN_SAL_IN_DATE_TO,BEN_SAL_IO_DATE_TO,BEN_SAL_IN_ELEMENT_ENTRY_ID,BEN_SAL_IO_ELEMENT_ENTRY_ID,BEN_SAL_IN_FORCED_RANKING,BEN_SAL_IO_FORCED_RANKING,BEN_SAL_IN_PERFORMANCE_RATING,BEN_SAL_IO_PERFORMANCE_RATING,BEN_SAL_IN_PERFORMANCE_REVIEW_ID,BEN_SAL_IO_PERFORMANCE_REVIEW_ID,BEN_SAL_IN_REVIEW_DATE,BEN_SAL_IO_REVIEW_DATE,BEN_SAL_IN_SALARY_APPROVED,BEN_SAL_IO_SALARY_APPROVED",
        "Context": "HR_RELATIONSHIP_ID,HR_TERM_ID,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,LEGAL_EMPLOYER_ID,DATE_EARNED,HR_ASSIGNMENT_ID,BUSINESS_GROUP_ID,PERSON_ID,JOB_ID,EFFECTIVE_DATE,PAYROLL_ASSIGNMENT_ID,LEGISLATIVE_DATA_GROUP_ID,ORGANIZATION_ID,BENEFIT_RELATION_ID",
        "Return Variables": "L_RETURN"
    },
    {
        "Code": "Coverage_Amount_Limit",
        "FORMULA_TYPE_NAME": "Coverage Amount Limit",
        "Input Values": " ",
        "Context": "BUSINESS_GROUP_ID ( ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID",
        "Return Variables": "L_MN_CVG_RQD_AMT, L_MX_CVG_ALWD_AMT, L_MX_CVG_WCFN_AMT, L_MX_CVG_INCR_ALWD_AMT, L_MX_CVG_INCR_WCF_ALWD_AMT"
    },
    {
        "Code": "Participation_And_Rate_Eligibility",
        "FORMULA_TYPE_NAME": "Participation And Rate Eligibility",
        "Input Values": " ",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,LER_ID,OPT_ID,ORGANIZATION_ID,PGM_ID,PL_ID,PL_TYP_ID,PERSON_ID",
        "Return Variables": "ELIGIBLE"
    },
    {
        "Code": "Rounding",
        "FORMULA_TYPE_NAME": "Rounding",
        "Input Values": "VALUE",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID",
        "Return Variables": "L_OUTPUT"
    },
    {
        "Code": "Age_Calculation",
        "FORMULA_TYPE_NAME": "Age Calculation",
        "Input Values": "PERSON_ID",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID,PERSON_ID",
        "Return Variables": "L_OUTPUT"
    },
    {
        "Code": "Person_Selection",
        "FORMULA_TYPE_NAME": "Person Selection",
        "Input Values": "BEN_IV_PERSON_ID",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID",
        "Return Variables": "L_OUTPUT"
    },
    {
        "Code": "Benefits_Extract_Custom_Data_Rule",
        "FORMULA_TYPE_NAME": "Benefits Extract Custom Data Rule",
        "Input Values": "",
        "Context": "HR_RELATIONSHIP_ID,HR_TERM_ID,LC_DATE_FROM,LC_DATE_TO,PAYROLL_RELATIONSHIP_ID,PAYROLL_TERM_ID,LEGAL_EMPLOYER_ID,DATE_EARNED,HR_ASSIGNMENT_ID BUSINESS_GROUP_ID,PERSON_ID JOB_ID,EFFECTIVE_DATE PAYROLL_ASSIGNMENT_ID,PAYROLL_ID,LEGISLATIVE_DATA_GROUP_ID,LER_ID,OPT_ID,ORGANIZATION_ID ACTY_BASE_RT_ID,PGM_ID,PL_ID,PL_TYP_ID,BENEFIT_RELATION_ID,PER_IN_LER_ID",
        "Return Variables": ""
    },
    {
        "Code": "Life_Event_Reason_Timeliness",
        "FORMULA_TYPE_NAME": "Life Event Reason Timeliness",
        "Input Values": "",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID",
        "Return Variables": "L_LIFEEVENT_VOIDED"
    },
    {
        "Code": "Coverage_Amount_Calculation",
        "FORMULA_TYPE_NAME": "Coverage Amount Calculation",
        "Input Values": "",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "L_CVG"
    },
    {
        "Code": "Rate_Value_Calculation",
        "FORMULA_TYPE_NAME": "Rate Value Calculation",
        "Input Values": "",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "L_RATE"
    },
    {
        "Code": "Dependent_Eligibility",
        "FORMULA_TYPE_NAME": "Dependent Eligibility",
        "Input Values": "CON_PERSON_ID",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "L_output"
    },
    {
        "Code": "Age_Determination_Date",
        "FORMULA_TYPE_NAME": "Age Determination Date",
        "Input Values": "",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "l_output"
    },
    {
        "Code": "Enrollment_Opportunity",
        "FORMULA_TYPE_NAME": "Enrollment Opportunity",
        "Input Values": "",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "L_ENRT_OPP"
    },
    {
        "Code": "Beneficiary_Certification_Required",
        "FORMULA_TYPE_NAME": "Beneficiary Certification Required",
        "Input Values": "",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "L_OUTPUT"
    },
    {
        "Code": "Compensation_Calculation",
        "FORMULA_TYPE_NAME": "Compensation Calculation",
        "Input Values": "PERSON_ID",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "L_OUTPUT"
    },
    {
        "Code": "Post_Election_Edit",
        "FORMULA_TYPE_NAME": "Post Election Edit",
        "Input Values": "",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "Successful ,Error Message"
    },
    {
        "Code": "Enrollment_Period_Start_Date",
        "FORMULA_TYPE_NAME": "Enrollment Period Start Date",
        "Input Values": "",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "L_start_dt"
    },
    {
        "Code": "Default_Enrollment",
        "FORMULA_TYPE_NAME": "Default Enrollment",
        "Input Values": "NUM_ELIG_DPNT",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "AUTO_DFLT_VAL,CARRY_FORWARD_ELIG_DPNT"
    },
    {
        "Code": "Evaluate_Life_Event",
        "FORMULA_TYPE_NAME": "Evaluate Life Event",
        "Input Values": "BEN_PPL_IV_LF_EVT_OCRD_DT,BEN_PPL_IV_PTNL_LER_FOR_PER_STAT_CD,BEN_PPL_IV_NTFN_DT,BEN_PPL_IV_DTCTD_DT",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "LIFE_EVENT_OCCURRED_DATE,LIFE_EVENT_HAPPENED,LIFE_EVENT_NOTIFICATION_DATE,LIFE_EVENT_VOIDED_DATE,LIFE_EVENT_MANUAL_DATE,LIFE_EVENT_STATUS_CODE,LIFE_EVENT_DETECTED_DATE"
    },
    {
        "Code":"Rate_Periodization",
        "FORMULA_TYPE_NAME": "Rate Periodization",
        "Input Values": "BEN_IV_CONVERT_FROM,BEN_IV_CONVERT_FROM_VAL",
        "Context": "BUSINESS_GROUP_ID (ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,PGM_ID,PL_ID,PL_TYP_ID",
        "Return Variables": "DFND_VAL,ANN_VAL,CMCD_VAL"
    },
    {
        "Code": "Waiting_Period_Value_And_UOM",
        "FORMULA_TYPE_NAME": "Waiting Period Value And UOM",
        "Input Values": "",
        "Context": "BUSINESS_GROUP_ID,EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID,LER_ID,ORGANIZATION_ID,JURISDICTION_CODE,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,LER_ID",
        "Return Variables": "WAIT_PERD_VAL ,WAIT_PERD_UOM"
    },
    {
        "Code" : "Recruiting_Job_Requisition",
        "FORMULA_TYPE_NAME": "Recruiting Job Requisition",
        "Input Values": "",
        "Context": "IRC_JOB_REQUISITION_ID,IRC_JOB_LANGUAGE",
        "Return Variables": "CONDITION_RESULT"
    },
    {
        "Code" : "Recruiting_Candidate_Selection_Process",
        "FORMULA_TYPE_NAME": "Recruiting Candidate Selection Process",
        "Input Values": "",
        "Context": "SUBMISSION_ID,LANGUAGE,IRC_INTRVW_SCHEDULE_ID,IRC_INTRVW_ID,IRC_INTRVW_REQUEST_ID,IRC_INTRVW_OPERATION,INTERVIEW_REQUEST_SENT ,INTERVIEW_SCHEDULED ,INTERVIEW_CANCELLED ,INTERVIEW_RESCHEDULED ,INTERVIEW_UPDATED ,INTERVIEW_COMPLETED",
        "Return Variables": "CONDITION_RESULT, CONDITION_MESSAGE"
    },
    {
        "Code" : "Coverage_Upper_Limit",
        "FORMULA_TYPE_NAME": "Coverage Upper Limit",
        "Input Values": "",
        "Context" : "BUSINESS_GROUP_ID ( ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PERSON_ID",
        "Return Variables": "L_MN_CVG_RQD_AMT, L_MX_CVG_ALWD_AMT, L_MX_CVG_WCFN_AMT, L_MX_CVG_INCR_ALWD_AMT, L_MX_CVG_INCR_WCF_ALWD_AMT"
    },
    # {
    #     "Code" : "Participation_And_Rate_Eligibility",
    #     "FORMULA_TYPE_NAME": "Participation And Rate Eligibility",
    #     "Input Values": "",
    #     "Context" : "BUSINESS_GROUP_ID ( ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,LER_ID,OPT_ID,ORGANIZATION_ID,PGM_ID,PL_ID,PL_TYP_ID,PERSON_ID",
    #     "Return Variables": "ELIGIBLE"
    # },
    {
        "Code" : "Extra_Input",
        "FORMULA_TYPE_NAME": "Extra Input",
        "Input Values": "",
        "Context" : "BUSINESS_GROUP_ID (ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,PERSON_ID,LER_ID,BENEFIT_RELATION_ID,ACTY_BASE_RT_ID,ORGANIZATION_ID",
        "Return Variables": "L_OUTPUT"
    },
    {
        "Code" : "Default_To_Assign_Pending_Action",
        "FORMULA_TYPE_NAME": "Default To Assign Pending Action",
        "Input Values": "BEN_PEN_IV_PRTT_ENRT_RSLT_ID",
        "Context" : "BUSINESS_GROUP_ID ( ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,PERSON_ID,LER_ID,BENEFIT_RELATION_ID,ELIG_PER_ELCTBL_CHC_ID,ORGANIZATION_ID",
        "Return Variables": "L_OUTPUT,L_BNFT_AMOUNT,L_EPE_ID"

    },
    {
        "Code" : "Enrollment_Coverage_Start_Date",
        "FORMULA_TYPE_NAME": "Enrollment Coverage Start Date",
        "Input Values": "",
        "Context" : "BUSINESS_GROUP_ID ( ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,PERSON_ID,LER_ID,BENEFIT_RELATION_ID,ELIG_PER_ELCTBL_CHC_ID,ORGANIZATION_ID",
        "Return Variables": "L_start_dt"
    },
    {
        "Code" : "Rate_Start_Date",
        "FORMULA_TYPE_NAME": "Rate Start Date",
        "Input Values": "",
        "Context" : "BUSINESS_GROUP_ID ( ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PGM_ID,PL_ID,PER_IN_LER_ID,PERSON_ID,LER_ID,BENEFIT_RELATION_ID,ELIG_PER_ELCTBL_CHC_ID",
        "Return Variables": "L_OUTPUT"
    },
    {
        "Code" : "Dependent Certification Required",
        "FORMULA_TYPE_NAME": "Dependent_Certification_Required",
        "Input Values": "BEN_PEN_IV_PRTT_ENRT_RSLT_ID",
        "Context" : "BUSINESS_GROUP_ID ( ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PGM_ID,PL_ID,PL_TYP_ID,OPT_ID,PERSON_ID,LER_ID,BENEFIT_RELATION_ID,ELIG_PER_ELCTBL_CHC_ID,ORGANIZATION_ID",
        "Return Variables": "L_OUTPUT,L_BNFT_AMOUNT,L_EPE_ID"
    },
    {
        "Code" : "Enrollment_Certification_Required",
        "FORMULA_TYPE_NAME": "Enrollment Certification Required",
        "Input Values": "",
        "Context" : "BUSINESS_GROUP_ID ( ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PGM_ID,PL_ID,PER_IN_LER_ID,ER_ID,BENEFIT_RELATION_ID,PL_TYP_ID,OPT_ID,ORGANIZATION_ID,ELIG_PER_ELCTBL_CHC_ID,ENRT_CTFN_TYP_CD",
        "Return Variables": "L_OUTPUT"
    },
    {
        "Code" : "Length_Of_Service_Date_To_Use",
        "FORMULA_TYPE_NAME": "Length Of Service Date To Use",
        "Input Values": "BEN_IV_PERSON_ID,BEN_IV_RT_STRT_DT,BEN_IV_CVG_STRT_DT",
        "Context" : "BUSINESS_GROUP_ID ( ENTERPRISE_ID),EFFECTIVE_DATE,HR_ASSIGNMENT_ID,PGM_ID,PL_ID,PER_IN_LER_ID,LER_ID,BENEFIT_RELATION_ID,PL_TYP_ID,OPT_ID,ORGANIZATION_ID",
        "Return Variables": "L_OUTPUT"
    }
    

]
  

def get_formula_details(formula_type: str):
    """Retrieve details of a given Formula Type Name."""
    for formula in FORMULA_DATA:
        if formula["Code"] == formula_type:
            return formula
    return None

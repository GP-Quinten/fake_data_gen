# ===================================================
# DATA PATHs
# ===================================================
DATABASE = "ppmi"
BUCKET_NAME = "syn-aws-par-s3-common-dev-bucket"
PATH_RAW_DATA = "raw_data/"
FILE_PPMI_RAW_DATA = "ppmi/PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv"

# =================================================
# Columns
# =================================================
COL_LIST = ['PATNO',
            'APPRDX',
            'EVENT_ID', 
            'age',
            'gen',
            'symptom1',
            'symptom2',
            'updrs3_score',
            'updrs1_score',
            'updrs_totscore',
            'ess',
            'PUTAMEN_R',
            'PUTAMEN_L']

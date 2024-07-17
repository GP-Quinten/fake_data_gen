# ===================================================
# DATA PATHs
# ===================================================

DATABASE = "mimic_iii"

# table names
FILENAME_ADMISSION = "ADMISSIONS.csv.gz"
FILENAME_PATIENTS = "PATIENTS.csv.gz"
FILENAME_DIAGNOSES = "DIAGNOSES_ICD.csv.gz"
FILENAME_DATETIMEEVENTS = "DATETIMEEVENTS.csv.gz"
FILENAME_PRESCRIPTIONS = "PRESCRIPTIONS.csv.gz"
FILENAME_ADMISSIONS = "ADMISSIONS.csv.gz"
FILENAME_LABS = "LABEVENTS.csv.gz"
FILENAME_LABS_DESC = "D_LABITEMS.csv.gz"

# ===================================================
# PREPROCESSING
# ===================================================
MIN_PREV = 10

### Admission
COL_TO_KEEP_PATIENTS = ["SUBJECT_ID", "GENDER", "DOB", "DOD", "EXPIRE_FLAG"]

### Patients    
COL_TO_KEEP_ADMISSION = [
    "SUBJECT_ID",
    "ETHNICITY",
    "FIRST_ADMISSION_DATE",
    "LAST_ADMISSION_DATE",
]
VALUE_GENDER_MALE = "Male"
VALUE_GENDER_FEMALE = "Female"

### Demog
COL_TO_KEEP_DEMOG = [
    "SUBJECT_ID",
    "AGE_AT_LAST_ADMISSION",
    "GENDER",
    "FOLLOWUP_PERIOD",
    "ETHNICITY",
]

### Labs
LABELS_TO_LOINC = {
    "PLATELET_COUNT": ["777-3"],
    "GLUCOSE": ["2339-0", "2345-7"],
    "HEMOGLOBIN": ["718-7"],
    "HDL": ["2085-9"],
    "LDL": ["2090-9", "18262-6"],
}

"""Project-defined reaction categories used for provisional NLP categorisation.

IMPORTANT
---------
* These categories are an **organ-system style grouping defined by this
  project**. They are NOT MedDRA System Organ Classes and no official MedDRA
  hierarchy is reproduced or implied. FAERS reaction terms are MedDRA
  Preferred Terms; mapping them to MedDRA SOCs officially requires a MedDRA
  licence (MSSO), which this project does not use.
* Each category is defined by a short description and a handful of generic
  "anchor" phrases authored by the project team. The anchors were written
  before the evaluation reference (NCI CTCAE v5.0) was inspected; any anchor
  that happens to coincide exactly with a reference term is excluded from the
  evaluation set (see evaluate.py) to avoid leakage.
* A prediction is a *provisional* similarity-based grouping for exploration,
  not a clinical coding decision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

TAXONOMY_VERSION = "pv-cat-1.0"


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    description: str
    anchors: tuple[str, ...]


CATEGORIES: tuple[Category, ...] = (
    Category("cardiovascular", "Cardiovascular", "Heart rhythm, heart muscle, coronary and blood-vessel problems, blood pressure.",
             ("cardiac arrhythmia", "heart failure", "myocardial infarction", "chest pain of cardiac origin",
              "palpitations", "tachycardia", "bradycardia", "hypertension", "hypotension", "deep vein thrombosis",
              "vasculitis", "cardiac arrest", "stroke due to blood clot in artery", "peripheral vascular disorder")),
    Category("respiratory", "Respiratory", "Airways, lungs, pleura and breathing.",
             ("shortness of breath", "cough", "pneumonitis", "asthma attack", "pulmonary embolism",
              "respiratory failure", "interstitial lung disease", "pleural effusion", "bronchospasm",
              "nasal congestion", "wheezing", "hypoxia")),
    Category("gastrointestinal", "Gastrointestinal", "Mouth, oesophagus, stomach, intestines, pancreas and digestion.",
             ("nausea", "vomiting", "diarrhoea", "constipation", "abdominal pain", "gastrointestinal haemorrhage",
              "pancreatitis", "colitis", "dyspepsia", "mouth ulceration", "bowel obstruction", "dysphagia")),
    Category("hepatobiliary", "Hepatobiliary", "Liver and bile-duct disorders.",
             ("hepatotoxicity", "drug-induced liver injury", "hepatitis", "jaundice", "hepatic failure",
              "cholestasis", "cholecystitis", "gallstones", "liver cirrhosis", "hepatic steatosis")),
    Category("renal_urinary", "Renal & urinary", "Kidney and urinary tract disorders.",
             ("acute kidney injury", "renal failure", "nephrotoxicity", "proteinuria", "haematuria",
              "urinary retention", "urinary incontinence", "nephritis", "dysuria", "urinary frequency")),
    Category("nervous_system", "Nervous system", "Brain, spinal cord, peripheral nerves, headache, seizures, sensation and movement.",
             ("headache", "dizziness", "seizure", "peripheral neuropathy", "paraesthesia", "tremor",
              "somnolence", "syncope", "encephalopathy", "cerebral haemorrhage", "migraine", "memory impairment",
              "dysgeusia", "movement disorder")),
    Category("psychiatric", "Psychiatric", "Mood, behaviour, perception and sleep-related mental disorders.",
             ("depression", "anxiety", "insomnia", "suicidal ideation", "hallucination", "confusional state",
              "agitation", "psychosis", "mood swings", "abnormal dreams", "substance dependence")),
    Category("skin", "Skin & subcutaneous", "Skin, hair, nails and subcutaneous tissue.",
             ("rash", "pruritus", "urticaria", "eczema", "alopecia", "erythema", "dry skin",
              "stevens-johnson syndrome", "blister", "skin exfoliation", "hyperhidrosis", "psoriasis", "acne")),
    Category("musculoskeletal", "Musculoskeletal", "Muscles, joints, bones, tendons and connective tissue.",
             ("arthralgia", "myalgia", "back pain", "muscle spasms", "rhabdomyolysis", "osteoporosis",
              "arthritis", "pain in extremity", "joint swelling", "tendon rupture", "osteonecrosis of jaw",
              "muscular weakness")),
    Category("blood", "Blood & lymphatic", "Blood cells, bone marrow, coagulation and lymph nodes.",
             ("anaemia", "neutropenia", "thrombocytopenia", "leukopenia", "febrile neutropenia",
              "pancytopenia", "bone marrow failure", "lymphadenopathy", "haemolysis",
              "disseminated intravascular coagulation")),
    Category("immune_allergic", "Immune & hypersensitivity", "Allergic, anaphylactic and other immune-system reactions.",
             ("hypersensitivity", "anaphylactic reaction", "allergic reaction", "drug allergy", "angioedema",
              "cytokine release syndrome", "serum sickness", "immunosuppression", "transplant rejection",
              "infusion-related allergic reaction")),
    Category("infections", "Infections", "Bacterial, viral, fungal and parasitic infections and infestations.",
             ("pneumonia", "urinary tract infection", "sepsis", "covid-19", "herpes zoster", "cellulitis",
              "nasopharyngitis", "fungal infection", "influenza", "bacterial infection", "viral infection",
              "tuberculosis")),
    Category("metabolic", "Metabolism & nutrition", "Glucose, electrolytes, fluid balance, appetite and nutrition.",
             ("hypoglycaemia", "hyperglycaemia", "diabetes mellitus", "hyponatraemia", "hypokalaemia",
              "dehydration", "decreased appetite", "lactic acidosis", "hyperlipidaemia", "malnutrition",
              "tumour lysis syndrome")),
    Category("endocrine", "Endocrine", "Hormone-producing glands: thyroid, adrenal, pituitary and others.",
             ("hypothyroidism", "hyperthyroidism", "adrenal insufficiency", "thyroiditis", "hypophysitis",
              "cushing's syndrome", "hyperparathyroidism", "inappropriate antidiuretic hormone secretion")),
    Category("eye", "Eye", "Eyes and vision.",
             ("blurred vision", "visual impairment", "cataract", "conjunctivitis", "dry eye", "glaucoma",
              "retinal detachment", "eye pain", "uveitis", "macular oedema")),
    Category("ear", "Ear & labyrinth", "Hearing and balance organs.",
             ("tinnitus", "hearing loss", "vertigo", "ear pain", "deafness", "ear infection of inner ear")),
    Category("reproductive", "Reproductive & breast", "Reproductive organs, menstruation, sexual function and breast.",
             ("erectile dysfunction", "menstrual disorder", "heavy menstrual bleeding", "breast pain",
              "gynaecomastia", "vaginal haemorrhage", "infertility", "amenorrhoea", "testicular pain")),
    Category("pregnancy", "Pregnancy & perinatal", "Pregnancy, childbirth, foetal and neonatal conditions.",
             ("exposure during pregnancy", "spontaneous abortion", "premature delivery", "foetal growth restriction",
              "stillbirth", "pre-eclampsia", "neonatal complication", "maternal exposure during breast feeding")),
    Category("neoplasms", "Neoplasms", "Benign, malignant and unspecified tumours, cancer progression.",
             ("malignant neoplasm", "cancer progression", "metastasis", "lymphoma", "leukaemia",
              "breast cancer", "lung cancer", "skin cancer", "benign tumour", "second primary malignancy")),
    Category("general", "General & administration site", "Whole-body symptoms, administration-site reactions and death.",
             ("fatigue", "pyrexia", "malaise", "asthenia", "oedema peripheral", "chills", "pain",
              "injection site reaction", "infusion site swelling", "death", "feeling abnormal",
              "condition aggravated", "influenza like illness")),
    Category("investigations", "Investigations (test results)", "Abnormal laboratory values, vital-sign measurements and test results.",
             ("alanine aminotransferase increased", "blood creatinine increased", "weight decreased",
              "weight increased", "platelet count decreased", "white blood cell count decreased",
              "blood pressure increased", "electrocardiogram qt prolonged", "haemoglobin decreased",
              "blood glucose increased", "liver function test abnormal", "heart rate increased")),
    Category("injury_procedural", "Injury, poisoning & procedural", "Falls, fractures, wounds, overdose, toxicity and procedure complications.",
             ("fall", "fracture", "contusion", "overdose", "poisoning", "laceration", "procedural complication",
              "post procedural haemorrhage", "toxicity to various agents", "radiation injury", "burn")),
    Category("product_use", "Product use, errors & efficacy", "Medication errors, off-label use, product quality/device problems and lack of effect.",
             ("off label use", "drug ineffective", "medication error", "wrong dose administered",
              "product dose omission", "intentional product misuse", "product quality issue",
              "device malfunction", "therapeutic response decreased", "drug interaction",
              "product use in unapproved indication", "treatment noncompliance", "inappropriate schedule of drug administration")),
)

CATEGORY_BY_ID = {c.id: c for c in CATEGORIES}
ABSTAIN = "abstain"


def taxonomy_hash() -> str:
    payload = json.dumps([asdict(c) for c in CATEGORIES], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def anchor_table() -> list[tuple[str, str]]:
    """Flat list of (category_id, anchor_text)."""
    return [(c.id, a) for c in CATEGORIES for a in c.anchors]


def as_dicts() -> list[dict]:
    return [
        {"id": c.id, "label": c.label, "description": c.description, "anchors": list(c.anchors)}
        for c in CATEGORIES
    ]


# Project-defined crosswalk used ONLY for evaluation against NCI CTCAE v5.0,
# whose terms are grouped under MedDRA SOC names. Reference groups without a
# counterpart here (e.g. congenital, social circumstances, surgical procedures)
# are excluded from evaluation and reported as such.
CTCAE_SOC_TO_CATEGORY = {
    "cardiac disorders": "cardiovascular",
    "vascular disorders": "cardiovascular",
    "respiratory, thoracic and mediastinal disorders": "respiratory",
    "gastrointestinal disorders": "gastrointestinal",
    "hepatobiliary disorders": "hepatobiliary",
    "renal and urinary disorders": "renal_urinary",
    "nervous system disorders": "nervous_system",
    "psychiatric disorders": "psychiatric",
    "skin and subcutaneous tissue disorders": "skin",
    "musculoskeletal and connective tissue disorders": "musculoskeletal",
    "blood and lymphatic system disorders": "blood",
    "immune system disorders": "immune_allergic",
    "infections and infestations": "infections",
    "metabolism and nutrition disorders": "metabolic",
    "endocrine disorders": "endocrine",
    "eye disorders": "eye",
    "ear and labyrinth disorders": "ear",
    "reproductive system and breast disorders": "reproductive",
    "pregnancy, puerperium and perinatal conditions": "pregnancy",
    "neoplasms benign, malignant and unspecified (incl cysts and polyps)": "neoplasms",
    "general disorders and administration site conditions": "general",
    "investigations": "investigations",
    "injury, poisoning and procedural complications": "injury_procedural",
}

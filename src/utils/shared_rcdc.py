"""Splitting RCDC's mixed vocabulary into diseases vs cross-cutting research areas.

RCDC (NIH's Research, Condition and Disease Categorisation) is a flat list of ~300 tags
that mixes three quite different things:

    genuine conditions      Cardiovascular, Obesity, Alzheimer's Disease, Breast Cancer
    research areas/methods  Clinical Research, Prevention, Machine Learning and AI
    populations/exposures   Women's Health, Aging, Minority Health, Tobacco, Nutrition

A raw ranking is therefore dominated by the second and third groups — in the UK Biobank
corpus, `Genetics`, `Prevention`, `Human Genome`, `Clinical Research` and `Aging` are the
five biggest tags, and none of them names a disease. They are legitimate answers to "what
kind of research is this?" but not to "what is it about?", so the two questions need two
different views of the same counts.

The split is a STOP LIST, deliberately explicit rather than a heuristic, so it can be read,
argued with and extended. It was first written for the clinical-trials notebook
(05_non_academic_01_clinical_trials, §3.4) against the 157 tags those trials carry; the
academic corpus carries ~320, so the terms that vocabulary adds are grouped separately
below and marked, to keep the provenance of each decision visible.

Usage:

    from utils.shared_rcdc import is_disease, split_terms, classify

    is_disease("Cardiovascular")     -> True
    is_disease("Prevention")         -> False
    classify("Prevention")           -> "cross-cutting"

Judgement calls worth knowing about, because they are the ones a reader will query:

    Nutrition, Tobacco       exposures/lifestyle, not conditions -> cross-cutting
    Pregnancy, Menopause     life stages, not conditions         -> cross-cutting
    Suicide, Childhood Injury  outcomes ICD codes as conditions  -> disease
    Alcoholism, Substance Misuse  named disorders                -> disease
    Cancer Genomics          a research programme, not a cancer  -> cross-cutting
"""

from __future__ import annotations

from typing import Iterable, List

# =============================================================================
# THE STOP LIST — every tag here is NOT a disease
# =============================================================================
# Group 1: from 05_non_academic_01_clinical_trials §3.4, unchanged.
_STOP_TRIALS = {
    # research-activity / portfolio labels
    "Clinical Research", "Clinical Trials and Supportive Activities", "Prevention",
    "Comparative Effectiveness Research", "Cost Effectiveness Research",
    "Dissemination and Implementation Research", "Patient Safety", "Health Services",
    "Primary Health Care", "Emergency Care", "Telehealth", "Rehabilitation",
    "Physical Rehabilitation", "Precision Medicine", "Orphan Drug", "Transplantation",
    "Organ Transplantation", "Regenerative Medicine",
    # methods / platforms / technology
    "Biomedical Imaging", "Bioengineering", "Biotechnology", "Biodefense",
    "Machine Learning and Artificial Intelligence", "Data Science", "Assistive Technology",
    "Networking and Information Technology R&D (NITRD)",
    # biology / mechanism — not a disease axis
    "Genetics", "Human Genome", "Genetic Testing", "Microbiome", "Neurosciences",
    "Hematology", "Endocannabinoid System Research", "Cannabinoid Research",
    # behavioural / lifestyle / exposure research areas
    "Behavioral and Social Science", "Basic Behavioral and Social Science",
    "Physical Activity", "Dietary Supplements", "Complementary and Integrative Health",
    "Sleep Research", "Breastfeeding, Lactation and Breast Milk", "Contraception/Reproduction",
    # populations / equity / life-stage framings
    "Aging", "Women's Health", "Maternal Health", "Minority Health", "Health Disparities",
    "Health Disparities and Racial or Ethnic Minority Health Research",
    "Social Determinants of Health", "Pediatric Research Initiative", "Rural Health",
    "Rare Diseases", "Infant Mortality", "Vaccine Related", "Immunization",
    "Pain Research", "Chronic Pain Management", "Behavioral Pain Management",
}

# Group 2: tags the ACADEMIC corpus carries that the 151 trials never did. Same reasoning,
# reviewed against the full ~320-tag vocabulary of counts.rcdc.*.
_STOP_ACADEMIC = {
    # exposures and lifestyle — what a paper studies the effect OF, not a condition
    "Nutrition", "Tobacco", "Tobacco Smoke and Health", "Estrogen",
    "Climate Change", "Climate-Related Exposures and Conditions", "Endocrine Disruptors",
    "Health Effects of Indoor Air Pollution", "Health Effects of Household Energy Combustion",
    # interventions / methods / research programmes
    "Immunotherapy", "Gene Therapy", "Gene Therapy Clinical Trials", "Radiation Oncology",
    "Natural Products", "Therapeutic Cannabinoid Research", "Wound Healing and Care",
    "Cancer Genomics", "Burden of Illness",
    # prevention programmes (the condition itself stays in, the programme does not)
    "Substance Abuse Prevention", "Suicide Prevention", "Youth Violence Prevention",
    "Underage Drinking - Prevention & Treatment (NIAAA Only)",
    # social-research areas
    "Violence Research", "Violence Against Women", "Youth Violence",
    "Child Abuse and Neglect Research",
    # life stages and physiology
    "Pregnancy", "Teenage Pregnancy", "Menopause",
    # populations / equity
    "American Indian or Alaska Native", "Sexual and Gender Minorities (SGM/LGBT*)",
    "Workforce Diversity and Outreach",
}

RCDC_STOP = _STOP_TRIALS | _STOP_ACADEMIC

# RCDC spells some research programmes out into a dozen near-identical variants
# ("Stem Cell Research - Induced Pluripotent Stem Cell - Non-Human"). Matching the family
# by prefix keeps the list readable and survives NIH adding another variant.
RCDC_STOP_PREFIXES = ("Stem Cell Research",)


def is_disease(term: str) -> bool:
    """True when an RCDC tag names a condition rather than a research area."""
    term = (term or "").strip()
    if not term or term in RCDC_STOP:
        return False
    return not term.startswith(RCDC_STOP_PREFIXES)


def classify(term: str) -> str:
    """'disease' or 'cross-cutting' — the label form of is_disease, for grouping."""
    return "disease" if is_disease(term) else "cross-cutting"


def split_terms(terms: Iterable[str]) -> tuple[List[str], List[str]]:
    """Partition an iterable of tags into (diseases, cross_cutting), order preserved."""
    diseases, cross = [], []
    for t in terms:
        (diseases if is_disease(t) else cross).append(t)
    return diseases, cross

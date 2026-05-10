"""
Build three-category evaluation test set.

  test_set/positives.jsonl              — 50 Health Feedback 2024 reviews
  test_set/negatives_facts.jsonl        — 35 synthesised established-fact claims
  test_set/negatives_unfalsifiable.jsonl— 20 synthesised unfalsifiable opinions
  test_set/test_set.jsonl               — merged file

Usage:
    python scripts/build_test_set.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT         = Path(__file__).parent.parent
JSONL_PATH   = ROOT / "data" / "raw" / "claimreview_all.jsonl"
OUT_DIR      = ROOT / "test_set"
DATE_CUTOFF  = "2024-01-01"
POS_LIMIT    = 50

# Two-tier keyword filter.
#
# STRONG (1 match sufficient): terms almost exclusively used in clinical /
#   public-health contexts — false-positive rate is very low.
# WEAK   (2+ matches required): broader terms that can appear in politics,
#   economics, or general news — require co-occurrence to confirm health topic.
#
# Removed from previous single-tier set:
#   "health"  → matches "UnitedHealthcare", "financial health", etc.
#   "gene"/"dna" → matches genetics in political/legal contexts
#   "doctor"/"hospital" → appear in crime/legal stories
#   "radiation" → appears in nuclear/military contexts
#   "blood"/"heart"/"brain" → too generic in metaphorical usage

HEALTH_STRONG: frozenset[str] = frozenset({
    "vaccine", "vaccination", "immunization",
    "cancer", "tumor", "carcinoma", "chemotherapy", "oncology",
    "diabetes", "insulin", "glucose",
    "antibiotic", "antiviral", "antifungal",
    "pandemic", "epidemic", "covid",
    "cholesterol", "hypertension",
    "surgery", "surgical",
    "allergy", "allergic",
    "immune", "immunity", "antibody",
    "supplement", "vitamin",
    "obesity",
    "symptom", "diagnosis",
    "medication", "dosage",
    "infection", "infected",
    "pathogen", "bacterium", "bacteria",
    "clinical trial", "placebo",
    "mental health", "depression", "anxiety",
    "heart disease", "blood pressure",
})

HEALTH_WEAK: frozenset[str] = frozenset({
    "disease", "virus", "drug", "medicine", "treatment",
    "therapy", "patient", "clinical", "flu",
    "cure", "toxin", "lung", "liver", "kidney",
    "nutrition", "diet",
})


# ── Category 2: synthesised established facts ──────────────────────────────────
# Style: CDC / WHO / MedlinePlus authoritative statements

NEGATIVES_FACTS: list[dict] = [
    # Vaccines & Immunity
    {
        "claim_text": "The MMR vaccine is safe and does not cause autism; this has been confirmed by numerous large-scale studies.",
        "category": "vaccines_and_immunity",
    },
    {
        "claim_text": "Vaccines stimulate the immune system to produce antibodies without causing the disease itself.",
        "category": "vaccines_and_immunity",
    },
    {
        "claim_text": "Annual influenza vaccination is recommended by the CDC for everyone aged 6 months and older.",
        "category": "vaccines_and_immunity",
    },
    {
        "claim_text": "Herd immunity reduces the likelihood of infection for unimmunised individuals when a sufficient proportion of a community is immune.",
        "category": "vaccines_and_immunity",
    },
    {
        "claim_text": "COVID-19 vaccines authorised by the FDA underwent rigorous clinical trials and are continuously monitored for safety.",
        "category": "vaccines_and_immunity",
    },
    # Cardiovascular Health
    {
        "claim_text": "High blood pressure is a major risk factor for heart disease and stroke.",
        "category": "cardiovascular_health",
    },
    {
        "claim_text": "At least 150 minutes per week of moderate-intensity aerobic exercise reduces the risk of cardiovascular disease.",
        "category": "cardiovascular_health",
    },
    {
        "claim_text": "Smoking is a leading cause of heart disease, and quitting significantly reduces cardiovascular risk within months.",
        "category": "cardiovascular_health",
    },
    {
        "claim_text": "Diets high in saturated and trans fats are associated with elevated LDL cholesterol and greater cardiovascular risk.",
        "category": "cardiovascular_health",
    },
    {
        "claim_text": "Aspirin for primary prevention of cardiovascular disease is no longer routinely recommended for adults over 60 due to bleeding risk.",
        "category": "cardiovascular_health",
    },
    # Nutrition & Diet
    {
        "claim_text": "Adults should consume at least five servings of fruits and vegetables daily, according to WHO dietary guidelines.",
        "category": "nutrition_and_diet",
    },
    {
        "claim_text": "Excess sodium intake is associated with elevated blood pressure; the recommended limit is less than 2,300 mg per day for most adults.",
        "category": "nutrition_and_diet",
    },
    {
        "claim_text": "Whole grains contain more fibre, vitamins, and minerals than refined grains and are associated with reduced chronic disease risk.",
        "category": "nutrition_and_diet",
    },
    {
        "claim_text": "Dietary fibre from fruits, vegetables, and whole grains supports digestive health and reduces colorectal cancer risk.",
        "category": "nutrition_and_diet",
    },
    {
        "claim_text": "Breastfeeding provides optimal infant nutrition and is associated with reduced risk of infections, allergies, and obesity.",
        "category": "nutrition_and_diet",
    },
    # Infectious Disease Prevention
    {
        "claim_text": "Handwashing with soap and water for at least 20 seconds is one of the most effective ways to prevent the spread of infectious diseases.",
        "category": "infectious_disease_prevention",
    },
    {
        "claim_text": "Consistent and correct condom use reduces the risk of sexually transmitted infections including HIV.",
        "category": "infectious_disease_prevention",
    },
    {
        "claim_text": "Tuberculosis is an airborne disease caused by Mycobacterium tuberculosis and can be effectively treated with a course of antibiotics.",
        "category": "infectious_disease_prevention",
    },
    {
        "claim_text": "Antibiotics are effective against bacterial infections but do not treat viral infections such as the common cold or influenza.",
        "category": "infectious_disease_prevention",
    },
    {
        "claim_text": "Malaria is transmitted through the bites of infected female Anopheles mosquitoes and is preventable with appropriate protective measures.",
        "category": "infectious_disease_prevention",
    },
    # Cancer Screening
    {
        "claim_text": "Regular colorectal cancer screening starting at age 45 can detect cancer early or prevent it by removing precancerous polyps.",
        "category": "cancer_screening",
    },
    {
        "claim_text": "Mammography screening for women aged 40 to 74 can detect breast cancer before symptoms appear.",
        "category": "cancer_screening",
    },
    {
        "claim_text": "Pap smears can detect precancerous cervical changes that may develop into cancer if left untreated.",
        "category": "cancer_screening",
    },
    {
        "claim_text": "Annual skin examinations by a dermatologist are recommended for early detection of melanoma and other skin cancers.",
        "category": "cancer_screening",
    },
    {
        "claim_text": "Low-dose CT scanning is recommended for annual lung cancer screening in adults aged 50 to 80 with a significant smoking history.",
        "category": "cancer_screening",
    },
    # Diabetes Management
    {
        "claim_text": "Type 2 diabetes can often be prevented or delayed through weight management, regular physical activity, and a healthy diet.",
        "category": "diabetes_management",
    },
    {
        "claim_text": "People with type 1 diabetes require daily insulin therapy because their pancreas does not produce insulin.",
        "category": "diabetes_management",
    },
    {
        "claim_text": "Maintaining blood glucose within target ranges helps prevent long-term complications of diabetes, including nerve damage and kidney disease.",
        "category": "diabetes_management",
    },
    {
        "claim_text": "The A1C test measures average blood sugar levels over the past two to three months and is used to monitor diabetes management.",
        "category": "diabetes_management",
    },
    {
        "claim_text": "People with diabetes face higher cardiovascular risk and require regular monitoring of blood pressure and cholesterol levels.",
        "category": "diabetes_management",
    },
    # Mental Health Basics
    {
        "claim_text": "Depression is a common and treatable medical condition, not a personal weakness or character flaw.",
        "category": "mental_health_basics",
    },
    {
        "claim_text": "Cognitive behavioural therapy is an evidence-based treatment effective for depression, anxiety disorders, and other mental health conditions.",
        "category": "mental_health_basics",
    },
    {
        "claim_text": "Suicide is preventable, and people experiencing suicidal thoughts should seek professional help immediately.",
        "category": "mental_health_basics",
    },
    {
        "claim_text": "Sleep deprivation is associated with increased risk of depression, anxiety, and impaired cognitive function.",
        "category": "mental_health_basics",
    },
    {
        "claim_text": "Regular physical exercise has been shown in clinical studies to reduce symptoms of depression and anxiety.",
        "category": "mental_health_basics",
    },
]

# ── Category 3: synthesised unfalsifiable opinions ─────────────────────────────

NEGATIVES_UNFALSIFIABLE: list[dict] = [
    # Vague lifestyle advice
    {
        "claim_text": "Listening to your body is the most important guide to good health.",
        "type": "vague_lifestyle_advice",
    },
    {
        "claim_text": "A positive mindset can make a significant difference to your overall well-being.",
        "type": "vague_lifestyle_advice",
    },
    {
        "claim_text": "Living in harmony with nature is the foundation of true health.",
        "type": "vague_lifestyle_advice",
    },
    {
        "claim_text": "Taking time for yourself and practising self-care is essential for maintaining good health.",
        "type": "vague_lifestyle_advice",
    },
    {
        "claim_text": "Your body knows what it needs, so it is important to tune in to its signals.",
        "type": "vague_lifestyle_advice",
    },
    # Unquantifiable subjective feelings
    {
        "claim_text": "Natural remedies feel gentler on the body than pharmaceutical medications.",
        "type": "unquantifiable_subjective_feeling",
    },
    {
        "claim_text": "Many people feel more energised and alive after switching to an organic diet.",
        "type": "unquantifiable_subjective_feeling",
    },
    {
        "claim_text": "Yoga and meditation create a sense of inner balance that conventional medicine cannot replicate.",
        "type": "unquantifiable_subjective_feeling",
    },
    {
        "claim_text": "There is something deeply calming about herbal teas that prescription sleep aids cannot provide.",
        "type": "unquantifiable_subjective_feeling",
    },
    {
        "claim_text": "Spending time in nature feels restorative in a way that cannot be fully captured by science.",
        "type": "unquantifiable_subjective_feeling",
    },
    # Value judgments
    {
        "claim_text": "People should have the right to choose their own medical treatments without government interference.",
        "type": "value_judgment",
    },
    {
        "claim_text": "Healthcare should be treated as a human right, not a commodity available only to those who can afford it.",
        "type": "value_judgment",
    },
    {
        "claim_text": "Informed consent is more important than compliance when it comes to medical decisions.",
        "type": "value_judgment",
    },
    {
        "claim_text": "Doctors should respect patients' wishes to explore alternative approaches before resorting to conventional treatments.",
        "type": "value_judgment",
    },
    {
        "claim_text": "No one should be compelled to take a medication or vaccine against their will.",
        "type": "value_judgment",
    },
    # Overly broad generalizations
    {
        "claim_text": "A balanced diet is the key to good health.",
        "type": "overly_broad_generalization",
    },
    {
        "claim_text": "Everything in moderation is the best approach to nutrition.",
        "type": "overly_broad_generalization",
    },
    {
        "claim_text": "Staying active and eating well can prevent most lifestyle diseases.",
        "type": "overly_broad_generalization",
    },
    {
        "claim_text": "Good sleep is one of the best medicines there is.",
        "type": "overly_broad_generalization",
    },
    {
        "claim_text": "Stress is harmful to health and should be avoided whenever possible.",
        "type": "overly_broad_generalization",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge_field(record: dict, all_keys: list[str]) -> dict:
    return {k: record.get(k) for k in all_keys}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Category 1: health-related English reviews from 2024 onwards ────────────
    print("Scanning claimreview_all.jsonl for health-related 2024 records…")
    positives: list[dict] = []
    raw_scanned = 0

    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_scanned += 1
            r = json.loads(line)

            if r.get("language") != "English":
                continue
            if (r.get("date") or "") < DATE_CUTOFF:
                continue

            claim_lower = (r.get("claim_text") or "").lower()
            strong_hits = sum(1 for kw in HEALTH_STRONG if kw in claim_lower)
            weak_hits   = sum(1 for kw in HEALTH_WEAK   if kw in claim_lower)
            if strong_hits < 1 and weak_hits < 2:
                continue

            idx = len(positives) + 1
            positives.append({
                "id":          f"pos_{idx:03d}",
                "claim_text":  r.get("claim_text", ""),
                "gold_label":  "RECOMMEND_NEW_CHECK",
                "is_variant":  False,
                "source":      f"{r.get('publisher', 'unknown')}_2024",
                "review_url":  r.get("review_url", ""),
            })

            if len(positives) >= POS_LIMIT:
                break

    print(f"  Scanned {raw_scanned:,} records → {len(positives)} positives (health-related, 2024+)")
    write_jsonl(OUT_DIR / "positives.jsonl", positives)
    print(f"  Saved → test_set/positives.jsonl")

    # ── Category 2: established facts ─────────────────────────────────────────
    neg_facts = [
        {
            "id":         f"neg_fact_{i+1:03d}",
            "claim_text": item["claim_text"],
            "gold_label": "DO_NOT_CHECK",
            "category":   item["category"],
            "source":     "simulated_cdc_who",
        }
        for i, item in enumerate(NEGATIVES_FACTS)
    ]
    write_jsonl(OUT_DIR / "negatives_facts.jsonl", neg_facts)
    print(f"  Saved → test_set/negatives_facts.jsonl  ({len(neg_facts)} records)")

    # ── Category 3: unfalsifiable opinions ────────────────────────────────────
    neg_unfal = [
        {
            "id":         f"neg_unfal_{i+1:03d}",
            "claim_text": item["claim_text"],
            "gold_label": "DO_NOT_CHECK",
            "type":       item["type"],
            "source":     "synthesized",
        }
        for i, item in enumerate(NEGATIVES_UNFALSIFIABLE)
    ]
    write_jsonl(OUT_DIR / "negatives_unfalsifiable.jsonl", neg_unfal)
    print(f"  Saved → test_set/negatives_unfalsifiable.jsonl  ({len(neg_unfal)} records)")

    # ── Merge ──────────────────────────────────────────────────────────────────
    all_keys = [
        "id", "claim_text", "gold_label",
        "is_variant", "category", "type", "source", "review_url",
    ]
    merged = [
        merge_field(r, all_keys)
        for r in positives + neg_facts + neg_unfal
    ]
    write_jsonl(OUT_DIR / "test_set.jsonl", merged)

    print(f"\n{'='*50}")
    print(f"  positives                : {len(positives):>4}")
    print(f"  negatives_facts          : {len(neg_facts):>4}")
    print(f"  negatives_unfalsifiable  : {len(neg_unfal):>4}")
    print(f"  {'─'*30}")
    print(f"  total                    : {len(merged):>4}")
    print(f"  Saved → test_set/test_set.jsonl")


if __name__ == "__main__":
    main()

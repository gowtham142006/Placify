import numpy as np
import pandas as pd
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------

RANDOM_SEED = 42
N_STUDENTS = 10_000
OUTPUT_FILENAME = "student_placement_prediction_dataset_v3.csv"

rng = np.random.default_rng(RANDOM_SEED)


# -----------------------------
# Dataset columns (unchanged)
# -----------------------------

FEATURE_COLUMNS = [
    "age",
    "gender",
    "cgpa",
    "branch",
    "college_tier",
    "internships_count",
    "projects_count",
    "certifications_count",
    "coding_skill_score",
    "communication_skill_score",
    "aptitude_score",
    "logical_reasoning_score",
    "mock_interview_score",
]

TARGET_COLUMN = "placement_status"

ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


# =====================================================
# 1. Latent student quality factor
# =====================================================
# This is the key improvement: a single latent variable that drives
# realistic correlations between related features. It is NOT included
# in the final dataset (no data leakage).
# Standard normal: mean=0, std=1. Higher = better student.

latent_quality = rng.normal(loc=0.0, scale=1.0, size=N_STUDENTS)


# =====================================================
# 2. Non-predictive features (independent of quality)
# =====================================================

# Age: realistic distribution skewed toward 20-22
age = rng.choice(
    np.arange(18, 29),
    size=N_STUDENTS,
    p=[0.12, 0.16, 0.18, 0.18, 0.14, 0.09, 0.06, 0.03, 0.02, 0.01, 0.01]
)

# Gender: explicitly NOT used in placement score — no demographic bias
gender = rng.choice(
    ["Male", "Female", "Other"],
    size=N_STUDENTS,
    p=[0.55, 0.44, 0.01]
)


# =====================================================
# 3. Features correlated with latent quality
# =====================================================

# --- CGPA ---
# Strong students get higher CGPAs. Formula:
# base CGPA ~ 7.5, shifted by latent quality (scaled by 0.9)
# plus individual noise (std=0.5)
cgpa = np.clip(
    7.5 + 0.9 * latent_quality + rng.normal(0, 0.5, N_STUDENTS),
    5.0,
    10.0
).round(2)


# --- Branch ---
# Branch has a small but realistic effect on placement.
# CSE/IT students have slightly better placement odds in practice.
branch = rng.choice(
    ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"],
    size=N_STUDENTS,
    p=[0.30, 0.20, 0.18, 0.12, 0.12, 0.08]
)

# Branch advantage mapping (small effect, not dominant)
branch_bonus = np.where(
    np.isin(branch, ["CSE", "IT"]), 0.10,
    np.where(np.isin(branch, ["ECE", "EEE"]), 0.0, -0.05)
)


# --- College Tier ---
college_tier = rng.choice(
    [1, 2, 3],
    size=N_STUDENTS,
    p=[0.25, 0.50, 0.25]
)


# --- Internships, Projects, Certifications ---
# Higher quality students accumulate more experience.
# Lambda for Poisson is tied to latent quality.

# Internships: base lambda=1.0, influenced by quality
internship_lambda = np.clip(1.0 + 0.5 * latent_quality, 0.2, 3.5)
internships_count = np.clip(
    rng.poisson(lam=internship_lambda),
    0, 4
)

# Projects: base lambda=2.0, influenced by quality
project_lambda = np.clip(2.0 + 0.6 * latent_quality, 0.3, 5.0)
projects_count = np.clip(
    rng.poisson(lam=project_lambda),
    0, 6
)

# Certifications: base lambda=2.0, influenced by quality
cert_lambda = np.clip(2.0 + 0.4 * latent_quality, 0.3, 5.0)
certifications_count = np.clip(
    rng.poisson(lam=cert_lambda),
    0, 8
)


# --- Backlogs ---
# Weaker students have more backlogs.
# Lambda inversely tied to quality.
backlog_lambda = np.clip(0.8 - 0.5 * latent_quality, 0.05, 3.0)
backlogs = np.clip(
    rng.poisson(lam=backlog_lambda),
    0, 5
)


# =====================================================
# 4. Skill scores (correlated with each other via latent)
# =====================================================
# Each skill score = base (from latent quality) + individual noise.
# This creates realistic inter-skill correlations (~0.3-0.5).

def generate_skill_score(latent, base_mean, latent_weight, noise_std, rng):
    """Generate a skill score correlated with latent quality.
    
    Args:
        latent: latent quality array
        base_mean: center of the score distribution
        latent_weight: how strongly latent quality affects score
        noise_std: individual noise standard deviation
        rng: random number generator
    
    Returns:
        Integer skill scores clipped to [40, 95]
    """
    score = base_mean + latent_weight * latent + rng.normal(0, noise_std, len(latent))
    return np.clip(score, 40, 95).round().astype(int)


# Coding: most influenced by latent quality
coding_skill_score = generate_skill_score(
    latent_quality, base_mean=68, latent_weight=10, noise_std=7, rng=rng
)

# Communication: moderately influenced (some students are good coders but poor communicators)
communication_skill_score = generate_skill_score(
    latent_quality, base_mean=70, latent_weight=7, noise_std=9, rng=rng
)

# Aptitude: strongly influenced
aptitude_score = generate_skill_score(
    latent_quality, base_mean=69, latent_weight=9, noise_std=7, rng=rng
)

# Logical reasoning: strongly influenced
logical_reasoning_score = generate_skill_score(
    latent_quality, base_mean=68, latent_weight=9, noise_std=7, rng=rng
)

# Mock interview: moderately influenced (interview skill varies more)
mock_interview_score = generate_skill_score(
    latent_quality, base_mean=67, latent_weight=8, noise_std=9, rng=rng
)


# =====================================================
# 5. Placement probability (stronger but still probabilistic)
# =====================================================

# Normalize features to approximately 0-1 range
cgpa_norm = (cgpa - 5.0) / 5.0  # 0.0 to 1.0

coding_norm = coding_skill_score / 100.0
communication_norm = communication_skill_score / 100.0
aptitude_norm = aptitude_score / 100.0
logical_norm = logical_reasoning_score / 100.0
interview_norm = mock_interview_score / 100.0

internship_norm = np.minimum(internships_count / 4.0, 1.0)
project_norm = np.minimum(projects_count / 6.0, 1.0)
certification_norm = np.minimum(certifications_count / 8.0, 1.0)

college_norm = (3 - college_tier) / 2.0  # Tier1=1.0, Tier2=0.5, Tier3=0.0
backlog_penalty = np.minimum(backlogs / 5.0, 1.0)


# Weighted placement score
# NOTE: gender is deliberately excluded — no demographic bias
placement_score = (
    0.22 * cgpa_norm
    + 0.18 * coding_norm
    + 0.10 * communication_norm
    + 0.10 * aptitude_norm
    + 0.08 * logical_norm
    + 0.15 * interview_norm
    + 0.05 * internship_norm
    + 0.03 * project_norm
    + 0.02 * certification_norm
    + 0.03 * college_norm
    + 0.04 * branch_bonus           # Small branch effect
    - 0.18 * backlog_penalty         # Stronger backlog penalty
)


# Add controlled noise (reduced from v2's 0.08 to 0.04)
noise = rng.normal(0, 0.04, N_STUDENTS)
placement_score = placement_score + noise


# Convert score to probability via sigmoid
# Steeper sigmoid (k=12 vs v2's k=8) gives better separation.
# Threshold=0.52 centers the distribution around ~55-60% placed.
placement_probability = 1.0 / (
    1.0 + np.exp(-12.0 * (placement_score - 0.52))
)


# Generate final target via Bernoulli draw (probabilistic, not deterministic)
placement_status = np.where(
    rng.random(N_STUDENTS) < placement_probability,
    "Placed",
    "Not Placed"
)


# =====================================================
# 6. Create DataFrame
# =====================================================

df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "cgpa": cgpa,
    "branch": branch,
    "college_tier": college_tier,
    "internships_count": internships_count,
    "projects_count": projects_count,
    "certifications_count": certifications_count,
    "coding_skill_score": coding_skill_score,
    "communication_skill_score": communication_skill_score,
    "aptitude_score": aptitude_score,
    "logical_reasoning_score": logical_reasoning_score,
    "mock_interview_score": mock_interview_score,
    "backlogs": backlogs,
    "placement_status": placement_status,
})

# NOTE: latent_quality is NOT in the DataFrame — no data leakage.


# =====================================================
# 7. Save dataset
# =====================================================

output_path = Path(__file__).resolve().parent.parent / "data" / OUTPUT_FILENAME
df.to_csv(output_path, index=False)


# =====================================================
# 8. Validation output
# =====================================================

print("=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)

print(f"\nOutput: {output_path}")
print(f"\nShape: {df.shape}")

print(f"\nMissing values:\n{df.isnull().sum()}")

print(f"\nDuplicate rows: {df.duplicated().sum()}")

print(f"\nTarget distribution:")
target_counts = df["placement_status"].value_counts()
target_pcts = df["placement_status"].value_counts(normalize=True) * 100
for label in target_counts.index:
    print(f"  {label}: {target_counts[label]} ({target_pcts[label]:.1f}%)")

print(f"\nFeature ranges:")
for col in df.columns:
    if df[col].dtype in ["int64", "float64", "int32"]:
        print(f"  {col}: [{df[col].min()}, {df[col].max()}]")
    else:
        vals = df[col].unique()
        print(f"  {col}: {sorted(vals)}")

# Placement rate by CGPA groups
print(f"\nPlacement rate by CGPA groups:")
cgpa_bins = pd.cut(df["cgpa"], bins=[0, 6, 7, 8, 10], labels=["<6", "6-7", "7-8", "8+"])
cgpa_placement = df.groupby(cgpa_bins, observed=False)["placement_status"].apply(
    lambda x: (x == "Placed").mean() * 100
)
for group, rate in cgpa_placement.items():
    count = (cgpa_bins == group).sum()
    print(f"  CGPA {group}: {rate:.1f}% placed ({count} students)")

# Placement rate by coding skill groups
print(f"\nPlacement rate by coding skill groups:")
coding_bins = pd.cut(df["coding_skill_score"], bins=[0, 50, 65, 80, 100],
                     labels=["Low(<=50)", "Med(51-65)", "Good(66-80)", "High(81+)"])
coding_placement = df.groupby(coding_bins, observed=False)["placement_status"].apply(
    lambda x: (x == "Placed").mean() * 100
)
for group, rate in coding_placement.items():
    count = (coding_bins == group).sum()
    print(f"  Coding {group}: {rate:.1f}% placed ({count} students)")

# Placement rate by mock interview groups
print(f"\nPlacement rate by mock interview groups:")
mock_bins = pd.cut(df["mock_interview_score"], bins=[0, 50, 65, 80, 100],
                   labels=["Low(<=50)", "Med(51-65)", "Good(66-80)", "High(81+)"])
mock_placement = df.groupby(mock_bins, observed=False)["placement_status"].apply(
    lambda x: (x == "Placed").mean() * 100
)
for group, rate in mock_placement.items():
    count = (mock_bins == group).sum()
    print(f"  Mock Interview {group}: {rate:.1f}% placed ({count} students)")

# Placement rate by backlog count
print(f"\nPlacement rate by backlog count:")
for b in sorted(df["backlogs"].unique()):
    subset = df[df["backlogs"] == b]
    rate = (subset["placement_status"] == "Placed").mean() * 100
    print(f"  Backlogs={b}: {rate:.1f}% placed ({len(subset)} students)")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE — Ready for model training")
print("=" * 60)
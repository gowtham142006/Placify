import numpy as np
import pandas as pd
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------

RANDOM_SEED = 42
N_STUDENTS = 10_000

rng = np.random.default_rng(RANDOM_SEED)


# -----------------------------
# Dataset columns
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

# -----------------------------
# Generate student features
# -----------------------------

age = rng.choice(
    np.arange(18, 29),
    size=N_STUDENTS,
    p=[0.12, 0.16, 0.18, 0.18, 0.14, 0.09, 0.06, 0.03, 0.02, 0.01, 0.01]
)

gender = rng.choice(
    ["Male", "Female", "Other"],
    size=N_STUDENTS,
    p=[0.55, 0.44, 0.01]
)

cgpa = np.clip(
    rng.normal(loc=7.5, scale=1.1, size=N_STUDENTS),
    5.0,
    10.0
).round(2)

branch = rng.choice(
    ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"],
    size=N_STUDENTS,
    p=[0.30, 0.20, 0.18, 0.12, 0.12, 0.08]
)

college_tier = rng.choice(
    [1, 2, 3],
    size=N_STUDENTS,
    p=[0.25, 0.50, 0.25]
)

internships_count = np.clip(
    rng.poisson(lam=1.2, size=N_STUDENTS),
    0,
    4
)

projects_count = np.clip(
    rng.poisson(lam=2.2, size=N_STUDENTS),
    0,
    6
)

certifications_count = np.clip(
    rng.poisson(lam=2.0, size=N_STUDENTS),
    0,
    8
)

backlogs = np.clip(
    rng.poisson(lam=0.5, size=N_STUDENTS),
    0,
    5
)


# -----------------------------
# Skill scores
# -----------------------------

coding_skill_score = np.clip(
    rng.normal(loc=68, scale=15, size=N_STUDENTS),
    40,
    95
).round().astype(int)

communication_skill_score = np.clip(
    rng.normal(loc=70, scale=14, size=N_STUDENTS),
    40,
    95
).round().astype(int)

aptitude_score = np.clip(
    rng.normal(loc=69, scale=15, size=N_STUDENTS),
    40,
    95
).round().astype(int)

logical_reasoning_score = np.clip(
    rng.normal(loc=68, scale=15, size=N_STUDENTS),
    40,
    95
).round().astype(int)

mock_interview_score = np.clip(
    rng.normal(loc=67, scale=16, size=N_STUDENTS),
    40,
    95
).round().astype(int)

# -----------------------------
# Generate placement probability
# -----------------------------

# Normalize important scores to approximately 0-1
cgpa_score = (cgpa - 5.0) / 5.0

coding_score = coding_skill_score / 100
communication_score = communication_skill_score / 100
aptitude_score_norm = aptitude_score / 100
logical_score = logical_reasoning_score / 100
interview_score = mock_interview_score / 100

internship_score = np.minimum(internships_count / 4, 1.0)
project_score = np.minimum(projects_count / 6, 1.0)
certification_score = np.minimum(certifications_count / 8, 1.0)

college_score = (3 - college_tier) / 2
backlog_penalty = np.minimum(backlogs / 5, 1.0)


# Weighted placement score
placement_score = (
    0.20 * cgpa_score
    + 0.18 * coding_score
    + 0.12 * communication_score
    + 0.12 * aptitude_score_norm
    + 0.10 * logical_score
    + 0.15 * interview_score
    + 0.05 * internship_score
    + 0.03 * project_score
    + 0.02 * certification_score
    + 0.02 * college_score
    - 0.15 * backlog_penalty
)


# Add controlled randomness
noise = rng.normal(0, 0.08, N_STUDENTS)

placement_score = placement_score + noise


# Convert score to probability
placement_probability = 1 / (
    1 + np.exp(-8 * (placement_score - 0.50))
)


# Generate final target
placement_status = np.where(
    rng.random(N_STUDENTS) < placement_probability,
    "Placed",
    "Not Placed"
)

# -----------------------------
# Create dataset
# -----------------------------

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


# -----------------------------
# Save dataset
# -----------------------------

output_path = Path(__file__).resolve().parent.parent / "data" / "student_placement_prediction_dataset_v2.csv"

df.to_csv(output_path, index=False)

print(f"Dataset generated successfully: {output_path}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("\nPlacement distribution:")
print(df["placement_status"].value_counts())
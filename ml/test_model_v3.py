
import joblib
import pandas as pd

MODEL_PATH = "ml/models/placement_model_v3.joblib"

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame([{
    "age": 22,
    "gender": "Male",
    "cgpa": 8.0,
    "branch": "CSE",
    "college_tier": 1,
    "internships_count": 2,
    "projects_count": 3,
    "certifications_count": 3,
    "coding_skill_score": 80,
    "communication_skill_score": 75,
    "aptitude_score": 78,
    "logical_reasoning_score": 76,
    "mock_interview_score": 80,
    "backlogs": 0
}])

probabilities = model.predict_proba(sample)[0]
classes = model.classes_

print("Model loaded successfully")
print("Classes:", classes)
print("Probabilities:", probabilities)

placed_index = list(classes).index("Placed")
placed_probability = probabilities[placed_index]

threshold = 0.55
prediction = "Placed" if placed_probability >= threshold else "Not Placed"

print("Placement probability:", round(float(placed_probability), 4))
print("Prediction at threshold", threshold, ":", prediction)

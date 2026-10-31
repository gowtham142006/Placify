import { useState } from "react";
import { Sparkles, Loader2 } from "lucide-react";
import FormInput from "./FormInput";
import FormSelect from "./FormSelect";
import FormSlider from "./FormSlider";
import PredictionResult from "./PredictionResult";
import { predictPlacement } from "../services/predictionService";

/**
 * Converts form state into the exact JSON payload expected by Spring Boot.
 * All numeric fields are explicitly cast to numbers.
 * Empty‑string number fields are converted to null so validation can catch them.
 */
function preparePayload(formData) {
    const toInt = (v) => (v === "" || v === null || v === undefined ? null : Number(v));
    const toFloat = (v) => (v === "" || v === null || v === undefined ? null : parseFloat(v));

    return {
        age: toInt(formData.age),
        gender: formData.gender,
        cgpa: toFloat(formData.cgpa),
        branch: formData.branch,
        collegeTier: toInt(formData.collegeTier),
        internshipsCount: toInt(formData.internshipsCount),
        projectsCount: toInt(formData.projectsCount),
        certificationsCount: toInt(formData.certificationsCount),
        codingSkillScore: Number(formData.codingSkillScore),
        communicationSkillScore: Number(formData.communicationSkillScore),
        aptitudeScore: Number(formData.aptitudeScore),
        logicalReasoningScore: Number(formData.logicalReasoningScore),
        mockInterviewScore: Number(formData.mockInterviewScore),
        backlogs: toInt(formData.backlogs),
    };
}

function PredictionForm() {
    const [formData, setFormData] = useState({
        age: "",
        gender: "",
        cgpa: "",
        branch: "",
        collegeTier: "",
        internshipsCount: "",
        projectsCount: "",
        certificationsCount: "",
        codingSkillScore: 50,
        communicationSkillScore: 50,
        aptitudeScore: 50,
        logicalReasoningScore: 50,
        mockInterviewScore: 50,
        backlogs: "",
    });

    const [errors, setErrors] = useState({});
    const [predictionResult, setPredictionResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [apiError, setApiError] = useState(null);

    const handleChange = (e) => {
        const { name, value, type } = e.target;

        setFormData((prev) => ({
            ...prev,
            [name]: type === "range" || type === "number" ? Number(value) : value,
        }));

        // Clear error when field is modified
        if (errors[name]) {
            setErrors((prev) => ({ ...prev, [name]: "" }));
        }

        // Clear API error when user edits the form
        if (apiError) {
            setApiError(null);
        }
    };

    const validate = () => {
        const newErrors = {};

        // Required fields & bounds
        if (!formData.age && formData.age !== 0) {
            newErrors.age = "Age is required";
        } else if (formData.age < 16 || formData.age > 35) {
            newErrors.age = "Age must be between 16 and 35";
        }

        if (!formData.gender) {
            newErrors.gender = "Gender is required";
        }

        if (formData.cgpa === "" || formData.cgpa === null || formData.cgpa === undefined) {
            newErrors.cgpa = "CGPA is required";
        } else if (formData.cgpa < 0 || formData.cgpa > 10) {
            newErrors.cgpa = "CGPA must be between 0 and 10";
        }

        if (!formData.branch) {
            newErrors.branch = "Branch is required";
        }

        if (!formData.collegeTier) {
            newErrors.collegeTier = "College Tier is required";
        }

        if (formData.internshipsCount === "" || formData.internshipsCount === null) {
            newErrors.internshipsCount = "Internships count is required";
        } else if (formData.internshipsCount < 0) {
            newErrors.internshipsCount = "Cannot be negative";
        }

        if (formData.projectsCount === "" || formData.projectsCount === null) {
            newErrors.projectsCount = "Projects count is required";
        } else if (formData.projectsCount < 0) {
            newErrors.projectsCount = "Cannot be negative";
        }

        if (formData.certificationsCount === "" || formData.certificationsCount === null) {
            newErrors.certificationsCount = "Certifications count is required";
        } else if (formData.certificationsCount < 0) {
            newErrors.certificationsCount = "Cannot be negative";
        }

        if (formData.backlogs === "" || formData.backlogs === null) {
            newErrors.backlogs = "Active backlogs is required";
        } else if (formData.backlogs < 0) {
            newErrors.backlogs = "Cannot be negative";
        }

        // Sliders bounds
        if (formData.codingSkillScore < 0 || formData.codingSkillScore > 100) {
            newErrors.codingSkillScore = "Must be between 0 and 100";
        }

        if (formData.communicationSkillScore < 0 || formData.communicationSkillScore > 100) {
            newErrors.communicationSkillScore = "Must be between 0 and 100";
        }

        if (formData.aptitudeScore < 0 || formData.aptitudeScore > 100) {
            newErrors.aptitudeScore = "Must be between 0 and 100";
        }

        if (formData.logicalReasoningScore < 0 || formData.logicalReasoningScore > 100) {
            newErrors.logicalReasoningScore = "Must be between 0 and 100";
        }

        if (formData.mockInterviewScore < 0 || formData.mockInterviewScore > 100) {
            newErrors.mockInterviewScore = "Must be between 0 and 100";
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        // Prevent duplicate submissions
        if (isLoading) return;

        if (!validate()) return;

        setIsLoading(true);
        setApiError(null);
        setPredictionResult(null);

        try {
            const payload = preparePayload(formData);
            const result = await predictPlacement(payload);
            setPredictionResult(result);
        } catch (error) {
            setApiError(error.message || "An unexpected error occurred. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleReset = () => {
        setPredictionResult(null);
        setApiError(null);
        window.scrollTo({ top: 0, behavior: "smooth" });
    };

    return (
        <div className="w-full max-w-4xl mx-auto bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden">
            <div className="p-8 md:p-12">
                <h2 className="text-2xl md:text-3xl font-extrabold text-slate-900 mb-2 tracking-tight">
                    Student Details
                </h2>
                <p className="text-slate-500 mb-10 text-lg">
                    Fill in your academic and skill details to predict your placement chances.
                </p>

                <form onSubmit={handleSubmit} className="space-y-12">
                    {/* Personal & Academic Details */}
                    <div>
                        <h3 className="text-xl font-bold text-slate-900 mb-6 pb-3 border-b border-slate-100 flex items-center gap-2">
                            Personal &amp; Academic Details
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-8">
                            <FormInput
                                id="age"
                                name="age"
                                label="Age"
                                type="number"
                                min="16"
                                value={formData.age}
                                onChange={handleChange}
                                error={errors.age}
                            />
                            <FormSelect
                                id="gender"
                                name="gender"
                                label="Gender"
                                value={formData.gender}
                                onChange={handleChange}
                                error={errors.gender}
                                options={[
                                    { value: "Male", label: "Male" },
                                    { value: "Female", label: "Female" },
                                ]}
                            />
                            <FormInput
                                id="cgpa"
                                name="cgpa"
                                label="CGPA (out of 10)"
                                type="number"
                                step="0.01"
                                min="0"
                                value={formData.cgpa}
                                onChange={handleChange}
                                error={errors.cgpa}
                            />
                            <FormSelect
                                id="branch"
                                name="branch"
                                label="Branch"
                                value={formData.branch}
                                onChange={handleChange}
                                error={errors.branch}
                                options={[
                                    { value: "CSE", label: "CSE" },
                                    { value: "IT", label: "IT" },
                                    { value: "ECE", label: "ECE" },
                                    { value: "EEE", label: "EEE" },
                                    { value: "Mechanical", label: "Mechanical" },
                                    { value: "Civil", label: "Civil" },
                                    { value: "Other", label: "Other" },
                                ]}
                            />
                            <FormSelect
                                id="collegeTier"
                                name="collegeTier"
                                label="College Tier"
                                value={formData.collegeTier}
                                onChange={handleChange}
                                error={errors.collegeTier}
                                options={[
                                    { value: "1", label: "Tier 1" },
                                    { value: "2", label: "Tier 2" },
                                    { value: "3", label: "Tier 3" },
                                ]}
                            />
                            <FormInput
                                id="backlogs"
                                name="backlogs"
                                label="Active Backlogs"
                                type="number"
                                min="0"
                                value={formData.backlogs}
                                onChange={handleChange}
                                error={errors.backlogs}
                            />
                        </div>
                    </div>

                    {/* Experience & Projects */}
                    <div>
                        <h3 className="text-xl font-bold text-slate-900 mb-6 pb-3 border-b border-slate-100 flex items-center gap-2">
                            Experience &amp; Projects
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-x-6 gap-y-8">
                            <FormInput
                                id="internshipsCount"
                                name="internshipsCount"
                                label="Internships Count"
                                type="number"
                                min="0"
                                value={formData.internshipsCount}
                                onChange={handleChange}
                                error={errors.internshipsCount}
                            />
                            <FormInput
                                id="projectsCount"
                                name="projectsCount"
                                label="Projects Count"
                                type="number"
                                min="0"
                                value={formData.projectsCount}
                                onChange={handleChange}
                                error={errors.projectsCount}
                            />
                            <FormInput
                                id="certificationsCount"
                                name="certificationsCount"
                                label="Certifications Count"
                                type="number"
                                min="0"
                                value={formData.certificationsCount}
                                onChange={handleChange}
                                error={errors.certificationsCount}
                            />
                        </div>
                    </div>

                    {/* Skills Assessment */}
                    <div>
                        <h3 className="text-xl font-bold text-slate-900 mb-6 pb-3 border-b border-slate-100 flex items-center gap-2">
                            Skills Assessment
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-10">
                            <FormSlider
                                id="codingSkillScore"
                                name="codingSkillScore"
                                label="Coding Skill Score"
                                value={formData.codingSkillScore}
                                onChange={handleChange}
                                error={errors.codingSkillScore}
                            />
                            <FormSlider
                                id="communicationSkillScore"
                                name="communicationSkillScore"
                                label="Communication Skill Score"
                                value={formData.communicationSkillScore}
                                onChange={handleChange}
                                error={errors.communicationSkillScore}
                            />
                            <FormSlider
                                id="aptitudeScore"
                                name="aptitudeScore"
                                label="Aptitude Score"
                                value={formData.aptitudeScore}
                                onChange={handleChange}
                                error={errors.aptitudeScore}
                            />
                            <FormSlider
                                id="logicalReasoningScore"
                                name="logicalReasoningScore"
                                label="Logical Reasoning Score"
                                value={formData.logicalReasoningScore}
                                onChange={handleChange}
                                error={errors.logicalReasoningScore}
                            />
                            <FormSlider
                                id="mockInterviewScore"
                                name="mockInterviewScore"
                                label="Mock Interview Score"
                                value={formData.mockInterviewScore}
                                onChange={handleChange}
                                error={errors.mockInterviewScore}
                            />
                        </div>
                    </div>

                    {/* API Error Display */}
                    {apiError && (
                        <div className="rounded-2xl border border-red-200 bg-red-50 p-5 flex items-start gap-4">
                            <div className="flex items-center justify-center w-10 h-10 rounded-full bg-red-100 shrink-0">
                                <svg
                                    className="w-5 h-5 text-red-600"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                    strokeWidth={2}
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
                                    />
                                </svg>
                            </div>
                            <div className="flex-grow">
                                <h4 className="text-sm font-bold text-red-800">
                                    Prediction Failed
                                </h4>
                                <p className="mt-1 text-sm text-red-700">{apiError}</p>
                                <button
                                    type="button"
                                    onClick={() => setApiError(null)}
                                    className="mt-3 text-sm font-semibold text-red-600 hover:text-red-800 underline underline-offset-2 transition-colors"
                                >
                                    Dismiss
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Submit Button */}
                    <div className="pt-8">
                        <button
                            type="submit"
                            disabled={isLoading}
                            className={`w-full flex items-center justify-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-primary-600 to-accent-500 text-white font-bold text-lg shadow-lg shadow-primary-500/30 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-primary-500/30 ${
                                isLoading
                                    ? "opacity-70 cursor-not-allowed"
                                    : "hover:shadow-xl hover:shadow-primary-500/40 hover:-translate-y-1"
                            }`}
                        >
                            {isLoading ? (
                                <>
                                    <Loader2 className="w-6 h-6 animate-spin" />
                                    Predicting...
                                </>
                            ) : (
                                <>
                                    <Sparkles className="w-6 h-6 animate-pulse" />
                                    Predict Placement
                                </>
                            )}
                        </button>
                    </div>

                    {/* Prediction Result */}
                    <PredictionResult result={predictionResult} onReset={handleReset} />
                </form>
            </div>
        </div>
    );
}

export default PredictionForm;

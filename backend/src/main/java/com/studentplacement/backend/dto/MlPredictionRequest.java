package com.studentplacement.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigDecimal;

/**
 * DTO sent to the FastAPI ML service.
 * Field names use {@code @JsonProperty} to produce snake_case JSON keys.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MlPredictionRequest {

    @JsonProperty("age")
    private Integer age;

    @JsonProperty("gender")
    private String gender;

    @JsonProperty("cgpa")
    private BigDecimal cgpa;

    @JsonProperty("branch")
    private String branch;

    @JsonProperty("college_tier")
    private Integer collegeTier;

    @JsonProperty("internships_count")
    private Integer internshipsCount;

    @JsonProperty("projects_count")
    private Integer projectsCount;

    @JsonProperty("certifications_count")
    private Integer certificationsCount;

    @JsonProperty("coding_skill_score")
    private Integer codingSkillScore;

    @JsonProperty("communication_skill_score")
    private Integer communicationSkillScore;

    @JsonProperty("aptitude_score")
    private Integer aptitudeScore;

    @JsonProperty("logical_reasoning_score")
    private Integer logicalReasoningScore;

    @JsonProperty("mock_interview_score")
    private Integer mockInterviewScore;

    @JsonProperty("backlogs")
    private Integer backlogs;
}

package com.studentplacement.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * DTO representing the JSON response returned by the FastAPI ML service.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MlPredictionResponse {

    @JsonProperty("prediction")
    private String prediction;

    @JsonProperty("placement_probability")
    private Double placementProbability;

    @JsonProperty("threshold")
    private Double threshold;
}

package com.studentplacement.backend.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PredictionResponse {

    private Boolean prediction;

    private Double confidence;

    private String message;

    /** Probability of placement (0.0–1.0) returned by the ML model. */
    private Double placementProbability;

    /** Decision threshold used by the ML model. */
    private Double threshold;
}

package com.studentplacement.backend.service;

import com.studentplacement.backend.dto.MlPredictionRequest;
import com.studentplacement.backend.dto.MlPredictionResponse;
import com.studentplacement.backend.dto.PredictionRequest;
import com.studentplacement.backend.dto.PredictionResponse;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

@Service
public class PredictionService {

    private static final Logger log = LoggerFactory.getLogger(PredictionService.class);

    private final RestClient mlRestClient;

    public PredictionService(RestClient mlRestClient) {
        this.mlRestClient = mlRestClient;
    }

    /**
     * Sends student features to the FastAPI ML service and translates
     * the response into the backend's public API format.
     */
    public PredictionResponse predict(PredictionRequest request) {

        // Map incoming camelCase DTO → snake_case DTO for FastAPI
        MlPredictionRequest mlRequest = MlPredictionRequest.builder()
                .age(request.getAge())
                .gender(request.getGender())
                .cgpa(request.getCgpa())
                .branch(request.getBranch())
                .collegeTier(request.getCollegeTier())
                .internshipsCount(request.getInternshipsCount())
                .projectsCount(request.getProjectsCount())
                .certificationsCount(request.getCertificationsCount())
                .codingSkillScore(request.getCodingSkillScore())
                .communicationSkillScore(request.getCommunicationSkillScore())
                .aptitudeScore(request.getAptitudeScore())
                .logicalReasoningScore(request.getLogicalReasoningScore())
                .mockInterviewScore(request.getMockInterviewScore())
                .backlogs(request.getBacklogs())
                .build();

        MlPredictionResponse mlResponse;
        try {
            mlResponse = mlRestClient.post()
                    .uri("/predict")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(mlRequest)
                    .retrieve()
                    .body(MlPredictionResponse.class);
        } catch (RestClientException ex) {
            log.error("ML service call failed: {}", ex.getMessage(), ex);
            throw new RuntimeException(
                    "Prediction service is unavailable. Please ensure the ML service is running.", ex);
        }

        if (mlResponse == null || mlResponse.getPrediction() == null) {
            log.error("ML service returned an empty or invalid response");
            throw new RuntimeException("ML service returned an invalid response.");
        }

        boolean isPlaced = "Placed".equalsIgnoreCase(mlResponse.getPrediction());
        double probability = mlResponse.getPlacementProbability() != null
                ? mlResponse.getPlacementProbability() : 0.0;

        return PredictionResponse.builder()
                .prediction(isPlaced)
                .confidence(Math.round(probability * 10000.0) / 100.0)   // e.g. 0.8741 → 87.41
                .message(isPlaced ? "Likely to be placed" : "Unlikely to be placed")
                .placementProbability(probability)
                .threshold(mlResponse.getThreshold())
                .build();
    }
}


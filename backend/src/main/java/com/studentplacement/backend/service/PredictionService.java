package com.studentplacement.backend.service;

import com.studentplacement.backend.dto.MlPredictionRequest;
import com.studentplacement.backend.dto.MlPredictionResponse;
import com.studentplacement.backend.dto.PredictionRequest;
import com.studentplacement.backend.dto.PredictionResponse;
import com.studentplacement.backend.entity.Student;
import com.studentplacement.backend.repository.StudentRepository;

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
    private final StudentRepository studentRepository;

    public PredictionService(RestClient mlRestClient, StudentRepository studentRepository) {
        this.mlRestClient = mlRestClient;
        this.studentRepository = studentRepository;
    }

    /**
     * Sends student features to the FastAPI ML service, translates the response,
     * and persists the student details with the prediction outcome to Supabase PostgreSQL.
     *
     * Note on transaction management: The external network call to FastAPI is executed
     * outside of any database transaction so connection pool resources are not held hostage
     * during HTTP I/O. The subsequent save operation is executed transactionally by
     * Spring Data JPA's SimpleJpaRepository.
     */
    public PredictionResponse predict(PredictionRequest request) {

        // 1. Map incoming camelCase DTO → snake_case DTO for FastAPI
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

        // 2. Call external FastAPI ML service (no DB transaction held during network I/O)
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

        // 3. Map all 14 input features and the predicted placementStatus to Student entity
        Student student = Student.builder()
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
                .placementStatus(isPlaced)
                .build();

        // 4. Persist to Supabase database (only after successful ML prediction)
        Student saved = studentRepository.save(student);
        log.info("Persisted student prediction record (id: {}, placementStatus: {})",
                saved != null ? saved.getId() : null, isPlaced);

        // 5. Return standard response to preserve React frontend contract
        return PredictionResponse.builder()
                .prediction(isPlaced)
                .confidence(Math.round(probability * 10000.0) / 100.0)   // e.g. 0.8741 → 87.41
                .message(isPlaced ? "Likely to be placed" : "Unlikely to be placed")
                .placementProbability(probability)
                .threshold(mlResponse.getThreshold())
                .build();
    }
}

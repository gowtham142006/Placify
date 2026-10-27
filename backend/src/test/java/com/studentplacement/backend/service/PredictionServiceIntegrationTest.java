package com.studentplacement.backend.service;

import com.studentplacement.backend.dto.PredictionRequest;
import com.studentplacement.backend.dto.PredictionResponse;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import java.math.BigDecimal;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration test for PredictionService -> FastAPI.
 *
 * Prerequisites:
 *   FastAPI must be running at http://127.0.0.1:8000
 *   (start with: python -m uvicorn ml.main:app --host 127.0.0.1 --port 8000)
 */
class PredictionServiceIntegrationTest {

    private PredictionService service;

    @BeforeEach
    void setUp() {
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(Duration.ofSeconds(5));
        requestFactory.setReadTimeout(Duration.ofSeconds(10));

        RestClient mlRestClient = RestClient.builder()
                .baseUrl("http://127.0.0.1:8000")
                .requestFactory(requestFactory)
                .build();
        service = new PredictionService(mlRestClient);
    }

    @Test
    void strongCandidate_shouldBePredictedAsPlaced() {
        PredictionRequest request = PredictionRequest.builder()
                .age(22)
                .gender("Male")
                .cgpa(BigDecimal.valueOf(8.0))
                .branch("CSE")
                .collegeTier(1)
                .internshipsCount(2)
                .projectsCount(3)
                .certificationsCount(3)
                .codingSkillScore(80)
                .communicationSkillScore(75)
                .aptitudeScore(78)
                .logicalReasoningScore(76)
                .mockInterviewScore(80)
                .backlogs(0)
                .build();

        PredictionResponse response = service.predict(request);

        assertNotNull(response);
        assertTrue(response.getPrediction(), "Strong candidate should be predicted as Placed");
        assertNotNull(response.getPlacementProbability());
        assertTrue(response.getPlacementProbability() > 0.55,
                "Placement probability should exceed threshold");
        assertEquals(0.55, response.getThreshold());
        assertEquals("Likely to be placed", response.getMessage());
        assertTrue(response.getConfidence() > 55.0, "Confidence should reflect probability");
    }

    @Test
    void weakCandidate_shouldBePredictedAsNotPlaced() {
        PredictionRequest request = PredictionRequest.builder()
                .age(22)
                .gender("Male")
                .cgpa(BigDecimal.valueOf(4.5))
                .branch("ME")
                .collegeTier(3)
                .internshipsCount(0)
                .projectsCount(0)
                .certificationsCount(0)
                .codingSkillScore(20)
                .communicationSkillScore(25)
                .aptitudeScore(30)
                .logicalReasoningScore(28)
                .mockInterviewScore(20)
                .backlogs(5)
                .build();

        PredictionResponse response = service.predict(request);

        assertNotNull(response);
        assertFalse(response.getPrediction(), "Weak candidate should not be predicted as Placed");
        assertNotNull(response.getPlacementProbability());
        assertTrue(response.getPlacementProbability() < 0.55,
                "Placement probability should be below threshold");
        assertEquals(0.55, response.getThreshold());
        assertEquals("Unlikely to be placed", response.getMessage());
    }

    @Test
    void serviceUnavailable_shouldThrowClearException() {
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(Duration.ofMillis(500));
        requestFactory.setReadTimeout(Duration.ofMillis(500));

        // Connect to a port where nothing is listening
        RestClient brokenClient = RestClient.builder()
                .baseUrl("http://127.0.0.1:59999")
                .requestFactory(requestFactory)
                .build();
        PredictionService brokenService = new PredictionService(brokenClient);

        PredictionRequest request = PredictionRequest.builder()
                .age(22)
                .gender("Male")
                .cgpa(BigDecimal.valueOf(8.0))
                .branch("CSE")
                .collegeTier(1)
                .internshipsCount(2)
                .projectsCount(3)
                .certificationsCount(3)
                .codingSkillScore(80)
                .communicationSkillScore(75)
                .aptitudeScore(78)
                .logicalReasoningScore(76)
                .mockInterviewScore(80)
                .backlogs(0)
                .build();

        RuntimeException ex = assertThrows(RuntimeException.class, () -> brokenService.predict(request));
        assertTrue(ex.getMessage().contains("Prediction service is unavailable"),
                "Exception message should clearly indicate service unavailability: " + ex.getMessage());
    }
}

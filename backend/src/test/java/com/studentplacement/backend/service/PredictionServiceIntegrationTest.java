package com.studentplacement.backend.service;

import com.studentplacement.backend.dto.PredictionRequest;
import com.studentplacement.backend.dto.PredictionResponse;
import com.studentplacement.backend.entity.Student;
import com.studentplacement.backend.repository.StudentRepository;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import java.math.BigDecimal;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Integration and unit tests for PredictionService -> FastAPI and StudentRepository persistence.
 *
 * The StudentRepository is mocked to ensure test isolation and independence from a live
 * Supabase database connection.
 *
 * For tests requiring live FastAPI (e.g. strongCandidate / weakCandidate):
 *   FastAPI must be running at http://127.0.0.1:8000
 *   (start with: python -m uvicorn ml.main:app --host 127.0.0.1 --port 8000)
 */
class PredictionServiceIntegrationTest {

    private PredictionService service;
    private StudentRepository studentRepository;

    @BeforeEach
    void setUp() {
        studentRepository = mock(StudentRepository.class);
        when(studentRepository.save(any(Student.class))).thenAnswer(invocation -> invocation.getArgument(0));

        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(Duration.ofSeconds(5));
        requestFactory.setReadTimeout(Duration.ofSeconds(10));

        RestClient mlRestClient = RestClient.builder()
                .baseUrl("http://127.0.0.1:8000")
                .requestFactory(requestFactory)
                .build();

        service = new PredictionService(mlRestClient, studentRepository);
    }

    @Test
    void strongCandidate_shouldBePredictedAsPlaced_andSavedCorrectly() {
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

        // 1. Verify prediction response
        assertNotNull(response);
        assertTrue(response.getPrediction(), "Strong candidate should be predicted as Placed");
        assertNotNull(response.getPlacementProbability());
        assertTrue(response.getPlacementProbability() > 0.55,
                "Placement probability should exceed threshold");
        assertEquals(0.55, response.getThreshold());
        assertEquals("Likely to be placed", response.getMessage());
        assertTrue(response.getConfidence() > 55.0, "Confidence should reflect probability");

        // 2. Verify repository save was called exactly once
        ArgumentCaptor<Student> captor = ArgumentCaptor.forClass(Student.class);
        verify(studentRepository, times(1)).save(captor.capture());

        // 3. Verify all 14 input fields and placementStatus were correctly mapped
        Student savedStudent = captor.getValue();
        assertNotNull(savedStudent, "Saved student entity must not be null");
        assertEquals(Integer.valueOf(22), savedStudent.getAge(), "age must match");
        assertEquals("Male", savedStudent.getGender(), "gender must match");
        assertEquals(BigDecimal.valueOf(8.0), savedStudent.getCgpa(), "cgpa must match");
        assertEquals("CSE", savedStudent.getBranch(), "branch must match");
        assertEquals(Integer.valueOf(1), savedStudent.getCollegeTier(), "collegeTier must match");
        assertEquals(Integer.valueOf(2), savedStudent.getInternshipsCount(), "internshipsCount must match");
        assertEquals(Integer.valueOf(3), savedStudent.getProjectsCount(), "projectsCount must match");
        assertEquals(Integer.valueOf(3), savedStudent.getCertificationsCount(), "certificationsCount must match");
        assertEquals(Integer.valueOf(80), savedStudent.getCodingSkillScore(), "codingSkillScore must match");
        assertEquals(Integer.valueOf(75), savedStudent.getCommunicationSkillScore(), "communicationSkillScore must match");
        assertEquals(Integer.valueOf(78), savedStudent.getAptitudeScore(), "aptitudeScore must match");
        assertEquals(Integer.valueOf(76), savedStudent.getLogicalReasoningScore(), "logicalReasoningScore must match");
        assertEquals(Integer.valueOf(80), savedStudent.getMockInterviewScore(), "mockInterviewScore must match");
        assertEquals(Integer.valueOf(0), savedStudent.getBacklogs(), "backlogs must match");
        assertTrue(savedStudent.getPlacementStatus(), "placementStatus must be true for strong candidate");
    }

    @Test
    void weakCandidate_shouldBePredictedAsNotPlaced_andSavedCorrectly() {
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

        // 1. Verify prediction response
        assertNotNull(response);
        assertFalse(response.getPrediction(), "Weak candidate should not be predicted as Placed");
        assertNotNull(response.getPlacementProbability());
        assertTrue(response.getPlacementProbability() < 0.55,
                "Placement probability should be below threshold");
        assertEquals(0.55, response.getThreshold());
        assertEquals("Unlikely to be placed", response.getMessage());

        // 2. Verify repository save was called exactly once with placementStatus = false
        ArgumentCaptor<Student> captor = ArgumentCaptor.forClass(Student.class);
        verify(studentRepository, times(1)).save(captor.capture());

        Student savedStudent = captor.getValue();
        assertNotNull(savedStudent);
        assertFalse(savedStudent.getPlacementStatus(), "placementStatus must be false for weak candidate");
        assertEquals(Integer.valueOf(5), savedStudent.getBacklogs());
    }

    @Test
    void serviceUnavailable_shouldThrowClearException_andNeverSaveStudent() {
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(Duration.ofMillis(500));
        requestFactory.setReadTimeout(Duration.ofMillis(500));

        // Connect to a port where nothing is listening
        RestClient brokenClient = RestClient.builder()
                .baseUrl("http://127.0.0.1:59999")
                .requestFactory(requestFactory)
                .build();

        StudentRepository brokenRepo = mock(StudentRepository.class);
        PredictionService brokenService = new PredictionService(brokenClient, brokenRepo);

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

        // Verify that in case of ML service failure, save() was NEVER invoked
        verify(brokenRepo, never()).save(any(Student.class));
    }
}

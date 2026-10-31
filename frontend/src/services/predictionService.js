import API from "./api";

/**
 * Sends the 14 student features to Spring Boot for placement prediction.
 *
 * @param {Object} formData - The prepared prediction payload.
 * @returns {Promise<Object>} The prediction response from Spring Boot.
 * @throws {Error} A user-friendly error with a descriptive message.
 */
export const predictPlacement = async (formData) => {
    try {
        const response = await API.post("/predict", formData);
        return response.data;
    } catch (error) {
        if (error.code === "ECONNABORTED") {
            throw new Error(
                "Request timed out. Please check if the server is running and try again.",
                { cause: error }
            );
        }

        if (!error.response) {
            // Network error — server unreachable
            throw new Error(
                "Unable to connect to the prediction server. Please ensure the backend is running at " +
                    (import.meta.env.VITE_API_BASE_URL || "http://localhost:8080") +
                    " and try again.",
                { cause: error }
            );
        }

        // Server responded with an error status
        const status = error.response.status;
        const data = error.response.data;

        if (status === 400 || status === 422) {
            const detail =
                data?.message || data?.detail || "Invalid input data. Please check your entries.";
            throw new Error(detail, { cause: error });
        }

        if (status === 500) {
            const detail =
                data?.message ||
                "The prediction service encountered an internal error. The ML service may be unavailable.";
            throw new Error(detail, { cause: error });
        }

        throw new Error(
            data?.message || `Unexpected server error (HTTP ${status}). Please try again later.`,
            { cause: error }
        );
    }
};

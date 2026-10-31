import { CheckCircle, XCircle, TrendingUp, Target, MessageSquare } from "lucide-react";

/**
 * Displays the prediction result returned from Spring Boot.
 *
 * Expected props.result shape:
 * {
 *   prediction: boolean,
 *   confidence: number,          // already a percentage, e.g. 87.41
 *   message: string,
 *   placementProbability: number, // 0.0–1.0
 *   threshold: number             // e.g. 0.55
 * }
 */
function PredictionResult({ result, onReset }) {
    if (!result) return null;

    const isPlaced = result.prediction === true;
    const probabilityPercent = (result.placementProbability * 100).toFixed(2);
    const confidence = result.confidence?.toFixed(2) ?? "—";
    const thresholdPercent = result.threshold
        ? (result.threshold * 100).toFixed(0)
        : null;

    return (
        <div className="mt-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Main Status Card */}
            <div
                className={`rounded-2xl border-2 p-8 text-center transition-all duration-300 ${
                    isPlaced
                        ? "border-emerald-200 bg-gradient-to-br from-emerald-50 to-teal-50"
                        : "border-red-200 bg-gradient-to-br from-red-50 to-orange-50"
                }`}
            >
                {/* Status Icon */}
                <div className="flex justify-center mb-4">
                    <div
                        className={`flex items-center justify-center w-16 h-16 rounded-full ${
                            isPlaced
                                ? "bg-emerald-100 text-emerald-600"
                                : "bg-red-100 text-red-600"
                        }`}
                    >
                        {isPlaced ? (
                            <CheckCircle className="w-8 h-8" strokeWidth={2.2} />
                        ) : (
                            <XCircle className="w-8 h-8" strokeWidth={2.2} />
                        )}
                    </div>
                </div>

                {/* Status Heading */}
                <h3
                    className={`text-2xl md:text-3xl font-extrabold tracking-tight ${
                        isPlaced ? "text-emerald-800" : "text-red-800"
                    }`}
                >
                    {isPlaced ? "Likely to be Placed 🎉" : "Unlikely to be Placed"}
                </h3>

                {/* Backend Message */}
                <div className="flex items-center justify-center gap-2 mt-3">
                    <MessageSquare
                        className={`w-4 h-4 ${isPlaced ? "text-emerald-500" : "text-red-500"}`}
                        strokeWidth={2}
                    />
                    <p
                        className={`text-base font-medium ${
                            isPlaced ? "text-emerald-700" : "text-red-700"
                        }`}
                    >
                        {result.message}
                    </p>
                </div>
            </div>

            {/* Metrics Grid */}
            <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
                {/* Placement Probability */}
                <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                    <div className="flex items-center gap-2 mb-3">
                        <TrendingUp className="w-4 h-4 text-primary-500" strokeWidth={2.2} />
                        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                            Placement Probability
                        </span>
                    </div>
                    <p className="text-2xl font-bold text-slate-900 tabular-nums">
                        {probabilityPercent}%
                    </p>
                    {/* Progress Bar */}
                    <div className="mt-3 w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all duration-700 ease-out ${
                                isPlaced
                                    ? "bg-gradient-to-r from-emerald-400 to-teal-500"
                                    : "bg-gradient-to-r from-red-400 to-orange-500"
                            }`}
                            style={{ width: `${probabilityPercent}%` }}
                        />
                    </div>
                </div>

                {/* Confidence */}
                <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                    <div className="flex items-center gap-2 mb-3">
                        <Target className="w-4 h-4 text-accent-500" strokeWidth={2.2} />
                        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                            Confidence
                        </span>
                    </div>
                    <p className="text-2xl font-bold text-slate-900 tabular-nums">
                        {confidence}%
                    </p>
                    <div className="mt-3 w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                        <div
                            className="h-full rounded-full bg-gradient-to-r from-primary-400 to-accent-500 transition-all duration-700 ease-out"
                            style={{ width: `${confidence}%` }}
                        />
                    </div>
                </div>

                {/* Decision Threshold */}
                <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                    <div className="flex items-center gap-2 mb-3">
                        <svg
                            className="w-4 h-4 text-slate-400"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            strokeWidth={2.2}
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                d="M3 6h18M3 12h18M3 18h18"
                            />
                        </svg>
                        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                            Decision Threshold
                        </span>
                    </div>
                    <p className="text-2xl font-bold text-slate-900 tabular-nums">
                        {thresholdPercent ? `${thresholdPercent}%` : "—"}
                    </p>
                    <p className="mt-2 text-xs text-slate-400">
                        Probability above this threshold → Placed
                    </p>
                </div>
            </div>

            {/* Predict Again Button */}
            {onReset && (
                <div className="mt-8 text-center">
                    <button
                        type="button"
                        onClick={onReset}
                        className="inline-flex items-center gap-2 px-6 py-3 rounded-xl border-2 border-slate-200 text-slate-600 font-semibold text-sm hover:border-primary-300 hover:text-primary-600 hover:bg-primary-50/50 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary-500/30"
                    >
                        <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            strokeWidth={2.2}
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                            />
                        </svg>
                        Predict Again
                    </button>
                </div>
            )}
        </div>
    );
}

export default PredictionResult;


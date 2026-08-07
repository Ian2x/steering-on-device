import Foundation
import SteeringKit

enum GenerationPane: String, Sendable {
    case baseline
    case steered
    case actAdd
}

enum ResidualDirectionMode: String, Sendable, Codable {
    case semantic
    case randomMatchedNorm = "random-matched-norm"
}

enum Stage3RunMode: String, Sendable, Codable {
    case calibrateLogit = "calibrate-logit"
    case calibrateActAdd = "calibrate-actadd"
    case evaluate
    case evaluateRandom = "evaluate-random"
}

/// How the two live arms are held to a KL cost.
///
/// The app shipped only `greedyCumulativeCap`, and only on the logit arm. That combination is the
/// subject of `docs/phase6/on-device-rho`: a single cumulative budget, spent first-come-first-served
/// with no lookahead and nothing reserved for later steps. On those eight packets the first
/// generated step consumed 97.3620711503% to 99.8667704858% of the eight-nat cap, so the arms were
/// matched on total cost and unmatched on *when* that cost was paid. Total KL is the controlled
/// variable; the temporal schedule is not, and it moves with the treatment.
///
/// `matchedPerStep` is the audit's discipline instead: a fixed scalar applied at every step with no
/// cap and no adaptive rescaling, with the scalar chosen so the arm's *mean* teacher-forced KL per
/// step hits a target. It is what `docs/phase6/teacher-forced-comparison` froze, and it is the
/// default here.
enum KLDiscipline: String, Sendable, Codable, CaseIterable, Identifiable {
    /// Fixed scalar, applied every step, metered but never attenuated.
    case matchedPerStep = "matched-per-step"
    /// One cumulative budget, spent greedily until exhausted. Retained as a demonstration.
    case greedyCumulativeCap = "greedy-cumulative-cap"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .matchedPerStep: "Matched per-step"
        case .greedyCumulativeCap: "Greedy cumulative cap"
        }
    }

    /// True when a cumulative budget attenuates the intervention. Recorded per arm in the run
    /// report, because "was a cap active" is the one fact a reader needs to interpret a KL trace.
    var capsInterventions: Bool { self == .greedyCumulativeCap }
}

/// One arm's calibration result: the scalar whose mean teacher-forced KL per step came closest to
/// the target on a shared, base-generated continuation.
struct CalibrationOutcome: Sendable, Codable {
    let method: String
    let selectedScalar: Double
    let achievedMeanNatsPerStep: Double
    let target: Double
    let lowerBracket: Double
    let upperBracket: Double
    let iterations: Int
    let promptCount: Int
    let continuationTokenCount: Int

    var signedError: Double { achievedMeanNatsPerStep - target }
}

struct TeacherForcedKLResult: Sendable, Codable {
    let method: String
    let meanNatsPerStep: Double
    let perStepNats: [Double]
    let continuationTokenCount: Int
    let appliedScalar: Double
    let directionDiagnostics: ResidualDirectionDiagnostics?
}

struct ResidualDirectionDiagnostics: Sendable, Codable {
    let mode: String
    let positiveTokenCount: Int
    let negativeTokenCount: Int
    let alignedPositionCount: Int
    let historicalFinalVectorNorm: Double
    let semanticMatrixNorm: Double
    let appliedMatrixNorm: Double
    let semanticPerPositionNorms: [Double]
    let appliedPerPositionNorms: [Double]
    let alignment: String
    let injection: String
}

struct PaneState: Sendable {
    var text = ""
    var tokenIDs: [Int] = []
    var tokenCount = 0
    var tokensPerSecond = 0.0
    var residentMemoryBytes: UInt64 = 0
    var topicScore: Double?
    var baseModelNLL: Double?
    var isActive = false

    static let empty = PaneState()
}

struct GenerationUpdate: Sendable {
    let pane: GenerationPane
    let text: String
    let tokenCount: Int
    let tokensPerSecond: Double
    let residentMemoryBytes: UInt64
    let klReading: KLReading?
}

struct GenerationSummary: Sendable {
    let pane: GenerationPane
    let text: String
    let tokenIDs: [Int]
    let tokenCount: Int
    let seconds: Double
    let residentMemoryBytes: UInt64
    let klHistory: [KLReading]
    let droppedTokenStrings: [String]
    let baseModelNLL: Double?
    let appliedCoefficient: Double?
    let directionDiagnostics: ResidualDirectionDiagnostics?
}

enum DemoError: LocalizedError {
    case missingResource(String)
    case missingCoreMLOutput
    case invalidCentroid(String)
    case invalidKLDivergence(String)
    case invalidActAddLayer(Int, Int)
    case calibrationBracketTooLow(arm: String, achieved: Double, target: Double)
    case calibrationToleranceFailed(arm: String, achieved: Double, target: Double, tolerance: Double)
    case snapshotRenderingFailed

    var errorDescription: String? {
        switch self {
        case .missingResource(let name): "Missing bundled resource: \(name)."
        case .missingCoreMLOutput: "The Core ML topic encoder returned no embedding."
        case .invalidCentroid(let id): "No valid topic centroid exists for \(id)."
        case .invalidKLDivergence(let reason): "KL measurement failed: \(reason)"
        case .invalidActAddLayer(let layer, let count):
            "ActAdd layer \(layer) is invalid for a model with \(count) layers."
        case .calibrationBracketTooLow(let arm, let achieved, let target):
            """
            The \(arm) arm cannot reach \(target) nats/step on this prompt: its bracket's upper \
            endpoint only reached \(achieved). Calibration stopped without changing any scalar.
            """
        case .calibrationToleranceFailed(let arm, let achieved, let target, let tolerance):
            """
            The \(arm) arm's closest candidate landed at \(achieved) nats/step, outside the \
            \(tolerance) tolerance around \(target). Calibration stopped without changing any scalar.
            """
        case .snapshotRenderingFailed: "Could not render the app evidence snapshot."
        }
    }
}

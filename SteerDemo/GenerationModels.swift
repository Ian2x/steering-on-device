import Foundation
import SteeringKit

enum GenerationPane: String, Sendable {
    case baseline
    case steered
    case actAdd
}

struct PaneState: Sendable {
    var text = ""
    var tokenCount = 0
    var tokensPerSecond = 0.0
    var residentMemoryBytes: UInt64 = 0
    var topicScore: Double?
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
    let tokenCount: Int
    let seconds: Double
    let residentMemoryBytes: UInt64
    let klHistory: [KLReading]
    let droppedTokenStrings: [String]
}

enum DemoError: LocalizedError {
    case missingResource(String)
    case missingCoreMLOutput
    case invalidCentroid(String)
    case invalidKLDivergence(String)
    case invalidActAddLayer(Int, Int)
    case snapshotRenderingFailed

    var errorDescription: String? {
        switch self {
        case .missingResource(let name): "Missing bundled resource: \(name)."
        case .missingCoreMLOutput: "The Core ML topic encoder returned no embedding."
        case .invalidCentroid(let id): "No valid topic centroid exists for \(id)."
        case .invalidKLDivergence(let reason): "KL measurement failed: \(reason)"
        case .invalidActAddLayer(let layer, let count):
            "ActAdd layer \(layer) is invalid for a model with \(count) layers."
        case .snapshotRenderingFailed: "Could not render the app evidence snapshot."
        }
    }
}

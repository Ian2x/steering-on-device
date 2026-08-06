import Foundation
import MLX
import MLXLMCommon
import SteeringKit

final class KLMetricsSink: @unchecked Sendable {
    private let lock = NSLock()
    private var meter: KLMeter

    init(budget: Double) {
        meter = KLMeter(budget: budget)
    }

    func append(_ value: Double) {
        lock.lock()
        defer { lock.unlock() }
        _ = try? meter.record(divergence: value)
    }

    func latest() -> KLReading? {
        lock.lock()
        defer { lock.unlock() }
        return meter.history.last
    }

    func history() -> [KLReading] {
        lock.lock()
        defer { lock.unlock() }
        return meter.history
    }

    func isExhausted() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let latest = meter.history.last, latest.budget > 0 else { return false }
        return latest.cumulative >= latest.budget
    }

    func remainingBudget() -> Double {
        lock.lock()
        defer { lock.unlock() }
        return max(0, meter.budget - meter.cumulative)
    }
}

struct SparseBiasProcessor: LogitProcessor {
    let tokenIDs: [UInt32]
    let biasValues: [Float]
    let inverseTemperature: Float
    let sink: KLMetricsSink

    init(biases: [TokenBias], temperature: Float, sink: KLMetricsSink) {
        precondition(temperature > 0)
        tokenIDs = biases.map { UInt32($0.tokenID) }
        biasValues = biases.map { Float($0.value) }
        inverseTemperature = 1 / temperature
        self.sink = sink
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard !tokenIDs.isEmpty else { return logits }
        guard !sink.isExhausted() else { return logits }

        let base = logits.asType(.float32)
        let indices = MLXArray(tokenIDs)
        let sampledBase = base * inverseTemperature
        let baseLogProb = sampledBase - logSumExp(sampledBase, axis: -1, keepDims: true)

        func candidate(scale: Float) -> (logits: MLXArray, divergence: Double) {
            // MLXArray has reference semantics. This elementwise op creates an
            // independent graph value rather than editing the baseline alias.
            let edited = base + zeros(like: base)
            edited[0..., indices] = edited[0..., indices]
                + MLXArray(biasValues.map { $0 * scale })
            let sampledEdited = edited * inverseTemperature
            let editedLogProb = sampledEdited
                - logSumExp(sampledEdited, axis: -1, keepDims: true)
            let probability = exp(editedLogProb)
            let value = sum(probability * (editedLogProb - baseLogProb), axis: -1)
            eval(value)
            return (edited, max(0, Double(value.item(Float.self))))
        }

        let remaining = sink.remainingBudget()
        guard remaining > 1e-8 else { return logits }
        let full = candidate(scale: 1)
        if full.divergence <= remaining {
            sink.append(full.divergence)
            return full.logits.asType(logits.dtype)
        }

        // The last active step is rescaled by bisection so cumulative KL does
        // not overshoot the selected budget. KL is monotone along this fixed
        // sparse-bias direction.
        var lower: Float = 0
        var upper: Float = 1
        var selected = candidate(scale: 0)
        for _ in 0 ..< 14 {
            let middle = (lower + upper) / 2
            let trial = candidate(scale: middle)
            if trial.divergence <= remaining {
                lower = middle
                selected = trial
            } else {
                upper = middle
            }
        }
        sink.append(selected.divergence)
        return selected.logits.asType(logits.dtype)
    }

    mutating func didSample(token: MLXArray) {}
}

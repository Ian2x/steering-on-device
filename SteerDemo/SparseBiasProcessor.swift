import Foundation
import MLX
import MLXLMCommon
import SteeringKit

final class KLMetricsSink: @unchecked Sendable {
    private let lock = NSLock()
    private var meter: KLMeter
    private var failure: Error?

    init(budget: Double) {
        meter = KLMeter(budget: budget)
    }

    func append(_ value: Double) {
        lock.lock()
        defer { lock.unlock() }
        do {
            _ = try meter.record(divergence: value)
        } catch {
            failure = error
        }
    }

    func fail(_ error: Error) {
        lock.lock()
        failure = error
        lock.unlock()
    }

    func throwIfFailed() throws {
        lock.lock()
        defer { lock.unlock() }
        if let failure { throw failure }
    }

    func reading(forReturnedTokenCount count: Int) -> KLReading? {
        lock.lock()
        defer { lock.unlock() }
        return meter.reading(forReturnedTokenCount: count)
    }

    func history(forReturnedTokenCount count: Int) -> [KLReading] {
        lock.lock()
        defer { lock.unlock() }
        return meter.readings(forReturnedTokenCount: count)
    }

    func remainingBudget() -> Double {
        lock.lock()
        defer { lock.unlock() }
        let remaining = max(0, meter.budget - meter.cumulative)
        // A nanonat-scale budget tail cannot produce a meaningful float32
        // logit edit; treating it as exhausted avoids repeated CPU readback.
        return remaining <= 1e-8 ? 0 : remaining
    }
}

struct SparseBiasProcessor: LogitProcessor {
    let biases: [TokenBias]
    let tokenIDs: [UInt32]
    let biasValues: [Float]
    let temperature: Double
    let sink: KLMetricsSink

    init(biases: [TokenBias], temperature: Float, sink: KLMetricsSink) {
        precondition(temperature > 0)
        self.biases = biases
        tokenIDs = biases.map { UInt32($0.tokenID) }
        biasValues = biases.map { Float($0.value) }
        self.temperature = Double(temperature)
        self.sink = sink
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard !tokenIDs.isEmpty else { return logits }
        let remaining = sink.remainingBudget()
        guard remaining > 0 else { return logits }

        let base = logits.asType(.float32)
        eval(base)
        do {
            let decision = try BiasBudgetSelector.select(
                baseLogits: base.asArray(Float.self).map(Double.init),
                biases: biases,
                remaining: remaining,
                temperature: temperature
            )
            let edited = base + zeros(like: base)
            if decision.scale > 0 {
                let indices = MLXArray(tokenIDs)
                edited[0..., indices] = edited[0..., indices]
                    + MLXArray(biasValues.map { $0 * Float(decision.scale) })
            }
            sink.append(decision.divergence)
            return edited.asType(logits.dtype)
        } catch {
            sink.fail(DemoError.invalidKLDivergence(error.localizedDescription))
            return logits
        }
    }

    mutating func didSample(token: MLXArray) {}
}

/// Applies the calibrated sparse bias at every step. Unlike
/// `SparseBiasProcessor`, this processor has no cumulative budget or adaptive
/// rescaling; it is used only by the teacher-forced Phase 6 comparison.
struct FixedSparseBiasProcessor: LogitProcessor {
    let tokenIDs: [UInt32]
    let biasValues: [Float]
    let temperature: Double
    let sink: KLMetricsSink

    init(biases: [TokenBias], temperature: Float, sink: KLMetricsSink) {
        precondition(temperature > 0)
        tokenIDs = biases.map { UInt32($0.tokenID) }
        biasValues = biases.map { Float($0.value) }
        self.temperature = Double(temperature)
        self.sink = sink
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard !tokenIDs.isEmpty else { return logits }
        let base = logits.asType(.float32)
        let edited = base + zeros(like: base)
        let indices = MLXArray(tokenIDs)
        edited[0..., indices] = edited[0..., indices] + MLXArray(biasValues)
        eval(base, edited)
        do {
            let inverseTemperature = 1 / temperature
            let divergence = try KLMeter.divergence(
                biasedLogits: edited.asArray(Float.self).map { Double($0) * inverseTemperature },
                baseLogits: base.asArray(Float.self).map { Double($0) * inverseTemperature }
            )
            sink.append(divergence)
        } catch {
            sink.fail(DemoError.invalidKLDivergence(error.localizedDescription))
        }
        return edited.asType(logits.dtype)
    }

    mutating func didSample(token: MLXArray) {}
}

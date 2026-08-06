import Foundation

public enum KLMeterError: Error, Equatable {
    case emptyLogits
    case shapeMismatch(biased: Int, base: Int)
    case nonFiniteLogit
}

public struct KLReading: Codable, Equatable, Sendable {
    public let step: Int
    public let perStep: Double
    public let cumulative: Double
    public let budget: Double

    public var budgetFraction: Double {
        guard budget > 0 else { return cumulative > 0 ? 1 : 0 }
        return cumulative / budget
    }
}

/// Numerically stable KL(biased || base) tracking for generation logits.
public struct KLMeter: Sendable {
    public private(set) var history: [KLReading] = []
    public private(set) var cumulative: Double = 0
    public let budget: Double

    public init(budget: Double) {
        self.budget = max(0, budget)
    }

    @discardableResult
    public mutating func record(
        biasedLogits: [Double],
        baseLogits: [Double]
    ) throws -> KLReading {
        let value = try Self.divergence(
            biasedLogits: biasedLogits,
            baseLogits: baseLogits
        )
        return try record(divergence: value)
    }

    /// Records a KL value computed on an accelerator without copying its full
    /// vocabulary logits back to the CPU.
    @discardableResult
    public mutating func record(divergence value: Double) throws -> KLReading {
        guard value.isFinite else { throw KLMeterError.nonFiniteLogit }
        let clamped = max(0, value)
        cumulative += clamped
        let reading = KLReading(
            step: history.count + 1,
            perStep: clamped,
            cumulative: cumulative,
            budget: budget
        )
        history.append(reading)
        return reading
    }

    public static func divergence(
        biasedLogits: [Double],
        baseLogits: [Double]
    ) throws -> Double {
        guard !biasedLogits.isEmpty, !baseLogits.isEmpty else {
            throw KLMeterError.emptyLogits
        }
        guard biasedLogits.count == baseLogits.count else {
            throw KLMeterError.shapeMismatch(
                biased: biasedLogits.count,
                base: baseLogits.count
            )
        }
        guard biasedLogits.allSatisfy(\.isFinite), baseLogits.allSatisfy(\.isFinite) else {
            throw KLMeterError.nonFiniteLogit
        }

        let biasedLogZ = logSumExp(biasedLogits)
        let baseLogZ = logSumExp(baseLogits)
        var value = 0.0

        for index in biasedLogits.indices {
            let logP = biasedLogits[index] - biasedLogZ
            let logQ = baseLogits[index] - baseLogZ
            value += Foundation.exp(logP) * (logP - logQ)
        }

        // Roundoff can make mathematically-zero KL a tiny negative value.
        return max(0, value)
    }

    private static func logSumExp(_ values: [Double]) -> Double {
        let maximum = values.max()!
        let shifted = values.reduce(0.0) { partial, value in
            partial + Foundation.exp(value - maximum)
        }
        return maximum + Foundation.log(shifted)
    }
}

import Foundation

public enum KLMeterError: LocalizedError, Equatable {
    case emptyLogits
    case shapeMismatch(biased: Int, base: Int)
    case nonFiniteLogit

    public var errorDescription: String? {
        switch self {
        case .emptyLogits:
            "KL divergence requires two non-empty logit vectors."
        case .shapeMismatch(let biased, let base):
            "KL divergence requires matching logit shapes; received \(biased) and \(base)."
        case .nonFiniteLogit:
            "KL divergence received a non-finite logit or divergence value."
        }
    }
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

    /// TokenIterator may evaluate one token ahead of the token it returns.
    /// This view excludes KL readings for tokens that were never delivered.
    public func readings(forReturnedTokenCount count: Int) -> [KLReading] {
        Array(history.prefix(max(0, count)))
    }

    public func reading(forReturnedTokenCount count: Int) -> KLReading? {
        guard count > 0, count <= history.count else { return nil }
        return history[count - 1]
    }

    /// Records a KL value computed by a validated distribution-comparison path.
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
        var sum = 0.0
        var compensation = 0.0

        for index in biasedLogits.indices {
            let p = Foundation.exp(biasedLogits[index] - biasedLogZ)
            let q = Foundation.exp(baseLogits[index] - baseLogZ)
            let term: Double
            if p > 0, q > 0 {
                // q * ((1+r) log(1+r) - r) is algebraically equivalent to
                // KL after sum(p-q)=0, but avoids subtracting nearly equal
                // log probabilities when the distributions are close.
                let r = (p - q) / q
                term = q * ((1 + r) * Foundation.log1p(r) - r)
            } else {
                let logP = biasedLogits[index] - biasedLogZ
                let logQ = baseLogits[index] - baseLogZ
                term = p * (logP - logQ)
            }
            let corrected = term - compensation
            let next = sum + corrected
            compensation = (next - sum) - corrected
            sum = next
        }

        return max(0, sum)
    }

    private static func logSumExp(_ values: [Double]) -> Double {
        let maximum = values.max()!
        let shifted = values.reduce(0.0) { partial, value in
            partial + Foundation.exp(value - maximum)
        }
        return maximum + Foundation.log(shifted)
    }
}

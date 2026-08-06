import Foundation

public enum BiasBudgetSelectorError: LocalizedError, Equatable {
    case emptyLogits
    case nonFiniteLogit
    case invalidTemperature(Double)
    case invalidRemainingBudget(Double)
    case tokenIDOutOfRange(Int)
    case nonFiniteBias

    public var errorDescription: String? {
        switch self {
        case .emptyLogits:
            "Cannot enforce a KL budget on an empty logit vector."
        case .nonFiniteLogit:
            "KL-budget selection received a non-finite model logit."
        case .invalidTemperature(let value):
            "Sampling temperature must be finite and positive; received \(value)."
        case .invalidRemainingBudget(let value):
            "Remaining KL budget must be finite and non-negative; received \(value)."
        case .tokenIDOutOfRange(let tokenID):
            "Bias token ID \(tokenID) is outside the model vocabulary."
        case .nonFiniteBias:
            "KL-budget selection received a non-finite steering bias."
        }
    }
}

public struct BiasBudgetDecision: Equatable, Sendable {
    public let scale: Double
    public let divergence: Double

    public init(scale: Double, divergence: Double) {
        self.scale = scale
        self.divergence = divergence
    }
}

/// Selects the largest scale on a fixed sparse-bias direction whose
/// temperature-scaled KL(biased || base) fits in the remaining budget.
///
/// This pure implementation is shared by the app and `swift test`. A parity
/// test ties its sparse-distribution identity to the general KL implementation
/// covered by the hand-derived and high-precision fixtures.
public enum BiasBudgetSelector {
    private struct SparseDistribution {
        let tokenMasses: [Double]
        let biasDeltas: [Double]

        init(baseLogits: [Double], biases: [TokenBias], temperature: Double) throws {
            guard !baseLogits.isEmpty else {
                throw BiasBudgetSelectorError.emptyLogits
            }
            guard baseLogits.allSatisfy(\.isFinite) else {
                throw BiasBudgetSelectorError.nonFiniteLogit
            }
            guard temperature.isFinite, temperature > 0 else {
                throw BiasBudgetSelectorError.invalidTemperature(temperature)
            }
            guard biases.allSatisfy({ $0.value.isFinite }) else {
                throw BiasBudgetSelectorError.nonFiniteBias
            }

            var combined: [Int: Double] = [:]
            for bias in biases {
                guard baseLogits.indices.contains(bias.tokenID) else {
                    throw BiasBudgetSelectorError.tokenIDOutOfRange(bias.tokenID)
                }
                combined[bias.tokenID, default: 0] += bias.value / temperature
            }

            let scaled = baseLogits.map { $0 / temperature }
            let maximum = scaled.max()!
            var shiftedSum = 0.0
            var compensation = 0.0
            for value in scaled {
                let term = Foundation.exp(value - maximum)
                let corrected = term - compensation
                let next = shiftedSum + corrected
                compensation = (next - shiftedSum) - corrected
                shiftedSum = next
            }
            let logNormalizer = maximum + Foundation.log(shiftedSum)
            let entries = combined.sorted { $0.key < $1.key }
            tokenMasses = entries.map { Foundation.exp(scaled[$0.key] - logNormalizer) }
            biasDeltas = entries.map(\.value)
        }

        func divergence(scale: Double) throws -> Double {
            guard scale.isFinite else {
                throw BiasBudgetSelectorError.nonFiniteBias
            }
            var normalizerRatio = 1.0
            var biasedDeltaMoment = 0.0
            for (mass, delta) in zip(tokenMasses, biasDeltas) {
                let scaledDelta = scale * delta
                let multiplier = Foundation.exp(scaledDelta)
                normalizerRatio += mass * Foundation.expm1(scaledDelta)
                biasedDeltaMoment += mass * multiplier * scaledDelta
            }
            guard normalizerRatio.isFinite, normalizerRatio > 0,
                  biasedDeltaMoment.isFinite
            else {
                throw BiasBudgetSelectorError.nonFiniteLogit
            }
            return max(0, biasedDeltaMoment / normalizerRatio - Foundation.log(normalizerRatio))
        }
    }

    public static func candidateDivergence(
        baseLogits: [Double],
        biases: [TokenBias],
        scale: Double,
        temperature: Double
    ) throws -> Double {
        try SparseDistribution(
            baseLogits: baseLogits,
            biases: biases,
            temperature: temperature
        ).divergence(scale: scale)
    }

    public static func select(
        baseLogits: [Double],
        biases: [TokenBias],
        remaining: Double,
        temperature: Double,
        iterations: Int = 20
    ) throws -> BiasBudgetDecision {
        guard remaining.isFinite, remaining >= 0 else {
            throw BiasBudgetSelectorError.invalidRemainingBudget(remaining)
        }
        guard !biases.isEmpty, remaining > 0 else {
            return BiasBudgetDecision(scale: 0, divergence: 0)
        }

        let distribution = try SparseDistribution(
            baseLogits: baseLogits,
            biases: biases,
            temperature: temperature
        )
        return try bisect(
            remaining: remaining,
            iterations: iterations,
            divergence: distribution.divergence(scale:)
        )
    }

    /// Selects a feasible scale whose candidate logits fit the remaining KL
    /// budget. Sparse output bias is monotone along its fixed direction. A
    /// nonlinear dense closure, such as a transformer tail, is not guaranteed
    /// monotone, so this method does not claim a globally largest feasible
    /// scale for arbitrary closures. Both paths share the safety invariant:
    /// only a probe at or below the remaining budget can become `selected`.
    public static func select(
        baseLogits: [Double],
        remaining: Double,
        temperature: Double,
        iterations: Int = 20,
        candidateLogits: (Double) throws -> [Double]
    ) throws -> BiasBudgetDecision {
        guard remaining.isFinite, remaining >= 0 else {
            throw BiasBudgetSelectorError.invalidRemainingBudget(remaining)
        }
        guard !baseLogits.isEmpty else {
            throw BiasBudgetSelectorError.emptyLogits
        }
        guard baseLogits.allSatisfy(\.isFinite) else {
            throw BiasBudgetSelectorError.nonFiniteLogit
        }
        guard temperature.isFinite, temperature > 0 else {
            throw BiasBudgetSelectorError.invalidTemperature(temperature)
        }
        guard remaining > 0 else {
            return BiasBudgetDecision(scale: 0, divergence: 0)
        }

        let scaledBase = baseLogits.map { $0 / temperature }
        func divergence(at scale: Double) throws -> Double {
            let candidate = try candidateLogits(scale)
            guard candidate.count == baseLogits.count else {
                throw KLMeterError.shapeMismatch(
                    biased: candidate.count,
                    base: baseLogits.count
                )
            }
            guard candidate.allSatisfy(\.isFinite) else {
                throw BiasBudgetSelectorError.nonFiniteLogit
            }
            return try KLMeter.divergence(
                biasedLogits: candidate.map { $0 / temperature },
                baseLogits: scaledBase
            )
        }

        return try bisect(
            remaining: remaining,
            iterations: iterations,
            divergence: divergence(at:)
        )
    }

    /// Shared interval bisection. `selected` is updated only by a probe that
    /// satisfies the remaining budget, preserving cumulative <= budget even
    /// when an arbitrary candidate closure is not monotone. Maximality then is
    /// not guaranteed; callers must not describe the result as globally largest.
    private static func bisect(
        remaining: Double,
        iterations: Int,
        divergence: (Double) throws -> Double
    ) throws -> BiasBudgetDecision {
        let full = try divergence(1)
        if full <= remaining {
            return BiasBudgetDecision(scale: 1, divergence: full)
        }

        var lower = 0.0
        var upper = 1.0
        var selected = BiasBudgetDecision(scale: 0, divergence: 0)
        for _ in 0 ..< max(1, iterations) {
            let middle = (lower + upper) / 2
            let candidateDivergence = try divergence(middle)
            if candidateDivergence <= remaining {
                lower = middle
                selected = BiasBudgetDecision(scale: middle, divergence: candidateDivergence)
            } else {
                upper = middle
            }
            if upper - lower <= 1e-7 { break }
        }
        return selected
    }
}

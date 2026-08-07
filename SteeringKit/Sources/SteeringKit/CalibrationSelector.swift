import Foundation

public enum CalibrationSelectorError: LocalizedError, Equatable {
    case invalidBracket(Double)
    case invalidIterations(Int)
    case nonFiniteMean(Double)
    case bracketBelowTarget(achieved: Double, target: Double)
    case toleranceExceeded(achieved: Double, target: Double, tolerance: Double)

    public var errorDescription: String? {
        switch self {
        case .invalidBracket(let value):
            "Calibration needs a finite, positive upper bracket; received \(value)."
        case .invalidIterations(let value):
            "Calibration needs at least one bisection iteration; received \(value)."
        case .nonFiniteMean(let value):
            "Calibration received a non-finite mean KL: \(value)."
        case .bracketBelowTarget(let achieved, let target):
            "The upper bracket only reached \(achieved) nats/step, below the target \(target)."
        case .toleranceExceeded(let achieved, let target, let tolerance):
            "The closest candidate landed at \(achieved) nats/step, outside \(tolerance) of \(target)."
        }
    }
}

public struct CalibrationCandidate: Equatable, Sendable {
    public let scalar: Double
    public let meanNatsPerStep: Double

    public init(scalar: Double, meanNatsPerStep: Double) {
        self.scalar = scalar
        self.meanNatsPerStep = meanNatsPerStep
    }
}

public struct CalibrationSelection: Equatable, Sendable {
    public let scalar: Double
    public let meanNatsPerStep: Double
    public let lowerBracket: Double
    public let upperBracket: Double
    public let tested: [CalibrationCandidate]
}

/// Chooses the scalar whose mean teacher-forced KL per step comes closest to a target.
///
/// This is the frozen Phase 6 algorithm, lifted out of the runner so the app and `swift test` share
/// one implementation:
///
/// 1. Evaluate the upper bracket. If it does not reach the target, fail without selecting.
/// 2. Run `iterations` midpoint bisections. A candidate at or above the target replaces the upper
///    endpoint; below it replaces the lower.
/// 3. Select the tested candidate with the **minimum absolute error** from the target, ties going
///    to the lower scalar, then check it against the tolerance.
///
/// Step 3 is Amendment 1. The original protocol selected the final tested upper endpoint, which
/// could only ever land at or above the target; the amendment is what makes below-target arms
/// reachable, and three of the four committed arms are below target. See
/// `docs/phase6/teacher-forced-comparison/amendment-1.md`.
///
/// The curve is **not** assumed monotone. The original protocol did assume it, and that assumption
/// is what failed on the complete wedding calibration: the residual mean fell from `0.435236843732678`
/// at coefficient `6.7822265625` to `0.43477367707994063` at `6.78466796875`. Bisection still drives
/// the search here, but a reversal is recorded rather than treated as a stop condition, and the
/// selection reads every tested point instead of trusting the final bracket.
public enum CalibrationSelector {
    public static func select(
        target: Double,
        tolerance: Double,
        upperBracket: Double,
        iterations: Int,
        onCandidate: (Int, Int, CalibrationCandidate) -> Void = { _, _, _ in },
        meanKL: (Double) async throws -> Double
    ) async throws -> CalibrationSelection {
        guard upperBracket.isFinite, upperBracket > 0 else {
            throw CalibrationSelectorError.invalidBracket(upperBracket)
        }
        guard iterations >= 1 else {
            throw CalibrationSelectorError.invalidIterations(iterations)
        }

        var low = 0.0
        var high = upperBracket
        let total = iterations + 1

        let highMean = try await meanKL(high)
        guard highMean.isFinite else {
            throw CalibrationSelectorError.nonFiniteMean(highMean)
        }
        guard highMean >= target else {
            throw CalibrationSelectorError.bracketBelowTarget(
                achieved: highMean,
                target: target
            )
        }
        var tested = [CalibrationCandidate(scalar: high, meanNatsPerStep: highMean)]
        onCandidate(1, total, tested[0])

        for index in 0 ..< iterations {
            try Task.checkCancellation()
            let midpoint = (low + high) / 2
            let mean = try await meanKL(midpoint)
            guard mean.isFinite else {
                throw CalibrationSelectorError.nonFiniteMean(mean)
            }
            let candidate = CalibrationCandidate(scalar: midpoint, meanNatsPerStep: mean)
            tested.append(candidate)
            if mean >= target { high = midpoint } else { low = midpoint }
            onCandidate(index + 2, total, candidate)
        }

        // Minimum absolute error; exact ties choose the lower scalar.
        let selected = tested.min { first, second in
            let firstError = abs(first.meanNatsPerStep - target)
            let secondError = abs(second.meanNatsPerStep - target)
            return firstError == secondError
                ? first.scalar < second.scalar
                : firstError < secondError
        }!
        guard abs(selected.meanNatsPerStep - target) <= tolerance else {
            throw CalibrationSelectorError.toleranceExceeded(
                achieved: selected.meanNatsPerStep,
                target: target,
                tolerance: tolerance
            )
        }

        return CalibrationSelection(
            scalar: selected.scalar,
            meanNatsPerStep: selected.meanNatsPerStep,
            lowerBracket: low,
            upperBracket: high,
            tested: tested
        )
    }
}

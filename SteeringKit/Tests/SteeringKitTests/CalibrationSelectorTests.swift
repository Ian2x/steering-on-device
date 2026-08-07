import Foundation
import Testing
@testable import SteeringKit

struct CalibrationFixture: Codable {
    struct Candidate: Codable {
        let label: String
        let promptCount: Int
        let scalar: Double
        let meanNatsPerStep: Double
    }

    struct Arm: Codable {
        let topic: String
        let method: String
        let upperBracket: Double
        let iterations: Int
        let expectedScalar: Double
        let expectedMeanNatsPerStep: Double
        let candidates: [Candidate]
    }

    let source: String
    let target: Double
    let tolerance: Double
    let arms: [Arm]
}

/// Replays the four committed Phase 6 calibration curves through the selector and requires it to
/// reproduce the frozen scalars in `docs/phase6/teacher-forced-comparison/results.md`.
///
/// The fixture is derived from the 304 committed calibration packets by
/// `Scripts/generate_calibration_fixture.py`. If the app's selector ever drifts from the algorithm
/// that produced the published numbers, this fails.
@Test func reproducesFrozenCalibrationScalars() async throws {
    let url = try #require(
        Bundle.module.url(
            forResource: "calibration_golden",
            withExtension: "json",
            subdirectory: "Fixtures"
        )
    )
    let fixture = try JSONDecoder().decode(
        CalibrationFixture.self,
        from: Data(contentsOf: url)
    )
    #expect(fixture.arms.count == 4)
    #expect(abs(fixture.target - 0.435_238_018_732_847_95) < 1e-18)

    for arm in fixture.arms {
        let name = "\(arm.topic)/\(arm.method)"
        // The runner evaluated each candidate on four prompts and compared their mean; the
        // fixture stores that mean, so a lookup here is the same quantity the frozen run used.
        let curve = Dictionary(
            uniqueKeysWithValues: arm.candidates.map { ($0.scalar, $0.meanNatsPerStep) }
        )
        #expect(arm.candidates.allSatisfy { $0.promptCount == 4 }, "\(name) prompt count")

        var visited: [Double] = []
        let selection = try await CalibrationSelector.select(
            target: fixture.target,
            tolerance: fixture.tolerance,
            upperBracket: arm.upperBracket,
            iterations: arm.iterations
        ) { scalar in
            visited.append(scalar)
            // A miss means the Swift bisection walked somewhere the frozen runner never did.
            return try #require(curve[scalar], "\(name) visited untested scalar \(scalar)")
        }

        #expect(visited.count == arm.iterations + 1, "\(name) candidate count")
        #expect(
            selection.scalar == arm.expectedScalar,
            "\(name) selected \(selection.scalar), frozen run selected \(arm.expectedScalar)"
        )
        #expect(
            abs(selection.meanNatsPerStep - arm.expectedMeanNatsPerStep) < 1e-12,
            "\(name) achieved \(selection.meanNatsPerStep)"
        )
        #expect(abs(selection.meanNatsPerStep - fixture.target) <= fixture.tolerance)
    }
}

/// Three of the four committed arms landed *below* the target. Under the original protocol's
/// "final tested upper endpoint" rule they were unreachable, which is why Amendment 1 exists.
@Test func selectsBelowTargetCandidateWhenItIsCloser() async throws {
    // A deliberately coarse curve: the nearest point sits just under the target.
    let curve: [Double: Double] = [8: 1.0, 4: 0.30, 6: 0.44, 5: 0.4299]
    let selection = try await CalibrationSelector.select(
        target: 0.43,
        tolerance: 0.01,
        upperBracket: 8,
        iterations: 3
    ) { scalar in
        try #require(curve[scalar])
    }
    #expect(selection.scalar == 5)
    #expect(selection.meanNatsPerStep < 0.43)
}

@Test func toleranceFailureDoesNotSelect() async {
    await #expect(throws: CalibrationSelectorError.self) {
        _ = try await CalibrationSelector.select(
            target: 0.43,
            tolerance: 0.001,
            upperBracket: 8,
            iterations: 2
        ) { scalar in scalar }
    }
}

@Test func bracketBelowTargetFailsBeforeAnyBisection() async {
    var calls = 0
    await #expect(throws: CalibrationSelectorError.self) {
        _ = try await CalibrationSelector.select(
            target: 5,
            tolerance: 0.1,
            upperBracket: 2,
            iterations: 18
        ) { scalar in
            calls += 1
            return scalar * 0.1
        }
    }
    #expect(calls == 1)
}

/// A non-monotone curve must not be treated as a stop condition. The frozen wedding residual
/// calibration reversed by `-0.000463166652737379` nats/step and the original protocol halted; the
/// amended rule keeps searching and reads every tested point.
@Test func toleratesANonMonotoneReversal() async throws {
    let curve: [Double: Double] = [
        40: 2.0,
        20: 0.9,
        10: 0.50,
        5: 0.20,
        7.5: 0.4352,
        6.25: 0.30,
        // Below its neighbour at 6.25 despite a larger scalar: the reversal.
        6.875: 0.29,
    ]
    let selection = try await CalibrationSelector.select(
        target: 0.435,
        tolerance: 0.001,
        upperBracket: 40,
        iterations: 6
    ) { scalar in
        try #require(curve[scalar])
    }
    #expect(selection.scalar == 7.5)
    #expect(selection.tested.count == 7)
}

import Foundation
import Testing
@testable import SteeringKit

struct KLFixture: Codable {
    struct Case: Codable {
        let name: String
        let biasedLogits: [Double]
        let baseLogits: [Double]
        let expectedKL: Double
    }

    let implementation: String
    let cases: [Case]
}

@Test func actAddCoefficientZeroUsesBitIdenticalBaselineRoute() {
    #expect(ActAddPassPlanner.route(coefficient: 0) == .baseline)
    #expect(ActAddPassPlanner.route(coefficient: -0.0) == .baseline)
    #expect(ActAddPassPlanner.route(coefficient: 1e-12) == .activationAddition)

    let baselineTokenBytes = [UInt32(17), 29, 4, 4, 91].withUnsafeBytes { Data($0) }
    let editedTokenBytes = [UInt32(17), 88, 6].withUnsafeBytes { Data($0) }
    var baselineCalls = 0
    var actAddCalls = 0
    let actual = ActAddPassPlanner.run(coefficient: 0) {
        baselineCalls += 1
        return baselineTokenBytes
    } activationAddition: {
        actAddCalls += 1
        return editedTokenBytes
    }

    #expect(actual == baselineTokenBytes)
    #expect(Array(actual) == Array(baselineTokenBytes))
    #expect(baselineCalls == 1)
    #expect(actAddCalls == 0)
}

@Test func handComputedUniformBaseFixture() throws {
    // p = [1/2, 1/4, 1/4], q = [1/3, 1/3, 1/3]
    let biased = [Foundation.log(0.5), Foundation.log(0.25), Foundation.log(0.25)]
    let base = [Foundation.log(1.0 / 3), Foundation.log(1.0 / 3), Foundation.log(1.0 / 3)]
    let expected = 0.5 * Foundation.log(1.5) + 0.5 * Foundation.log(0.75)

    let actual = try KLMeter.divergence(biasedLogits: biased, baseLogits: base)
    #expect(abs(actual - expected) < 1e-12)
}

@Test func identicalLogitsHaveZeroKL() throws {
    let actual = try KLMeter.divergence(
        biasedLogits: [1, 2, 3],
        baseLogits: [1, 2, 3]
    )
    #expect(actual < 1e-15)
}

@Test func goldenPythonParity() throws {
    let url = try #require(Bundle.module.url(forResource: "kl_golden", withExtension: "json", subdirectory: "Fixtures"))
    let fixture = try JSONDecoder().decode(KLFixture.self, from: Data(contentsOf: url))
    #expect(fixture.implementation == "python-decimal-direct-normalization-v2")

    for item in fixture.cases {
        let actual = try KLMeter.divergence(
            biasedLogits: item.biasedLogits,
            baseLogits: item.baseLogits
        )
        let relativeError = abs(actual - item.expectedKL) / item.expectedKL
        #expect(relativeError < 1e-5, "fixture: \(item.name), relative error: \(relativeError)")
    }
}

@Test func meterAccumulatesAndReportsBudget() throws {
    var meter = KLMeter(budget: 0.1)
    let divergence = try KLMeter.divergence(
        biasedLogits: [1, 0, 0],
        baseLogits: [0, 0, 0]
    )
    let first = try meter.record(divergence: divergence)
    let second = try meter.record(divergence: divergence)

    #expect(first.step == 1)
    #expect(second.step == 2)
    #expect(abs(second.cumulative - 2 * first.perStep) < 1e-12)
    #expect(abs(second.budgetFraction - second.cumulative / 0.1) < 1e-12)
}

@Test func candidatePathMatchesDirectKLAtTemperature() throws {
    let base = [0.2, -0.7, 1.3, 0.0]
    let biases = [
        TokenBias(tokenID: 1, tokenString: " topic", value: 2.5),
        TokenBias(tokenID: 3, tokenString: " ceremony", value: 1.25),
    ]
    let scale = 0.375
    let temperature = 0.7
    let candidate = try BiasBudgetSelector.candidateDivergence(
        baseLogits: base,
        biases: biases,
        scale: scale,
        temperature: temperature
    )

    var edited = base
    for bias in biases { edited[bias.tokenID] += bias.value * scale }
    let direct = try KLMeter.divergence(
        biasedLogits: edited.map { $0 / temperature },
        baseLogits: base.map { $0 / temperature }
    )
    #expect(abs(candidate - direct) < 1e-15)

    var state: UInt64 = 0x71e5_7ca1_e5ca_1e5d
    func unit() -> Double {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Double(state >> 11) / Double(1 << 53)
    }
    for index in 0 ..< 100 {
        let count = index.isMultiple(of: 17) ? 1 : 2 + Int(unit() * 30)
        let randomBase = (0 ..< count).map { _ in (unit() - 0.5) * 24 }
        let randomBiases = (0 ..< min(count, 1 + Int(unit() * 6))).map {
            TokenBias(
                tokenID: $0,
                tokenString: "token-\($0)",
                value: (unit() - 0.5) * 18
            )
        }
        let randomScale = unit()
        let randomTemperature = 0.2 + unit() * 1.8
        let sparse = try BiasBudgetSelector.candidateDivergence(
            baseLogits: randomBase,
            biases: randomBiases,
            scale: randomScale,
            temperature: randomTemperature
        )
        var randomEdited = randomBase
        for bias in randomBiases {
            randomEdited[bias.tokenID] += bias.value * randomScale
        }
        let general = try KLMeter.divergence(
            biasedLogits: randomEdited.map { $0 / randomTemperature },
            baseLogits: randomBase.map { $0 / randomTemperature }
        )
        let tolerance = max(1e-12, general * 1e-10)
        #expect(abs(sparse - general) <= tolerance, "random parity case \(index)")
    }
}

@Test func randomizedBisectionNeverOvershootsBudget() throws {
    var state: UInt64 = 0x5eed_cafe_f00d_beef
    func unit() -> Double {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Double(state >> 11) / Double(1 << 53)
    }

    let budgets = [0.0, 1e-12, 1e-8, 0.01, 0.5, 4.0]
    for budget in budgets {
        var meter = KLMeter(budget: budget)
        for step in 0 ..< 40 {
            let vocabularySize = step.isMultiple(of: 11) ? 1 : 2 + Int(unit() * 31)
            let base = (0 ..< vocabularySize).map { _ in (unit() - 0.5) * 16 }
            let biasCount = min(vocabularySize, 1 + Int(unit() * 4))
            let biases = (0 ..< biasCount).map { index in
                TokenBias(
                    tokenID: index,
                    tokenString: "token-\(index)",
                    value: (unit() - 0.25) * 20
                )
            }
            let remaining = max(0, meter.budget - meter.cumulative)
            let decision = try BiasBudgetSelector.select(
                baseLogits: base,
                biases: biases,
                remaining: remaining,
                temperature: 0.2 + unit() * 1.8
            )
            #expect(decision.divergence <= remaining)
            _ = try meter.record(divergence: decision.divergence)
            #expect(meter.cumulative <= meter.budget)
        }
    }
}

@Test func randomizedDenseBisectionNeverOvershootsBudget() throws {
    var state: UInt64 = 0xaced_0add_cafe_babe
    func unit() -> Double {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Double(state >> 11) / Double(1 << 53)
    }

    for budget in [0.0, 1e-12, 1e-8, 0.01, 0.5, 4.0] {
        var meter = KLMeter(budget: budget)
        for step in 0 ..< 40 {
            let count = step.isMultiple(of: 11) ? 1 : 2 + Int(unit() * 31)
            let base = (0 ..< count).map { _ in (unit() - 0.5) * 16 }
            let direction = (0 ..< count).map { _ in (unit() - 0.5) * 20 }
            let remaining = max(0, meter.budget - meter.cumulative)
            var closureCalls = 0
            let decision = try BiasBudgetSelector.select(
                baseLogits: base,
                remaining: remaining,
                temperature: 0.2 + unit() * 1.8
            ) { scale in
                closureCalls += 1
                // Alternate an affine path with a deliberately nonlinear,
                // non-monotone gain. The selector promises feasibility for
                // arbitrary dense closures, not global maximality.
                let gain = step.isMultiple(of: 2)
                    ? scale
                    : scale + 0.3 * Foundation.sin(4 * .pi * scale)
                return zip(base, direction).map { $0 + gain * $1 }
            }
            if remaining == 0 {
                #expect(closureCalls == 0)
                #expect(decision == BiasBudgetDecision(scale: 0, divergence: 0))
            }
            #expect(decision.divergence <= remaining)
            _ = try meter.record(divergence: decision.divergence)
            #expect(meter.cumulative <= meter.budget)
        }
    }
}

@Test func returnedTokenViewDropsSpeculativeReadAhead() throws {
    var meter = KLMeter(budget: 10)
    _ = try meter.record(divergence: 0.1)
    _ = try meter.record(divergence: 0.2)
    _ = try meter.record(divergence: 0.3)

    let delivered = meter.readings(forReturnedTokenCount: 2)
    #expect(delivered.count == 2)
    #expect(abs((delivered.last?.cumulative ?? 0) - 0.3) < 1e-12)
    #expect(meter.reading(forReturnedTokenCount: 2) == delivered.last)
    #expect(meter.reading(forReturnedTokenCount: 4) == nil)
}

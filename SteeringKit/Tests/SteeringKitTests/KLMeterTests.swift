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
    #expect(fixture.implementation == "python-logsumexp-kl-v1")

    for item in fixture.cases {
        let actual = try KLMeter.divergence(
            biasedLogits: item.biasedLogits,
            baseLogits: item.baseLogits
        )
        #expect(abs(actual - item.expectedKL) < 1e-12, "fixture: \(item.name)")
    }
}

@Test func meterAccumulatesAndReportsBudget() throws {
    var meter = KLMeter(budget: 0.1)
    let first = try meter.record(biasedLogits: [1, 0, 0], baseLogits: [0, 0, 0])
    let second = try meter.record(biasedLogits: [1, 0, 0], baseLogits: [0, 0, 0])

    #expect(first.step == 1)
    #expect(second.step == 2)
    #expect(abs(second.cumulative - 2 * first.perStep) < 1e-12)
    #expect(abs(second.budgetFraction - second.cumulative / 0.1) < 1e-12)
}


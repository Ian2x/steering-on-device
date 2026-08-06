import Foundation
import Testing
@testable import SteeringKit

@Test func loadsLexiconsAndBuildsSingleTokenSupport() throws {
    let data = Data(
        """
        [{
          "id": "wedding",
          "name": "Wedding",
          "terms": ["wedding", "bride"],
          "tokenStrings": [" wedding", " bride", " wedding day", " duplicate"],
          "tokenWeights": {" wedding": 2.0, " bride": -0.5}
        }]
        """.utf8
    )
    let lexicon = try #require(LexiconBias.load(from: data).first)
    let tokenMap = [" wedding": [10], " bride": [11], " wedding day": [12, 13], " duplicate": [10]]
    let construction = try LexiconBias.build(for: lexicon, strength: 1.5) {
        tokenMap[$0] ?? []
    }

    #expect(construction.biases == [
        TokenBias(tokenID: 10, tokenString: " wedding", value: 3.0),
        TokenBias(tokenID: 11, tokenString: " bride", value: -0.75),
    ])
    #expect(construction.droppedTokenStrings == [" wedding day"])
}

@Test func rejectsLexiconWithoutSingleTokenEntries() throws {
    let lexicon = SteeringLexicon(
        id: "multi",
        name: "Multi",
        terms: ["two words"],
        tokenStrings: [" two words"]
    )

    #expect(throws: LexiconBiasError.noSingleTokenEntries(lexiconID: "multi")) {
        try LexiconBias.build(for: lexicon, strength: 1) { _ in [1, 2] }
    }
}


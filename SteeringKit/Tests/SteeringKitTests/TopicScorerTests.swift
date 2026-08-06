import Testing
@testable import SteeringKit

@Test func overlapScorerMatchesWholeWords() throws {
    let lexicon = SteeringLexicon(
        id: "wedding",
        name: "Wedding",
        terms: ["bride", "groom", "vows", "altar"],
        tokenStrings: []
    )
    let score = try LexiconOverlapScorer().score(
        text: "The bride and groom exchanged vows.",
        lexicon: lexicon
    )
    #expect(score == 0.75)
}


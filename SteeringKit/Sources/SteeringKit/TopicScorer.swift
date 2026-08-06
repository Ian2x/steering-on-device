import Foundation

public protocol TopicScorer: Sendable {
    func score(text: String, lexicon: SteeringLexicon) throws -> Double
}

/// A dependency-free test double and fallback diagnostic, not the app's judge.
public struct LexiconOverlapScorer: TopicScorer {
    public init() {}

    public func score(text: String, lexicon: SteeringLexicon) throws -> Double {
        let words = Set(
            text.lowercased().split { !$0.isLetter }.map(String.init)
        )
        let terms = Set(lexicon.terms.map { $0.lowercased() })
        guard !terms.isEmpty else { return 0 }
        return Double(words.intersection(terms).count) / Double(terms.count)
    }
}


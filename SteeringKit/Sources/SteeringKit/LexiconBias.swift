import Foundation

/// A topic lexicon and the token strings used by the sparse output controller.
///
/// `terms` are used for human-readable scoring. `tokenStrings` are deliberately
/// separate because the paper's controller uses leading-space strings and keeps
/// only entries that tokenize to exactly one token.
public struct SteeringLexicon: Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let name: String
    public let terms: [String]
    public let tokenStrings: [String]
    public let tokenWeights: [String: Double]?

    public init(
        id: String,
        name: String,
        terms: [String],
        tokenStrings: [String],
        tokenWeights: [String: Double]? = nil
    ) {
        self.id = id
        self.name = name
        self.terms = terms
        self.tokenStrings = tokenStrings
        self.tokenWeights = tokenWeights
    }
}

public struct TokenBias: Codable, Equatable, Sendable {
    public let tokenID: Int
    public let tokenString: String
    public let value: Double

    public init(tokenID: Int, tokenString: String, value: Double) {
        self.tokenID = tokenID
        self.tokenString = tokenString
        self.value = value
    }
}

public struct BiasConstruction: Equatable, Sendable {
    public let biases: [TokenBias]
    public let droppedTokenStrings: [String]

    public init(biases: [TokenBias], droppedTokenStrings: [String]) {
        self.biases = biases
        self.droppedTokenStrings = droppedTokenStrings
    }
}

public enum LexiconBiasError: Error, Equatable {
    case noSingleTokenEntries(lexiconID: String)
    case invalidStrength(Double)
}

public enum LexiconBias {
    public static func load(from data: Data) throws -> [SteeringLexicon] {
        try JSONDecoder().decode([SteeringLexicon].self, from: data)
    }

    public static func load(from url: URL) throws -> [SteeringLexicon] {
        try load(from: Data(contentsOf: url))
    }

    /// Builds the paper-style sparse bias support.
    ///
    /// The audit's `resolve_a0_token_ids` keeps a leading-space token string
    /// only when it encodes to one token; multi-token entries are reported and
    /// omitted. Duplicate token IDs are kept once. `tokenWeights` can carry
    /// signed regression-discovered values; demo lexicons default to +1.
    public static func build(
        for lexicon: SteeringLexicon,
        strength: Double,
        encode: (String) throws -> [Int]
    ) throws -> BiasConstruction {
        guard strength.isFinite else {
            throw LexiconBiasError.invalidStrength(strength)
        }

        var biases: [TokenBias] = []
        var dropped: [String] = []
        var seen = Set<Int>()

        for tokenString in lexicon.tokenStrings {
            let ids = try encode(tokenString)
            guard ids.count == 1 else {
                dropped.append(tokenString)
                continue
            }

            let tokenID = ids[0]
            guard seen.insert(tokenID).inserted else { continue }
            let weight = lexicon.tokenWeights?[tokenString] ?? 1
            biases.append(
                TokenBias(
                    tokenID: tokenID,
                    tokenString: tokenString,
                    value: strength * weight
                )
            )
        }

        guard !biases.isEmpty else {
            throw LexiconBiasError.noSingleTokenEntries(lexiconID: lexicon.id)
        }
        return BiasConstruction(biases: biases, droppedTokenStrings: dropped)
    }
}


import CoreML
import Foundation
import SteeringKit
import Tokenizers

@MainActor
final class CoreMLTopicScorer {
    private struct CentroidFile: Decodable {
        let modelID: String
        let dimensions: Int
        let centroids: [String: [Double]]

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case dimensions, centroids
        }
    }

    private let model: MLModel
    private let tokenizer: any Tokenizer
    private let centroids: [String: [Double]]
    private let dimensions: Int
    private let maxLength = 128

    static func load() async throws -> CoreMLTopicScorer {
        let model = try loadModel()
        guard let tokenizerFolder = Bundle.main.url(
            forResource: "MiniLMTokenizer",
            withExtension: nil
        ) else {
            throw DemoError.missingResource("MiniLMTokenizer")
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
        guard let centroidsURL = Bundle.main.url(
            forResource: "topic-centroids",
            withExtension: "json"
        ) else {
            throw DemoError.missingResource("topic-centroids.json")
        }
        let file = try JSONDecoder().decode(
            CentroidFile.self,
            from: Data(contentsOf: centroidsURL)
        )
        return CoreMLTopicScorer(
            model: model,
            tokenizer: tokenizer,
            centroids: file.centroids,
            dimensions: file.dimensions
        )
    }

    private static func loadModel() throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        if let compiled = Bundle.main.url(forResource: "TopicEncoder", withExtension: "mlmodelc") {
            return try MLModel(contentsOf: compiled, configuration: configuration)
        }
        guard let package = Bundle.main.url(forResource: "TopicEncoder", withExtension: "mlpackage") else {
            throw DemoError.missingResource("TopicEncoder.mlpackage")
        }
        let compiled = try MLModel.compileModel(at: package)
        return try MLModel(contentsOf: compiled, configuration: configuration)
    }

    private init(
        model: MLModel,
        tokenizer: any Tokenizer,
        centroids: [String: [Double]],
        dimensions: Int
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.centroids = centroids
        self.dimensions = dimensions
    }

    func score(text: String, lexicon: SteeringLexicon) throws -> Double {
        let embedding = try embed(text)
        guard let centroid = centroids[lexicon.id], centroid.count == dimensions else {
            throw DemoError.invalidCentroid(lexicon.id)
        }
        let dot = zip(embedding, centroid).reduce(0.0) { $0 + $1.0 * $1.1 }
        let leftNorm = sqrt(embedding.reduce(0.0) { $0 + $1 * $1 })
        let rightNorm = sqrt(centroid.reduce(0.0) { $0 + $1 * $1 })
        guard leftNorm > 0, rightNorm > 0 else { return 0 }
        return dot / (leftNorm * rightNorm)
    }

    private func embed(_ text: String) throws -> [Double] {
        guard let clsID = tokenizer.convertTokenToId("[CLS]"),
              let sepID = tokenizer.convertTokenToId("[SEP]")
        else {
            throw DemoError.missingResource("MiniLM [CLS]/[SEP] tokens")
        }
        let content = tokenizer.encode(text: text, addSpecialTokens: false)
        var ids = [clsID]
            + Array(content.prefix(maxLength - 2))
            + [sepID]
        let attentionCount = ids.count
        let padID = tokenizer.convertTokenToId("[PAD]") ?? 0
        ids += Array(repeating: padID, count: maxLength - ids.count)

        let inputIDs = try MLMultiArray(
            shape: [1, NSNumber(value: maxLength)],
            dataType: .int32
        )
        let attentionMask = try MLMultiArray(
            shape: [1, NSNumber(value: maxLength)],
            dataType: .int32
        )
        for index in 0 ..< maxLength {
            inputIDs[index] = NSNumber(value: Int32(ids[index]))
            attentionMask[index] = NSNumber(value: Int32(index < attentionCount ? 1 : 0))
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIDs,
            "attention_mask": attentionMask,
        ])
        let prediction = try model.prediction(from: provider)
        guard let array = prediction.featureValue(for: "embedding")?.multiArrayValue else {
            throw DemoError.missingCoreMLOutput
        }
        return (0 ..< dimensions).map { Double(truncating: array[$0]) }
    }
}

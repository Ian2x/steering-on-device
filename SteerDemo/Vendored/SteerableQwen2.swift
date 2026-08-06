// Adapted from mlx-swift-examples/Libraries/MLXLLM/Models/Qwen2.swift at
// 9bff95ca5f0b9e8c021acc4d71a2bbe4a7441631 (MIT License).
// Local changes rename the model and expose a residual-stream prompt boundary
// so a front-aligned per-position direction can be baked into the KV cache.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct SteerableQwen2Direction {
    let matrix: MLXArray
    let diagnostics: ResidualDirectionDiagnostics
}

// port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/qwen2.py

private class Attention: Module {
    let args: SteerableQwen2Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    public init(_ args: SteerableQwen2Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta,
            scale: ropeScale)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: SteerableQwen2Configuration) {
        _attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

private class SteerableQwen2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm

    public init(_ args: SteerableQwen2Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }

    func residualSequence(_ inputs: MLXArray, afterLayer layerIndex: Int) -> MLXArray {
        precondition(layers.indices.contains(layerIndex), "residual layer is outside the model")
        var hidden = embedTokens(inputs)
        let mask = createAttentionMask(h: hidden, cache: nil)
        for index in 0 ... layerIndex {
            hidden = layers[index](hidden, mask: mask, cache: nil)
        }
        return hidden
    }

    func prefill(
        _ inputs: MLXArray,
        cache: [KVCache],
        direction: MLXArray,
        coefficient: Double,
        afterLayer layerIndex: Int
    ) -> MLXArray {
        precondition(layers.indices.contains(layerIndex), "residual layer is outside the model")
        precondition(cache.count == layers.count, "one KV cache is required per layer")
        var hidden = embedTokens(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache[index])
            if index == layerIndex, coefficient != 0 {
                let positions = min(hidden.dim(1), direction.dim(0))
                let edited = hidden + zeros(like: hidden)
                edited[0..., ..<positions, 0...] =
                    edited[0..., ..<positions, 0...]
                    + direction[..<positions, 0...][.newAxis] * Float(coefficient)
                hidden = edited
            }
        }
        return norm(hidden)
    }
}

public class SteerableQwen2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: SteerableQwen2ModelInner
    let configuration: SteerableQwen2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: SteerableQwen2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = SteerableQwen2ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        project(model(inputs, cache: cache))
    }

    var hiddenLayerCount: Int { model.layers.count }

    func residualDirection(
        positiveTokens: [Int],
        negativeTokens: [Int],
        historicalPaddingTokenID: Int,
        afterLayer layerIndex: Int,
        mode: ResidualDirectionMode,
        randomSeed: UInt64
    ) -> SteerableQwen2Direction {
        precondition(!positiveTokens.isEmpty && !negativeTokens.isEmpty)
        let positiveHidden = model.residualSequence(
            MLXArray(positiveTokens)[.newAxis],
            afterLayer: layerIndex
        )[0]
        let negativeHidden = model.residualSequence(
            MLXArray(negativeTokens)[.newAxis],
            afterLayer: layerIndex
        )[0]
        let alignedPositions = max(positiveTokens.count, negativeTokens.count)
        let semantic = zeros([alignedPositions, configuration.hiddenSize])
        semantic[..<positiveTokens.count, 0...] = positiveHidden
        semantic[..<negativeTokens.count, 0...] =
            semantic[..<negativeTokens.count, 0...] - negativeHidden

        let positiveHistorical =
            Array(repeating: historicalPaddingTokenID, count: alignedPositions - positiveTokens.count)
            + positiveTokens
        let negativeHistorical =
            Array(repeating: historicalPaddingTokenID, count: alignedPositions - negativeTokens.count)
            + negativeTokens
        let historicalPositive = model.residualSequence(
            MLXArray(positiveHistorical)[.newAxis],
            afterLayer: layerIndex
        )[0, -1, 0...]
        let historicalNegative = model.residualSequence(
            MLXArray(negativeHistorical)[.newAxis],
            afterLayer: layerIndex
        )[0, -1, 0...]
        let historical = historicalPositive - historicalNegative

        let semanticNormArray = sqrt(sum(semantic * semantic))
        let applied: MLXArray
        if mode == .randomMatchedNorm {
            let random = MLXRandom.normal(
                semantic.shape,
                dtype: .float32,
                key: MLXRandom.key(randomSeed)
            )
            let randomNorm = sqrt(sum(random * random))
            applied = random * semanticNormArray / maximum(randomNorm, MLXArray(1e-12))
        } else {
            applied = semantic
        }

        let semanticPerPosition = sqrt(sum(semantic * semantic, axis: -1))
        let appliedPerPosition = sqrt(sum(applied * applied, axis: -1))
        let appliedNormArray = sqrt(sum(applied * applied))
        let historicalNormArray = sqrt(sum(historical * historical))
        eval(
            semantic,
            applied,
            semanticNormArray,
            appliedNormArray,
            historicalNormArray,
            semanticPerPosition,
            appliedPerPosition
        )
        let diagnostics = ResidualDirectionDiagnostics(
            mode: mode.rawValue,
            positiveTokenCount: positiveTokens.count,
            negativeTokenCount: negativeTokens.count,
            alignedPositionCount: alignedPositions,
            historicalFinalVectorNorm: Double(historicalNormArray.item(Float.self)),
            semanticMatrixNorm: Double(semanticNormArray.item(Float.self)),
            appliedMatrixNorm: Double(appliedNormArray.item(Float.self)),
            semanticPerPositionNorms: semanticPerPosition.asArray(Float.self).map(Double.init),
            appliedPerPositionNorms: appliedPerPosition.asArray(Float.self).map(Double.init),
            alignment: "front-aligned; an absent side contributes zero at suffix positions and is never tokenized",
            injection: "all aligned prompt positions after the selected block; persistent in downstream KV cache"
        )
        return SteerableQwen2Direction(matrix: applied, diagnostics: diagnostics)
    }

    func prefillLogits(
        _ inputs: MLXArray,
        cache: [KVCache],
        direction: MLXArray,
        coefficient: Double,
        afterLayer layerIndex: Int
    ) -> MLXArray {
        project(
            model.prefill(
                inputs,
                cache: cache,
                direction: direction,
                coefficient: coefficient,
                afterLayer: layerIndex
            )
        )[0, -1, 0...]
    }

    private func project(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // Remove unused precomputed rotary freqs
        return weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
}

public struct SteerableQwen2Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<SteerableQwen2Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: SteerableQwen2Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: SteerableQwen2Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: SteerableQwen2Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: SteerableQwen2Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: SteerableQwen2Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: SteerableQwen2Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: SteerableQwen2Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: SteerableQwen2Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: SteerableQwen2Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.ropeTraditional =
            try container.decodeIfPresent(
                Bool.self, forKey: SteerableQwen2Configuration.CodingKeys.ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: SteerableQwen2Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

// MARK: - LoRA

extension SteerableQwen2Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}

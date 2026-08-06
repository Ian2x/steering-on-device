import Darwin
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import SteeringKit

final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false

    func reset() {
        lock.lock()
        stopped = false
        lock.unlock()
    }

    func stop() {
        lock.lock()
        stopped = true
        lock.unlock()
    }

    func isStopped() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }
}

actor MLXGenerationService {
    static let modelID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    static let modelRevision = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
    static let samplingTemperature: Float = 0.7
    static let samplingSeed: UInt64 = 42
#if DEBUG
    static let buildConfiguration = "Debug"
#else
    static let buildConfiguration = "Release"
#endif

    private var container: ModelContainer?
    private var loadingTask: Task<ModelContainer, Error>?

    private static func registerSteerableQwen() {
        LLMTypeRegistry.shared.registerModelType("qwen2") { configurationURL in
            let configuration = try JSONDecoder().decode(
                SteerableQwen2Configuration.self,
                from: Data(contentsOf: configurationURL)
            )
            return SteerableQwen2Model(configuration)
        }
    }

    func loadModel(progress: @escaping @Sendable (Double) -> Void) async throws {
        guard container == nil else {
            progress(1)
            return
        }
        if let loadingTask {
            container = try await loadingTask.value
            progress(1)
            return
        }
        let configuration = ModelConfiguration(
            id: Self.modelID,
            revision: Self.modelRevision,
            defaultPrompt: "Describe a quiet morning routine."
        )
        Self.registerSteerableQwen()
        let task = Task {
            try await LLMModelFactory.shared.loadContainer(
                configuration: configuration
            ) { download in
                progress(download.fractionCompleted)
            }
        }
        loadingTask = task
        do {
            let loaded = try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
            container = loaded
            loadingTask = nil
            progress(1)
        } catch {
            loadingTask = nil
            throw error
        }
    }

    /// Compile and cache the model's Metal kernels before either measured
    /// pane runs. Without this untimed token, the first (baseline) pass pays
    /// one-time JIT work and its tokens/second is not comparable with the
    /// already-warm steered pass.
    func warmUp(prompt: String) async throws {
        guard let container else {
            throw DemoError.missingResource("loaded MLX model")
        }
        try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user(prompt)])
            )
            let iterator = try TokenIterator(
                input: input,
                model: context.model,
                processor: nil,
                sampler: SeededCategoricalSampler(
                    temperature: Self.samplingTemperature,
                    seed: Self.samplingSeed
                ),
                maxTokens: 16
            )
            for _ in iterator {
                try Task.checkCancellation()
                break
            }
            Stream.gpu.synchronize()
        }
    }

    func generate(
        pane: GenerationPane,
        prompt: String,
        lexicon: SteeringLexicon,
        strength: Double,
        actAddCoefficient: Double,
        actAddLayer: Int,
        maxTokens: Int,
        klBudget: Double,
        stopFlag: StopFlag,
        update: @escaping @Sendable (GenerationUpdate) -> Void
    ) async throws -> GenerationSummary {
        guard let container else {
            throw DemoError.missingResource("loaded MLX model")
        }

        return try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user(prompt)])
            )
            let actAddRoute = ActAddPassPlanner.run(
                coefficient: actAddCoefficient,
                baseline: { ActAddPassRoute.baseline },
                activationAddition: { ActAddPassRoute.activationAddition }
            )
            if pane == .actAdd, actAddRoute == .activationAddition
            {
                return try Self.generateActAdd(
                    context: context,
                    input: input,
                    lexicon: lexicon,
                    coefficient: actAddCoefficient,
                    layer: actAddLayer,
                    maxTokens: maxTokens,
                    klBudget: klBudget,
                    stopFlag: stopFlag,
                    update: update
                )
            }
            let construction: BiasConstruction
            if pane == .steered {
                construction = try LexiconBias.build(
                    for: lexicon,
                    strength: strength
                ) { tokenString in
                    context.tokenizer.encode(text: tokenString, addSpecialTokens: false)
                }
            } else {
                construction = BiasConstruction(biases: [], droppedTokenStrings: [])
            }

            let sink = KLMetricsSink(budget: klBudget)
            let iterator: TokenIterator
            if pane == .steered {
                iterator = try TokenIterator(
                    input: input,
                    model: context.model,
                    processor: SparseBiasProcessor(
                        biases: construction.biases,
                        temperature: Self.samplingTemperature,
                        sink: sink
                    ),
                    sampler: SeededCategoricalSampler(
                        temperature: Self.samplingTemperature,
                        seed: Self.samplingSeed
                    ),
                    maxTokens: maxTokens
                )
            } else {
                iterator = try TokenIterator(
                    input: input,
                    model: context.model,
                    processor: nil,
                    sampler: SeededCategoricalSampler(
                        temperature: Self.samplingTemperature,
                        seed: Self.samplingSeed
                    ),
                    maxTokens: maxTokens
                )
            }

            let extraEOS = Set(
                context.configuration.extraEOSTokens.compactMap {
                    context.tokenizer.convertTokenToId($0)
                }
            )
            var tokens: [Int] = []
            var lastReportedKLStep = 0
            let start = ContinuousClock.now

            for token in iterator {
                try Task.checkCancellation()
                try sink.throwIfFailed()
                if stopFlag.isStopped() { break }
                if token == context.tokenizer.unknownTokenId
                    || token == context.tokenizer.eosTokenId
                    || extraEOS.contains(token)
                {
                    break
                }
                tokens.append(token)
                let elapsed = max(0.001, start.duration(to: .now).seconds)
                let latestKL = pane == .steered
                    ? sink.reading(forReturnedTokenCount: tokens.count)
                    : nil
                let newKLReading: KLReading?
                if let latestKL, latestKL.step > lastReportedKLStep {
                    newKLReading = latestKL
                    lastReportedKLStep = latestKL.step
                } else {
                    newKLReading = nil
                }
                update(
                    GenerationUpdate(
                        pane: pane,
                        text: context.tokenizer.decode(tokens: tokens),
                        tokenCount: tokens.count,
                        tokensPerSecond: Double(tokens.count) / elapsed,
                        residentMemoryBytes: residentMemoryBytes(),
                        klReading: newKLReading
                    )
                )
            }
            try sink.throwIfFailed()
            try Task.checkCancellation()
            Stream.gpu.synchronize()
            let seconds = max(0.001, start.duration(to: .now).seconds)
            return GenerationSummary(
                pane: pane,
                text: context.tokenizer.decode(tokens: tokens),
                tokenIDs: tokens,
                tokenCount: tokens.count,
                seconds: seconds,
                residentMemoryBytes: residentMemoryBytes(),
                klHistory: sink.history(forReturnedTokenCount: tokens.count),
                droppedTokenStrings: construction.droppedTokenStrings
            )
        }
    }

    private nonisolated static func generateActAdd(
        context: ModelContext,
        input: LMInput,
        lexicon: SteeringLexicon,
        coefficient: Double,
        layer: Int,
        maxTokens: Int,
        klBudget: Double,
        stopFlag: StopFlag,
        update: @escaping @Sendable (GenerationUpdate) -> Void
    ) throws -> GenerationSummary {
        guard let model = context.model as? SteerableQwen2Model else {
            throw DemoError.missingResource("steerable Qwen2 model implementation")
        }
        guard (0 ..< model.hiddenLayerCount).contains(layer) else {
            throw DemoError.invalidActAddLayer(layer, model.hiddenLayerCount)
        }
        guard let positivePrompt = lexicon.actAddPositivePrompt,
              let negativePrompt = lexicon.actAddNegativePrompt
        else {
            throw DemoError.missingResource("ActAdd contrast prompts for \(lexicon.id)")
        }

        var positive = context.tokenizer.encode(text: positivePrompt, addSpecialTokens: false)
        var negative = context.tokenizer.encode(text: negativePrompt, addSpecialTokens: false)
        let alignedLength = max(positive.count, negative.count)
        guard let paddingID = context.tokenizer.eosTokenId else {
            throw DemoError.missingResource("tokenizer EOS token for ActAdd alignment")
        }
        positive = Array(repeating: paddingID, count: alignedLength - positive.count) + positive
        negative = Array(repeating: paddingID, count: alignedLength - negative.count) + negative
        let direction = model.residualVector(
            positiveTokens: MLXArray(positive)[.newAxis],
            negativeTokens: MLXArray(negative)[.newAxis],
            afterLayer: layer
        )

        let promptTokens = input.text.tokens.asArray(Int.self)
        let extraEOS = Set(
            context.configuration.extraEOSTokens.compactMap {
                context.tokenizer.convertTokenToId($0)
            }
        )
        let sampler = SeededCategoricalSampler(
            temperature: Self.samplingTemperature,
            seed: Self.samplingSeed
        )
        var tokens: [Int] = []
        var meter = KLMeter(budget: klBudget)
        let start = ContinuousClock.now

        for _ in 0 ..< maxTokens {
            try Task.checkCancellation()
            if stopFlag.isStopped() { break }

            let fullPrefix = MLXArray(promptTokens + tokens)[.newAxis]
            let tail = model.prepareTail(fullPrefix, afterLayer: layer)
            let baseLogits = model.logits(from: tail).asType(.float32)
            eval(baseLogits)
            let baseValues = baseLogits.asArray(Float.self).map(Double.init)
            let remaining = max(0, klBudget - meter.cumulative)
            let decision: BiasBudgetDecision
            if remaining <= 1e-8 {
                decision = BiasBudgetDecision(scale: 0, divergence: 0)
            } else {
                decision = try BiasBudgetSelector.select(
                    baseLogits: baseValues,
                    remaining: remaining,
                    temperature: Double(Self.samplingTemperature)
                ) { scale in
                    let candidate = model.logits(
                        from: tail,
                        direction: direction,
                        scale: coefficient * scale
                    ).asType(.float32)
                    eval(candidate)
                    return candidate.asArray(Float.self).map(Double.init)
                }
            }

            let selectedLogits: MLXArray
            let selectedDivergence: Double?
            if decision.scale > 0 {
                selectedLogits = model.logits(
                    from: tail,
                    direction: direction,
                    scale: coefficient * decision.scale
                )
                selectedDivergence = decision.divergence
            } else {
                selectedLogits = baseLogits
                selectedDivergence = nil
            }
            let sampled = sampler.sample(logits: selectedLogits)
            eval(sampled)
            let token = sampled.item(Int.self)
            if token == context.tokenizer.unknownTokenId
                || token == context.tokenizer.eosTokenId
                || extraEOS.contains(token)
            {
                break
            }
            if let selectedDivergence {
                _ = try meter.record(divergence: selectedDivergence)
            }
            tokens.append(token)
            let elapsed = max(0.001, start.duration(to: .now).seconds)
            update(
                GenerationUpdate(
                    pane: .actAdd,
                    text: context.tokenizer.decode(tokens: tokens),
                    tokenCount: tokens.count,
                    tokensPerSecond: Double(tokens.count) / elapsed,
                    residentMemoryBytes: residentMemoryBytes(),
                    klReading: meter.history.last
                )
            )
        }
        try Task.checkCancellation()
        Stream.gpu.synchronize()
        return GenerationSummary(
            pane: .actAdd,
            text: context.tokenizer.decode(tokens: tokens),
            tokenIDs: tokens,
            tokenCount: tokens.count,
            seconds: max(0.001, start.duration(to: .now).seconds),
            residentMemoryBytes: residentMemoryBytes(),
            klHistory: meter.history,
            droppedTokenStrings: []
        )
    }
}

/// Gives both panes the same stochastic draws while allowing a logit change to
/// alter the sampled token. Greedy decoding hid non-zero distribution shifts
/// until the bias was large enough to cross an argmax boundary.
private struct SeededCategoricalSampler: LogitSampler {
    private let inverseTemperature: Float
    private let randomState: MLXRandom.RandomState

    init(temperature: Float, seed: UInt64) {
        precondition(temperature > 0)
        inverseTemperature = 1 / temperature
        randomState = MLXRandom.RandomState(seed: seed)
    }

    func sample(logits: MLXArray) -> MLXArray {
        MLXRandom.categorical(logits * inverseTemperature, key: randomState)
    }
}

private func residentMemoryBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? UInt64(info.resident_size) : 0
}

private extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}

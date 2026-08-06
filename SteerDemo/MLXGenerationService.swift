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
        residualDirectionMode: ResidualDirectionMode,
        staticBiasMode: Bool,
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
                    directionMode: residualDirectionMode,
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

            let sink = KLMetricsSink(
                budget: staticBiasMode ? Double.greatestFiniteMagnitude : klBudget
            )
            let iterator: TokenIterator
            if pane == .steered {
                if staticBiasMode {
                    iterator = try TokenIterator(
                        input: input,
                        model: context.model,
                        processor: FixedSparseBiasProcessor(
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
                }
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
            let baseModelNLL = try Self.baseModelNLL(
                model: context.model,
                promptTokens: input.text.tokens.asArray(Int.self),
                continuationTokens: tokens
            )
            return GenerationSummary(
                pane: pane,
                text: context.tokenizer.decode(tokens: tokens),
                tokenIDs: tokens,
                tokenCount: tokens.count,
                seconds: seconds,
                residentMemoryBytes: residentMemoryBytes(),
                klHistory: sink.history(forReturnedTokenCount: tokens.count),
                droppedTokenStrings: construction.droppedTokenStrings,
                baseModelNLL: baseModelNLL,
                appliedCoefficient: pane == .actAdd ? 0 : nil,
                directionDiagnostics: nil
            )
        }
    }

    private nonisolated static func generateActAdd(
        context: ModelContext,
        input: LMInput,
        lexicon: SteeringLexicon,
        coefficient: Double,
        layer: Int,
        directionMode: ResidualDirectionMode,
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
        let direction = try residualDirection(
            context: context,
            model: model,
            lexicon: lexicon,
            layer: layer,
            mode: directionMode
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
        var baseNLLSum = 0.0
        let start = ContinuousClock.now

        let baseCache = model.newCache(parameters: nil)
        let editedCache = model.newCache(parameters: nil)
        let promptArray = MLXArray(promptTokens)[.newAxis]
        var baseLogits = model(promptArray, cache: baseCache)[0, -1, 0...].asType(.float32)
        var editedLogits = model.prefillLogits(
            promptArray,
            cache: editedCache,
            direction: direction.matrix,
            coefficient: coefficient,
            afterLayer: layer
        ).asType(.float32)

        for stepIndex in 0 ..< maxTokens {
            try Task.checkCancellation()
            if stopFlag.isStopped() { break }

            eval(baseLogits, editedLogits)
            let baseValues = baseLogits.asArray(Float.self).map(Double.init)
            let editedValues = editedLogits.asArray(Float.self).map(Double.init)
            let inverseTemperature = 1 / Double(Self.samplingTemperature)
            let divergence = try KLMeter.divergence(
                biasedLogits: editedValues.map { $0 * inverseTemperature },
                baseLogits: baseValues.map { $0 * inverseTemperature }
            )
            let sampled = sampler.sample(logits: editedLogits)
            eval(sampled)
            let token = sampled.item(Int.self)
            if token == context.tokenizer.unknownTokenId
                || token == context.tokenizer.eosTokenId
                || extraEOS.contains(token)
            {
                break
            }
            _ = try meter.record(divergence: divergence)
            baseNLLSum += negativeLogLikelihood(
                logits: baseValues,
                token: token,
                temperature: Double(Self.samplingTemperature)
            )
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
            if stepIndex + 1 < maxTokens {
                let tokenArray = MLXArray([token])[.newAxis]
                baseLogits = model(tokenArray, cache: baseCache)[0, -1, 0...].asType(.float32)
                editedLogits = model(tokenArray, cache: editedCache)[0, -1, 0...].asType(.float32)
            }
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
            droppedTokenStrings: [],
            baseModelNLL: tokens.isEmpty ? nil : baseNLLSum / Double(tokens.count),
            appliedCoefficient: coefficient,
            directionDiagnostics: direction.diagnostics
        )
    }

    private nonisolated static func negativeLogLikelihood(
        logits: [Double],
        token: Int,
        temperature: Double
    ) -> Double {
        guard logits.indices.contains(token), temperature > 0 else { return .infinity }
        let scaled = logits.map { $0 / temperature }
        let maximum = scaled.max() ?? 0
        let logNormalizer = maximum + Foundation.log(
            scaled.reduce(0.0) { $0 + Foundation.exp($1 - maximum) }
        )
        return logNormalizer - scaled[token]
    }

    func fixedBaselineContinuation(prompt: String, steps: Int) async throws -> [Int] {
        guard let container else {
            throw DemoError.missingResource("loaded MLX model")
        }
        precondition(steps > 0)
        return try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user(prompt)])
            )
            let cache = context.model.newCache(parameters: nil)
            var logits = context.model(input.text.tokens, cache: cache)[0, -1, 0...]
                .asType(.float32)
            let sampler = SeededCategoricalSampler(
                temperature: Self.samplingTemperature,
                seed: Self.samplingSeed
            )
            var tokens: [Int] = []
            tokens.reserveCapacity(steps)
            for index in 0 ..< steps {
                let sampled = sampler.sample(logits: logits)
                eval(sampled)
                let token = sampled.item(Int.self)
                tokens.append(token)
                if index + 1 < steps {
                    logits = context.model(MLXArray([token])[.newAxis], cache: cache)[0, -1, 0...]
                        .asType(.float32)
                }
            }
            return tokens
        }
    }

    func teacherForcedKL(
        method: GenerationPane,
        prompt: String,
        continuationTokens: [Int],
        lexicon: SteeringLexicon,
        strength: Double,
        actAddCoefficient: Double,
        actAddLayer: Int,
        residualDirectionMode: ResidualDirectionMode
    ) async throws -> TeacherForcedKLResult {
        guard let container else {
            throw DemoError.missingResource("loaded MLX model")
        }
        guard !continuationTokens.isEmpty else {
            throw DemoError.invalidKLDivergence("teacher-forced continuation is empty")
        }
        return try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user(prompt)])
            )
            switch method {
            case .steered:
                let construction = try LexiconBias.build(
                    for: lexicon,
                    strength: strength
                ) { tokenString in
                    context.tokenizer.encode(text: tokenString, addSpecialTokens: false)
                }
                return try teacherForcedLogitKL(
                    model: context.model,
                    promptTokens: input.text.tokens.asArray(Int.self),
                    continuationTokens: continuationTokens,
                    biases: construction.biases,
                    appliedStrength: strength
                )
            case .actAdd:
                guard let model = context.model as? SteerableQwen2Model else {
                    throw DemoError.missingResource("steerable Qwen2 model implementation")
                }
                guard (0 ..< model.hiddenLayerCount).contains(actAddLayer) else {
                    throw DemoError.invalidActAddLayer(actAddLayer, model.hiddenLayerCount)
                }
                let direction = try residualDirection(
                    context: context,
                    model: model,
                    lexicon: lexicon,
                    layer: actAddLayer,
                    mode: residualDirectionMode
                )
                return try teacherForcedActAddKL(
                    model: model,
                    promptTokens: input.text.tokens.asArray(Int.self),
                    continuationTokens: continuationTokens,
                    direction: direction,
                    coefficient: actAddCoefficient,
                    layer: actAddLayer
                )
            case .baseline:
                throw DemoError.invalidKLDivergence("baseline has no intervention KL")
            }
        }
    }

    private nonisolated static func teacherForcedLogitKL(
        model: any LanguageModel,
        promptTokens: [Int],
        continuationTokens: [Int],
        biases: [TokenBias],
        appliedStrength: Double
    ) throws -> TeacherForcedKLResult {
        let cache = model.newCache(parameters: nil)
        var logits = model(MLXArray(promptTokens)[.newAxis], cache: cache)[0, -1, 0...]
            .asType(.float32)
        let tokenIDs = MLXArray(biases.map { UInt32($0.tokenID) })
        let values = MLXArray(biases.map { Float($0.value) })
        let inverseTemperature = 1 / Double(Self.samplingTemperature)
        var perStep: [Double] = []
        perStep.reserveCapacity(continuationTokens.count)
        for (index, token) in continuationTokens.enumerated() {
            let edited = logits + zeros(like: logits)
            edited[0..., tokenIDs] = edited[0..., tokenIDs] + values
            eval(logits, edited)
            perStep.append(
                try KLMeter.divergence(
                    biasedLogits: edited.asArray(Float.self).map { Double($0) * inverseTemperature },
                    baseLogits: logits.asArray(Float.self).map { Double($0) * inverseTemperature }
                )
            )
            if index + 1 < continuationTokens.count {
                logits = model(MLXArray([token])[.newAxis], cache: cache)[0, -1, 0...]
                    .asType(.float32)
            }
        }
        return TeacherForcedKLResult(
            method: "static-logit-bias",
            meanNatsPerStep: perStep.reduce(0, +) / Double(perStep.count),
            perStepNats: perStep,
            continuationTokenCount: continuationTokens.count,
            appliedScalar: appliedStrength,
            directionDiagnostics: nil
        )
    }

    private nonisolated static func teacherForcedActAddKL(
        model: SteerableQwen2Model,
        promptTokens: [Int],
        continuationTokens: [Int],
        direction: SteerableQwen2Direction,
        coefficient: Double,
        layer: Int
    ) throws -> TeacherForcedKLResult {
        let baseCache = model.newCache(parameters: nil)
        let editedCache = model.newCache(parameters: nil)
        let prompt = MLXArray(promptTokens)[.newAxis]
        var baseLogits = model(prompt, cache: baseCache)[0, -1, 0...].asType(.float32)
        var editedLogits = model.prefillLogits(
            prompt,
            cache: editedCache,
            direction: direction.matrix,
            coefficient: coefficient,
            afterLayer: layer
        ).asType(.float32)
        let inverseTemperature = 1 / Double(Self.samplingTemperature)
        var perStep: [Double] = []
        perStep.reserveCapacity(continuationTokens.count)
        for (index, token) in continuationTokens.enumerated() {
            eval(baseLogits, editedLogits)
            perStep.append(
                try KLMeter.divergence(
                    biasedLogits: editedLogits.asArray(Float.self).map { Double($0) * inverseTemperature },
                    baseLogits: baseLogits.asArray(Float.self).map { Double($0) * inverseTemperature }
                )
            )
            if index + 1 < continuationTokens.count {
                let next = MLXArray([token])[.newAxis]
                baseLogits = model(next, cache: baseCache)[0, -1, 0...].asType(.float32)
                editedLogits = model(next, cache: editedCache)[0, -1, 0...].asType(.float32)
            }
        }
        return TeacherForcedKLResult(
            method: direction.diagnostics.mode == ResidualDirectionMode.semantic.rawValue
                ? "residual-edit" : "random-matched-norm",
            meanNatsPerStep: perStep.reduce(0, +) / Double(perStep.count),
            perStepNats: perStep,
            continuationTokenCount: continuationTokens.count,
            appliedScalar: coefficient,
            directionDiagnostics: direction.diagnostics
        )
    }

    private nonisolated static func residualDirection(
        context: ModelContext,
        model: SteerableQwen2Model,
        lexicon: SteeringLexicon,
        layer: Int,
        mode: ResidualDirectionMode
    ) throws -> SteerableQwen2Direction {
        guard let positivePrompt = lexicon.actAddPositivePrompt,
              let negativePrompt = lexicon.actAddNegativePrompt
        else {
            throw DemoError.missingResource("ActAdd contrast prompts for \(lexicon.id)")
        }
        let positive = context.tokenizer.encode(text: positivePrompt, addSpecialTokens: false)
        let negative = context.tokenizer.encode(text: negativePrompt, addSpecialTokens: false)
        guard let paddingID = context.tokenizer.eosTokenId else {
            throw DemoError.missingResource("tokenizer EOS token for historical direction diagnostics")
        }
        return model.residualDirection(
            positiveTokens: positive,
            negativeTokens: negative,
            historicalPaddingTokenID: paddingID,
            afterLayer: layer,
            mode: mode,
            randomSeed: 20_260_806
        )
    }

    private nonisolated static func baseModelNLL(
        model: any LanguageModel,
        promptTokens: [Int],
        continuationTokens: [Int]
    ) throws -> Double? {
        guard !continuationTokens.isEmpty else { return nil }
        let cache = model.newCache(parameters: nil)
        var logits = model(MLXArray(promptTokens)[.newAxis], cache: cache)[0, -1, 0...]
            .asType(.float32)
        var total = 0.0
        for (index, token) in continuationTokens.enumerated() {
            eval(logits)
            total += negativeLogLikelihood(
                logits: logits.asArray(Float.self).map(Double.init),
                token: token,
                temperature: Double(Self.samplingTemperature)
            )
            if index + 1 < continuationTokens.count {
                logits = model(MLXArray([token])[.newAxis], cache: cache)[0, -1, 0...]
                    .asType(.float32)
            }
        }
        return total / Double(continuationTokens.count)
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

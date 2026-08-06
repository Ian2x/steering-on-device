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

    private var container: ModelContainer?
    private var loadingTask: Task<ModelContainer, Error>?

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

    func generate(
        pane: GenerationPane,
        prompt: String,
        lexicon: SteeringLexicon,
        strength: Double,
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
                tokenCount: tokens.count,
                seconds: seconds,
                residentMemoryBytes: residentMemoryBytes(),
                klHistory: sink.history(forReturnedTokenCount: tokens.count),
                droppedTokenStrings: construction.droppedTokenStrings
            )
        }
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

import Foundation
import SteeringKit

@MainActor
final class DemoViewModel: ObservableObject {
    private struct RunConfiguration: Sendable {
        let prompt: String
        let lexicon: SteeringLexicon
        let strength: Double
        let actAddCoefficient: Double
        let actAddLayer: Int
        let maxTokens: Int
        let klBudget: Double
    }

    private struct RunReport: Encodable {
        struct Pane: Encodable {
            let text: String
            let tokenCount: Int
            let tokensPerSecond: Double
            let residentMemoryBytes: UInt64
            let topicScore: Double?
        }

        let modelID: String
        let modelRevision: String
        let buildConfiguration: String
        let timestamp: String
        let status: String
        let error: String?
        let prompt: String
        let lexicon: String
        let biasStrength: Double
        let actAddCoefficient: Double
        let actAddLayer: Int
        let maxTokens: Int
        let seed: UInt64
        let temperature: Double
        let klBudget: Double
        let cumulativeKL: Double
        let klHistory: [KLReading]
        let actAddCumulativeKL: Double
        let actAddKLHistory: [KLReading]
        let baseline: Pane
        let steered: Pane
        let actAdd: Pane
        let droppedTokenStrings: [String]
    }

    @Published var prompt = "Describe a quiet morning routine in two short paragraphs."
    @Published var strength = 14.0
    @Published var actAddCoefficient = 12.0
    @Published var actAddLayer = 11
    @Published var selectedLexiconID = "wedding"
    @Published var maxTokens = 96
    @Published var klBudget = 8.0
    @Published private(set) var baseline = PaneState.empty
    @Published private(set) var steered = PaneState.empty
    @Published private(set) var actAdd = PaneState.empty
    @Published private(set) var klHistory: [KLReading] = []
    @Published private(set) var actAddKLHistory: [KLReading] = []
    @Published private(set) var lexicons: [SteeringLexicon] = []
    @Published private(set) var modelProgress = 0.0
    @Published private(set) var status = "Ready to load the on-device model"
    @Published private(set) var errorMessage: String?
    @Published private(set) var droppedTokenStrings: [String] = []
    @Published private(set) var isGenerating = false

    private let service = MLXGenerationService()
    private let stopFlag = StopFlag()
    private var topicScorer: CoreMLTopicScorer?
    private var generationTask: Task<Void, Never>?
    private var activeRun: RunConfiguration?
    private var didStartAutorun = false
    private var stopAfterTokenCount: Int?

    init() {
        if let value = ProcessInfo.processInfo.environment["STEERDEMO_BIAS_STRENGTH"],
           let parsed = Double(value)
        {
            strength = min(20, max(0, parsed))
        }
        if let value = ProcessInfo.processInfo.environment["STEERDEMO_ACTADD_COEFFICIENT"],
           let parsed = Double(value)
        {
            actAddCoefficient = min(40, max(0, parsed))
        }
        if let value = ProcessInfo.processInfo.environment["STEERDEMO_ACTADD_LAYER"],
           let parsed = Int(value)
        {
            actAddLayer = min(23, max(0, parsed))
        }
        if let value = ProcessInfo.processInfo.environment["STEERDEMO_PROMPT"], !value.isEmpty {
            prompt = value
        }
        if let value = ProcessInfo.processInfo.environment["STEERDEMO_MAX_TOKENS"],
           let parsed = Int(value)
        {
            maxTokens = min(128, max(16, parsed))
        }
        if let value = ProcessInfo.processInfo.environment["STEERDEMO_KL_BUDGET"],
           let parsed = Double(value)
        {
            klBudget = min(20, max(0.1, parsed))
        }
        if let value = ProcessInfo.processInfo.environment["STEERDEMO_STOP_AFTER_TOKENS"],
           let parsed = Int(value), parsed > 0
        {
            stopAfterTokenCount = parsed
        }
        do {
            guard let url = Bundle.main.url(forResource: "lexicons", withExtension: "json") else {
                throw DemoError.missingResource("lexicons.json")
            }
            lexicons = try LexiconBias.load(from: url)
            if let requested = ProcessInfo.processInfo.environment["STEERDEMO_LEXICON"],
               lexicons.contains(where: { $0.id == requested })
            {
                selectedLexiconID = requested
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    var selectedLexicon: SteeringLexicon? {
        lexicons.first { $0.id == selectedLexiconID }
    }

    func startAutorunIfRequested() {
        let environment = ProcessInfo.processInfo.environment
        let requested = CommandLine.arguments.contains("--autorun")
            || environment["STEERDEMO_AUTORUN"] == "1"
        guard !didStartAutorun, requested else { return }
        didStartAutorun = true
        generate()
    }

    func generate() {
        guard !isGenerating, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let lexicon = selectedLexicon
        else { return }

        let configuration = RunConfiguration(
            prompt: prompt,
            lexicon: lexicon,
            strength: strength,
            actAddCoefficient: actAddCoefficient,
            actAddLayer: actAddLayer,
            maxTokens: maxTokens,
            klBudget: klBudget
        )
        activeRun = configuration

        baseline = .empty
        steered = .empty
        actAdd = .empty
        klHistory = []
        actAddKLHistory = []
        droppedTokenStrings = []
        errorMessage = nil
        isGenerating = true
        stopFlag.reset()

        generationTask = Task {
            writeFrameIfRequested(name: "00-start")
            do {
                status = "Loading Qwen2.5 locally via MLX"
                try await loadModel()
                try Task.checkCancellation()
                status = "Warming the matched prompt outside the timing window"
                try await service.warmUp(prompt: configuration.prompt)
                try Task.checkCancellation()
                if topicScorer == nil {
                    status = "Loading the Core ML topic judge"
                    topicScorer = try await CoreMLTopicScorer.load()
                }
                try Task.checkCancellation()

                status = "Generating baseline (fixed seed, temperature 0.7)"
                baseline.isActive = true
                let baselineSummary = try await run(
                    pane: .baseline,
                    configuration: configuration
                )
                baseline.isActive = false
                try apply(summary: baselineSummary, lexicon: configuration.lexicon)

                if !stopFlag.isStopped(), !Task.isCancelled {
                    status = "Generating logit-bias pass from the same prompt"
                    steered.isActive = true
                    let steeredSummary = try await run(
                        pane: .steered,
                        configuration: configuration
                    )
                    steered.isActive = false
                    droppedTokenStrings = steeredSummary.droppedTokenStrings
                    klHistory = steeredSummary.klHistory
                    try apply(summary: steeredSummary, lexicon: configuration.lexicon)
                }
                if !stopFlag.isStopped(), !Task.isCancelled {
                    status = "Generating activation-addition pass at matched KL"
                    actAdd.isActive = true
                    let actAddSummary = try await run(
                        pane: .actAdd,
                        configuration: configuration
                    )
                    actAdd.isActive = false
                    actAddKLHistory = actAddSummary.klHistory
                    try apply(summary: actAddSummary, lexicon: configuration.lexicon)
                }
                status = stopFlag.isStopped() ? "Stopped" : "Complete — all inference stayed on-device"
            } catch is CancellationError {
                status = "Stopped"
            } catch {
                if Task.isCancelled || (error as? URLError)?.code == .cancelled {
                    status = "Stopped"
                } else {
                    errorMessage = error.localizedDescription
                    status = "Generation failed"
                }
            }
            baseline.isActive = false
            steered.isActive = false
            actAdd.isActive = false
            isGenerating = false
            writeFrameIfRequested(name: "99-final")
            writeSnapshotIfRequested()
            writeRunReportIfRequested()
            activeRun = nil
            generationTask = nil
        }
    }

    func stop() {
        stopFlag.stop()
        generationTask?.cancel()
        status = modelProgress > 0 && modelProgress < 1
            ? "Cancelling model download"
            : "Stopping after the current token"
    }

    func dismissError() {
        errorMessage = nil
    }

    private func loadModel() async throws {
        let (stream, continuation) = AsyncStream.makeStream(
            of: Double.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        let producer = Task {
            do {
                try await service.loadModel { fraction in
                    continuation.yield(fraction)
                }
                continuation.finish()
            } catch {
                continuation.finish()
                throw error
            }
        }
        try await withTaskCancellationHandler {
            for await fraction in stream {
                try Task.checkCancellation()
                modelProgress = fraction
            }
            try await producer.value
        } onCancel: {
            producer.cancel()
            continuation.finish()
        }
    }

    private func run(
        pane: GenerationPane,
        configuration: RunConfiguration
    ) async throws -> GenerationSummary {
        let (stream, continuation) = AsyncStream.makeStream(of: GenerationUpdate.self)
        let producer = Task {
            do {
                let summary = try await service.generate(
                    pane: pane,
                    prompt: configuration.prompt,
                    lexicon: configuration.lexicon,
                    strength: configuration.strength,
                    actAddCoefficient: configuration.actAddCoefficient,
                    actAddLayer: configuration.actAddLayer,
                    maxTokens: configuration.maxTokens,
                    klBudget: configuration.klBudget,
                    stopFlag: stopFlag
                ) { update in
                    continuation.yield(update)
                }
                continuation.finish()
                return summary
            } catch {
                continuation.finish()
                throw error
            }
        }
        return try await withTaskCancellationHandler {
            for await update in stream {
                try Task.checkCancellation()
                apply(update: update, lexicon: configuration.lexicon)
            }
            return try await producer.value
        } onCancel: {
            stopFlag.stop()
            producer.cancel()
            continuation.finish()
        }
    }

    private func apply(update: GenerationUpdate, lexicon: SteeringLexicon) {
        var state: PaneState
        switch update.pane {
        case .baseline: state = baseline
        case .steered: state = steered
        case .actAdd: state = actAdd
        }
        state.text = update.text
        state.tokenCount = update.tokenCount
        state.tokensPerSecond = update.tokensPerSecond
        state.residentMemoryBytes = update.residentMemoryBytes
        if update.tokenCount % 8 == 0, let topicScorer {
            do {
                state.topicScore = try topicScorer.score(text: update.text, lexicon: lexicon)
            } catch {
                errorMessage = "Core ML topic judge failed: \(error.localizedDescription)"
                stop()
            }
        }
        switch update.pane {
        case .baseline:
            baseline = state
        case .steered:
            steered = state
            if let reading = update.klReading { klHistory.append(reading) }
        case .actAdd:
            actAdd = state
            if let reading = update.klReading,
               reading.step > (actAddKLHistory.last?.step ?? 0)
            {
                actAddKLHistory.append(reading)
            }
        }
        if update.tokenCount.isMultiple(of: 16) {
            let prefix: String
            switch update.pane {
            case .baseline: prefix = "10-baseline"
            case .steered: prefix = "20-logit-bias"
            case .actAdd: prefix = "30-actadd"
            }
            writeFrameIfRequested(
                name: "\(prefix)-\(String(format: "%03d", update.tokenCount))"
            )
        }
        if let stopAfterTokenCount, update.tokenCount >= stopAfterTokenCount {
            stop()
        }
    }

    private func apply(summary: GenerationSummary, lexicon: SteeringLexicon) throws {
        var state: PaneState
        switch summary.pane {
        case .baseline: state = baseline
        case .steered: state = steered
        case .actAdd: state = actAdd
        }
        state.text = summary.text
        state.tokenCount = summary.tokenCount
        state.tokensPerSecond = Double(summary.tokenCount) / summary.seconds
        state.residentMemoryBytes = summary.residentMemoryBytes
        guard let topicScorer else {
            throw DemoError.missingResource("Core ML topic judge")
        }
        state.topicScore = try topicScorer.score(text: summary.text, lexicon: lexicon)
        switch summary.pane {
        case .baseline:
            baseline = state
        case .steered:
            steered = state
        case .actAdd:
            actAdd = state
        }
    }

    private func writeRunReportIfRequested() {
        let argumentPath: String? = {
            guard let flag = CommandLine.arguments.firstIndex(of: "--report"),
                  CommandLine.arguments.indices.contains(flag + 1)
            else { return nil }
            return CommandLine.arguments[flag + 1]
        }()
        guard let reportPath = ProcessInfo.processInfo.environment["STEERDEMO_REPORT_PATH"]
            ?? argumentPath
        else { return }
        guard let activeRun else { return }
        let report = RunReport(
            modelID: MLXGenerationService.modelID,
            modelRevision: MLXGenerationService.modelRevision,
            buildConfiguration: MLXGenerationService.buildConfiguration,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            status: status,
            error: errorMessage,
            prompt: activeRun.prompt,
            lexicon: activeRun.lexicon.id,
            biasStrength: activeRun.strength,
            actAddCoefficient: activeRun.actAddCoefficient,
            actAddLayer: activeRun.actAddLayer,
            maxTokens: activeRun.maxTokens,
            seed: MLXGenerationService.samplingSeed,
            temperature: Double(MLXGenerationService.samplingTemperature),
            klBudget: activeRun.klBudget,
            cumulativeKL: klHistory.last?.cumulative ?? 0,
            klHistory: klHistory,
            actAddCumulativeKL: actAddKLHistory.last?.cumulative ?? 0,
            actAddKLHistory: actAddKLHistory,
            baseline: .init(
                text: baseline.text,
                tokenCount: baseline.tokenCount,
                tokensPerSecond: baseline.tokensPerSecond,
                residentMemoryBytes: baseline.residentMemoryBytes,
                topicScore: baseline.topicScore
            ),
            steered: .init(
                text: steered.text,
                tokenCount: steered.tokenCount,
                tokensPerSecond: steered.tokensPerSecond,
                residentMemoryBytes: steered.residentMemoryBytes,
                topicScore: steered.topicScore
            ),
            actAdd: .init(
                text: actAdd.text,
                tokenCount: actAdd.tokenCount,
                tokensPerSecond: actAdd.tokensPerSecond,
                residentMemoryBytes: actAdd.residentMemoryBytes,
                topicScore: actAdd.topicScore
            ),
            droppedTokenStrings: droppedTokenStrings
        )
        do {
            let data = try JSONEncoder.pretty.encode(report)
            try data.write(to: URL(fileURLWithPath: reportPath))
        } catch {
            errorMessage = "Could not write smoke report: \(error.localizedDescription)"
        }
    }

    private func writeSnapshotIfRequested() {
        guard let path = ProcessInfo.processInfo.environment["STEERDEMO_SNAPSHOT_PATH"] else {
            return
        }
        do {
            try SnapshotExporter.write(model: self, to: URL(fileURLWithPath: path))
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func writeFrameIfRequested(name: String) {
        guard let directory = ProcessInfo.processInfo.environment["STEERDEMO_FRAMES_DIR"] else {
            return
        }
        let url = URL(fileURLWithPath: directory)
            .appendingPathComponent("\(name).png")
        try? SnapshotExporter.write(model: self, to: url)
    }
}

private extension JSONEncoder {
    static var pretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}

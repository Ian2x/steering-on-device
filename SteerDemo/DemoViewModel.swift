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
        let residualDirectionMode: ResidualDirectionMode
        let staticBiasMode: Bool
        let klDiscipline: KLDiscipline
        let maxTokens: Int
        let klBudget: Double

        /// A cap is active only when the discipline asks for one *and* this is not a frozen Phase 6
        /// harness run. The harness protocol says no cap is active; no UI state may change that.
        var capActive: Bool { !staticBiasMode && klDiscipline.capsInterventions }
    }

    private struct RunReport: Encodable {
        struct Pane: Encodable {
            let text: String
            let tokenIDs: [Int]
            let tokenCount: Int
            let tokensPerSecond: Double
            let residentMemoryBytes: UInt64
            let topicScore: Double?
            let baseModelNLL: Double?
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
        let actAddAppliedCoefficient: Double?
        let actAddLayer: Int
        let actAddDirectionMode: String
        let actAddKLCapEnabled: Bool
        let actAddDirectionDiagnostics: ResidualDirectionDiagnostics?
        let controlOnly: Bool
        let stage3RunMode: String?
        let klDiscipline: String
        let staticBiasKLCapEnabled: Bool
        let liveCalibrationTargetKL: Double?
        let liveCalibrationLogit: CalibrationOutcome?
        let liveCalibrationActAdd: CalibrationOutcome?
        let teacherForcedTargetKL: Double?
        let teacherForcedContinuationTokenIDs: [Int]
        let teacherForcedLogit: TeacherForcedKLResult?
        let teacherForcedActAdd: TeacherForcedKLResult?
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

    private struct PreservedRunReport: Decodable {
        struct Pane: Decodable {
            let text: String
            let tokenCount: Int
            let tokensPerSecond: Double
            let residentMemoryBytes: UInt64
            let topicScore: Double?
            let baseModelNLL: Double?
        }

        let status: String
        let stage3RunMode: String?
        let prompt: String
        let lexicon: String
        let biasStrength: Double
        let actAddCoefficient: Double
        let actAddLayer: Int
        let maxTokens: Int
        let klBudget: Double
        let klHistory: [KLReading]
        let actAddKLHistory: [KLReading]
        let baseline: Pane
        let steered: Pane
        let actAdd: Pane
        let droppedTokenStrings: [String]
    }

    /// The one layer/coefficient cell that cleared every frozen gate in the 180-packet blocking
    /// control (`docs/phase6/blocking-control`). The app ships defaulted here. It previously
    /// shipped at coefficient 12, which is a *tested and failed* cell — the worst semantic
    /// base-model NLL of all fifteen, 3.6147 against a 0.9831 baseline. Anyone who opened the app
    /// and pressed Generate saw gate-failing output. Do not move these defaults off the validated
    /// cell without moving the blocking control with them.
    static let validatedLayer = 10
    static let validatedCoefficient = 4.0

    /// The upstream audit's teacher-forced KL target, in nats per generation step. Every scalar in
    /// `docs/phase6/teacher-forced-comparison` was calibrated to this number.
    static let auditTargetKL = 0.435_238_018_732_847_95
    /// The frozen protocol's brackets, iteration count, and tolerance, reused by live calibration.
    static let logitScalarBracket = 20.0
    static let residualScalarBracket = 40.0
    static let calibrationIterations = 18
    static let calibrationTolerance = 0.002

    @Published var prompt = "Describe a quiet morning routine in two short paragraphs."
    @Published var strength = 14.0
    @Published var actAddCoefficient = DemoViewModel.validatedCoefficient
    @Published var actAddLayer = DemoViewModel.validatedLayer
    @Published var selectedLexiconID = "wedding"
    @Published var maxTokens = 96
    @Published var klBudget = 8.0
    /// Defaults to the audit's discipline. The greedy cap is retained as a demonstration of the
    /// confound recorded in `docs/phase6/on-device-rho`, not as a working mode.
    @Published var klDiscipline = KLDiscipline.matchedPerStep
    @Published private(set) var isCalibrating = false
    @Published private(set) var calibrationProgress: String?
    @Published private(set) var calibrationLogit: CalibrationOutcome?
    @Published private(set) var calibrationActAdd: CalibrationOutcome?
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
    private var residualDirectionMode: ResidualDirectionMode = .semantic
    private var actAddAppliedCoefficient: Double?
    private var actAddDirectionDiagnostics: ResidualDirectionDiagnostics?
    private var controlOnly = false
    private var stage3RunMode: Stage3RunMode?
    private var staticBiasMode = false
    private var teacherForcedContinuationTokenIDs: [Int] = []
    private var teacherForcedLogit: TeacherForcedKLResult?
    private var teacherForcedActAdd: TeacherForcedKLResult?
    private let teacherForcedTargetKL = DemoViewModel.auditTargetKL
    private var calibrationTask: Task<Void, Never>?

    var usesStaticBias: Bool { staticBiasMode }

    /// True when a cumulative budget is attenuating the live arms. False for every frozen Phase 6
    /// replay, whose protocol declares no cap active.
    var capActive: Bool { !staticBiasMode && klDiscipline.capsInterventions }

    /// Mean KL per *returned* step, the quantity the audit matched on. Nil until an arm has run.
    ///
    /// Read this as a diagnostic, not as a match. Each arm's mean here is taken over its own
    /// free-running output, so the two arms are averaging over different token sequences. Matching
    /// these two numbers would not make the arms comparable; only the teacher-forced calibration,
    /// which scores both on one shared continuation, does that.
    var logitMeanNatsPerStep: Double? { Self.meanPerStep(klHistory) }
    var actAddMeanNatsPerStep: Double? { Self.meanPerStep(actAddKLHistory) }

    /// Everything a calibration result is only true of.
    ///
    /// Both sliders quantize on drag — bias strength to `0.5`, coefficient to `1` — so the first
    /// nudge after calibrating snaps a scalar off its calibrated value and silently voids the
    /// match. On the residual arm that rounds a calibrated `7.6` straight onto coefficient `8`,
    /// a cell that failed the matched-random floor at all three tested depths. Exact equality is
    /// the point: any change at all, in either direction, means these numbers no longer describe
    /// the sliders.
    private struct CalibrationContext: Equatable {
        let prompt: String
        let lexiconID: String
        let actAddLayer: Int
        let logitScalar: Double
        let actAddScalar: Double
    }

    private var calibrationContext: CalibrationContext?

    private var liveCalibrationContext: CalibrationContext {
        CalibrationContext(
            prompt: prompt,
            lexiconID: selectedLexiconID,
            actAddLayer: actAddLayer,
            logitScalar: strength,
            actAddScalar: actAddCoefficient
        )
    }

    /// False once any input the calibration was computed under has moved. The UI says so and the
    /// run report withholds the outcomes, because a report that carried them would be claiming a
    /// match the run did not run under.
    var calibrationIsCurrent: Bool {
        calibrationContext != nil && calibrationContext == liveCalibrationContext
    }

    private static func meanPerStep(_ history: [KLReading]) -> Double? {
        guard !history.isEmpty else { return nil }
        return history.reduce(0.0) { $0 + $1.perStep } / Double(history.count)
    }

    /// Steps on which the intervention had effectively been switched off because the cumulative
    /// budget was already spent. Only meaningful under the greedy cap; it is the collapse, counted.
    func exhaustedStepCount(_ history: [KLReading]) -> Int {
        history.filter { $0.perStep < 1e-6 }.count
    }

    /// The first step's share of everything the arm ever spent. On the eight committed
    /// `docs/phase6/on-device-rho` packets this ran from 97.3620711503% to 99.8667704858%.
    func firstStepShare(_ history: [KLReading]) -> Double? {
        guard let first = history.first, let total = history.last?.cumulative, total > 0 else {
            return nil
        }
        return first.perStep / total
    }

    /// True when the residual arm sits on the blocking control's selected cell. Off it, the app is
    /// running a configuration that either failed the frozen gates or was never tested at all.
    var isOnValidatedCell: Bool {
        actAddLayer == DemoViewModel.validatedLayer
            && actAddCoefficient == DemoViewModel.validatedCoefficient
    }

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
        if ProcessInfo.processInfo.environment["STEERDEMO_RESIDUAL_DIRECTION"]
            == ResidualDirectionMode.randomMatchedNorm.rawValue
        {
            residualDirectionMode = .randomMatchedNorm
        }
        controlOnly = ProcessInfo.processInfo.environment["STEERDEMO_CONTROL_ONLY"] == "1"
        if let value = ProcessInfo.processInfo.environment["STEERDEMO_STAGE3_MODE"] {
            stage3RunMode = Stage3RunMode(rawValue: value)
            staticBiasMode = stage3RunMode != nil
            controlOnly = stage3RunMode == .evaluateRandom
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
        if let value = ProcessInfo.processInfo.environment["STEERDEMO_KL_DISCIPLINE"],
           let parsed = KLDiscipline(rawValue: value)
        {
            klDiscipline = parsed
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
        if let path = ProcessInfo.processInfo.environment["STEERDEMO_REPLAY_REPORT_PATH"] {
            do {
                try loadPreservedReport(at: path)
            } catch {
                errorMessage = "Could not load preserved report: \(error.localizedDescription)"
            }
        }
    }

    var selectedLexicon: SteeringLexicon? {
        lexicons.first { $0.id == selectedLexiconID }
    }

    func startAutorunIfRequested() {
        let environment = ProcessInfo.processInfo.environment
        if environment["STEERDEMO_REPLAY_REPORT_PATH"] != nil {
            guard !didStartAutorun else { return }
            didStartAutorun = true
            writeFrameIfRequested(name: "99-final")
            writeSnapshotIfRequested()
            return
        }
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
            residualDirectionMode: residualDirectionMode,
            staticBiasMode: staticBiasMode,
            klDiscipline: klDiscipline,
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
        actAddAppliedCoefficient = nil
        actAddDirectionDiagnostics = nil
        teacherForcedContinuationTokenIDs = []
        teacherForcedLogit = nil
        teacherForcedActAdd = nil
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

                var shouldGenerate = true
                if let stage3RunMode {
                    status = "Building the fixed 64-step baseline continuation"
                    teacherForcedContinuationTokenIDs = try await service.fixedBaselineContinuation(
                        prompt: configuration.prompt,
                        steps: 64
                    )
                    if stage3RunMode == .calibrateLogit || stage3RunMode == .evaluate {
                        status = "Measuring teacher-forced static-bias KL"
                        teacherForcedLogit = try await service.teacherForcedKL(
                            method: .steered,
                            prompt: configuration.prompt,
                            continuationTokens: teacherForcedContinuationTokenIDs,
                            lexicon: configuration.lexicon,
                            strength: configuration.strength,
                            actAddCoefficient: configuration.actAddCoefficient,
                            actAddLayer: configuration.actAddLayer,
                            residualDirectionMode: configuration.residualDirectionMode
                        )
                    }
                    if stage3RunMode == .calibrateActAdd
                        || stage3RunMode == .evaluate
                        || stage3RunMode == .evaluateRandom
                    {
                        status = "Measuring teacher-forced residual-edit KL"
                        teacherForcedActAdd = try await service.teacherForcedKL(
                            method: .actAdd,
                            prompt: configuration.prompt,
                            continuationTokens: teacherForcedContinuationTokenIDs,
                            lexicon: configuration.lexicon,
                            strength: configuration.strength,
                            actAddCoefficient: configuration.actAddCoefficient,
                            actAddLayer: configuration.actAddLayer,
                            residualDirectionMode: configuration.residualDirectionMode
                        )
                        actAddAppliedCoefficient = teacherForcedActAdd?.appliedScalar
                        actAddDirectionDiagnostics = teacherForcedActAdd?.directionDiagnostics
                    }
                    shouldGenerate = stage3RunMode == .evaluate || stage3RunMode == .evaluateRandom
                }
                try Task.checkCancellation()

                if shouldGenerate {
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

                    if !controlOnly, !stopFlag.isStopped(), !Task.isCancelled {
                        status = configuration.staticBiasMode
                            ? "Generating sustained calibrated static-bias pass"
                            : configuration.capActive
                                ? "Generating logit-bias pass under a greedy cumulative cap"
                                : "Generating logit-bias pass at a fixed scalar, uncapped"
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
                        status = configuration.capActive
                            ? "Generating persistent residual prompt edit under a greedy cumulative cap"
                            : "Generating persistent residual prompt edit (direct coefficient; KL cap off)"
                        actAdd.isActive = true
                        let actAddSummary = try await run(
                            pane: .actAdd,
                            configuration: configuration
                        )
                        actAdd.isActive = false
                        actAddKLHistory = actAddSummary.klHistory
                        try apply(summary: actAddSummary, lexicon: configuration.lexicon)
                    }
                }
                status = stopFlag.isStopped()
                    ? "Stopped"
                    : shouldGenerate
                        ? "Complete — all inference stayed on-device"
                        : "Complete — teacher-forced calibration packet"
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

    /// Calibrates both live arms to the audit's teacher-forced KL target on the user's own prompt,
    /// then writes the selected scalars into the sliders.
    ///
    /// This is the honest form of matching, and the reason it needs its own button rather than
    /// happening during generation: the two arms are scored on one shared, base-generated
    /// continuation. Their means are therefore comparable. Averaging each arm's KL over its own
    /// free-running output — which is what a live "match the running means" control would do —
    /// compares two averages taken over different token sequences and matches nothing.
    ///
    /// It is a single-prompt analogue of the frozen calibration, not the frozen calibration. The
    /// four committed scalars averaged each candidate over four fixed prompts.
    func calibrateToAuditTarget() {
        guard !isGenerating, !isCalibrating, !staticBiasMode,
              let lexicon = selectedLexicon,
              !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }

        isCalibrating = true
        errorMessage = nil
        calibrationLogit = nil
        calibrationActAdd = nil
        calibrationContext = nil
        let capturedPrompt = prompt
        let capturedLayer = actAddLayer
        let capturedMode = residualDirectionMode

        calibrationTask = Task {
            do {
                status = "Loading Qwen2.5 locally via MLX"
                try await loadModel()
                try Task.checkCancellation()

                calibrationProgress = "Building the shared 64-step base continuation"
                let continuation = try await service.fixedBaselineContinuation(
                    prompt: capturedPrompt,
                    steps: 64
                )
                try Task.checkCancellation()

                // Both arms are calibrated before either scalar is written. A half-applied
                // calibration would leave one arm on the audit's target and the other wherever the
                // slider happened to be — the same asymmetry this whole change removed — and it
                // would make `DemoError`'s "stopped without changing any scalar" a lie.
                var outcomes: [GenerationPane: CalibrationOutcome] = [:]
                for arm in [GenerationPane.steered, .actAdd] {
                    let name = arm == .steered ? "static bias" : "residual edit"
                    let bracket = arm == .steered
                        ? Self.logitScalarBracket
                        : Self.residualScalarBracket
                    let outcome = try await service.calibrateScalar(
                        arm: arm,
                        prompt: capturedPrompt,
                        continuationTokens: continuation,
                        lexicon: lexicon,
                        actAddLayer: capturedLayer,
                        residualDirectionMode: capturedMode,
                        target: Self.auditTargetKL,
                        tolerance: Self.calibrationTolerance,
                        upperBracket: bracket,
                        iterations: Self.calibrationIterations
                    ) { [weak self] done, total in
                        Task { @MainActor in
                            // The run may have finished or been cancelled between this candidate
                            // and the hop to the main actor; without the guard a late update
                            // reinstates "Calibrating..." over a completed run.
                            guard let self, self.isCalibrating else { return }
                            self.calibrationProgress =
                                "Calibrating \(name): candidate \(done) of \(total)"
                        }
                    }
                    try Task.checkCancellation()
                    outcomes[arm] = outcome
                }

                if let logit = outcomes[.steered], let residual = outcomes[.actAdd] {
                    strength = logit.selectedScalar
                    actAddCoefficient = residual.selectedScalar
                    calibrationLogit = logit
                    calibrationActAdd = residual
                    // Recorded after the writes so it matches the sliders exactly.
                    calibrationContext = liveCalibrationContext
                }
                calibrationProgress = nil
                status = "Both arms calibrated to \(Self.auditTargetKL.formatted(.number.precision(.fractionLength(5)))) nats/step on this prompt"
            } catch is CancellationError {
                calibrationProgress = nil
                status = "Calibration stopped"
            } catch {
                calibrationProgress = nil
                errorMessage = error.localizedDescription
                status = "Calibration failed"
            }
            isCalibrating = false
            calibrationTask = nil
        }
    }

    func stop() {
        stopFlag.stop()
        generationTask?.cancel()
        calibrationTask?.cancel()
        status = modelProgress > 0 && modelProgress < 1
            ? "Cancelling model download"
            : "Stopping after the current token"
    }

    func dismissError() {
        errorMessage = nil
    }

    private func loadPreservedReport(at path: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let report = try JSONDecoder().decode(PreservedRunReport.self, from: data)
        staticBiasMode = report.stage3RunMode != nil
        prompt = report.prompt
        selectedLexiconID = report.lexicon
        strength = report.biasStrength
        actAddCoefficient = report.actAddCoefficient
        actAddLayer = report.actAddLayer
        maxTokens = report.maxTokens
        klBudget = report.klBudget
        klHistory = report.klHistory
        actAddKLHistory = report.actAddKLHistory
        droppedTokenStrings = report.droppedTokenStrings
        status = report.stage3RunMode == Stage3RunMode.evaluate.rawValue
            ? "Preserved teacher-forced packet — comparison withheld by NLL gate; no inference rerun"
            : "Preserved invalidated Phase 6 packet — no inference rerun"
        baseline = PaneState(
            text: report.baseline.text,
            tokenIDs: [],
            tokenCount: report.baseline.tokenCount,
            tokensPerSecond: report.baseline.tokensPerSecond,
            residentMemoryBytes: report.baseline.residentMemoryBytes,
            topicScore: report.baseline.topicScore,
            baseModelNLL: report.baseline.baseModelNLL,
            isActive: false
        )
        steered = PaneState(
            text: report.steered.text,
            tokenIDs: [],
            tokenCount: report.steered.tokenCount,
            tokensPerSecond: report.steered.tokensPerSecond,
            residentMemoryBytes: report.steered.residentMemoryBytes,
            topicScore: report.steered.topicScore,
            baseModelNLL: report.steered.baseModelNLL,
            isActive: false
        )
        actAdd = PaneState(
            text: report.actAdd.text,
            tokenIDs: [],
            tokenCount: report.actAdd.tokenCount,
            tokensPerSecond: report.actAdd.tokensPerSecond,
            residentMemoryBytes: report.actAdd.residentMemoryBytes,
            topicScore: report.actAdd.topicScore,
            baseModelNLL: report.actAdd.baseModelNLL,
            isActive: false
        )
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
                    residualDirectionMode: configuration.residualDirectionMode,
                    staticBiasMode: configuration.staticBiasMode,
                    discipline: configuration.klDiscipline,
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
        state.tokenIDs = summary.tokenIDs
        state.tokenCount = summary.tokenCount
        state.tokensPerSecond = Double(summary.tokenCount) / summary.seconds
        state.residentMemoryBytes = summary.residentMemoryBytes
        state.baseModelNLL = summary.baseModelNLL
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
            actAddAppliedCoefficient = summary.appliedCoefficient
            actAddDirectionDiagnostics = summary.directionDiagnostics
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
            actAddAppliedCoefficient: actAddAppliedCoefficient,
            actAddLayer: activeRun.actAddLayer,
            actAddDirectionMode: activeRun.residualDirectionMode.rawValue,
            // Both flags now report the same fact — whether a cumulative budget attenuated that
            // arm — instead of one hardcoded `false` and one inferred from the harness mode. Every
            // committed protocol path runs with the cap off, so both still emit `false` there,
            // which is what `run_teacher_forced_comparison.py`, `run_blocking_control.sh`,
            // `summarize_blocking_control.py`, and `verify_readme_claims.py` assert.
            actAddKLCapEnabled: activeRun.capActive,
            actAddDirectionDiagnostics: actAddDirectionDiagnostics,
            controlOnly: controlOnly,
            stage3RunMode: stage3RunMode?.rawValue,
            klDiscipline: activeRun.klDiscipline.rawValue,
            staticBiasKLCapEnabled: activeRun.capActive,
            // Withheld unless the sliders still hold exactly what calibration put there. A stale
            // outcome in a packet would assert a match that this run was not generated under.
            liveCalibrationTargetKL: calibrationIsCurrent ? Self.auditTargetKL : nil,
            liveCalibrationLogit: calibrationIsCurrent ? calibrationLogit : nil,
            liveCalibrationActAdd: calibrationIsCurrent ? calibrationActAdd : nil,
            teacherForcedTargetKL: stage3RunMode == nil ? nil : teacherForcedTargetKL,
            teacherForcedContinuationTokenIDs: teacherForcedContinuationTokenIDs,
            teacherForcedLogit: teacherForcedLogit,
            teacherForcedActAdd: teacherForcedActAdd,
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
                tokenIDs: baseline.tokenIDs,
                tokenCount: baseline.tokenCount,
                tokensPerSecond: baseline.tokensPerSecond,
                residentMemoryBytes: baseline.residentMemoryBytes,
                topicScore: baseline.topicScore,
                baseModelNLL: baseline.baseModelNLL
            ),
            steered: .init(
                text: steered.text,
                tokenIDs: steered.tokenIDs,
                tokenCount: steered.tokenCount,
                tokensPerSecond: steered.tokensPerSecond,
                residentMemoryBytes: steered.residentMemoryBytes,
                topicScore: steered.topicScore,
                baseModelNLL: steered.baseModelNLL
            ),
            actAdd: .init(
                text: actAdd.text,
                tokenIDs: actAdd.tokenIDs,
                tokenCount: actAdd.tokenCount,
                tokensPerSecond: actAdd.tokensPerSecond,
                residentMemoryBytes: actAdd.residentMemoryBytes,
                topicScore: actAdd.topicScore,
                baseModelNLL: actAdd.baseModelNLL
            ),
            droppedTokenStrings: droppedTokenStrings
        )
        do {
            let data = try JSONEncoder.pretty.encode(report)
            try data.write(to: URL(fileURLWithPath: reportPath), options: .atomic)
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

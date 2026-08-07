import Charts
import SteeringKit
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: DemoViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                promptCard
                panes
                paneLegend
                meters
                explanation
                footer
            }
            .padding(24)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .alert("SteerDemo could not continue", isPresented: errorBinding) {
            Button("Dismiss", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "Unknown error")
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("On-device steering, made visible")
                    .font(.system(size: 28, weight: .semibold, design: .rounded))
                // "a direct residual control" used to name the ActAdd arm here, which reads as
                // though ActAdd were the control condition. The control is the logit bias.
                Text(headerSubtitle)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            Label("Prompts stay on this Mac", systemImage: "lock.shield.fill")
                .font(.callout.weight(.medium))
                .foregroundStyle(.green)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(.green.opacity(0.12), in: Capsule())
        }
    }

    private var headerSubtitle: String {
        if model.usesStaticBias {
            return "One prompt. Three fixed-seed passes. A sustained static-bias control against a direct residual ActAdd edit."
        }
        return model.capActive
            ? "One prompt. Three fixed-seed passes. Both controllers under one greedy cumulative KL cap — the confound, shown deliberately."
            : "One prompt. Three fixed-seed passes. Both controllers at a fixed scalar, uncapped, calibrated to the audit's per-step KL target."
    }

    private var promptCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompt").font(.headline)
            TextEditor(text: $model.prompt)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(10)
                .frame(minHeight: 74, maxHeight: 100)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                .disabled(model.isGenerating)
                .accessibilityLabel("Prompt")

            HStack(spacing: 18) {
                Picker("Topic", selection: $model.selectedLexiconID) {
                    ForEach(model.lexicons) { lexicon in
                        Text(lexicon.name).tag(lexicon.id)
                    }
                }
                .frame(width: 190)
                .disabled(model.isGenerating)
                .accessibilityLabel("Topic lexicon")

                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text("Bias strength")
                        Spacer()
                        Text(model.strength, format: .number.precision(.fractionLength(1)))
                            .monospacedDigit()
                    }
                    Slider(value: $model.strength, in: 0 ... 20, step: 0.5)
                        .accessibilityLabel("Bias strength")
                        .accessibilityValue(model.strength.formatted(.number.precision(.fractionLength(1))))
                }
                .frame(maxWidth: 320)
                .disabled(model.isGenerating)

                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(model.capActive ? "Cumulative KL cap" : "Per-step KL target")
                        Spacer()
                        Text(model.capActive
                            ? "\(model.klBudget, specifier: "%.2f") nats total"
                            : "\(DemoViewModel.auditTargetKL, specifier: "%.5f") nats/step")
                            .monospacedDigit()
                    }
                    if model.capActive {
                        Slider(value: $model.klBudget, in: 0.1 ... 20, step: 0.1)
                            .accessibilityLabel("Cumulative KL cap")
                            .accessibilityValue("\(model.klBudget, specifier: "%.2f") nats")
                    } else {
                        Text(model.usesStaticBias
                            ? "Scalar calibrated on a shared continuation"
                            : "Fixed scalar, applied every step, never attenuated")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .frame(maxWidth: 300)
                .disabled(model.isGenerating)

                Spacer()
                if model.isGenerating || model.isCalibrating {
                    Button("Stop", role: .destructive, action: model.stop)
                        .buttonStyle(.bordered)
                } else {
                    Button(action: model.generate) {
                        Label("Generate", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command])
                }
            }

            if !model.usesStaticBias { disciplineRow }

            HStack(spacing: 18) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text("Residual-edit coefficient")
                        Spacer()
                        Text(model.actAddCoefficient, format: .number.precision(.fractionLength(1)))
                            .monospacedDigit()
                            .foregroundStyle(model.actAddCoefficient == DemoViewModel.validatedCoefficient ? Color.primary : Color.orange)
                    }
                    Slider(value: $model.actAddCoefficient, in: 0 ... 40, step: 1)
                    .accessibilityLabel("Residual-edit coefficient")
                        .accessibilityValue(model.actAddCoefficient.formatted(.number.precision(.fractionLength(1))))
                    Text("validated: \(DemoViewModel.validatedCoefficient, specifier: "%.0f")")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: 360)
                .disabled(model.isGenerating)

                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text("Residual layer")
                        Spacer()
                        Text("after block \(model.actAddLayer)")
                            .monospacedDigit()
                            .foregroundStyle(model.actAddLayer == DemoViewModel.validatedLayer ? Color.primary : Color.orange)
                    }
                    Slider(
                        value: Binding(
                            get: { Double(model.actAddLayer) },
                            set: { model.actAddLayer = Int($0.rounded()) }
                        ),
                        in: 0 ... 23,
                        step: 1
                    )
                    .accessibilityLabel("Residual-edit residual layer")
                    .accessibilityValue("after block \(model.actAddLayer)")
                    // Only 8, 10, and 12 were ever tested. The slider spans 0...23 because the
                    // model has 24 blocks, not because the other 21 blocks mean anything here.
                    Text("validated: block \(DemoViewModel.validatedLayer) — only blocks 8/10/12 were tested")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: 360)
                .disabled(model.isGenerating)

                validatedCellNote
            }

            HStack(spacing: 8) {
                if model.modelProgress > 0, model.modelProgress < 1 {
                    ProgressView(value: model.modelProgress)
                        .frame(width: 120)
                        .accessibilityLabel("Model download progress")
                        .accessibilityValue("\(Int(model.modelProgress * 100)) percent")
                }
                Text(model.status).font(.caption).foregroundStyle(.secondary)
            }
        }
        .cardStyle()
    }

    private var disciplineRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 18) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("KL discipline").font(.caption).foregroundStyle(.secondary)
                    Picker("KL discipline", selection: $model.klDiscipline) {
                        ForEach(KLDiscipline.allCases) { discipline in
                            Text(discipline.label).tag(discipline)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .accessibilityLabel("KL discipline")
                }
                .frame(maxWidth: 340)
                .disabled(model.isGenerating || model.isCalibrating)

                Button(action: model.calibrateToAuditTarget) {
                    Label("Calibrate to audit target", systemImage: "scope")
                }
                .buttonStyle(.bordered)
                .disabled(model.isGenerating || model.isCalibrating || model.capActive)
                .help("Bisects each arm's scalar until its mean teacher-forced KL per step matches the audit's target on one shared, base-generated continuation.")

                if model.isCalibrating {
                    ProgressView().controlSize(.small)
                }
                if let progress = model.calibrationProgress {
                    Text(progress)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            disciplineNote
            calibrationSummary
        }
    }

    private var disciplineNote: some View {
        Group {
            switch model.klDiscipline {
            case .matchedPerStep:
                Text("Each arm holds a **fixed scalar at every step** and nothing is attenuated. Calibrate picks each scalar by bisection so its *mean* teacher-forced KL per step lands on \(DemoViewModel.auditTargetKL, specifier: "%.5f") nats — the audit's target — measured on one shared base-generated continuation, so the two arms' means are taken over the same tokens. This is the discipline the frozen Phase 6 comparison used.")
                    .foregroundStyle(.secondary)
            case .greedyCumulativeCap:
                // This mode exists to be shown failing. Anyone who lands on it without reading
                // the meters would take the matched totals as a controlled comparison.
                Text("**Demonstration of a confound — not a working mode.** One cumulative budget, spent first-come-first-served with no lookahead and nothing held back for later steps. Both arms are attenuated until the budget runs out, then run unmodified. The totals match; the *schedule* does not, so any difference between the arms may come from when each intervened rather than from how. On the eight committed `on-device-rho` packets the first step took 97.4% to 99.9% of an eight-nat cap. Note also that a perfectly spread eight nats over 64 steps is 0.125 nats/step, about a third of the audit's target, so this cap is off on level as well as on schedule.")
                    .foregroundStyle(.orange)
            }
        }
        .font(.caption)
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var calibrationSummary: some View {
        Group {
            if model.calibrationLogit != nil || model.calibrationActAdd != nil {
                VStack(alignment: .leading, spacing: 3) {
                    if let outcome = model.calibrationLogit {
                        calibrationLine("Static bias", outcome)
                    }
                    if let outcome = model.calibrationActAdd {
                        calibrationLine("Residual edit", outcome)
                    }
                    if !model.calibrationIsCurrent {
                        // The sliders quantize on drag, so one nudge moves a scalar off its
                        // calibrated value. Saying nothing would leave the numbers above reading
                        // as a live match.
                        Text("**Superseded.** A slider, the prompt, the topic, or the layer changed since these were measured, so they no longer describe what will run. The run report withholds them. Calibrate again to restore the match.")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.top, 2)
                    }
                    // Without this the four-prompt scalars in results.md and the one-prompt
                    // scalars on screen look like the same quantity disagreeing.
                    Text("Calibrated on **this one prompt**. The frozen comparison averaged every candidate over four fixed prompts before comparing it to the target; its committed scalars are \(frozenScalarReference). A single-prompt run lands somewhere else, and that difference is the prompt-to-prompt spread, not an error.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 2)
                }
                .padding(.top, 2)
            }
        }
    }

    private func calibrationLine(_ name: String, _ outcome: CalibrationOutcome) -> some View {
        HStack(spacing: 6) {
            Text("\(name):").font(.caption)
            Text("scalar \(outcome.selectedScalar, specifier: "%.4f")")
                .font(.caption.monospacedDigit())
            Text("→").accessibilityHidden(true).font(.caption).foregroundStyle(.secondary)
            Text("\(outcome.achievedMeanNatsPerStep, specifier: "%.6f") nats/step")
                .font(.caption.monospacedDigit())
            Text("(\(outcome.signedError >= 0 ? "+" : "")\(outcome.signedError, specifier: "%.2e") vs target, \(outcome.iterations) iterations, \(outcome.continuationTokenCount) shared tokens)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
    }

    /// The frozen four-prompt scalars for whichever topic is selected, quoted from
    /// `docs/phase6/teacher-forced-comparison/results.md`.
    private var frozenScalarReference: String {
        switch model.selectedLexiconID {
        case "wedding": "11.1447906494 static and 6.7822265625 residual"
        case "ocean": "8.4908294678 static and 7.2952270508 residual"
        default: "recorded for wedding and ocean only"
        }
    }

    private var panes: some View {
        HStack(alignment: .top, spacing: 16) {
            GenerationPaneView(
                title: "Baseline",
                subtitle: "Unmodified logits",
                tint: .blue,
                state: model.baseline,
                baselineNLL: nil
            )
            // The arms are named for their ROLE first and their mechanism second. They used to read
            // "Logit bias" and "Residual edit (controlled)", which never said which arm was the
            // control and put the word "controlled" on the ActAdd pane, where it invites the exact
            // opposite reading. Keep the role prefixes.
            GenerationPaneView(
                title: "Control — logit bias",
                subtitle: model.usesStaticBias ? "Sustained static topic-token bias" : "Sparse topic-token bias",
                tint: .orange,
                state: model.steered,
                baselineNLL: model.baseline.baseModelNLL
            )
            GenerationPaneView(
                title: "ActAdd — residual edit",
                subtitle: "Persistent prompt edit after block \(model.actAddLayer)",
                tint: .purple,
                state: model.actAdd,
                baselineNLL: model.baseline.baseModelNLL
            )
        }
    }

    private var validatedCellNote: some View {
        // The grid was 3 layers x 5 coefficients and 13 of the 15 cells failed. Without this note
        // the sliders imply the whole space is usable, and a viewer who drags them is generating
        // gate-failing text with nothing on screen saying so.
        Group {
            if model.usesStaticBias {
                // Replay of a frozen teacher-forced packet. These sit at block 10 but at the
                // calibrated scalar rather than the blocking control's coefficient, so the
                // live-run warning below would call a committed measurement a demonstration.
                Text("Replaying a frozen teacher-forced packet: block \(model.actAddLayer) at the calibrated scalar that matched this arm to the audit's KL target, not the blocking control's selected coefficient. These are committed measurements. The comparison they belong to was withheld after the residual arm failed its frozen base-model NLL gate.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if model.isOnValidatedCell {
                Text("Residual arm is on the **validated cell** — block \(DemoViewModel.validatedLayer), coefficient \(DemoViewModel.validatedCoefficient, specifier: "%.0f"). It is the cell the predeclared tie-break selected from the 2/15 that cleared every frozen gate. A later teacher-forced KL comparison was still withheld, because the residual arm failed its frozen base-model NLL gate; no controller ratio is reported.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("**Off the validated cell.** Only block \(DemoViewModel.validatedLayer) / coefficient \(DemoViewModel.validatedCoefficient, specifier: "%.0f") cleared every frozen gate; 13 of the 15 tested cells failed and blocks other than 8/10/12 were never tested. Coefficient 8 failed the matched-random floor at all three tested depths, and coefficient 12 failed the base-model NLL gate at block 10 with a median semantic NLL of 3.6147 against a 0.9831 baseline. Output here is a live demonstration, not a measured result.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var paneLegend: some View {
        // Without this line the two capsules read as a scoreboard: a viewer sees a higher topic
        // score on one arm and calls it the winner. Topic score rises when an arm emits lexicon
        // words; NLL rises when the text stops being something the unmodified model would say.
        // An arm can win the first while destroying the second, and that is the usual outcome here.
        Text("**topic** is a Core ML cosine diagnostic, not a preference judgment — it rises when the arm emits lexicon words. **NLL** is the unmodified model's mean token surprise at the arm's own output: higher means less fluent. Read them together. The frozen teacher-forced gate allowed a *median* of +1.00 nat/token over baseline; a single run above that is flagged here but is not by itself a gate failure.")
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var meters: some View {
        HStack(alignment: .top, spacing: 16) {
            KLChartView(
                logitHistory: model.klHistory,
                actAddHistory: model.actAddKLHistory,
                budget: model.klBudget,
                capActive: model.capActive,
                target: DemoViewModel.auditTargetKL
            )
                .frame(maxWidth: .infinity)
            VStack(alignment: .leading, spacing: 12) {
                Text(model.capActive ? "Interface budget" : "Interface cost").font(.headline)
                let cumulative = model.klHistory.last?.cumulative ?? 0
                let actAddCumulative = model.actAddKLHistory.last?.cumulative ?? 0
                if model.capActive {
                    ProgressView(value: min(cumulative, model.klBudget), total: model.klBudget)
                        .tint(cumulative > model.klBudget ? .red : .orange)
                        .accessibilityLabel("Cumulative KL budget")
                        .accessibilityValue("\(cumulative, specifier: "%.3f") of \(model.klBudget, specifier: "%.2f") nats")
                }
                armMeter(
                    name: model.usesStaticBias ? "Static bias" : "Logit bias",
                    cumulative: cumulative,
                    mean: model.logitMeanNatsPerStep
                )
                armMeter(
                    name: "Residual edit",
                    cumulative: actAddCumulative,
                    mean: model.actAddMeanNatsPerStep
                )
                collapseCallout
                meterFootnote
            }
            .cardStyle()
            .frame(maxWidth: .infinity)
        }
    }

    private func armMeter(name: String, cumulative: Double, mean: Double?) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(name)
                Spacer()
                Text(model.capActive
                    ? "\(cumulative, specifier: "%.3f") / \(model.klBudget, specifier: "%.2f") nats"
                    : "\(cumulative, specifier: "%.3f") nats total")
                    .monospacedDigit()
            }
            .font(.callout)
            if let mean {
                let delta = mean - DemoViewModel.auditTargetKL
                HStack {
                    Text("mean per step").foregroundStyle(.secondary)
                    Spacer()
                    Text("\(mean, specifier: "%.5f")  (\(delta >= 0 ? "+" : "")\(delta, specifier: "%.5f") vs target)")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                .font(.caption)
                .accessibilityElement(children: .combine)
                .accessibilityLabel("\(name) mean KL per step")
                .accessibilityValue("\(mean.formatted(.number.precision(.fractionLength(5)))) nats")
            }
        }
    }

    /// The finding, measured on whatever just ran. Only meaningful under the greedy cap, where a
    /// single cumulative budget can be exhausted by one step.
    private var collapseCallout: some View {
        Group {
            if model.capActive {
                let logitShare = model.firstStepShare(model.klHistory)
                let actAddShare = model.firstStepShare(model.actAddKLHistory)
                if logitShare != nil || actAddShare != nil {
                    VStack(alignment: .leading, spacing: 3) {
                        if let share = logitShare {
                            Text("**Logit arm:** step 1 took \(share * 100, specifier: "%.1f")% of everything the arm spent. \(model.exhaustedStepCount(model.klHistory)) of \(model.klHistory.count) recorded steps ran with the budget already gone.")
                        }
                        if let share = actAddShare {
                            Text("**Residual arm:** step 1 took \(share * 100, specifier: "%.1f")% of everything the arm spent. \(model.exhaustedStepCount(model.actAddKLHistory)) of \(model.actAddKLHistory.count) recorded steps ran with the budget already gone.")
                        }
                        Text("Matched on total cost, unmatched on schedule. A step that spends nothing is the unmodified model continuing from a prefix its own first step displaced.")
                    }
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var meterFootnote: some View {
        Group {
            if !model.droppedTokenStrings.isEmpty {
                Text("Dropped multi-token support: \(model.droppedTokenStrings.joined(separator: ", "))")
            } else if model.usesStaticBias {
                Text("Both traces are diagnostic and uncapped; the frozen comparison was withheld after its residual NLL gate failed.")
            } else if model.capActive {
                Text("Each arm bisects its final active step to fit what the cap has left. The residual arm is attenuated in logit space, not by rescaling its coefficient, so this reproduces the greedy schedule rather than the committed `on-device-rho` packets.")
            } else {
                // The trap this whole mode could fall into: two means side by side look matched.
                // They are not, and nothing else on screen would say so.
                Text("Both arms are uncapped. Each mean above is taken over that arm's **own** free-running output, so the two are averages over different token sequences — matching them would not make the arms comparable. Only the teacher-forced calibration, scored on one shared continuation, does that.")
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }

    private var explanationBody: String {
        let shared = "The residual path front-aligns a per-position contrast direction, injects it after the selected block across the aligned prompt positions, and bakes the edit into downstream KV caches. Topic scores come from a separate MiniLM encoder running as a Core ML model on-device; they are diagnostic cosine similarities, not preference judgments."
        let gates = "A frozen 180-packet control found direction-dependent, on-target behaviour above a matched-random floor in 2/15 cells. A later teacher-forced comparison matched both controllers to 0.43524 nats/step but failed its residual base-model NLL gate, so it supports no ratio."

        if model.usesStaticBias {
            return "This preserved packet shows the same prompt under an unchanged baseline, sustained static topic-token bias, and a direct residual prompt edit. The scalars were calibrated teacher-forced to 0.43524 nats/step on a shared continuation, but the frozen comparison failed its residual base-model NLL gate, so it supports no controller ratio. \(shared)"
        }
        switch model.klDiscipline {
        case .matchedPerStep:
            return """
            The app runs the same prompt three times with identical seeded sampling: an unchanged \
            baseline, a sparse topic-token logit bias, and a direct residual prompt edit. Each \
            controller holds one fixed scalar at every step, with no cumulative cap and no \
            adaptive rescaling — the discipline the audit used. Calibrate to audit target picks \
            each scalar by bisection so its mean teacher-forced KL per step lands on 0.43524 nats, \
            measured on one shared base-generated continuation.

            The free-running panes below are illustrative, not a controlled comparison. \
            Calibration matches the arms on a shared token sequence; once each arm generates its \
            own text, their per-step means are averages over different sequences. \(gates) \(shared)
            """
        case .greedyCumulativeCap:
            return """
            This mode is retained to demonstrate a confound, not to produce a result. Both \
            controllers draw on one cumulative KL budget, spent first-come-first-served with no \
            lookahead and nothing reserved for later steps, and each is attenuated until the \
            budget is gone.

            The arms end up matched on total cost and unmatched on schedule, so a difference \
            between them may come from when each intervened rather than from how. On the eight \
            committed docs/phase6/on-device-rho packets the first generated step consumed 97.4% \
            to 99.9% of an eight-nat cap. The preregistered layer sweep died the same way: blocks \
            3 and 19 produced byte-identical text, because what they shared was the first-token \
            shock rather than any property of the blocks. \(gates) \(shared)
            """
        }
    }

    private var explanation: some View {
        DisclosureGroup("What am I looking at?") {
            Text(explanationBody)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .padding(.top, 8)
            Link(
                "Read the steering audit",
                destination: URL(string: "https://github.com/Ian2x/steering-output-equivalence-audit")!
            )
            .padding(.top, 6)
        }
        .cardStyle()
    }

    private var footer: some View {
        HStack {
            Text("Qwen2.5-0.5B-Instruct • 4-bit • seed 42 / T=0.7 • all LLM inference on-device via MLX (Metal-backed)")
            Spacer()
            Text("Core ML judge • macOS native")
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private var errorBinding: Binding<Bool> {
        Binding(
            get: { model.errorMessage != nil },
            set: { isPresented in
                if !isPresented { model.dismissError() }
            }
        )
    }
}

private struct GenerationPaneView: View {
    let title: String
    let subtitle: String
    let tint: Color
    let state: PaneState
    /// Baseline mean token NLL, or nil on the baseline pane itself. Supplied so an intervention
    /// pane can show its delta: the raw NLL alone is not interpretable without the arm it moved from.
    let baselineNLL: Double?

    /// The frozen teacher-forced gate's ceiling, in nats per token, over the baseline median.
    /// Applied here to a single run only as a visual flag — the gate itself is on medians.
    private static let nllDeltaCeiling = 1.0

    private var nllFormat: FloatingPointFormatStyle<Double> {
        .number.precision(.fractionLength(3))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline)
                    Text(subtitle).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if state.isActive {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("\(title) generation in progress")
                }
                if let score = state.topicScore {
                    Text("topic \(score, specifier: "%.3f")")
                        .font(.caption.monospacedDigit())
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(tint.opacity(0.13), in: Capsule())
                        .accessibilityLabel("\(title) topic score")
                        .accessibilityValue(score.formatted(.number.precision(.fractionLength(3))))
                }
                if let nll = state.baseModelNLL {
                    let delta = baselineNLL.map { nll - $0 }
                    let overCeiling = (delta ?? 0) > Self.nllDeltaCeiling
                    Text(delta.map { "NLL \(nll.formatted(nllFormat)) \($0 > 0 ? "+" : "")\($0.formatted(nllFormat))" }
                        ?? "NLL \(nll.formatted(nllFormat))")
                        .font(.caption.monospacedDigit())
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .foregroundStyle(overCeiling ? Color.red : Color.primary)
                        .background((overCeiling ? Color.red : Color.secondary).opacity(0.13), in: Capsule())
                        .accessibilityLabel("\(title) base-model mean token NLL")
                        .accessibilityValue(
                            delta.map {
                                "\(nll.formatted(nllFormat)), \($0.formatted(nllFormat)) over baseline"
                                    + (overCeiling ? ", above the frozen gate's one nat per token ceiling" : "")
                            } ?? nll.formatted(nllFormat)
                        )
                }
            }
            ScrollView {
                Text(state.text.isEmpty ? "Generation will appear here token by token." : state.text)
                    .foregroundStyle(state.text.isEmpty ? .tertiary : .primary)
                    .frame(maxWidth: .infinity, minHeight: 210, alignment: .topLeading)
                    .textSelection(.enabled)
            }
            .padding(12)
            .background(.quaternary.opacity(0.38), in: RoundedRectangle(cornerRadius: 10))
            .accessibilityElement(children: .combine)
            .accessibilityLabel("\(title) generated text")
            .accessibilityValue(state.text.isEmpty ? "No generation yet" : state.text)
            .accessibilityAddTraits(.updatesFrequently)

            HStack {
                Label("\(state.tokenCount) tokens", systemImage: "text.word.spacing")
                Spacer()
                if ProcessInfo.processInfo.environment["STEERDEMO_HIDE_RATES"] != "1" {
                    Text("\(state.tokensPerSecond, specifier: "%.1f") tok/s")
                    Text("•").accessibilityHidden(true)
                }
                Text(formatMemory(state.residentMemoryBytes))
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
        }
        .cardStyle()
        .overlay(alignment: .top) {
            Rectangle().fill(tint).frame(height: 3).clipShape(.rect(topLeadingRadius: 12, topTrailingRadius: 12))
        }
        .frame(maxWidth: .infinity)
    }

    private func formatMemory(_ bytes: UInt64) -> String {
        guard bytes > 0 else { return "— memory" }
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
}

private struct KLChartView: View {
    let logitHistory: [KLReading]
    let actAddHistory: [KLReading]
    let budget: Double
    let capActive: Bool
    let target: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Per-step KL").font(.headline)
            Chart {
                // The target is the quantity being matched, so it belongs on the axis that shows
                // whether it was. Under the greedy cap the rule is what the trace fails to sit on.
                RuleMark(y: .value("Audit target", target))
                    .foregroundStyle(.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .annotation(position: .top, alignment: .leading) {
                        Text("audit target \(target, specifier: "%.5f")")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                ForEach(logitHistory, id: \.step) { reading in
                    LineMark(
                        x: .value("Step", reading.step),
                        y: .value("KL", reading.perStep),
                        series: .value("Intervention", "Control — logit bias")
                    )
                    .foregroundStyle(.orange)
                    PointMark(
                        x: .value("Step", reading.step),
                        y: .value("KL", reading.perStep)
                    )
                    .foregroundStyle(.orange)
                }
                ForEach(actAddHistory, id: \.step) { reading in
                    LineMark(
                        x: .value("Step", reading.step),
                        y: .value("KL", reading.perStep),
                        series: .value("Intervention", "ActAdd")
                    )
                    .foregroundStyle(.purple)
                    PointMark(
                        x: .value("Step", reading.step),
                        y: .value("KL", reading.perStep)
                    )
                    .foregroundStyle(.purple)
                }
            }
            .chartXAxisLabel("generation step")
            .chartYAxisLabel("nats")
            .frame(height: 128)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Per-step KL chart")
            .accessibilityValue(chartAccessibilityValue)
                Text(caption)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .cardStyle()
    }

    private var caption: String {
        if logitHistory.isEmpty, actAddHistory.isEmpty {
            return "Traces start with the intervention passes. The dashed rule is the audit's per-step target."
        }
        return capActive
            ? "Orange: the logit-bias control. Purple: the ActAdd residual edit. Under the greedy cap a trace spikes once and then sits on zero, nowhere near the dashed target."
            : "Orange: the logit-bias control. Purple: the ActAdd residual edit. Calibrated arms scatter around the dashed target instead of spiking; returned tokens only."
    }

    private var chartAccessibilityValue: String {
        guard let latest = logitHistory.last ?? actAddHistory.last else { return "No measurements yet" }
        let discipline = capActive
            ? "greedy cumulative cap of \(budget.formatted(.number.precision(.fractionLength(2)))) nats"
            : "uncapped, target \(target.formatted(.number.precision(.fractionLength(5)))) nats per step"
        return "\(logitHistory.count) logit-bias steps and \(actAddHistory.count) activation-addition steps; latest KL \(latest.perStep.formatted(.number.precision(.fractionLength(4)))) nats; \(discipline)"
    }
}

private extension View {
    func cardStyle() -> some View {
        padding(16)
            .background(.background, in: RoundedRectangle(cornerRadius: 12))
            .overlay {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(.separator.opacity(0.45), lineWidth: 1)
            }
    }
}

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
                Text(model.usesStaticBias
                    // "a direct residual control" used to name the ActAdd arm here, which reads as
                    // though ActAdd were the control condition. The control is the logit bias.
                    ? "One prompt. Three fixed-seed passes. A sustained static-bias control against a direct residual ActAdd edit."
                    : "One prompt. Three fixed-seed passes. A sparse KL-capped bias control against a direct residual ActAdd edit.")
                    .foregroundStyle(.secondary)
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
                        Text(model.usesStaticBias ? "Static-bias KL" : "Logit-bias KL cap")
                        Spacer()
                        Text(model.usesStaticBias ? "uncapped" : "\(model.klBudget, specifier: "%.2f") nats")
                            .monospacedDigit()
                    }
                    if model.usesStaticBias {
                        Text("Scalar calibrated on a shared continuation")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        Slider(value: $model.klBudget, in: 0.1 ... 20, step: 0.1)
                            .accessibilityLabel("Logit-bias KL cap")
                            .accessibilityValue("\(model.klBudget, specifier: "%.2f") nats")
                    }
                }
                .frame(maxWidth: 300)
                .disabled(model.isGenerating)

                Spacer()
                if model.isGenerating {
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
                budget: model.klBudget
            )
                .frame(maxWidth: .infinity)
            VStack(alignment: .leading, spacing: 12) {
                Text(model.usesStaticBias ? "Interface cost" : "Interface budget").font(.headline)
                let cumulative = model.klHistory.last?.cumulative ?? 0
                let actAddCumulative = model.actAddKLHistory.last?.cumulative ?? 0
                if !model.usesStaticBias {
                    ProgressView(value: min(cumulative, model.klBudget), total: model.klBudget)
                        .tint(cumulative > model.klBudget ? .red : .orange)
                        .accessibilityLabel("Cumulative KL budget")
                        .accessibilityValue("\(cumulative, specifier: "%.3f") of \(model.klBudget, specifier: "%.2f") nats")
                }
                HStack {
                    Text(model.usesStaticBias ? "Static-bias KL (uncapped)" : "Logit-bias KL")
                    Spacer()
                    Text(model.usesStaticBias
                        ? "\(cumulative, specifier: "%.3f") nats"
                        : "\(cumulative, specifier: "%.3f") / \(model.klBudget, specifier: "%.2f") nats")
                        .monospacedDigit()
                }
                .font(.callout)
                HStack {
                    Text("Residual-edit KL (uncapped)")
                    Spacer()
                    Text("\(actAddCumulative, specifier: "%.3f") nats")
                        .monospacedDigit()
                        .foregroundStyle(!model.usesStaticBias && actAddCumulative > model.klBudget ? Color.red : Color.primary)
                }
                .font(.callout)
                // In sparse mode the logit arm is held to the slider's cap and the residual arm is
                // not held to anything. Nothing else on screen says so, and the two numbers sit
                // one above the other inviting a straight comparison. A run at 8 nats against a
                // run at 100 is not a comparison, and the viewer has to be told which one is free.
                if !model.usesStaticBias, actAddCumulative > model.klBudget {
                    Text("The residual arm spent \(actAddCumulative / max(model.klBudget, 0.0001), specifier: "%.1f")× the logit arm's cap. Only the logit arm is capped, so these two traces are not KL-matched and the panes are not a controlled comparison.")
                        .font(.caption)
                        .foregroundStyle(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if !model.droppedTokenStrings.isEmpty {
                    Text("Dropped multi-token support: \(model.droppedTokenStrings.joined(separator: ", "))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if model.usesStaticBias {
                    Text("Both traces are diagnostic and uncapped; the frozen comparison was withheld after its residual NLL gate failed.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Each controller bisects its final active step to respect the cap; logit bias uses only single-token entries.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .cardStyle()
            .frame(maxWidth: .infinity)
        }
    }

    private var explanation: some View {
        DisclosureGroup("What am I looking at?") {
            Text(model.usesStaticBias
                ? "This preserved packet shows the same prompt under an unchanged baseline, sustained static topic-token bias, and a direct residual prompt edit. The scalars were calibrated teacher-forced to 0.43524 nats/step on a shared continuation, but the frozen comparison failed its residual base-model NLL gate, so it supports no controller ratio. Topic scores come from a separate MiniLM encoder running as a Core ML model on-device; they are diagnostic cosine similarities, not preference judgments."
                : "The app runs the same prompt three times with identical seeded sampling: an unchanged baseline, a sparse topic-token logit bias under a cumulative cap, and a direct residual prompt edit. The residual path front-aligns a per-position contrast direction, injects it after the selected block across the aligned prompt positions, and bakes the edit into downstream KV caches. A frozen 180-packet control found direction-dependent, on-target behavior above a matched-random floor in 2/15 cells. A later teacher-forced comparison matched both controllers to 0.43524 nats/step but failed its residual base-model NLL gate, so it supports no ratio. Topic scores come from a separate MiniLM encoder running as a Core ML model on-device; they are diagnostic cosine similarities, not preference judgments.")
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

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Per-step KL").font(.headline)
            Chart {
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
                Text(logitHistory.isEmpty && actAddHistory.isEmpty ? "Traces start with the intervention passes." : "Orange: the logit-bias control. Purple: the ActAdd residual edit. Returned tokens only; cap status is shown alongside the chart.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .cardStyle()
    }

    private var chartAccessibilityValue: String {
        guard let latest = logitHistory.last ?? actAddHistory.last else { return "No measurements yet" }
        return "\(logitHistory.count) logit-bias steps and \(actAddHistory.count) activation-addition steps; latest KL \(latest.perStep.formatted(.number.precision(.fractionLength(4)))) nats; budget \(budget.formatted(.number.precision(.fractionLength(2)))) nats"
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

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
                Text("One prompt. Three fixed-seed passes. Sparse KL-capped bias plus a direct residual control.")
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
                        Text("Logit-bias KL cap")
                        Spacer()
                        Text("\(model.klBudget, specifier: "%.2f") nats")
                            .monospacedDigit()
                    }
                    Slider(value: $model.klBudget, in: 0.1 ... 20, step: 0.1)
                        .accessibilityLabel("Logit-bias KL cap")
                        .accessibilityValue("\(model.klBudget, specifier: "%.2f") nats")
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
                    }
                    Slider(value: $model.actAddCoefficient, in: 0 ... 40, step: 1)
                    .accessibilityLabel("Residual-edit coefficient")
                        .accessibilityValue(model.actAddCoefficient.formatted(.number.precision(.fractionLength(1))))
                }
                .frame(maxWidth: 360)
                .disabled(model.isGenerating)

                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text("Residual layer")
                        Spacer()
                        Text("after block \(model.actAddLayer)")
                            .monospacedDigit()
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
                }
                .frame(maxWidth: 360)
                .disabled(model.isGenerating)

                Text("Blocking control passed: 2/15 frozen layer/coefficient cells cleared direction, topic, random-floor, and degeneracy gates. The residual KL trace is diagnostic, not capped; a matched comparison is still pending.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
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
                state: model.baseline
            )
            GenerationPaneView(
                title: "Logit bias",
                subtitle: "Sparse topic-token bias",
                tint: .orange,
                state: model.steered
            )
            GenerationPaneView(
                title: "Residual edit (controlled)",
                subtitle: "Persistent prompt edit after block \(model.actAddLayer)",
                tint: .purple,
                state: model.actAdd
            )
        }
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
                Text("Interface budget").font(.headline)
                let cumulative = model.klHistory.last?.cumulative ?? 0
                let actAddCumulative = model.actAddKLHistory.last?.cumulative ?? 0
                ProgressView(value: min(cumulative, model.klBudget), total: model.klBudget)
                    .tint(cumulative > model.klBudget ? .red : .orange)
                    .accessibilityLabel("Cumulative KL budget")
                    .accessibilityValue("\(cumulative, specifier: "%.3f") of \(model.klBudget, specifier: "%.2f") nats")
                HStack {
                    Text("Logit-bias KL")
                    Spacer()
                    Text("\(cumulative, specifier: "%.3f") / \(model.klBudget, specifier: "%.2f") nats")
                        .monospacedDigit()
                }
                .font(.callout)
                HStack {
                    Text("Residual-edit KL (uncapped)")
                    Spacer()
                    Text("\(actAddCumulative, specifier: "%.3f") nats")
                        .monospacedDigit()
                }
                .font(.callout)
                if !model.droppedTokenStrings.isEmpty {
                    Text("Dropped multi-token support: \(model.droppedTokenStrings.joined(separator: ", "))")
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
            Text("The app runs the same prompt three times with identical seeded sampling: an unchanged baseline, a sparse topic-token logit bias under a cumulative cap, and a direct residual prompt edit. The residual path front-aligns a per-position contrast direction, injects it after the selected block across the aligned prompt positions, and bakes the edit into downstream KV caches. A frozen 180-packet control found direction-dependent, on-target behavior above a matched-random floor in 2/15 cells; that validates the path in this harness but is not yet a matched controller comparison. Topic scores come from a separate MiniLM encoder running as a Core ML model on-device; they are diagnostic cosine similarities, not preference judgments.")
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
                        series: .value("Intervention", "Logit bias")
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
                Text(logitHistory.isEmpty && actAddHistory.isEmpty ? "Traces start with the intervention passes." : "Orange: capped logit bias. Purple: uncapped residual control. Returned tokens only.")
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

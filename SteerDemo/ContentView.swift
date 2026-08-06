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
                Text("One prompt. Two fixed-seed passes. A measured interface budget.")
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
                        Text("KL budget")
                        Spacer()
                        Text("\(model.klBudget, specifier: "%.2f") nats")
                            .monospacedDigit()
                    }
                    Slider(value: $model.klBudget, in: 0.1 ... 20, step: 0.1)
                        .accessibilityLabel("KL budget")
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
                title: "Steered",
                subtitle: "Sparse topic-token bias",
                tint: .orange,
                state: model.steered
            )
        }
    }

    private var meters: some View {
        HStack(alignment: .top, spacing: 16) {
            KLChartView(history: model.klHistory, budget: model.klBudget)
                .frame(maxWidth: .infinity)
            VStack(alignment: .leading, spacing: 12) {
                Text("Interface budget").font(.headline)
                let cumulative = model.klHistory.last?.cumulative ?? 0
                ProgressView(value: min(cumulative, model.klBudget), total: model.klBudget)
                    .tint(cumulative > model.klBudget ? .red : .orange)
                    .accessibilityLabel("Cumulative KL budget")
                    .accessibilityValue("\(cumulative, specifier: "%.3f") of \(model.klBudget, specifier: "%.2f") nats")
                HStack {
                    Text("Cumulative KL")
                    Spacer()
                    Text("\(cumulative, specifier: "%.3f") / \(model.klBudget, specifier: "%.2f") nats")
                        .monospacedDigit()
                }
                .font(.callout)
                if !model.droppedTokenStrings.isEmpty {
                    Text("Dropped multi-token support: \(model.droppedTokenStrings.joined(separator: ", "))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("The last biased step is rescaled to respect the budget; only one-token entries receive it.")
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
            Text("The app runs the same prompt twice with identical seeded sampling. The baseline uses the model's logits unchanged; the steered pass applies a fixed sparse bias to single-token topic terms until its cumulative KL budget is exhausted, then continues unbiased from the resulting steered prefix. The orange trace is KL(biased ‖ base) computed from those two distributions before sampling. The topic score comes from a separate MiniLM encoder running as a Core ML model on-device. This is an interface demonstration of the audit's output-control result, not an ActAdd implementation or a training system.")
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
                Text("\(state.tokensPerSecond, specifier: "%.1f") tok/s")
                Text("•").accessibilityHidden(true)
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
    let history: [KLReading]
    let budget: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Per-step KL").font(.headline)
            Chart(history, id: \.step) { reading in
                AreaMark(
                    x: .value("Step", reading.step),
                    y: .value("KL", reading.perStep)
                )
                .foregroundStyle(.orange.opacity(0.16))
                LineMark(
                    x: .value("Step", reading.step),
                    y: .value("KL", reading.perStep)
                )
                .foregroundStyle(.orange)
                .interpolationMethod(.linear)
                PointMark(
                    x: .value("Step", reading.step),
                    y: .value("KL", reading.perStep)
                )
                .foregroundStyle(.orange)
            }
            .chartXAxisLabel("generation step")
            .chartYAxisLabel("nats")
            .frame(height: 128)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Per-step KL chart")
            .accessibilityValue(chartAccessibilityValue)
            Text(history.isEmpty ? "The trace starts with the steered pass." : "Measured before each fixed-seed sample.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .cardStyle()
    }

    private var chartAccessibilityValue: String {
        guard let latest = history.last else { return "No measurements yet" }
        return "\(history.count) measured steps; latest KL \(latest.perStep.formatted(.number.precision(.fractionLength(4)))) nats; cumulative \(latest.cumulative.formatted(.number.precision(.fractionLength(4)))) of \(budget.formatted(.number.precision(.fractionLength(2)))) nats"
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

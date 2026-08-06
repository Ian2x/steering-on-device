import SwiftUI

@main
struct SteerDemoApp: App {
    @StateObject private var model = DemoViewModel()

    var body: some Scene {
        Window("SteerDemo", id: "main") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 1_420, minHeight: 820)
                .task { model.startAutorunIfRequested() }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1_680, height: 1_000)
    }
}

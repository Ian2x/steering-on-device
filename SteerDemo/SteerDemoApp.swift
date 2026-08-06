import SwiftUI

@main
struct SteerDemoApp: App {
    @StateObject private var model = DemoViewModel()

    var body: some Scene {
        Window("SteerDemo", id: "main") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 1_080, minHeight: 760)
                .task { model.startAutorunIfRequested() }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1_240, height: 860)
    }
}

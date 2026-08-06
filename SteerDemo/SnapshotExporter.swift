import AppKit
import SwiftUI

@MainActor
enum SnapshotExporter {
    static func write(model: DemoViewModel, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let content = ContentView()
            .environmentObject(model)
            .frame(width: 1680, height: 1180)
        let hostingView = NSHostingView(rootView: content)
        hostingView.frame = NSRect(x: 0, y: 0, width: 1680, height: 1180)
        hostingView.appearance = NSAppearance(named: .darkAqua)
        hostingView.layoutSubtreeIfNeeded()
        guard let bitmap = hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds)
        else {
            throw DemoError.snapshotRenderingFailed
        }
        hostingView.cacheDisplay(in: hostingView.bounds, to: bitmap)
        guard let data = bitmap.representation(using: .png, properties: [:]) else {
            throw DemoError.snapshotRenderingFailed
        }
        try data.write(to: url, options: .atomic)
    }
}

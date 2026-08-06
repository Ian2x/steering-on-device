import AppKit
import CoreGraphics
import Foundation

guard CommandLine.arguments.count == 3 else {
    fputs("usage: composite_app_icon.swift INPUT.png OUTPUT.png\n", stderr)
    exit(2)
}

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
guard let source = NSImage(contentsOf: inputURL),
      let bitmap = NSBitmapImageRep(
          bitmapDataPlanes: nil,
          pixelsWide: 1024,
          pixelsHigh: 1024,
          bitsPerSample: 8,
          samplesPerPixel: 4,
          hasAlpha: true,
          isPlanar: false,
          colorSpaceName: .deviceRGB,
          bitmapFormat: [],
          bytesPerRow: 1024 * 4,
          bitsPerPixel: 32
      ), let context = NSGraphicsContext(bitmapImageRep: bitmap)
else {
    fputs("could not create icon canvas\n", stderr)
    exit(1)
}

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = context
context.cgContext.clear(CGRect(x: 0, y: 0, width: 1024, height: 1024))

// Big Sur and later icons occupy roughly 82% of the 1024-point canvas.
// A sampled superellipse gives the continuous macOS squircle without
// redrawing or altering the supplied artwork inside that platform mask.
let iconRect = CGRect(x: 92, y: 92, width: 840, height: 840)
let exponent = 5.0
let points = 720
let path = CGMutablePath()
for index in 0 ..< points {
    let theta = 2 * Double.pi * Double(index) / Double(points)
    let cosine = cos(theta)
    let sine = sin(theta)
    let x = pow(abs(cosine), 2 / exponent) * (cosine < 0 ? -1 : 1)
    let y = pow(abs(sine), 2 / exponent) * (sine < 0 ? -1 : 1)
    let point = CGPoint(
        x: iconRect.midX + x * iconRect.width / 2,
        y: iconRect.midY + y * iconRect.height / 2
    )
    index == 0 ? path.move(to: point) : path.addLine(to: point)
}
path.closeSubpath()

context.cgContext.saveGState()
context.cgContext.setShadow(
    offset: CGSize(width: 0, height: -12),
    blur: 24,
    color: NSColor.black.withAlphaComponent(0.38).cgColor
)
context.cgContext.addPath(path)
context.cgContext.setFillColor(NSColor.black.cgColor)
context.cgContext.fillPath()
context.cgContext.restoreGState()

context.cgContext.addPath(path)
context.cgContext.clip()
source.draw(
    in: iconRect,
    from: CGRect(origin: .zero, size: source.size),
    operation: .copy,
    fraction: 1,
    respectFlipped: false,
    hints: [.interpolation: NSImageInterpolation.high]
)

NSGraphicsContext.restoreGraphicsState()
guard let data = bitmap.representation(using: .png, properties: [:]) else {
    fputs("could not encode icon PNG\n", stderr)
    exit(1)
}
try data.write(to: outputURL, options: .atomic)

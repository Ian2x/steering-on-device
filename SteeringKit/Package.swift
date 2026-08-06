// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "SteeringKit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "SteeringKit", targets: ["SteeringKit"]),
    ],
    targets: [
        .target(name: "SteeringKit"),
        .testTarget(
            name: "SteeringKitTests",
            dependencies: ["SteeringKit"],
            resources: [.copy("Fixtures")]
        ),
    ]
)


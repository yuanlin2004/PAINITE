// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "PAINITE",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "PAINITE", targets: ["PAINITE"]),
        .library(name: "PainiteKit", targets: ["PainiteKit"]),
    ],
    targets: [
        // Service client, configuration, file handling. No UI, so it is unit tested.
        .target(name: "PainiteKit"),
        // The app: AppKit lifecycle with SwiftUI content.
        .executableTarget(name: "PAINITE", dependencies: ["PainiteKit"]),
        .testTarget(name: "PainiteKitTests", dependencies: ["PainiteKit"]),
    ]
)

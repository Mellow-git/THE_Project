// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PersonalAI",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "PersonalAI",
            targets: ["PersonalAI"]
        ),
    ],
    dependencies: [
        // No external dependencies needed for basic SwiftUI app
    ],
    targets: [
        .executableTarget(
            name: "PersonalAI",
            dependencies: [],
            path: "Sources",
            resources: [
                .process("Assets.xcassets")
            ]
        ),
    ]
)

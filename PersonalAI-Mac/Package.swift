// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "PersonalAI",
    platforms: [.macOS(.v12)],
    products: [
        .executable(name: "PersonalAI", targets: ["PersonalAI"]) 
    ],
    targets: [
        .executableTarget(
            name: "PersonalAI",
            dependencies: [],
            path: "PersonalAI"
        )
    ]
)

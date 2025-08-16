import SwiftUI
import AppKit

@main
struct PersonalAIApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .onAppear {
                    print("ContentView appeared")
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .defaultSize(width: 800, height: 600)
    }
    
    init() {
        print("PersonalAIApp initializing...")
        // Ensure the app is properly initialized for macOS
        NSApplication.shared.activate(ignoringOtherApps: true)
        print("App activated")
    }
} 
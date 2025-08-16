import SwiftUI

struct ContentView: View {
    @State private var hasCompletedOnboarding = false
    
    var body: some View {
        if hasCompletedOnboarding {
            MainAppView()
        } else {
            OnboardingView(hasCompletedOnboarding: $hasCompletedOnboarding)
        }
    }
}

struct MainAppView: View {
    var body: some View {
        VStack(spacing: 20) {
            Text("Your Personal AI Assistant")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("AI is ready to learn from your data")
                .font(.title2)
                .foregroundColor(.secondary)
            
            // Placeholder for future AI functionality
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.blue.opacity(0.1))
                .frame(height: 200)
                .overlay(
                    Text("AI Interface Coming Soon")
                        .font(.headline)
                        .foregroundColor(.blue)
                )
        }
        .padding()
    }
} 
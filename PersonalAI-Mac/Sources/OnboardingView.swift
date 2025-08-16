import SwiftUI

struct OnboardingView: View {
    @Binding var hasCompletedOnboarding: Bool
    @State private var currentStep = 0
    @State private var selectedDirectories: [URL] = []
    @State private var safariAccess = false
    @State private var chromeAccess = false
    
    let steps = [
        "Welcome to Your Personal AI Assistant!",
        "Select directories to scan (with your permission)",
        "Choose browser data access (Safari, Chrome)",
        "Your privacy is our priority - all data stays local",
        "Give consent and learn the Copilot Key",
        "Setup complete! Your AI is ready to learn."
    ]

    var body: some View {
        VStack(spacing: 30) {
            // Progress indicator
            ProgressView(value: Double(currentStep + 1), total: Double(steps.count))
                .progressViewStyle(LinearProgressViewStyle())
                .padding(.horizontal)
            
            // Step content
            VStack(spacing: 20) {
                Text(steps[currentStep])
                    .font(.title2)
                    .multilineTextAlignment(.center)
                    .padding()
                
                // Show directory selection on step 1
                if currentStep == 1 {
                    DirectorySelectionView(selectedDirectories: $selectedDirectories)
                }
                
                // Show browser selection on step 2
                if currentStep == 2 {
                    BrowserSelectionView(safariAccess: $safariAccess, chromeAccess: $chromeAccess)
                }
                
                // Show privacy info on step 3
                if currentStep == 3 {
                    PrivacyInfoView()
                }
                
                // Show consent and Copilot Key info on step 4
                if currentStep == 4 {
                    ConsentView()
                }
            }
            
            Spacer()
            
            // Navigation buttons
            HStack {
                if currentStep > 0 {
                    Button("Back") {
                        withAnimation {
                            currentStep -= 1
                        }
                    }
                    .buttonStyle(.bordered)
                }

                Spacer()

                if currentStep < steps.count - 1 {
                    Button("Next") {
                        withAnimation {
                            currentStep += 1
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(currentStep == 1 && selectedDirectories.isEmpty)
                } else {
                    Button("Get Started") {
                        withAnimation {
                            hasCompletedOnboarding = true
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding(.horizontal)
        }
        .padding()
        .frame(maxWidth: 600, maxHeight: 700)
    }
}

struct DirectorySelectionView: View {
    @Binding var selectedDirectories: [URL]
    
    var body: some View {
        VStack(spacing: 15) {
            Text("Select folders to scan")
                .font(.headline)
            
            Button("Choose Folders") {
                selectDirectories()
            }
            .buttonStyle(.borderedProminent)
            
            if !selectedDirectories.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Selected folders:")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    ForEach(selectedDirectories, id: \.self) { url in
                        HStack {
                            Text(url.lastPathComponent)
                                .lineLimit(1)
                            Spacer()
                            Button("Remove") {
                                selectedDirectories.removeAll { $0 == url }
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(6)
                    }
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.05))
        .cornerRadius(12)
    }
    
    private func selectDirectories() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.message = "Select folders to scan"
        panel.prompt = "Choose"
        
        if panel.runModal() == .OK {
            selectedDirectories.append(contentsOf: panel.urls)
        }
    }
}

struct BrowserSelectionView: View {
    @Binding var safariAccess: Bool
    @Binding var chromeAccess: Bool
    
    var body: some View {
        VStack(spacing: 15) {
            Text("Browser Data Access")
                .font(.headline)
            
            Text("Choose which browsers to access (with your permission)")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            VStack(spacing: 12) {
                HStack {
                    Text("Safari")
                        .font(.subheadline)
                    Spacer()
                    Toggle("", isOn: $safariAccess)
                        .labelsHidden()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.gray.opacity(0.05))
                .cornerRadius(8)
                
                HStack {
                    Text("Chrome")
                        .font(.subheadline)
                    Spacer()
                    Toggle("", isOn: $chromeAccess)
                        .labelsHidden()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.gray.opacity(0.05))
                .cornerRadius(8)
            }
        }
        .padding()
        .background(Color.gray.opacity(0.05))
        .cornerRadius(12)
    }
}

struct PrivacyInfoView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "lock.shield")
                .font(.system(size: 60))
                .foregroundColor(.green)
            
            VStack(spacing: 12) {
                Text("Privacy First")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Text("• All data stays on your device")
                Text("• No data sent to external servers")
                Text("• You control what gets scanned")
                Text("• Local AI processing only")
            }
            .font(.subheadline)
            .multilineTextAlignment(.center)
        }
        .padding()
        .background(Color.green.opacity(0.1))
        .cornerRadius(12)
    }
}

struct ConsentView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "hand.raised")
                .font(.system(size: 50))
                .foregroundColor(.blue)
            
            VStack(spacing: 15) {
                Text("Final Consent")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Text("By proceeding, you consent to:")
                    .font(.subheadline)
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("• Local scanning of selected folders")
                    Text("• Browser data access (if selected)")
                    Text("• Local AI processing of your data")
                    Text("• No external data transmission")
                }
                .font(.caption)
                .multilineTextAlignment(.leading)
                
                Text("Copilot Key: Press ⌘+Shift+P to activate AI")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.top, 8)
            }
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }
} 
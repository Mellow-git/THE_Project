import SwiftUI
import Cocoa

struct OnboardingView: View {
    @State private var currentStep = 0
    @State private var selectedDirectories: [URL] = []
    @State private var allowBrowserData = false
    let onComplete: () -> Void
    
    private let steps = [
        ("Welcome", "Your Personal Privacy-First AI Assistant"),
        ("Directories", "Select folders to personalize your AI"),
        ("Browser", "Allow browsing data for better context"),
        ("Privacy", "Your data stays encrypted on your device"),
        ("Ready", "Setup complete! Ready to use Copilot Key")
    ]
    
    var body: some View {
        VStack(spacing: 40) {
            HStack {
                ForEach(0..<steps.count, id: \.self) { index in
                    Circle()
                        .fill(index <= currentStep ? Color.blue : Color.gray.opacity(0.3))
                        .frame(width: 12, height: 12)
                }
            }
            
            VStack(spacing: 30) {
                Text(steps[currentStep].0)
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text(steps[currentStep].1)
                    .font(.title2)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                
                Group {
                    if currentStep == 1 {
                        DirectorySelectionCard(selectedDirectories: $selectedDirectories)
                    } else if currentStep == 2 {
                        BrowserPermissionCard(allowBrowserData: $allowBrowserData)
                    } else if currentStep == 3 {
                        PrivacyInfoCard()
                    } else if currentStep == 4 {
                        CompletionCard()
                    }
                }
            }
            
            Spacer()
            
            HStack {
                if currentStep > 0 {
                    Button("Back") {
                        currentStep -= 1
                    }
                    .buttonStyle(.bordered)
                }
                
                Spacer()
                
                if currentStep < steps.count - 1 {
                    Button("Continue") {
                        currentStep += 1
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    Button("Start Using AI") {
                        UserDefaults.standard.set(selectedDirectories.map { $0.path }, forKey: "selectedDirectories")
                        UserDefaults.standard.set(allowBrowserData, forKey: "allowBrowserData")
                        onComplete()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding(60)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct DirectorySelectionCard: View {
    @Binding var selectedDirectories: [URL]
    
    var body: some View {
        VStack(spacing: 20) {
            if selectedDirectories.isEmpty {
                VStack {
                    Image(systemName: "folder.badge.plus")
                        .font(.system(size: 48))
                        .foregroundColor(.blue)
                    Text("No folders selected yet")
                        .foregroundColor(.secondary)
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Selected Folders:")
                        .font(.headline)
                    ForEach(selectedDirectories.prefix(3), id: \.self) { url in
                        HStack {
                            Image(systemName: "folder.fill")
                                .foregroundColor(.blue)
                            Text(url.lastPathComponent)
                            Spacer()
                            Text("✓")
                                .foregroundColor(.green)
                        }
                        .padding(.vertical, 2)
                    }
                    if selectedDirectories.count > 3 {
                        Text("+ \(selectedDirectories.count - 3) more...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            Button("Choose Folders") {
                selectDirectories()
            }
            .buttonStyle(.bordered)
        }
        .padding(20)
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
    
    private func selectDirectories() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.prompt = "Select"
        panel.message = "Choose folders your AI can learn from"
        
        if panel.runModal() == .OK {
            selectedDirectories = panel.urls
        }
    }
}

struct BrowserPermissionCard: View {
    @Binding var allowBrowserData: Bool
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: allowBrowserData ? "checkmark.circle.fill" : "globe")
                .font(.system(size: 48))
                .foregroundColor(allowBrowserData ? .green : .blue)
            
            Toggle("Allow browsing history access", isOn: $allowBrowserData)
                .toggleStyle(.switch)
            
            Text("Help your AI understand your interests and provide better context-aware assistance")
                .font(.caption)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
        }
        .padding(20)
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
}

struct PrivacyInfoCard: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "lock.shield.fill")
                .font(.system(size: 48))
                .foregroundColor(.green)
            
            VStack(spacing: 8) {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("All data stays on your Mac")
                }
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Encrypted local storage only")
                }
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("You control what gets shared")
                }
            }
            .font(.system(size: 14))
        }
        .padding(20)
        .background(Color.green.opacity(0.1))
        .cornerRadius(12)
    }
}

struct CompletionCard: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "sparkles")
                .font(.system(size: 48))
                .foregroundColor(.purple)
            
            Text("Press ⌘+Shift+C anywhere for instant AI help!")
                .font(.headline)
                .multilineTextAlignment(.center)
        }
        .padding(20)
        .background(Color.purple.opacity(0.1))
        .cornerRadius(12)
    }
}

import SwiftUI

struct DashboardView: View {
    var body: some View {
        VStack(spacing: 30) {
            Text("🎉 Your AI Assistant is Ready!")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("Press ⌘+Shift+C anywhere for instant help")
                .font(.title2)
                .foregroundColor(.secondary)
            
            VStack(spacing: 16) {
                Text("Quick Setup Summary:")
                    .font(.headline)
                
                if let dirs = UserDefaults.standard.array(forKey: "selectedDirectories") as? [String] {
                    Text("📁 \(dirs.count) folders selected for learning")
                }
                
                if UserDefaults.standard.bool(forKey: "allowBrowserData") {
                    Text("🌐 Browser data enabled")
                } else {
                    Text("🌐 Browser data disabled")
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(12)
            
            Spacer()
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

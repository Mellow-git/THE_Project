import SwiftUI
import AppKit

struct DirectorySelector: View {
    @Binding var selectedDirectories: [URL]
    let title: String
    let message: String
    
    init(selectedDirectories: Binding<[URL]>, title: String = "Select Directories", message: String = "Choose folders to scan") {
        self._selectedDirectories = selectedDirectories
        self.title = title
        self.message = message
    }
    
    var body: some View {
        VStack(spacing: 15) {
            Button(title) {
                selectDirectories()
            }
            .buttonStyle(.borderedProminent)
            
            if !selectedDirectories.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Selected directories:")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    ForEach(selectedDirectories, id: \.self) { url in
                        HStack {
                            Image(systemName: "folder")
                                .foregroundColor(.blue)
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
    }
    
    private func selectDirectories() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.message = message
        panel.prompt = "Choose"
        
        if panel.runModal() == .OK {
            selectedDirectories.append(contentsOf: panel.urls)
        }
    }
} 
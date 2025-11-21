import SwiftUI
import AppKit

struct SafariExportView: View {
    @StateObject private var safariCollector = SafariDataCollector()
    @State private var selectedDays = 30
    @State private var isShowingSavePanel = false
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Safari Debug Export")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("Step 1: Close Safari completely (⌘+Q)")
                .font(.headline)
                .foregroundColor(.orange)
            
            Text("Step 2: Enable Full Disk Access in System Settings")
                .font(.headline)
                .foregroundColor(.orange)
            
            Button("Test Safari Access") {
                safariCollector.fetchHistory(maxAgeDays: selectedDays)
            }
            .buttonStyle(.borderedProminent)
            .disabled(safariCollector.isFetching)

            Button("Copy Safari History.db…") {
                showSavePanelAndCopy()
            }
            .buttonStyle(.bordered)
            
            if safariCollector.isFetching {
                ProgressView("Checking Safari database...")
            }
            
            if !safariCollector.exportStatus.isEmpty {
                Text(safariCollector.exportStatus)
                    .font(.headline)
                    .foregroundColor(safariCollector.exportStatus.contains("✅") ? .green : .red)
                    .multilineTextAlignment(.center)
            }

            if !safariCollector.copyStatus.isEmpty {
                Text(safariCollector.copyStatus)
                    .font(.subheadline)
                    .foregroundColor(safariCollector.copyStatus.contains("✅") ? .green : .red)
                    .multilineTextAlignment(.center)
            }
            
            if !safariCollector.debugInfo.isEmpty {
                ScrollView {
                    Text("Debug Information:")
                        .font(.headline)
                        .padding(.bottom, 5)
                    
                    Text(safariCollector.debugInfo)
                        .font(.system(.caption, design: .monospaced))
                        .padding()
                        .background(Color.black.opacity(0.1))
                        .cornerRadius(8)
                }
                .frame(maxHeight: 220)
            }
            
            if !safariCollector.historyItems.isEmpty {
                List(safariCollector.historyItems.prefix(5)) { item in
                    VStack(alignment: .leading) {
                        Text(item.title)
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Text(item.url)
                            .font(.caption)
                            .foregroundColor(.blue)
                    }
                }
                .frame(height: 150)
            }
            
            Spacer()
        }
        .padding()
    }
}

private extension SafariExportView {
    func showSavePanelAndCopy() {
        let panel = NSSavePanel()
        panel.title = "Save a copy of Safari History.db"
        panel.canCreateDirectories = true
        panel.showsHiddenFiles = false
        panel.nameFieldStringValue = "History.db"
        panel.allowedContentTypes = []

        let response = panel.runModal()
        if response == .OK, let url = panel.url {
            safariCollector.copySafariHistory(to: url)
        }
    }
}

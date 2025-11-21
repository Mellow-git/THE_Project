import Foundation
import SwiftUI
import SQLite3

class SafariDataCollector: ObservableObject {
    @Published var historyItems: [SafariHistoryItem] = []
    @Published var isFetching = false
    @Published var exportStatus = ""
    @Published var debugInfo = ""
    @Published var copyStatus = ""
    
    struct SafariHistoryItem: Identifiable, Codable {
        let id = UUID()
        let url: String
        let title: String
        let visitDate: Date
    }
    
    func fetchHistory(maxAgeDays: Int) {
        isFetching = true
        historyItems.removeAll()
        exportStatus = "Checking Safari database access..."
        debugInfo = ""
        
        DispatchQueue.global(qos: .userInitiated).async {
            let fetchedItems = self.extractRealSafariHistory(maxAgeDays: maxAgeDays)
            
            DispatchQueue.main.async {
                self.historyItems = fetchedItems
                self.isFetching = false
                
                if fetchedItems.isEmpty {
                    self.exportStatus = "❌ Found 0 items - Check debug info below"
                } else {
                    self.exportStatus = "✅ Found \(fetchedItems.count) real Safari items"
                    self.saveHistoryAsCSV(items: fetchedItems, maxAgeDays: maxAgeDays)
                }
            }
        }
    }
    
    private func extractRealSafariHistory(maxAgeDays: Int) -> [SafariHistoryItem] {
        var items: [SafariHistoryItem] = []
        
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let historyPath = homeDir.appendingPathComponent("Library/Safari/History.db").path
        
        DispatchQueue.main.async {
            self.debugInfo += "🔍 Checking path: \(historyPath)\n"
        }
        
        // Check if file exists
        if !FileManager.default.fileExists(atPath: historyPath) {
            DispatchQueue.main.async {
                self.debugInfo += "❌ Safari History.db not found\n"
                self.debugInfo += "💡 Make sure you have Full Disk Access enabled\n"
            }
            return items
        }
        
        DispatchQueue.main.async {
            self.debugInfo += "✅ Safari History.db file found\n"
        }
        
        // Check file permissions
        if !FileManager.default.isReadableFile(atPath: historyPath) {
            DispatchQueue.main.async {
                self.debugInfo += "❌ Cannot read Safari History.db\n"
                self.debugInfo += "💡 Enable Full Disk Access in System Settings\n"
            }
            return items
        }
        
        DispatchQueue.main.async {
            self.debugInfo += "✅ File is readable\n"
        }
        
        var db: OpaquePointer?
        
        if sqlite3_open_v2(historyPath, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK {
            DispatchQueue.main.async {
                self.debugInfo += "✅ SQLite database opened successfully\n"
            }
            
            // Enumerate tables
            let tableQuery = "SELECT name FROM sqlite_master WHERE type='table'"
            var tableStmt: OpaquePointer?
            var tables: [String] = []
            if sqlite3_prepare_v2(db, tableQuery, -1, &tableStmt, nil) == SQLITE_OK {
                while sqlite3_step(tableStmt) == SQLITE_ROW {
                    if let namePtr = sqlite3_column_text(tableStmt, 0) {
                        let tableName = String(cString: namePtr)
                        tables.append(tableName)
                    }
                }
                sqlite3_finalize(tableStmt)
            }
            DispatchQueue.main.async {
                self.debugInfo += "📋 Tables found: \(tables.joined(separator: ", "))\n"
            }
            
            // Try different query approaches based on Safari version
            let queries = [
                // Attempt 1: modern Safari (columns may vary; this may fail which is fine for debugging)
                """
                SELECT hv.url, hi.url, hi.domain_expansion, hv.title, hv.visit_time
                FROM history_visits hv
                JOIN history_items hi ON hv.history_item = hi.id
                ORDER BY hv.visit_time DESC
                LIMIT 100
                """,
                // Attempt 2: older Safari
                """
                SELECT url, title, visit_count
                FROM history_items
                ORDER BY visit_count DESC
                LIMIT 100
                """
            ]
            
            for (index, query) in queries.enumerated() {
                var stmt: OpaquePointer?
                if sqlite3_prepare_v2(db, query, -1, &stmt, nil) == SQLITE_OK {
                    DispatchQueue.main.async {
                        self.debugInfo += "✅ Query \(index + 1) prepared successfully\n"
                    }
                    var rowCount = 0
                    while sqlite3_step(stmt) == SQLITE_ROW && rowCount < 50 {
                        var url = ""
                        if let urlPtr = sqlite3_column_text(stmt, 0) {
                            url = String(cString: urlPtr)
                        }
                        var title = "Untitled"
                        if let titlePtr = sqlite3_column_text(stmt, 1) {
                            title = String(cString: titlePtr)
                        }
                        if !url.isEmpty {
                            items.append(SafariHistoryItem(url: url, title: title, visitDate: Date()))
                            rowCount += 1
                        }
                    }
                    DispatchQueue.main.async {
                        self.debugInfo += "📊 Query \(index + 1) returned \(rowCount) rows\n"
                    }
                    sqlite3_finalize(stmt)
                    if !items.isEmpty { break }
                } else {
                    let errorMessage = String(cString: sqlite3_errmsg(db))
                    DispatchQueue.main.async {
                        self.debugInfo += "❌ Query \(index + 1) failed: \(errorMessage)\n"
                    }
                }
            }
            
            sqlite3_close(db)
        } else {
            let errorMessage = String(cString: sqlite3_errmsg(db))
            DispatchQueue.main.async {
                self.debugInfo += "❌ Failed to open SQLite database: \(errorMessage)\n"
            }
        }
        
        DispatchQueue.main.async {
            self.debugInfo += "🎯 Final result: \(items.count) items extracted\n"
        }
        
        return items
    }
    
    func saveHistoryAsCSV(items: [SafariHistoryItem], maxAgeDays: Int) {
        let fileName = "SafariHistory_Debug_\(Date().formatted(.iso8601.year().month().day())).csv"
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            exportStatus = "Failed to access Documents folder"
            return
        }
        
        let fileURL = documentsDir.appendingPathComponent(fileName)
        
        var csvText = "URL,Title,VisitDate\n"
        for item in items {
            let dateString = ISO8601DateFormatter().string(from: item.visitDate)
            let cleanedTitle = item.title.replacingOccurrences(of: "\"", with: "\"\"")
            let line = "\"\(item.url)\",\"\(cleanedTitle)\",\"\(dateString)\"\n"
            csvText += line
        }
        
        do {
            try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
            exportStatus = "✅ Exported \(items.count) items to \(fileName)"
        } catch {
            exportStatus = "❌ Export failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Copy Safari History.db
    func historyDatabaseURL() -> URL {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        return homeDir.appendingPathComponent("Library/Safari/History.db")
    }

    func copySafariHistory(to destinationURL: URL, includeSidecars: Bool = true) {
        let sourceURL = historyDatabaseURL()

        DispatchQueue.main.async {
            self.copyStatus = ""
        }

        // Validate source exists and readable
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            DispatchQueue.main.async { self.copyStatus = "❌ Source History.db not found" }
            return
        }
        guard FileManager.default.isReadableFile(atPath: sourceURL.path) else {
            DispatchQueue.main.async { self.copyStatus = "❌ Cannot read History.db (enable Full Disk Access)" }
            return
        }

        // Perform copy in background
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.replaceItem(at: destinationURL, with: sourceURL)

                if includeSidecars {
                    let sidecarExtensions = ["-wal", "-shm"]
                    for ext in sidecarExtensions {
                        let sidecarSource = sourceURL.deletingPathExtension().appendingPathExtension("db\(ext)")
                        let sidecarDest = destinationURL.deletingPathExtension().appendingPathExtension("db\(ext)")
                        if FileManager.default.fileExists(atPath: sidecarSource.path) {
                            try? self.replaceItem(at: sidecarDest, with: sidecarSource)
                        }
                    }
                }

                DispatchQueue.main.async {
                    self.copyStatus = "✅ Copied to \(destinationURL.lastPathComponent)"
                }
            } catch {
                DispatchQueue.main.async {
                    self.copyStatus = "❌ Copy failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func replaceItem(at destinationURL: URL, with sourceURL: URL) throws {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        } else {
            let parentDir = destinationURL.deletingLastPathComponent()
            try fileManager.createDirectory(at: parentDir, withIntermediateDirectories: true)
        }
        try fileManager.copyItem(at: sourceURL, to: destinationURL)
    }
}

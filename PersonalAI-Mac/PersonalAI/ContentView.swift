import SwiftUI

struct ContentView: View {
    @State private var showOnboarding = true
    @State private var selectedTab = 0
    
    var body: some View {
        if showOnboarding {
            OnboardingView(onComplete: {
                showOnboarding = false
            })
        } else {
            TabView(selection: $selectedTab) {
                EnhancedDashboardView()
                    .tabItem {
                        Image(systemName: "house.fill")
                        Text("Dashboard")
                    }
                    .tag(0)
                
                SafariExportView()
                    .tabItem {
                        Image(systemName: "safari.fill")
                        Text("Safari Export")
                    }
                    .tag(1)
            }
        }
    }
}

# PersonalAI - Privacy-First AI Assistant for Mac

A personalized AI assistant that learns from your local data while maintaining complete privacy. Built with SwiftUI for macOS.

## 🚀 Features

- **Privacy-First Design**: All data stays on your device
- **Multi-Step Onboarding**: User-friendly setup process
- **Directory Selection**: Choose which folders to scan
- **Browser Integration**: Optional access to Safari/Chrome data
- **Local AI Processing**: No data sent to external servers
- **Copilot Key Support**: Quick access with ⌘+Shift+P

## 🛠️ Requirements

- macOS 14.0 or later
- Xcode 15.0 or later
- Swift 5.9+

## 📱 Installation & Setup

### Option 1: Open in Xcode
1. Clone or download this repository
2. Open `PersonalAI.xcodeproj` in Xcode
3. Select your development team in project settings
4. Build and run (⌘+R)

### Option 2: Build from Command Line
```bash
cd PersonalAI-Mac
xcodebuild -project PersonalAI.xcodeproj -scheme PersonalAI -configuration Debug
```

## 🔐 Privacy & Permissions

The app requests the following permissions:
- **File Access**: User-selected directories only
- **Browser Data**: Optional access to Safari/Chrome (user choice)
- **Network**: No external connections (local only)

All permissions are clearly explained during onboarding and can be modified later.

## 🏗️ Project Structure

```
PersonalAI-Mac/
├── PersonalAI.xcodeproj/          # Xcode project file
├── PersonalAI/                     # Source code
│   ├── PersonalAIApp.swift        # Main app entry point
│   ├── ContentView.swift          # Main app interface
│   ├── OnboardingView.swift       # Multi-step onboarding
│   ├── DirectorySelector.swift    # Directory selection component
│   └── Info.plist                 # App configuration
├── Assets.xcassets/               # App icons and colors
└── README.md                      # This file
```

## 🎯 Development Roadmap

### Phase 1 (Current) ✅
- [x] Basic app structure
- [x] Onboarding flow
- [x] Directory selection
- [x] Privacy-first messaging

### Phase 2 (Week 1-2)
- [ ] Browser data access implementation
- [ ] Secure local storage
- [ ] Privacy dashboard
- [ ] File scanning logic

### Phase 3 (Week 3-4)
- [ ] AI model integration
- [ ] Copilot Key functionality
- [ ] Data processing pipeline
- [ ] User interface enhancements

### Phase 4 (Week 5-6)
- [ ] Performance optimization
- [ ] Advanced privacy controls
- [ ] User feedback integration
- [ ] Final testing and polish

## 🔧 Customization

### Adding New Permission Types
1. Update `OnboardingView.swift` with new step
2. Add corresponding UI component
3. Update `Info.plist` with permission descriptions
4. Test permission flow

### Modifying Privacy Settings
- Edit `PrivacyInfoView.swift` for messaging
- Update `ConsentView.swift` for consent text
- Modify permission descriptions in `Info.plist`

## 🐛 Troubleshooting

### Common Issues

**Build Errors**: Ensure Xcode 15.0+ and macOS 14.0+
**Permission Denied**: Check app permissions in System Preferences
**Directory Access**: Verify folder selection in onboarding

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export PERSONALAI_DEBUG=1
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues:
- Create an issue in this repository
- Check the troubleshooting section above
- Review the development roadmap

---

**Built with ❤️ for privacy-conscious Mac users** 
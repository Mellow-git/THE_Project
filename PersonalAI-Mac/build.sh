#!/bin/bash

# PersonalAI Mac Build Script
# This script helps build and test the PersonalAI Mac app

echo "🚀 Building PersonalAI Mac App..."

# Check if we're in the right directory
if [ ! -f "PersonalAI.xcodeproj/project.pbxproj" ]; then
    echo "❌ Error: Please run this script from the PersonalAI-Mac directory"
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
xcodebuild clean -project PersonalAI.xcodeproj -scheme PersonalAI

# Build the project
echo "🔨 Building project..."
xcodebuild build -project PersonalAI.xcodeproj -scheme PersonalAI -configuration Debug

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "📱 Next steps:"
    echo "1. Open PersonalAI.xcodeproj in Xcode"
    echo "2. Select your development team in project settings"
    echo "3. Press ⌘+R to build and run"
    echo ""
    echo "🔐 Note: You may need to grant permissions in System Preferences > Security & Privacy"
else
    echo "❌ Build failed. Please check the error messages above."
    exit 1
fi 
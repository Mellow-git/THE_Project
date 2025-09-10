# Mellow — Project README (Detailed Progress and Architecture)

Mellow is a privacy‑first, on‑device personal AI for macOS. It learns from user‑selected data sources and live on‑screen context to deliver instant, relevant assistance—without sending information to the cloud. It’s designed for developers, students, and privacy‑conscious professionals who want fast, offline, screen‑aware help for coding, research, and daily workflows.

## Current Status (Snapshot)
- Core concept: on‑device assistant with a “Copilot Key” (global hotkey) for screen/context capture
- Messaging: problems, solutions, target audience, market sizing (TAM/SAM), SWOT, “why now”
- Implementation tracks active:
  - macOS app skeleton (Xcode), hotkey service, sandbox/entitlements, permissions prompts
  - Data pipeline MVP: local source registration → background ingestion → semantic index → retrieval
  - Assist pipeline MVP: context capture → RAG → local inference with adapters → streaming responses
  - UI shell: compact panel + command palette; ephemeral OCR and privacy controls

***

## Architecture Overview

- App layers:
  - Shell (macOS app, menu bar helper, global hotkey)
  - Context layer (active window text, selection, optional screenshot + OCR)
  - Data layer (ingestion, chunking, embeddings, vector search)
  - Assist layer (prompt assembly, tools, inference, streaming)
  - Privacy layer (permissions, local‑only network policy, purge, logging)

- Data residency:
  - All processing is local by default. No cloud calls unless explicitly enabled in dev mode.
  - Sensitive artifacts (screenshots/OCR) are ephemeral by default.

***

## Xcode Project Structure and Approaches

- Targets
  - Mellow.app (main sandboxed app)
  - MellowHelper (privileged helper for hotkeys and background ingestion)
  - MellowCLI (optional CLI for power users and testing)
- Languages/Frameworks
  - Swift 5.10+, SwiftUI for primary UI
  - AppKit for low‑level macOS APIs (event taps, accessibility, screen capture)
  - Combine for reactive event streams
- Minimum macOS
  - macOS 13+ (Ventura) recommended; 12+ with guarded API paths
- Entitlements and Capabilities
  - App Sandbox enabled
  - User Selected File access (security‑scoped bookmarks)
  - Hardened Runtime
  - Optional: com.apple.security.files.user-selected.read-write for data sources
- Permissions Flow (system)
  - Accessibility (AX) for reading selected text/active UI element
  - Screen Recording (if screenshot/OCR is enabled)
  - Full Disk Access (optional, only if user wants broad indexing; otherwise scoped)
- Global Hotkey Options
  - Approach A (recommended): Event tap + Carbon Hotkey API via a small Obj‑C/Swift bridge for reliable system‑wide capture
  - Approach B: NSEvent.addGlobalMonitorForEvents (limited reliability when app not frontmost)
  - Helper: LaunchAgent or SMAppService to keep the hotkey responder resident
- Inter‑process Coordination
  - XPC between app and helper for separation of privileges and responsiveness
  - App Group container for shared small state (e.g., last context id)

***

## UI: Dependencies and Structure

- UI Stack
  - SwiftUI for views
  - AppKit bridges for menu bar item, window level control, and focus handling
- Dependencies
  - Optional: KeyboardShortcuts (for quick setup), Defaults (lightweight settings), LaunchAtLogin
  - Markdown rendering for responses (attributed text or a lightweight renderer)
- Components
  - Menu bar icon with quick actions (Capture Context, Last Results, Privacy)
  - Compact panel (floating window) with:
    - Prompt input
    - Live streaming output
    - Tabs: Assist | Data | Privacy
  - Command palette (bring‑up via Copilot Key)
  - Context preview strip: shows source (Active Window, Selection, URL), with “remove” toggles
  - Privacy badge: indicates local‑only mode; click‑through to settings
- UX Policies
  - Ephemeral screenshots/OCR (off by default; session‑only if on)
  - Clear indicators when any capture is active
  - One‑click purge of local data and embeddings

***

## Data Pipeline (Current Implementation)

- Source Registration
  - Types: folder, file, repo, notes, browser history (if explicitly enabled)
  - Security‑scoped bookmarks for persistent, user‑granted access
  - Include/exclude patterns (e.g., “**/*.py”, “**/*.md”, exclude “**/node_modules/**”)
- Ingestion
  - Background worker (helper process) with throttling and OS quality‑of‑service set to “utility”
  - File types supported now: .md, .txt, .pdf (text layer), .py/.js/.ts/.java/.go, .ipynb (text cells)
  - Change detection: file mtime hash; incremental re‑indexing
- Chunking
  - Token‑aware chunking (~512–1,024 token windows) with 10–15% overlap
  - Metadata: path, source_id, language, linespan, hash
- Embeddings
  - Local embedding model (small, CPU/NPU‑friendly); int8/int4 quant where supported
  - Cosine similarity; normalized vectors; approximate search (HNSW) for speed
- Index Storage
  - Lightweight local store (SQLite + FTS5 or SQLite + sidecar .vec files; prototyping with SQLite)
  - Namespace per source_id; global index for multi‑source queries
- Search
  - Top‑k retrieval with optional filters (source_id, path prefix, filetype, freshness)
  - Reranking (optional) using a compact local reranker model
- Privacy Controls
  - Per‑source privacy flags (private, ephemeral)
  - One‑click purge (deletes docs, chunks, embeddings, caches)

***

## Context Capture (Current Implementation)

- Sources
  - Active window title and app bundle id
  - Selected text via AX (when available)
  - Clipboard (opt‑in)
  - Optional: screenshot + OCR (disabled by default)
- OCR
  - Local OCR pipeline; process in memory, discard image unless user saves
- URL/App Metadata
  - Bundle ID, file URL (if accessible), language mode (from app or heuristics)
- Context Object Example
  {
    "context_id": "ctx_2025_08_24_001",
    "app": "Xcode",
    "title": "Mellow.xcodeproj — Build Logs",
    "text": "Linker error: undefined symbol ...",
    "metadata": {
      "bundle_id": "com.apple.dt.Xcode",
      "path": "/Users/me/project/...",
      "lang": "swift",
      "ts": 1692870000
    }
  }

***

## Assist Pipeline (Current Implementation)

- Steps
  1) Capture context (shortcut or UI)
  2) Semantic search over local index (top‑k, optional rerank)
  3) Prompt assembly: system + user prompt + context snippets + tool hints
  4) Local inference with streaming tokens
- Tools (MVP)
  - code_explain (explain selected error/log/code)
  - summarize_pdf (summarize a retrieved PDF section)
  - search_local (query index with keyword+semantic)
- Models
  - Base: compact chat model (2–4B params, int4) for on‑device responsiveness
  - Optional adapter: LoRA/adapter weights for local personalization
- Output
  - Token streaming to UI; copy/export; insert into editor (if safe and permitted)

***

## Inference and Personalization

- Adapters
  - LoRA/adapter methods for targeted tasks (coding, summarization tone)
  - Apply/revert instantly; adapters stored locally, user‑scoped
- Tuning
  - Lightweight local runs with cap on epochs/steps to control time and thermals
  - Background priority; pause/resume when on battery or high thermals
- Performance
  - Preference for NPU/Metal backends where available; fallback to CPU
  - Quantized models to balance latency and quality

***

## Privacy and Security

- Defaults
  - Local‑only network mode; telemetry off by default
  - Ephemeral screenshots/OCR; short‑lived caches
- Controls
  - Privacy page exposing permissions state, toggles, and purge
  - Granular source‑level visibility and exclude patterns
- Logging
  - Minimal local logs for debugging; sensitive content redacted where possible

***

## Developer API (Local)

- Localhost API (dev mode): http://localhost:4310
- Key endpoints (subject to change as we stabilize):
  - POST /v1/context/capture
  - POST /v1/assist
  - POST /v1/data/sources
  - POST /v1/data/ingest
  - GET  /v1/data/search
  - POST /v1/tune/start
  - GET  /v1/privacy/status
- CLI (optional)
  - mellow context capture --ocr
  - mellow assist -p "Explain selection"
  - mellow data add ~/Projects --pattern "**/*.py"
  - mellow data ingest --source src_101

***

## Build & Run (Dev)

- Requirements
  - Xcode 15+, macOS 13+
  - Swift 5.10+
- Steps
  1) Clone repo; open Mellow.xcodeproj
  2) Select “Mellow” scheme; run once to grant initial permissions
  3) Enable helper (SMAppService) for global hotkey reliability
  4) In app, add a test folder as a data source and run ingestion
  5) Trigger Copilot Key; verify context preview and streaming output

***

## Roadmap (Near Term)

- macOS Alpha
  - End‑to‑end capture → retrieval → assist stable
  - Minimal crash handling; safe fallbacks if permissions missing
- UI/UX
  - Command palette polish; quick intents (Explain, Summarize, Fix)
  - Response actions: copy, insert, open file at linespan
- Data
  - More parsers (CSV, HTML, PPTX), better PDF handling
  - Faster incremental indexing; background thermal management
- Models
  - Add code‑specialized adapter; optional reranker
  - Auto‑selection of best local backend (NPU/Metal/CPU)
- Privacy
  - Per‑session “burn after reading” mode for OCR/context

***

## Contributing

- Branching: feature/*, fix/*, docs/*
- PRs: small, testable; include UI gifs/logs when relevant
- Security: report privately; do not attach sensitive logs or paths
- Style: SwiftLint optional; formatter rules in repo

***

## License

- To be decided (MIT/Apache‑2.0 recommended). Will be finalized before public beta.

***
## Next up!!
- next steps will be exciting as ollama or some other low rank adapted llm will be integrated in the software workflow,,,,,,,, STAY TUNED

## Contact

Author: Mellow
mehylekyle@gmail.com

import PainiteKit
import SwiftUI

struct DocumentView: View {
    @ObservedObject var model: DocumentModel
    @ObservedObject var server = ServerStore.shared
    var onSaveAs: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                header
                Divider()
                ZStack {
                    CompareViewRepresentable(model: model)
                    if model.image == nil {
                        if let err = model.loadError {
                            Text(err).foregroundStyle(.secondary).padding()
                        } else {
                            ProgressView("Opening…")
                        }
                    }
                }
            }
            Divider()
            sidebar
                .frame(width: 280)
        }
        .frame(minWidth: 900, minHeight: 560)
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 0) {
            Text("Original")
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, 12)
            HStack(spacing: 6) {
                Text("Enhanced")
                previewBadge
                Spacer()
                zoomControls
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
        }
        .font(.callout)
        .frame(height: 30)
        .background(.bar)
    }

    @ViewBuilder private var previewBadge: some View {
        switch model.previewState {
        case .rendering(let p):
            ProgressView(value: p).progressViewStyle(.linear).frame(width: 60)
            Text("Updating…").foregroundStyle(.secondary)
        case .failed:
            Label("Preview failed", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .labelStyle(.titleAndIcon)
        case .ready where !model.previewIsCurrent:
            Text("Out of date").foregroundStyle(.secondary)
        case .ready:
            if let ms = model.preview?.serverMs {
                Text("server \(Int(ms)) ms").foregroundStyle(.secondary).monospacedDigit()
            }
        case .idle:
            EmptyView()
        }
    }

    private var zoomControls: some View {
        HStack(spacing: 4) {
            Button { model.viewport?.zoomOut() } label: { Image(systemName: "minus.magnifyingglass") }
                .help("Zoom Out (⌘−)")
            Text(model.zoomPercent > 0 ? "\(Int(model.zoomPercent.rounded()))%" : "–")
                .monospacedDigit()
                .frame(minWidth: 44)
            Button { model.viewport?.zoomIn() } label: { Image(systemName: "plus.magnifyingglass") }
                .help("Zoom In (⌘+)")
            Button("Fit") { model.viewport?.fit() }.help("Zoom to Fit (⌘0)")
            Button("100%") { model.viewport?.actualSize() }.help("Actual Size (⌘1)")
        }
        .buttonStyle(.borderless)
    }

    // MARK: Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 16) {
            if model.handoff.isLightroom {
                lightroomNote
            }
            enhanceSection
            Divider()
            exportSection
            Spacer(minLength: 0)
            Divider()
            statusSection
        }
        .padding(14)
        .disabled(model.loadError != nil)
    }

    private var lightroomNote: some View {
        Label {
            Text("Opened from Lightroom").font(.callout.weight(.semibold))
        } icon: {
            Image(systemName: "arrow.triangle.turn.up.right.diamond")
        }
    }

    private var enhanceSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Enhancement").font(.headline)
            Picker("Operation", selection: $model.operation) {
                ForEach(PainiteKit.Operation.allCases) { op in
                    Text(op.title).tag(op)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .disabled(model.export.isRunning)

            if model.operationHasStrength {
                HStack {
                    Text("Strength")
                    Slider(value: $model.strength, in: model.strengthRange)
                    Text("\(Int((model.strength * 100).rounded()))")
                        .monospacedDigit()
                        .frame(width: 30, alignment: .trailing)
                }
                .disabled(model.export.isRunning)
            } else {
                Text("Doubles the width and height.").font(.callout).foregroundStyle(.secondary)
            }

            if let modelID = model.preview?.model ?? server.capabilities?.operation(model.operation)?.defaultModel {
                HStack(spacing: 6) {
                    Text(modelID).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                    if server.capabilities?.model(modelID)?.placeholder == true {
                        Text("placeholder")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Capsule().fill(Color.orange.opacity(0.25)))
                            .help("A stand-in model for testing. Denoise and sharpen return the original; upscale is a plain resize.")
                    }
                }
            }

            if case .failed(let msg) = model.previewState {
                VStack(alignment: .leading, spacing: 6) {
                    Text(msg).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    Button("Regenerate Preview") { model.regeneratePreview() }
                }
            }
        }
    }

    private var exportSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(model.handoff.isLightroom ? "Return to Lightroom" : "Export").font(.headline)

            if !model.handoff.isLightroom {
                Picker("Format", selection: $model.exportFormat) {
                    Text("TIFF").tag(ImageFormat.tiff)
                    Text("JPEG").tag(ImageFormat.jpeg)
                }
                .pickerStyle(.segmented)
                .disabled(model.export.isRunning)
            }

            HStack {
                if model.canSaveInPlace {
                    Button("Save") { model.saveInPlace() }
                        .buttonStyle(.borderedProminent)
                        .help("Replace the copy Lightroom handed over, then click Finish in Lightroom.")
                }
                Button("Save As…") { onSaveAs() }
            }
            .disabled(model.export.isRunning || model.image == nil)

            exportStatus
        }
    }

    @ViewBuilder private var exportStatus: some View {
        switch model.export {
        case .idle:
            if model.handoff.isLightroom {
                Text(model.canSaveInPlace
                     ? "Save replaces Lightroom's copy; then click Finish in Lightroom. For a RAW photo, use Save As and pick the file with Select… in Lightroom."
                     : "Lightroom's copy can't be written. Use Save As, then choose the file with Browse… or Select… in Lightroom.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        case .running(let phase, let p):
            VStack(alignment: .leading, spacing: 4) {
                if let p {
                    ProgressView(value: p) { Text(phase) }
                } else {
                    ProgressView { Text(phase) }.progressViewStyle(.linear)
                }
                Button("Cancel") { model.cancelExport() }
                    .keyboardShortcut(.cancelAction)
            }
        case .done(let url, let inPlace):
            VStack(alignment: .leading, spacing: 4) {
                Label("Saved \(url.lastPathComponent)", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                if inPlace {
                    Text("Switch to Lightroom and click Finish.").font(.callout).foregroundStyle(.secondary)
                } else if model.handoff.isLightroom {
                    Text("In Lightroom, click Browse… (or Select… for a RAW photo) and choose this file.")
                        .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                } else {
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                        .buttonStyle(.link)
                }
            }
        case .failed(let msg):
            Text(msg).font(.callout).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            uploadLine
            serverLine
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    @ViewBuilder private var uploadLine: some View {
        switch model.upload {
        case .idle: EmptyView()
        case .hashing: Text("Checking the server for this photo…")
        case .uploading(let p):
            ProgressView(value: p) { Text("Uploading original… \(Int(p * 100))%") }
        case .ready: Text("Original is on the server")
        case .failed(let msg): Text(msg).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private var serverLine: some View {
        switch server.status {
        case .unconfigured:
            Button("Set up the server in Settings…") { AppDelegate.shared?.showSettings(nil) }.buttonStyle(.link)
        case .checking:
            Text("Connecting to \(server.hostLabel)…")
        case .connected:
            Text("Connected to \(server.hostLabel)")
        case .failed(let msg):
            Text(msg).fixedSize(horizontal: false, vertical: true)
        }
    }
}

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
                .frame(width: 300)
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
                Text(isEnhancing ? "Enhancing…" : "Enhanced")
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

    private var isEnhancing: Bool {
        if case .rendering = model.previewState { return true }
        return false
    }

    @ViewBuilder private var previewBadge: some View {
        switch model.previewState {
        case .rendering(let p):
            ProgressView(value: p).progressViewStyle(.linear).frame(width: 60)
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
            Toggle(isOn: $model.isSelectingArea) {
                Label("Select Area", systemImage: "viewfinder")
            }
            .toggleStyle(.button)
            .help("Drag on the photo to choose the preview area, or click to move it there (⌘E). Shift-drag works at any time.")
            Button { model.previewVisibleArea() } label: { Image(systemName: "rectangle.dashed") }
                .help("Preview the area in view (⇧⌘E)")
            Divider().frame(height: 16).padding(.horizontal, 4)
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
            HStack {
                Text("Steps").font(.headline)
                Spacer()
                addStepMenu
            }
            if model.steps.isEmpty {
                Text("Add a step to enhance the photo. Steps run top to bottom in one pass on the server.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            ScrollView {
                VStack(spacing: 8) {
                    ForEach(Array(model.steps.enumerated()), id: \.element.id) { index, step in
                        StepCard(model: model, step: step, index: index, count: model.steps.count,
                                 enabledIndex: enabledIndex(of: step))
                    }
                }
            }
            .frame(maxHeight: 360)
            .fixedSize(horizontal: false, vertical: true)

            if model.signature.items.count > 1 {
                Text(Recipe.label(model.signature))
                    .font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }

            if !model.steps.isEmpty {
                Text("The preview covers the box on the photo. Zooming and panning don't change it. To preview another area, use Select Area (⌘E) and drag or click on the photo, or Shift-drag.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }

            if case .failed(let msg) = model.previewState {
                VStack(alignment: .leading, spacing: 6) {
                    Text(msg).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    Button("Regenerate Preview") { model.regeneratePreview() }
                }
            }
        }
        .disabled(model.export.isRunning)
    }

    private var addStepMenu: some View {
        Menu {
            ForEach(PainiteKit.Operation.allCases) { op in
                Button(op.title) { model.addStep(op) }
                    .disabled(!model.canAdd(op))
            }
        } label: {
            Label("Add Step", systemImage: "plus")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .disabled(model.steps.count >= Recipe.maxSteps)
        .help(model.steps.count >= Recipe.maxSteps ? "Up to \(Recipe.maxSteps) steps" : "Add an enhancement step")
    }

    /// Position among enabled steps, which is how the server numbers them.
    private func enabledIndex(of step: RecipeStep) -> Int? {
        guard step.enabled else { return nil }
        return model.steps.filter(\.enabled).firstIndex { $0.id == step.id }
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
            ProgressView(value: p) { Text("Uploading photo… \(Int(p * 100))%") }
        case .ready: Text("Photo uploaded to server")
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

/// One step: operation, strength, model, and controls to toggle, reorder, or remove it.
struct StepCard: View {
    @ObservedObject var model: DocumentModel
    @ObservedObject var server = ServerStore.shared
    let step: RecipeStep
    let index: Int
    let count: Int
    let enabledIndex: Int?

    private var binding: Binding<RecipeStep> {
        Binding(
            get: { model.steps.first { $0.id == step.id } ?? step },
            set: { new in
                if let i = model.steps.firstIndex(where: { $0.id == step.id }) { model.steps[i] = new }
            })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text("\(index + 1)")
                    .font(.caption.weight(.bold).monospacedDigit())
                    .frame(width: 18, height: 18)
                    .background(Circle().fill(step.enabled ? Color.accentColor.opacity(0.25) : Color.secondary.opacity(0.15)))
                Toggle(isOn: binding.enabled) {
                    Text(step.operation.title).font(.callout.weight(.semibold))
                }
                .toggleStyle(.checkbox)
                .help(step.enabled ? "Turn this step off without removing it" : "Turn this step on")
                Spacer(minLength: 4)
                Button { model.moveStep(step.id, by: -1) } label: { Image(systemName: "chevron.up") }
                    .disabled(index == 0).help("Move up")
                Button { model.moveStep(step.id, by: 1) } label: { Image(systemName: "chevron.down") }
                    .disabled(index == count - 1).help("Move down")
                Button { model.removeStep(step.id) } label: { Image(systemName: "xmark") }
                    .help("Remove step")
            }
            .buttonStyle(.borderless)

            if model.takesStrength(step.operation) {
                HStack {
                    Slider(value: binding.strength, in: model.strengthRange(step.operation)) {
                        Text("Strength")
                    }
                    .controlSize(.small)
                    Text("\(Int((step.strength * 100).rounded()))")
                        .font(.callout.monospacedDigit())
                        .frame(width: 28, alignment: .trailing)
                }
                .disabled(!step.enabled)
            } else if step.operation.scale > 1 {
                Text("Doubles the width and height.").font(.caption).foregroundStyle(.secondary)
            }

            if let i = enabledIndex, let id = model.modelID(forEnabledStep: i) {
                HStack(spacing: 6) {
                    Text(id).font(.caption.monospaced()).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                    if server.capabilities?.model(id)?.placeholder == true {
                        Text("placeholder")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Capsule().fill(Color.orange.opacity(0.25)))
                            .help("A stand-in model for testing; it does not really enhance.")
                    }
                }
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.secondary.opacity(0.2)))
        .opacity(step.enabled ? 1 : 0.6)
    }
}

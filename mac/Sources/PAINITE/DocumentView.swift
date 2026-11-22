import PainiteKit
import SwiftUI

/// The editing window's content: the compare canvas and the floating inspector.
/// The toolbar lives in DocumentWindowController.
struct DocumentView: View {
    @ObservedObject var model: DocumentModel
    @ObservedObject var server = ServerStore.shared

    var body: some View {
        HStack(spacing: 0) {
            canvas
            if model.inspectorVisible {
                InspectorView(model: model)
                    .frame(width: 296)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.08), lineWidth: 0.5))
                    .shadow(color: .black.opacity(0.12), radius: 12, y: 4)
                    .padding(8)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.2), value: model.inspectorVisible)
        .tint(.painite)
        .frame(minWidth: 960, minHeight: 600)
    }

    private var canvas: some View {
        ZStack(alignment: .top) {
            CompareViewRepresentable(model: model)
            if model.image == nil {
                Group {
                    if let err = model.loadError {
                        Text(err).foregroundStyle(.secondary).padding()
                    } else {
                        ProgressView("Opening…")
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                HStack(alignment: .top) {
                    CanvasTag { Text("Before").fontWeight(.semibold) }
                    Spacer()
                    AfterTag(model: model)
                }
                .padding(12)
            }
            if let banner = serverBanner {
                banner.padding(.top, 52)
            }
        }
    }

    /// Shown over the photo when the server can't be reached.
    private var serverBanner: AnyView? {
        switch server.status {
        case .failed(let msg):
            return AnyView(CanvasTag {
                HStack(spacing: 6) {
                    Image(systemName: "wifi.exclamationmark").foregroundStyle(.orange)
                    Text("Can't reach the server").fontWeight(.semibold)
                    Text(msg).foregroundStyle(.white.opacity(0.7)).lineLimit(1)
                    Button("Try Again") { server.refresh() }.buttonStyle(.link)
                }
            }.frame(maxWidth: 560))
        case .unconfigured:
            return AnyView(CanvasTag {
                HStack(spacing: 6) {
                    Text("No server set up").fontWeight(.semibold)
                    Button("Connect to Server…") { AppDelegate.shared?.showSettings(nil) }.buttonStyle(.link)
                }
            })
        default:
            return nil
        }
    }
}

/// A small dark capsule that sits on the photo.
struct CanvasTag<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        HStack(spacing: 6) { content }
            .font(.system(size: 11.5))
            .foregroundStyle(.white)
            .padding(.horizontal, 10)
            .frame(height: 24)
            .background(.black.opacity(0.55), in: Capsule())
            .background(.ultraThinMaterial.opacity(0.6), in: Capsule())
            .environment(\.colorScheme, .dark)
    }
}

/// "After │ Denoise 60 · Sharpen 25", "After │ Updating · 40%", or a failure with Try Again.
struct AfterTag: View {
    @ObservedObject var model: DocumentModel

    var body: some View {
        CanvasTag {
            Text("After").fontWeight(.semibold)
            Rectangle().fill(.white.opacity(0.3)).frame(width: 1, height: 12)
            switch model.afterStatus {
            case .noSteps:
                Text("No steps").foregroundStyle(.white.opacity(0.7))
            case .current(let label):
                Text(label).foregroundStyle(.white.opacity(0.75)).lineLimit(1)
                    .help(serverTime)
            case .updating(let p):
                ProgressView().controlSize(.mini)
                Text(p.map { "Updating · \(Int(($0 * 100).rounded()))%" } ?? "Updating")
                    .foregroundStyle(.white.opacity(0.75)).monospacedDigit()
            case .failed(let msg):
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                Text("Couldn't render").foregroundStyle(.white.opacity(0.85)).help(msg)
                Button("Try Again") { model.regeneratePreview() }
                    .buttonStyle(.link)
            }
        }
    }

    private var serverTime: String {
        guard let ms = model.preview?.serverMs else { return "" }
        return "Rendered on the server in \(Int(ms)) ms"
    }
}

// MARK: Inspector

struct InspectorView: View {
    @ObservedObject var model: DocumentModel
    @ObservedObject var server = ServerStore.shared

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 14) {
                if model.handoff.isLightroom { lightroomCard }
                presetRow
                stepsHeader
            }
            .padding([.horizontal, .top], 14)
            .padding(.bottom, 8)

            StepList(model: model)
                .disabled(model.export.isRunning)

            exportStatus
                .padding(.horizontal, 14)
            footer
        }
        .disabled(model.loadError != nil)
    }

    private var lightroomCard: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "arrow.up.forward.app").foregroundStyle(Color.painite)
            VStack(alignment: .leading, spacing: 2) {
                Text("Opened from Lightroom").font(.callout.weight(.semibold))
                Text(model.canSaveInPlace
                     ? "Save to Lightroom replaces the edit copy. Then click Finish in Lightroom."
                     : "Lightroom's copy can't be replaced. Export, then use Browse… in Lightroom.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.painite.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private var presetRow: some View {
        HStack(spacing: 8) {
            Menu {
                ForEach(Preset.builtIn) { p in
                    Button(p.name) { model.applyPreset(p) }
                }
            } label: {
                let t = model.presetTitle
                if let from = t.from {
                    Text("\(t.name)  ").fontWeight(.semibold) + Text("· from \(from)").foregroundColor(.secondary)
                } else {
                    Text(t.name).fontWeight(.semibold)
                }
            }
            .menuStyle(.button)
            .frame(maxWidth: .infinity)
            .help("Presets set the whole list of steps")

            Menu {
                Button("Reset to \(model.basePreset ?? "Preset")") { model.resetToPreset() }
                    .disabled(model.basePreset == nil || model.currentPreset != nil)
            } label: {
                Image(systemName: "ellipsis")
            }
            .menuStyle(.button)
            .menuIndicator(.hidden)
            .fixedSize()
        }
        .controlSize(.large)
    }

    private var stepsHeader: some View {
        HStack {
            Text("STEPS")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.tertiary)
                .tracking(0.5)
            Spacer()
            Menu {
                ForEach(PainiteKit.Operation.addable) { op in
                    Button {
                        model.addStep(op)
                    } label: {
                        Label(op.shortTitle, systemImage: op.symbolName)
                    }
                    .disabled(!model.canAdd(op))
                }
            } label: {
                Label("Add Step", systemImage: "plus")
            }
            .menuStyle(.button)
            .controlSize(.small)
            .fixedSize()
            .disabled(model.steps.count >= Recipe.maxSteps || model.export.isRunning)
            .help(model.steps.count >= Recipe.maxSteps ? "Up to \(Recipe.maxSteps) steps" : "Add an enhancement step")
        }
    }

    @ViewBuilder private var exportStatus: some View {
        switch model.export {
        case .idle:
            EmptyView()
        case .running(let phase, let p):
            VStack(alignment: .leading, spacing: 6) {
                if let p {
                    ProgressView(value: p) { Text(phase).font(.callout) }
                } else {
                    ProgressView { Text(phase).font(.callout) }.progressViewStyle(.linear)
                }
                Button("Cancel") { model.cancelExport() }
                    .keyboardShortcut(.cancelAction)
                    .controlSize(.small)
            }
            .padding(.vertical, 10)
        case .done(let url, let inPlace):
            VStack(alignment: .leading, spacing: 4) {
                Label("Saved \(url.lastPathComponent)", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.callout)
                if inPlace {
                    Text("Saved. Click Finish in Lightroom.").font(.caption).foregroundStyle(.secondary)
                } else if model.handoff.isLightroom {
                    Text("In Lightroom, click Browse… (or Select… for a RAW photo) and choose this file.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                } else {
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                        .buttonStyle(.link).font(.caption)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 10)
        case .failed(let msg):
            Text(msg).font(.callout).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 10)
        }
    }

    private var footer: some View {
        VStack(spacing: 5) {
            Divider().padding(.bottom, 6)
            if case .uploading(let p) = model.upload {
                row("Upload") { ProgressView(value: p).frame(width: 120) }
            } else if case .failed(let msg) = model.upload {
                Text(msg).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            row("Output") { Text(outputLabel).monospacedDigit() }
            row("Server") {
                HStack(spacing: 5) {
                    Circle().fill(serverColor).frame(width: 7, height: 7)
                    Text(server.status == .unconfigured ? "Not set up" : server.hostLabel)
                }
                .help(serverHelp)
                .onTapGesture { AppDelegate.shared?.showSettings(nil) }
            }
            if server.usesPlaceholderModels {
                Text("Test models in use").foregroundStyle(.orange).frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .padding(.bottom, 12)
    }

    private func row<V: View>(_ title: String, @ViewBuilder _ value: () -> V) -> some View {
        HStack {
            Text(title)
            Spacer()
            value().foregroundStyle(.primary)
        }
    }

    private var outputLabel: String {
        guard model.imageSize.width > 0 else { return "–" }
        let s = CGFloat(Recipe.scale(model.signature))
        let w = Int(model.imageSize.width * s), h = Int(model.imageSize.height * s)
        let fmt = model.handoff.isLightroom ? (model.sourceFormat ?? .tiff) : model.exportFormat
        return "\(w) × \(h) · \(fmt == .jpeg ? "JPEG" : "TIFF")"
    }

    private var serverColor: Color {
        switch server.status {
        case .connected: .green
        case .checking: .yellow
        case .failed: .red
        case .unconfigured: .gray
        }
    }

    private var serverHelp: String {
        switch server.status {
        case .connected: "Connected to \(server.hostLabel). Click for server settings."
        case .checking: "Connecting to \(server.hostLabel)…"
        case .failed(let msg): msg
        case .unconfigured: "Set up the server in Settings"
        }
    }
}

// MARK: Steps

/// The ordered, reorderable list of steps. Drag a card to move it.
struct StepList: View {
    @ObservedObject var model: DocumentModel

    var body: some View {
        VStack(spacing: 0) {
            if model.steps.isEmpty {
                VStack(spacing: 6) {
                    Image(systemName: "wand.and.sparkles").font(.title2).foregroundStyle(.tertiary)
                    Text("Pick a preset or add a step.").font(.callout).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 28)
                Spacer(minLength: 0)
            } else {
                List {
                    ForEach(model.steps) { step in
                        StepCard(model: model, step: step)
                            .listRowInsets(EdgeInsets(top: 4, leading: 14, bottom: 4, trailing: 14))
                            .listRowSeparator(.hidden)
                            .listRowBackground(Color.clear)
                    }
                    .onMove { from, to in model.moveSteps(from: from, to: to) }
                    Text(model.steps.count > 1 ? "Steps run top to bottom · drag to reorder" : "Steps run top to bottom")
                        .font(.caption).foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity)
                        .listRowSeparator(.hidden)
                        .listRowBackground(Color.clear)
                        .moveDisabled(true)
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .environment(\.defaultMinListRowHeight, 10)
            }
        }
    }
}

/// One step: icon, name, on/off switch and its control. The grip and the … menu
/// show on hover.
struct StepCard: View {
    @ObservedObject var model: DocumentModel
    let step: RecipeStep
    @State private var hovering = false

    private var binding: Binding<RecipeStep> {
        Binding(
            get: { model.steps.first { $0.id == step.id } ?? step },
            set: { new in
                if let i = model.steps.firstIndex(where: { $0.id == step.id }) { model.steps[i] = new }
            })
    }

    private var index: Int { model.steps.firstIndex { $0.id == step.id } ?? 0 }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "line.3.horizontal")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.tertiary)
                    .frame(width: 10)
                    .opacity(hovering && model.steps.count > 1 ? 1 : 0)
                    .help("Drag to reorder")
                iconTile
                Text(step.operation.shortTitle)
                    .font(.system(size: 13, weight: step.enabled ? .semibold : .medium))
                    .foregroundStyle(step.enabled ? .primary : .secondary)
                    .lineLimit(1)
                    .fixedSize()
                Spacer(minLength: 4)
                if !hovering, let meta { Text(meta).font(.caption).foregroundStyle(.tertiary).lineLimit(1) }
                if hovering { moreMenu }
                Toggle("", isOn: binding.enabled)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .labelsHidden()
                    .help(step.enabled ? "Turn this step off without removing it" : "Turn this step on")
            }
            .frame(height: 26)

            if step.enabled, hasBody {
                VStack(alignment: .leading, spacing: 8) { controls }
                    .padding(.leading, 52)
            }
        }
        .padding(.vertical, 8)
        .padding(.leading, 4)
        .padding(.trailing, 10)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous)
            .fill(Color.primary.opacity(hovering ? 0.075 : 0.045)))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
            .strokeBorder(Color.primary.opacity(0.07), lineWidth: 0.5))
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .contextMenu { menuItems }
    }

    private var iconTile: some View {
        Image(systemName: step.operation.symbolName)
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(step.enabled ? Color.painite : Color.secondary)
            .frame(width: 26, height: 26)
            .background(RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(step.enabled ? Color.painite.opacity(0.16) : Color.primary.opacity(0.07)))
    }

    private var hasBody: Bool {
        step.operation.isUnblur || model.takesStrength(step.operation) || step.operation == .face
    }

    /// Right-aligned detail in the header row.
    private var meta: String? {
        if step.operation.scale > 1, model.imageSize.width > 0 {
            return "\(Int(model.imageSize.width) * 2) × \(Int(model.imageSize.height) * 2)"
        }
        return nil
    }

    @ViewBuilder private var controls: some View {
        if step.operation.isUnblur {
            Picker("Blur", selection: binding.operation) {
                Text("Motion").tag(PainiteKit.Operation.unblurMotion)
                Text("Focus").tag(PainiteKit.Operation.unblurFocus)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.small)
            .fixedSize()
            .help("Motion fixes camera shake and moving subjects. Focus fixes a photo that is out of focus.")
        }
        if model.takesStrength(step.operation) {
            HStack(spacing: 10) {
                Slider(value: binding.strength, in: model.strengthRange(step.operation),
                       onEditingChanged: { editing in model.isAdjusting = editing })
                .controlSize(.small)
                Text("\(Int((step.strength * 100).rounded()))")
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 26, alignment: .trailing)
            }
        }
        if step.operation == .face {
            faceControls
        }
    }

    @ViewBuilder private var faceControls: some View {
        let faces = model.preview.flatMap { $0.signature == model.signature ? $0.faces : nil }
        HStack(spacing: 6) {
            Toggle("Show faces", isOn: $model.showFaces)
                .toggleStyle(.checkbox)
                .controlSize(.small)
                .help("Outline the faces this step found: green if enhanced, dashed orange if skipped")
            Spacer(minLength: 4)
            if let faces {
                Text(faceSummary(faces)).font(.caption).foregroundStyle(.secondary)
                    .help(skipNote(faces))
            }
        }
    }

    private func faceSummary(_ faces: [FaceInfo]) -> String {
        if faces.isEmpty { return "None found" }
        let enhanced = faces.filter { $0.status == .enhanced }.count
        let skipped = faces.count - enhanced
        return skipped > 0 ? "\(enhanced) enhanced · \(skipped) skipped" : "\(enhanced) enhanced"
    }

    private func skipNote(_ faces: [FaceInfo]) -> String {
        let small = faces.filter { $0.status == .tooSmall }.count
        let large = faces.filter { $0.status == .tooLarge }.count
        var parts: [String] = []
        if small > 0 { parts.append("\(small) too small: the model would invent a different face") }
        if large > 0 { parts.append("\(large) too large: the photo already has more real detail than the model adds") }
        return parts.isEmpty ? "" : "Skipped (dashed): " + parts.joined(separator: "; ") + "."
    }

    private var moreMenu: some View {
        Menu { menuItems } label: {
            Image(systemName: "ellipsis").font(.system(size: 11, weight: .semibold))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .foregroundStyle(.secondary)
    }

    @ViewBuilder private var menuItems: some View {
        Button("Move Up") { model.moveStep(step.id, by: -1) }.disabled(index == 0)
        Button("Move Down") { model.moveStep(step.id, by: 1) }.disabled(index == model.steps.count - 1)
        Divider()
        Button("Duplicate") { model.duplicateStep(step.id) }.disabled(!model.canDuplicate(step.id))
        Button("Solo") { model.soloStep(step.id) }
        Divider()
        Button("Remove", role: .destructive) { model.removeStep(step.id) }
    }
}

extension Color {
    /// The brand accent from the logo: #F58A3D in dark mode, #E0582F in light mode for contrast.
    static let painite = Color(nsColor: .painite)
}

extension NSColor {
    static let painite = NSColor(name: "painite") { appearance in
        appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            ? NSColor(srgbRed: 0xF5 / 255, green: 0x8A / 255, blue: 0x3D / 255, alpha: 1)
            : NSColor(srgbRed: 0xE0 / 255, green: 0x58 / 255, blue: 0x2F / 255, alpha: 1)
    }
}

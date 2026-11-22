import AppKit
import PainiteKit
import QuickLookThumbnailing
import SwiftUI
import UniformTypeIdentifiers

struct SettingsView: View {
    @ObservedObject var server = ServerStore.shared
    @State private var endpoint = ""
    @State private var token = ""
    @State private var caFile = ""
    @State private var saveError: String?
    @State private var fingerprint = ""
    @State private var trustMessage: String?
    @State private var trusting = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Form {
                TextField("Server", text: $endpoint, prompt: Text("https://spark1.local:8443"))
                SecureField("Token", text: $token, prompt: Text("from painite-server token create"))
                HStack {
                    TextField("CA certificate", text: $caFile, prompt: Text("optional PEM file"))
                    Button("Choose…", action: chooseCA)
                }
            }
            if endpoint.lowercased().hasPrefix("https://") {
                HStack {
                    TextField("CA fingerprint", text: $fingerprint, prompt: Text("SHA256:… from painite-server admin fingerprint"))
                    Button("Trust", action: trust).disabled(trusting)
                }
                if let trustMessage {
                    Text(trustMessage).font(.callout).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                }
            }
            Text("Shared with the painite command-line tool: \(server.configPath)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)

            HStack {
                Button("Save and Connect", action: save)
                    .keyboardShortcut(.defaultAction)
                Button("Check Again") { server.refresh() }
                    .disabled(server.client == nil)
            }
            if let saveError {
                Text(saveError).foregroundStyle(.red)
            }

            Divider()
            status
        }
        .padding(20)
        .frame(width: 520)
        .onAppear(perform: loadFields)
    }

    @ViewBuilder private var status: some View {
        switch server.status {
        case .unconfigured:
            Text(server.configError ?? "Enter the server address and a token.").foregroundStyle(.secondary)
        case .checking:
            ProgressView("Connecting…").controlSize(.small)
        case .failed(let msg):
            Label(msg, systemImage: "xmark.octagon.fill").foregroundStyle(.red)
        case .connected:
            VStack(alignment: .leading, spacing: 6) {
                Label("Connected", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                if let h = server.health {
                    Text("Server \(h.version) · GPU: \(h.gpu.available ? (h.gpu.name ?? "available") : "none") · queue \(h.queue.queued + h.queue.running)/\(h.queue.capacity)")
                        .font(.callout)
                }
                if let caps = server.capabilities {
                    Text("Models").font(.callout.weight(.semibold)).padding(.top, 4)
                    ForEach(caps.models) { m in
                        HStack {
                            Text(m.id).font(.callout.monospaced())
                            if m.placeholder == true {
                                Text("placeholder").font(.caption).foregroundStyle(.orange)
                            }
                        }
                    }
                }
            }
        }
    }

    private func loadFields() {
        endpoint = server.config.endpoint
        token = server.config.token
        caFile = server.config.caFile
    }

    private func save() {
        do {
            try server.update(ClientConfig(endpoint: endpoint.trimmingCharacters(in: .whitespaces),
                                           token: token.trimmingCharacters(in: .whitespacesAndNewlines),
                                           caFile: caFile.trimmingCharacters(in: .whitespaces)))
            saveError = nil
        } catch {
            saveError = error.localizedDescription
        }
    }

    /// Like `painite trust`: fetch the server's CA, check its fingerprint, save it, and use it.
    private func trust() {
        trusting = true
        trustMessage = "Fetching the server's CA…"
        let endpoint = endpoint
        let wanted = fingerprint
        Task {
            defer { trusting = false }
            do {
                let (pem, got) = try await CATrust.fetch(endpoint: endpoint)
                if wanted.trimmingCharacters(in: .whitespaces).isEmpty {
                    trustMessage = "The server presents \(got). Compare it with `painite-server admin fingerprint` on the server, paste it above, and click Trust."
                    return
                }
                guard CATrust.normalize(wanted) == got else {
                    trustMessage = "Fingerprint mismatch: the server presented \(got). Not trusted."
                    return
                }
                let path = CATrust.defaultCAPath()
                try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
                try pem.write(to: path, options: .atomic)
                caFile = path.path
                save()
                trustMessage = saveError == nil ? "Trusted. Saved the CA to \(path.path)." : nil
            } catch {
                trustMessage = error.localizedDescription
            }
        }
    }

    private func chooseCA() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            caFile = url.path
        }
    }
}

struct WelcomeView: View {
    @ObservedObject var server = ServerStore.shared
    var open: () -> Void
    var openURLs: ([URL]) -> Void
    @State private var targeted = false
    @State private var recent: [URL] = NSDocumentController.shared.recentDocumentURLs

    var body: some View {
        VStack(spacing: 18) {
            dropZone
            if !recent.isEmpty { recentRow }
        }
        .padding(20)
        .frame(width: 720)
        .tint(.painite)
        .onDrop(of: [.fileURL], isTargeted: $targeted) { providers in
            for provider in providers {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    if let url { DispatchQueue.main.async { openURLs([url]) } }
                }
            }
            return true
        }
        .onAppear { recent = NSDocumentController.shared.recentDocumentURLs }
    }

    private var unconfigured: Bool { server.status == .unconfigured }

    private var dropZone: some View {
        VStack(spacing: 12) {
            if let icon = NSApp.applicationIconImage {
                Image(nsImage: icon).resizable().frame(width: 72, height: 72)
            }
            Text(unconfigured ? "Connect to your server" : "Drop a photo here")
                .font(.system(size: 22, weight: .bold))
            Text(unconfigured ? "PAINITE enhances photos on your PAINITE server. Set it up to start."
                              : "JPEG or TIFF, up to 16-bit. Or open one from Lightroom with Edit in.")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Group {
                if unconfigured {
                    Button("Connect to Server…") { AppDelegate.shared?.showSettings(nil) }
                } else {
                    Button("Open Photo…", action: open)
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .keyboardShortcut(.defaultAction)
            .padding(.top, 4)
            statusLine
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 36)
        .background(RoundedRectangle(cornerRadius: 16, style: .continuous)
            .fill(targeted ? Color.painite.opacity(0.08) : Color.clear))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
            .strokeBorder(targeted ? Color.painite : Color.secondary.opacity(0.4),
                          style: StrokeStyle(lineWidth: targeted ? 2 : 1, dash: [6, 4])))
    }

    /// Only shown when something is wrong.
    @ViewBuilder private var statusLine: some View {
        switch server.status {
        case .failed(let msg):
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                Text(msg).lineLimit(2)
                Button("Settings…") { AppDelegate.shared?.showSettings(nil) }.buttonStyle(.link)
            }
            .font(.callout).foregroundStyle(.secondary)
            .padding(.horizontal, 24)
        case .connected where server.usesPlaceholderModels:
            Text("Test models in use").font(.callout).foregroundStyle(.orange)
        default:
            EmptyView()
        }
    }

    private var recentRow: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("RECENT").font(.caption.weight(.semibold)).foregroundStyle(.tertiary).tracking(0.5)
                Spacer()
                Button("Clear") {
                    NSDocumentController.shared.clearRecentDocuments(nil)
                    recent = []
                }
                .buttonStyle(.link).font(.caption)
            }
            HStack(alignment: .top, spacing: 14) {
                ForEach(recent.prefix(4), id: \.self) { url in
                    RecentPhoto(url: url) { openURLs([url]) }
                }
                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, 4)
    }
}

/// A recent photo: Quick Look thumbnail and name.
struct RecentPhoto: View {
    let url: URL
    let open: () -> Void
    @State private var thumb: NSImage?

    var body: some View {
        Button(action: open) {
            VStack(alignment: .leading, spacing: 6) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.primary.opacity(0.06))
                    if let thumb {
                        Image(nsImage: thumb).resizable().aspectRatio(contentMode: .fill)
                    } else {
                        Image(systemName: "photo").foregroundStyle(.tertiary)
                    }
                }
                .frame(width: 156, height: 96)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                Text(url.lastPathComponent).font(.callout).lineLimit(1).truncationMode(.middle)
                    .frame(width: 156, alignment: .leading)
            }
        }
        .buttonStyle(.plain)
        .help(url.path)
        .task {
            let size = CGSize(width: 312, height: 192)
            let request = QLThumbnailGenerator.Request(fileAt: url, size: size, scale: 1, representationTypes: .thumbnail)
            if let rep = try? await QLThumbnailGenerator.shared.generateBestRepresentation(for: request) {
                thumb = rep.nsImage
            }
        }
    }
}

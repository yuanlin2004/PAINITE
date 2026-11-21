import AppKit
import PainiteKit
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

    var body: some View {
        VStack(spacing: 18) {
            if let icon = NSApp.applicationIconImage {
                Image(nsImage: icon).resizable().frame(width: 96, height: 96)
            }
            Text("PAINITE").font(.largeTitle.weight(.semibold))
            Text("Open a JPEG or TIFF photo to denoise, sharpen, or upscale it.")
                .foregroundStyle(.secondary)
            Button("Open Photo…", action: open)
                .keyboardShortcut(.defaultAction)
                .controlSize(.large)
            Group {
                switch server.status {
                case .unconfigured:
                    Button("Set up the server…") { AppDelegate.shared?.showSettings(nil) }.buttonStyle(.link)
                case .checking:
                    Text("Connecting to \(server.hostLabel)…")
                case .connected:
                    Text("Connected to \(server.hostLabel)\(server.usesPlaceholderModels ? " · placeholder models" : "")")
                case .failed(let msg):
                    VStack(spacing: 4) {
                        Text(msg).multilineTextAlignment(.center)
                        Button("Settings…") { AppDelegate.shared?.showSettings(nil) }.buttonStyle(.link)
                    }
                }
            }
            .font(.callout)
            .foregroundStyle(.secondary)
        }
        .padding(40)
        .frame(width: 480, height: 400)
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for provider in providers {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    if let url { DispatchQueue.main.async { openURLs([url]) } }
                }
            }
            return true
        }
    }
}

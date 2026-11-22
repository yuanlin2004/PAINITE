import Foundation
import PainiteKit

/// The app-wide connection: configuration, client, and what the server offers.
@MainActor
final class ServerStore: ObservableObject {
    static let shared = ServerStore()

    enum Status: Equatable {
        case unconfigured
        case checking
        case connected
        case failed(String)
    }

    @Published private(set) var config: ClientConfig
    @Published private(set) var configError: String?
    @Published private(set) var status: Status = .unconfigured
    @Published private(set) var health: Health?
    @Published private(set) var capabilities: Capabilities?

    private(set) var client: APIClient?
    private var refreshTask: Task<Void, Never>?

    private init() {
        do {
            config = try ClientConfig.load()
        } catch {
            config = ClientConfig()
            configError = error.localizedDescription
        }
        rebuildClient()
        refresh()
    }

    var configPath: String { ClientConfig.defaultPath.path }

    /// Saves new settings to the shared config file and reconnects.
    func update(_ new: ClientConfig) throws {
        try new.save()
        config = new
        configError = nil
        rebuildClient()
        refresh()
    }

    func refresh() {
        refreshTask?.cancel()
        guard let client else {
            status = .unconfigured
            return
        }
        status = .checking
        refreshTask = Task {
            do {
                let h = try await client.health()
                let caps = try await client.capabilities()
                guard !Task.isCancelled else { return }
                health = h
                capabilities = caps
                status = .connected
            } catch is CancellationError {
            } catch {
                guard !Task.isCancelled else { return }
                health = nil
                status = .failed(error.localizedDescription)
            }
        }
    }

    func requireClient() throws -> APIClient {
        if let client { return client }
        throw APIError(code: "not_configured", message: configError ?? "No server is configured. Add its address and token in Settings (⌘,).")
    }

    /// Current capabilities, fetching them if the last check failed.
    func currentCapabilities() async throws -> Capabilities {
        if let capabilities { return capabilities }
        let client = try requireClient()
        let caps = try await client.retrying { try await client.capabilities() }
        capabilities = caps
        status = .connected
        return caps
    }

    /// Whether any operation's default model is a stand-in that does not really enhance.
    var usesPlaceholderModels: Bool {
        !(capabilities?.placeholderOperations.isEmpty ?? true)
    }

    var hostLabel: String {
        URL(string: config.endpoint)?.host ?? config.endpoint
    }

    private func rebuildClient() {
        guard config.isComplete else {
            client = nil
            capabilities = nil
            return
        }
        do {
            client = try APIClient(config: config)
            configError = nil
        } catch {
            client = nil
            configError = error.localizedDescription
        }
        capabilities = nil
    }
}

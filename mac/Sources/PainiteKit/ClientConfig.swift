// Server address and credentials, shared with the painite CLI.
//
// The CLI keeps them in <user config dir>/painite/config.json, which on macOS is
// ~/Library/Application Support/painite/config.json (Go's os.UserConfigDir).
// The app reads and writes the same file, so configuring either client
// configures both. PAINITE_CONFIG overrides the path, as it does for the CLI.

import Foundation

public struct ClientConfig: Equatable, Sendable {
    public var endpoint: String
    public var token: String
    public var caFile: String

    public init(endpoint: String = "", token: String = "", caFile: String = "") {
        self.endpoint = endpoint
        self.token = token
        self.caFile = caFile
    }

    public var isComplete: Bool { !endpoint.isEmpty && !token.isEmpty }

    public static var defaultPath: URL {
        if let p = ProcessInfo.processInfo.environment["PAINITE_CONFIG"], !p.isEmpty {
            return URL(fileURLWithPath: p)
        }
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return support.appendingPathComponent("painite/config.json")
    }

    /// Reads the file. A missing file is an empty config.
    public static func load(from path: URL = defaultPath) throws -> ClientConfig {
        let raw = try readObject(path)
        return ClientConfig(
            endpoint: raw["endpoint"] as? String ?? "",
            token: raw["token"] as? String ?? "",
            caFile: raw["ca_file"] as? String ?? ""
        )
    }

    /// Writes the file with owner-only permissions, keeping keys this app does not know.
    public func save(to path: URL = ClientConfig.defaultPath) throws {
        var raw = try ClientConfig.readObject(path)
        for (key, value) in [("endpoint", endpoint), ("token", token), ("ca_file", caFile)] {
            raw[key] = value.isEmpty ? nil : value
        }
        let data = try JSONSerialization.data(withJSONObject: raw, options: [.prettyPrinted, .sortedKeys])
        let fm = FileManager.default
        try fm.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true,
                               attributes: [.posixPermissions: 0o700])
        let tmp = path.deletingLastPathComponent().appendingPathComponent(".config-\(UUID().uuidString).tmp")
        guard fm.createFile(atPath: tmp.path, contents: data + Data("\n".utf8), attributes: [.posixPermissions: 0o600]) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: tmp.path])
        }
        if rename(tmp.path, path.path) != 0 {
            let err = POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            try? fm.removeItem(at: tmp)
            throw err
        }
    }

    private static func readObject(_ path: URL) throws -> [String: Any] {
        guard let data = try? Data(contentsOf: path) else { return [:] }
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw APIError(code: "not_configured", message: "\(path.path) is not a JSON object")
        }
        return obj
    }
}

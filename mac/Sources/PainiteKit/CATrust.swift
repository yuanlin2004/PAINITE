// Pinning the server's private CA, as `painite trust` does in the CLI.
//
// `painite-server admin init` creates a private CA and serves it at
// GET /v1/ca.pem. The client fetches it without verifying TLS, checks its
// SHA-256 fingerprint against the one the operator got from
// `painite-server admin fingerprint`, and only then saves it as ca_file.

import CryptoKit
import Foundation
import Security

public enum CATrust {
    /// Downloads the server's CA certificate. Not verified: the caller must
    /// compare the returned fingerprint with the operator's before trusting it.
    public static func fetch(endpoint: String) async throws -> (pem: Data, fingerprint: String) {
        var base = endpoint.trimmingCharacters(in: .whitespaces)
        while base.hasSuffix("/") { base.removeLast() }
        guard base.lowercased().hasPrefix("https://"), let url = URL(string: base + "/v1/ca.pem") else {
            throw APIError(code: "not_configured", message: "The server address must start with https:// to trust its certificate.")
        }
        let delegate = AcceptAnyServer()
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let (data, resp): (Data, URLResponse)
        do {
            (data, resp) = try await session.data(from: url)
        } catch {
            throw APIClient.wrap(error)
        }
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else { throw APIClient.error(status: status, body: data) }
        guard let der = firstCertificateDER(in: data) else {
            throw APIError(code: "tls_untrusted", message: "The server sent no PEM certificate.")
        }
        let pem = "-----BEGIN CERTIFICATE-----\n" + der.base64EncodedString(options: [.lineLength64Characters, .endLineWithLineFeed])
            + "\n-----END CERTIFICATE-----\n"
        return (Data(pem.utf8), fingerprint(der: der))
    }

    /// "SHA256:<hex>" of a certificate's DER bytes, the format `painite-server admin fingerprint` prints.
    public static func fingerprint(der: Data) -> String {
        "SHA256:" + SHA256.hash(data: der).map { String(format: "%02x", $0) }.joined()
    }

    /// Accepts "SHA256:ab12…", "ab:12:…", any case.
    public static func normalize(_ fingerprint: String) -> String {
        var s = fingerprint.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if s.hasPrefix("sha256:") { s.removeFirst(7) }
        return "SHA256:" + s.replacingOccurrences(of: ":", with: "")
    }

    /// Only the first certificate counts: anything appended after it would be
    /// trusted without being covered by the fingerprint.
    static func firstCertificateDER(in pem: Data) -> Data? {
        guard let text = String(data: pem, encoding: .utf8),
              let begin = text.range(of: "-----BEGIN CERTIFICATE-----"),
              let end = text.range(of: "-----END CERTIFICATE-----", range: begin.upperBound..<text.endIndex) else { return nil }
        let b64 = text[begin.upperBound..<end.lowerBound].filter { !$0.isWhitespace }
        guard let der = Data(base64Encoded: String(b64)), SecCertificateCreateWithData(nil, der as CFData) != nil else { return nil }
        return der
    }

    /// Where the app saves a trusted CA: next to the shared config file, like the CLI.
    public static func defaultCAPath(configPath: URL = ClientConfig.defaultPath) -> URL {
        configPath.deletingLastPathComponent().appendingPathComponent("ca.pem")
    }
}

private final class AcceptAnyServer: NSObject, URLSessionDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        if challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
           let trust = challenge.protectionSpace.serverTrust {
            completionHandler(.useCredential, URLCredential(trust: trust))
        } else {
            completionHandler(.performDefaultHandling, nil)
        }
    }
}

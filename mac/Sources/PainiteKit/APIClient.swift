// Client for the PAINITE service contract (api/openapi.yaml).
// Mirrors cli/internal/api/client.go: same retry rules and error codes.

import Foundation
import Security

public final class APIClient: @unchecked Sendable {
    public let baseURL: URL
    private let token: String
    private let session: URLSession
    private let trust: TrustDelegate
    /// Retries for retryable failures, with exponential backoff capped at 10 s.
    public var retries = 6

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.outputFormatting = [.sortedKeys]
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    /// - Parameter protocolClasses: test hook for stubbing the network.
    public init(config: ClientConfig, protocolClasses: [AnyClass]? = nil) throws {
        let endpoint = config.endpoint.trimmingCharacters(in: .whitespaces)
        guard !endpoint.isEmpty else {
            throw APIError(code: "not_configured", message: "No server is configured. Add its address and token in Settings.")
        }
        guard var url = URL(string: endpoint), let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https", url.host != nil else {
            throw APIError(code: "not_configured", message: "The server address must look like https://spark1.local:8443")
        }
        while url.path.hasSuffix("/") { url.deleteLastPathComponent() }
        baseURL = url.appendingPathComponent("v1")
        token = config.token.trimmingCharacters(in: .whitespacesAndNewlines)

        var anchors: [SecCertificate] = []
        if !config.caFile.isEmpty {
            anchors = try APIClient.loadCertificates(path: config.caFile)
        }
        trust = TrustDelegate(anchors: anchors)
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 60
        cfg.httpMaximumConnectionsPerHost = 6
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.waitsForConnectivity = false
        if let protocolClasses { cfg.protocolClasses = protocolClasses }
        session = URLSession(configuration: cfg, delegate: trust, delegateQueue: nil)
    }

    deinit { session.finishTasksAndInvalidate() }

    // MARK: Routes

    public func health() async throws -> Health {
        try await send(request("GET", "health", auth: false))
    }

    public func capabilities() async throws -> Capabilities {
        try await send(request("GET", "capabilities"))
    }

    /// Whether this token already uploaded the source, without sending bytes.
    public func sourceExists(sha256: String) async throws -> Bool {
        let (_, resp) = try await data(for: request("HEAD", "sources/\(sha256)"))
        switch resp.statusCode {
        case 200: return true
        case 404: return false
        default: throw APIClient.error(status: resp.statusCode, body: Data())
        }
    }

    /// Uploads a file addressed by the SHA-256 of its bytes. Re-uploading is a no-op on the server.
    public func uploadSource(file: URL, sha256: String,
                             progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> Source {
        var req = request("PUT", "sources/\(sha256)")
        req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 300
        let delegate = ProgressDelegate(upload: progress)
        let (body, resp): (Data, URLResponse)
        do {
            (body, resp) = try await session.upload(for: req, fromFile: file, delegate: delegate)
        } catch {
            throw APIClient.wrap(error)
        }
        return try APIClient.decode(body, resp)
    }

    /// Creates a job. Retrying with the same key and body returns the same job.
    public func createJob(_ job: JobRequest, idempotencyKey: String) async throws -> Job {
        var req = request("POST", "jobs")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue(idempotencyKey, forHTTPHeaderField: "Idempotency-Key")
        req.httpBody = try APIClient.encoder.encode(job)
        return try await send(req)
    }

    public func job(id: String) async throws -> Job {
        try await send(request("GET", "jobs/\(id)"))
    }

    @discardableResult
    public func cancelJob(id: String) async throws -> Job {
        try await send(request("POST", "jobs/\(id)/cancel"))
    }

    public func deleteResult(jobID: String) async throws {
        let (body, resp) = try await data(for: request("DELETE", "jobs/\(jobID)/result"))
        if resp.statusCode >= 400 { throw APIClient.error(status: resp.statusCode, body: body) }
    }

    /// Downloads a job's result into `partial`, resuming from the bytes already
    /// there, and verifies the complete file against `sha256` when given.
    /// The caller renames `partial` into place.
    public func downloadResult(jobID: String, sha256: String?, to partial: URL,
                               progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: partial.path) {
            guard fm.createFile(atPath: partial.path, contents: nil) else {
                throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: partial.path])
            }
        }
        let have = (try? fm.attributesOfItem(atPath: partial.path)[.size] as? Int64) ?? 0
        var req = request("GET", "jobs/\(jobID)/result")
        req.timeoutInterval = 300
        if have > 0 { req.setValue("bytes=\(have)-", forHTTPHeaderField: "Range") }

        let sink = try FileSink(partial: partial, offset: have, progress: progress)
        let task = session.dataTask(with: req)
        task.delegate = sink
        let outcome: (Int, String?) = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { cont in
                sink.continuation = cont
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
        let (status, headerSHA) = outcome
        if status == 416 {
            // Range beyond the end: the partial file is wrong. Start over on the next attempt.
            try? fm.removeItem(at: partial)
            throw APIError(status: 416, code: "connection_failed", message: "download range rejected; retrying from the start")
        }
        if !(200..<300).contains(status) {
            throw APIClient.error(status: status, body: sink.errorBody)
        }
        let expected = sha256 ?? headerSHA
        if let expected, !expected.isEmpty {
            let actual = try FileDigest.sha256(of: partial)
            if actual != expected.lowercased() {
                try? fm.removeItem(at: partial)
                throw APIError(code: "connection_failed", message: "downloaded result failed its checksum; retrying")
            }
        }
    }

    // MARK: Retry

    /// Runs `body` until it succeeds, fails with a non-retryable error, or retries run out.
    public func retrying<T>(_ body: () async throws -> T) async throws -> T {
        var delay: Double = 0.5
        var attempt = 0
        while true {
            do {
                return try await body()
            } catch let e as APIError where e.isRetryable && attempt < retries {
                attempt += 1
                let sleep = delay / 2 + Double.random(in: 0..<delay)
                try await Task.sleep(nanoseconds: UInt64(sleep * 1e9))
                delay = min(delay * 2, 10)
            }
        }
    }

    // MARK: Plumbing

    private func request(_ method: String, _ path: String, auth: Bool = true) -> URLRequest {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = method
        if auth, !token.isEmpty { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        req.setValue("painite-mac", forHTTPHeaderField: "User-Agent")
        return req
    }

    private func data(for req: URLRequest) async throws -> (Data, HTTPURLResponse) {
        do {
            let (body, resp) = try await session.data(for: req)
            return (body, resp as! HTTPURLResponse)
        } catch {
            throw APIClient.wrap(error)
        }
    }

    private func send<T: Decodable>(_ req: URLRequest) async throws -> T {
        let (body, resp) = try await data(for: req)
        return try APIClient.decode(body, resp)
    }

    static func decode<T: Decodable>(_ body: Data, _ resp: URLResponse) throws -> T {
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else { throw error(status: status, body: body) }
        do {
            return try decoder.decode(T.self, from: body)
        } catch {
            throw APIError(status: status, code: "connection_failed", message: "unexpected response from server: \(error)")
        }
    }

    static func error(status: Int, body: Data) -> APIError {
        struct Envelope: Decodable { var error: Job.ErrorBody? }
        if let env = try? decoder.decode(Envelope.self, from: body), let e = env.error, !e.code.isEmpty {
            return APIError(status: status, code: e.code, message: e.message)
        }
        let code = switch status {
        case 401: "unauthorized"
        case 404: "not_found"
        default: "internal"
        }
        return APIError(status: status, code: code, message: "HTTP \(status)")
    }

    static func wrap(_ error: Error) -> Error {
        if error is CancellationError || error is APIError { return error }
        if let u = error as? URLError {
            if u.code == .cancelled { return CancellationError() }
            return APIError(code: "connection_failed", message: u.localizedDescription)
        }
        return APIError(code: "connection_failed", message: error.localizedDescription)
    }

    static func loadCertificates(path: String) throws -> [SecCertificate] {
        let expanded = (path as NSString).expandingTildeInPath
        guard let pem = try? String(contentsOfFile: expanded, encoding: .utf8) else {
            throw APIError(code: "not_configured", message: "Can't read the CA file at \(path).")
        }
        var certs: [SecCertificate] = []
        var inside = false
        var b64 = ""
        for line in pem.split(whereSeparator: \.isNewline) {
            if line.hasPrefix("-----BEGIN CERTIFICATE-----") { inside = true; b64 = ""; continue }
            if line.hasPrefix("-----END CERTIFICATE-----") {
                inside = false
                if let der = Data(base64Encoded: b64), let cert = SecCertificateCreateWithData(nil, der as CFData) {
                    certs.append(cert)
                }
                continue
            }
            if inside { b64 += line.trimmingCharacters(in: .whitespaces) }
        }
        guard !certs.isEmpty else {
            throw APIError(code: "not_configured", message: "The CA file at \(path) has no PEM certificates.")
        }
        return certs
    }
}

/// Trusts only the configured CA when one is set (the CLI's ca_file); otherwise the system roots.
private final class TrustDelegate: NSObject, URLSessionDelegate, @unchecked Sendable {
    let anchors: [SecCertificate]

    init(anchors: [SecCertificate]) { self.anchors = anchors }

    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let serverTrust = challenge.protectionSpace.serverTrust, !anchors.isEmpty else {
            completionHandler(.performDefaultHandling, nil)
            return
        }
        SecTrustSetAnchorCertificates(serverTrust, anchors as CFArray)
        SecTrustSetAnchorCertificatesOnly(serverTrust, true)
        if SecTrustEvaluateWithError(serverTrust, nil) {
            completionHandler(.useCredential, URLCredential(trust: serverTrust))
        } else {
            completionHandler(.cancelAuthenticationChallenge, nil)
        }
    }
}

private final class ProgressDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    let upload: @Sendable (Double) -> Void

    init(upload: @escaping @Sendable (Double) -> Void) { self.upload = upload }

    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        guard totalBytesExpectedToSend > 0 else { return }
        upload(Double(totalBytesSent) / Double(totalBytesExpectedToSend))
    }
}

/// Streams a result download into a file: appends on 206, rewrites on 200,
/// and collects the body of an error response instead.
private final class FileSink: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let handle: FileHandle
    private let offset: Int64
    private let progress: @Sendable (Double) -> Void
    private var status = 0
    private var expected: Int64 = 0
    private var written: Int64 = 0
    private var headerSHA: String?
    private var writeError: Error?
    var errorBody = Data()
    var continuation: CheckedContinuation<(Int, String?), Error>?

    init(partial: URL, offset: Int64, progress: @escaping @Sendable (Double) -> Void) throws {
        handle = try FileHandle(forWritingTo: partial)
        self.offset = offset
        self.progress = progress
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let http = response as? HTTPURLResponse
        status = http?.statusCode ?? 0
        headerSHA = http?.value(forHTTPHeaderField: "X-Painite-SHA256")
        do {
            if status == 200 {
                // Whole file: drop whatever partial content there was.
                try handle.truncate(atOffset: 0)
                expected = max(response.expectedContentLength, 0)
            } else if status == 206 {
                try handle.seekToEnd()
                expected = offset + max(response.expectedContentLength, 0)
                written = offset
            }
        } catch {
            writeError = error
            completionHandler(.cancel)
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard status == 200 || status == 206 else {
            if errorBody.count < 65536 { errorBody.append(data) }
            return
        }
        do {
            try handle.write(contentsOf: data)
            written += Int64(data.count)
            if expected > 0 { progress(Double(written) / Double(expected)) }
        } catch {
            writeError = error
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? handle.synchronize()
        try? handle.close()
        let cont = continuation
        continuation = nil
        if let writeError {
            cont?.resume(throwing: writeError)
        } else if let error {
            cont?.resume(throwing: APIClient.wrap(error))
        } else {
            cont?.resume(returning: (status, headerSHA))
        }
    }
}

import CryptoKit
import Foundation
@testable import PainiteKit
import XCTest

final class APIClientTests: XCTestCase {
    var client: APIClient!

    override func setUp() {
        StubProtocol.handler = nil
        StubProtocol.requests = []
        client = try! APIClient(config: ClientConfig(endpoint: "http://spark.test:8443/", token: "tok"),
                                protocolClasses: [StubProtocol.self])
        client.retries = 2
    }

    func testRejectsMissingOrBadEndpoint() {
        XCTAssertThrowsError(try APIClient(config: ClientConfig(endpoint: "", token: "t"))) { e in
            XCTAssertEqual((e as? APIError)?.code, "not_configured")
        }
        XCTAssertThrowsError(try APIClient(config: ClientConfig(endpoint: "spark1:8443", token: "t")))
    }

    func testDecodesCapabilitiesAndSendsToken() async throws {
        StubProtocol.handler = { req, _ in
            XCTAssertEqual(req.url?.absoluteString, "http://spark.test:8443/v1/capabilities")
            XCTAssertEqual(req.value(forHTTPHeaderField: "Authorization"), "Bearer tok")
            return (200, [:], Data("""
            {"operations":[{"name":"denoise","default_model":"denoise/identity@0.1.0",
              "params":{"strength":{"min":0,"max":1,"default":0.5}}},
              {"name":"upscale2x","default_model":"upscale2x/bilinear@0.1.0","params":{}}],
             "models":[{"id":"denoise/identity@0.1.0","operation":"denoise","version":"0.1.0","scale":1,"placeholder":true}],
             "limits":{"max_upload_bytes":536870912,"max_input_pixels":48000000,"max_output_pixels":192000000,
               "queue_capacity":32,"result_retention_seconds":86400,"tile_size":512},
             "formats":{"input":["jpeg","tiff"],"output":["jpeg","tiff"]}}
            """.utf8))
        }
        let caps = try await client.capabilities()
        XCTAssertEqual(caps.operation(.denoise)?.params["strength"]?.default, 0.5)
        XCTAssertEqual(caps.operation(.upscale2x)?.params.isEmpty, true)
        XCTAssertEqual(caps.model("denoise/identity@0.1.0")?.placeholder, true)
        XCTAssertEqual(caps.limits.tileSize, 512)
    }

    func testJobRequestEncodingMatchesContract() throws {
        let req = JobRequest(sourceId: "sha256:" + String(repeating: "a", count: 64), kind: .preview, operation: .denoise,
                             strength: 0.6, model: "denoise/identity@0.1.0", region: Region(x: 1, y: 2, w: 3, h: 4),
                             format: .tiff)
        let obj = try JSONSerialization.jsonObject(with: APIClient.encoder.encode(req)) as! [String: Any]
        XCTAssertEqual(Set(obj.keys), ["source_id", "kind", "operation", "params", "model", "region", "output"])
        XCTAssertEqual((obj["params"] as! [String: Any])["strength"] as? Double, 0.6)
        XCTAssertEqual((obj["output"] as! [String: Any])["format"] as? String, "tiff")

        let full = JobRequest(sourceId: "sha256:x", kind: .full, operation: .upscale2x, strength: nil, model: nil,
                              region: nil, format: nil)
        let obj2 = try JSONSerialization.jsonObject(with: APIClient.encoder.encode(full)) as! [String: Any]
        XCTAssertEqual(Set(obj2.keys), ["source_id", "kind", "operation"])
    }

    func testCreateJobSendsIdempotencyKeyAndDecodesJob() async throws {
        StubProtocol.handler = { req, body in
            XCTAssertEqual(req.httpMethod, "POST")
            XCTAssertEqual(req.value(forHTTPHeaderField: "Idempotency-Key"), "mac-key-123")
            XCTAssertNotNil(try? JSONSerialization.jsonObject(with: body))
            return (201, [:], Data(APIClientTests.jobJSON(state: "queued").utf8))
        }
        let job = try await client.createJob(
            JobRequest(sourceId: "sha256:x", kind: .full, operation: .denoise, strength: 0.5, model: nil, region: nil, format: nil),
            idempotencyKey: "mac-key-123")
        XCTAssertEqual(job.id, "job_01")
        XCTAssertEqual(job.state, .queued)
        XCTAssertNil(job.result)
    }

    func testErrorEnvelopeKeepsStableCode() async {
        StubProtocol.handler = { _, _ in
            (409, [:], Data(#"{"error":{"code":"model_unavailable","message":"model x is not installed"}}"#.utf8))
        }
        do {
            _ = try await client.job(id: "job_01")
            XCTFail("expected an error")
        } catch let e as APIError {
            XCTAssertEqual(e.code, "model_unavailable")
            XCTAssertEqual(e.status, 409)
            XCTAssertFalse(e.isRetryable)
        } catch {
            XCTFail("unexpected \(error)")
        }
    }

    func testRetriesRetryableFailuresOnly() async throws {
        var calls = 0
        StubProtocol.handler = { _, _ in
            calls += 1
            if calls < 3 { return (503, [:], Data(#"{"error":{"code":"queue_full","message":"busy"}}"#.utf8)) }
            return (200, [:], Data(APIClientTests.jobJSON(state: "running").utf8))
        }
        let job = try await client.retrying { try await self.client.job(id: "job_01") }
        XCTAssertEqual(job.state, .running)
        XCTAssertEqual(calls, 3)

        calls = 0
        StubProtocol.handler = { _, _ in
            calls += 1
            return (401, [:], Data(#"{"error":{"code":"unauthorized","message":"bad token"}}"#.utf8))
        }
        do {
            _ = try await client.retrying { try await self.client.job(id: "job_01") }
            XCTFail("expected an error")
        } catch {
            XCTAssertEqual(calls, 1)
        }
    }

    func testNetworkFailureIsConnectionFailed() async {
        StubProtocol.handler = { _, _ in (0, [:], Data()) }
        client.retries = 0
        do {
            _ = try await client.health()
            XCTFail("expected an error")
        } catch {
            XCTAssertEqual((error as? APIError)?.code, "connection_failed")
        }
    }

    func testSourceExists() async throws {
        StubProtocol.handler = { req, _ in
            XCTAssertEqual(req.httpMethod, "HEAD")
            return (req.url!.lastPathComponent == "aa" ? 200 : 404, [:], Data())
        }
        let yes = try await client.sourceExists(sha256: "aa")
        let no = try await client.sourceExists(sha256: "bb")
        XCTAssertTrue(yes)
        XCTAssertFalse(no)
    }

    func testDownloadResumesWithRangeAndVerifiesDigest() async throws {
        let full = Data((0..<10000).map { UInt8($0 % 251) })
        let digest = SHA256.hash(data: full).map { String(format: "%02x", $0) }.joined()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let partial = dir.appendingPathComponent("out.part")
        try full.prefix(4000).write(to: partial)

        StubProtocol.handler = { req, _ in
            XCTAssertEqual(req.value(forHTTPHeaderField: "Range"), "bytes=4000-")
            return (206, ["Content-Length": "6000", "X-Painite-SHA256": digest], full.suffix(from: 4000))
        }
        try await client.downloadResult(jobID: "job_01", sha256: nil, to: partial)
        XCTAssertEqual(try Data(contentsOf: partial), full)
    }

    func testDownloadWholeFileReplacesPartialAndRejectsBadDigest() async throws {
        let full = Data("hello result".utf8)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let partial = dir.appendingPathComponent("out.part")
        try Data("junk".utf8).write(to: partial)

        // Server ignores the range and sends the whole file.
        StubProtocol.handler = { _, _ in (200, [:], full) }
        let good = SHA256.hash(data: full).map { String(format: "%02x", $0) }.joined()
        try await client.downloadResult(jobID: "job_01", sha256: good, to: partial)
        XCTAssertEqual(try Data(contentsOf: partial), full)

        try FileManager.default.removeItem(at: partial)
        client.retries = 0
        do {
            try await client.downloadResult(jobID: "job_01", sha256: String(repeating: "0", count: 64), to: partial)
            XCTFail("expected a checksum error")
        } catch {
            XCTAssertEqual((error as? APIError)?.code, "connection_failed")
            XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
        }
    }

    func testResultGoneIsNotRetryable() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        StubProtocol.handler = { _, _ in
            (410, [:], Data(#"{"error":{"code":"result_unavailable","message":"expired"}}"#.utf8))
        }
        do {
            try await client.downloadResult(jobID: "job_01", sha256: nil, to: dir.appendingPathComponent("x.part"))
            XCTFail("expected an error")
        } catch let e as APIError {
            XCTAssertEqual(e.code, "result_unavailable")
            XCTAssertFalse(e.isRetryable)
        }
    }

    static func jobJSON(state: String) -> String {
        """
        {"id":"job_01","state":"\(state)","kind":"full","operation":"denoise","model":"denoise/identity@0.1.0",
         "source_id":"sha256:x","params":{"strength":0.5},"region":null,"progress":0,
         "created_at":"2026-11-21T00:00:00.000Z","started_at":null,"finished_at":null,
         "result":null,"timings_ms":{},"error":null}
        """
    }
}

// Types for the service contract in api/openapi.yaml.
// JSON keys are snake_case on the wire; the coders in APIClient convert them.

import Foundation

public enum Operation: String, Codable, CaseIterable, Sendable, Identifiable {
    case denoise, sharpen, upscale2x

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .denoise: "Denoise"
        case .sharpen: "Sharpen"
        case .upscale2x: "Upscale 2×"
        }
    }

    /// Output pixels per input pixel along each axis.
    public var scale: Int { self == .upscale2x ? 2 : 1 }
}

public enum ImageFormat: String, Codable, Sendable, CaseIterable {
    case jpeg, tiff

    public var fileExtension: String { self == .jpeg ? "jpg" : "tif" }
}

public struct Health: Codable, Sendable, Equatable {
    public struct GPU: Codable, Sendable, Equatable {
        public var available: Bool
        public var name: String?
    }

    public struct Queue: Codable, Sendable, Equatable {
        public var queued: Int
        public var running: Int
        public var capacity: Int
    }

    public var status: String
    public var version: String
    public var gpu: GPU
    public var queue: Queue
}

public struct ParamRange: Codable, Sendable, Equatable {
    public var min: Double
    public var max: Double
    public var `default`: Double
}

public struct Capabilities: Codable, Sendable, Equatable {
    public struct OperationInfo: Codable, Sendable, Equatable {
        /// Kept as a string so an operation added on the server does not break decoding.
        public var name: String
        public var defaultModel: String
        public var params: [String: ParamRange]
    }

    public struct ModelInfo: Codable, Sendable, Equatable, Identifiable {
        public var id: String
        public var operation: String
        public var version: String
        public var scale: Int
        public var placeholder: Bool?
    }

    public struct Limits: Codable, Sendable, Equatable {
        public var maxUploadBytes: Int64
        public var maxInputPixels: Int64
        public var maxOutputPixels: Int64
        public var queueCapacity: Int
        public var resultRetentionSeconds: Int?
        public var tileSize: Int?
    }

    public struct Formats: Codable, Sendable, Equatable {
        public var input: [String]
        public var output: [String]
    }

    public var operations: [OperationInfo]
    public var models: [ModelInfo]
    public var limits: Limits
    public var formats: Formats

    public func operation(_ op: Operation) -> OperationInfo? {
        operations.first { $0.name == op.rawValue }
    }

    public func model(_ id: String) -> ModelInfo? {
        models.first { $0.id == id }
    }
}

public struct Source: Codable, Sendable, Equatable {
    public var sourceId: String
    public var bytes: Int64
    public var width: Int
    public var height: Int
    public var format: String
    public var bitsPerChannel: Int?
    public var orientation: Int?
    public var hasIcc: Bool?
}

/// A rectangle in displayed (orientation-applied) source pixels.
public struct Region: Codable, Sendable, Hashable {
    public var x: Int
    public var y: Int
    public var w: Int
    public var h: Int

    public init(x: Int, y: Int, w: Int, h: Int) {
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    }
}

public enum JobKind: String, Codable, Sendable {
    case preview, full
}

public struct JobRequest: Codable, Sendable, Equatable {
    public struct Params: Codable, Sendable, Equatable {
        public var strength: Double?
    }

    public struct Output: Codable, Sendable, Equatable {
        public var format: ImageFormat?
        public var jpegQuality: Int?
    }

    public var sourceId: String
    public var kind: JobKind
    public var operation: Operation
    public var params: Params?
    public var model: String?
    public var region: Region?
    public var output: Output?

    public init(sourceId: String, kind: JobKind, operation: Operation, strength: Double?, model: String?,
                region: Region?, format: ImageFormat?, jpegQuality: Int? = nil) {
        self.sourceId = sourceId
        self.kind = kind
        self.operation = operation
        params = strength.map { Params(strength: $0) }
        self.model = model
        self.region = region
        if format != nil || jpegQuality != nil {
            output = Output(format: format, jpegQuality: jpegQuality)
        }
    }
}

public enum JobState: String, Codable, Sendable {
    case queued, running, completed, failed, canceled

    public var isTerminal: Bool { self == .completed || self == .failed || self == .canceled }
}

public struct Job: Codable, Sendable, Equatable {
    public struct Result: Codable, Sendable, Equatable {
        public var bytes: Int64
        public var sha256: String
        public var format: ImageFormat
        public var width: Int
        public var height: Int
        public var expiresAt: String?
        public var deleted: Bool?
    }

    public struct ErrorBody: Codable, Sendable, Equatable {
        public var code: String
        public var message: String
    }

    public var id: String
    public var state: JobState
    public var kind: JobKind
    public var operation: String
    public var model: String
    public var sourceId: String
    public var params: [String: Double]?
    public var region: Region?
    public var progress: Double?
    public var result: Result?
    public var timingsMs: [String: Double]?
    public var error: ErrorBody?
}

/// A failure with a stable code. Server errors carry the contract's codes;
/// client-side failures use `connection_failed`, `not_configured`, and the like.
public struct APIError: Error, Sendable, Equatable, LocalizedError {
    public var status: Int
    public var code: String
    public var message: String

    public init(status: Int = 0, code: String, message: String) {
        self.status = status
        self.code = code
        self.message = message
    }

    /// Whether trying the same request again may succeed.
    public var isRetryable: Bool {
        switch code {
        case "connection_failed", "queue_full", "internal": return true
        default: return status == 502 || status == 503 || status == 504
        }
    }

    /// A sentence for the photographer, with what to do next where there is something to do.
    public var errorDescription: String? {
        switch code {
        case "not_configured": return message
        case "connection_failed": return "Can't reach the PAINITE server. Check the server address and that the server is running. (\(message))"
        case "unauthorized": return "The server rejected the access token. Check it in Settings."
        case "unsupported_format": return "The server can't read this image. PAINITE supports RGB JPEG and TIFF files without alpha."
        case "too_many_pixels": return "This image is larger than the server allows. (\(message))"
        case "upload_too_large": return "This file is larger than the server accepts. (\(message))"
        case "disk_full": return "The server is out of storage. Try again later."
        case "model_unavailable": return "The model used for the preview is no longer installed on the server. Regenerate the preview, then export again."
        case "queue_full": return "The server is busy. Try again in a moment."
        case "result_unavailable": return "The result has expired or was deleted on the server. Run it again."
        case "processing_failed": return "The server could not process this image. (\(message))"
        case "digest_mismatch": return "The upload was corrupted in transit. Try again."
        case "canceled": return "Canceled."
        default: return message.isEmpty ? "Something went wrong (\(code))." : message
        }
    }
}

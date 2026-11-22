import AppKit
import Foundation
import PainiteKit

/// One open photo: its upload, the live preview, and export.
@MainActor
final class DocumentModel: ObservableObject {
    enum UploadState: Equatable {
        case idle
        case hashing
        case uploading(Double)
        case ready
        case failed(String)
    }

    enum PreviewState: Equatable {
        case idle
        case rendering(Double)
        case ready
        case failed(String)
    }

    enum ExportState: Equatable {
        case idle
        case running(String, Double?)
        case done(URL, inPlace: Bool)
        case failed(String)

        var isRunning: Bool {
            if case .running = self { return true }
            return false
        }
    }

    struct Preview {
        let image: CGImage
        /// Displayed source pixels the preview covers (snapped to the server's tile grid).
        let region: Region
        let signature: RecipeSignature
        /// The model each step used, in order.
        let models: [String]
        let serverMs: Double?
    }

    let url: URL
    let handoff: HandoffInfo
    let server = ServerStore.shared

    @Published private(set) var image: CGImage?
    @Published private(set) var imageSize: CGSize = .zero
    @Published private(set) var loadError: String?
    @Published private(set) var sourceFormat: ImageFormat?
    @Published private(set) var bitsPerChannel = 8

    /// The enhancement steps, run in order in one server job.
    @Published var steps: [RecipeStep] = [RecipeStep(operation: .denoise)] {
        didSet { if oldValue != steps { settingsChanged() } }
    }
    @Published var exportFormat: ImageFormat = .tiff

    @Published private(set) var upload: UploadState = .idle
    @Published private(set) var preview: Preview? {
        didSet { updatePreviewBox() }
    }
    @Published private(set) var previewState: PreviewState = .idle {
        didSet { if oldValue != previewState { updatePreviewBox() } }
    }
    @Published private(set) var export: ExportState = .idle
    @Published private(set) var zoomPercent: Double = 0
    /// True while the next drag on the photo picks the preview area.
    @Published var isSelectingArea = false {
        didSet { viewport?.selectMode = isSelectingArea }
    }

    /// The compare view, for zoom commands.
    weak var viewport: CompareView?

    private var visible: CGRect = .zero
    /// The area the user picked for previews, in displayed source pixels. Zooming
    /// and panning never change it; only Select Area or Preview Visible Area do.
    private(set) var selection: CGRect?
    private var orientation = 1
    private var storedSize = (width: 0, height: 0)
    private var sha: String?
    private var uploadTask: Task<String, Error>?
    private var debounceTask: Task<Void, Never>?
    private var previewTask: Task<Void, Never>?
    private var previewJobID: String?
    private var previewSeq = 0
    private var exportTask: Task<Void, Never>?
    private var cache: [CacheKey: Preview] = [:]
    private var cacheOrder: [CacheKey] = []

    private struct CacheKey: Hashable {
        let signature: RecipeSignature
        let region: Region
    }

    init(url: URL) {
        self.url = url
        handoff = Handoff.inspect(url)
        Task { await load() }
    }

    var title: String { url.lastPathComponent }

    func takesStrength(_ op: PainiteKit.Operation) -> Bool {
        if let info = server.capabilities?.operation(op) { return info.params["strength"] != nil }
        return op != .upscale2x
    }

    func strengthRange(_ op: PainiteKit.Operation) -> ClosedRange<Double> {
        if let r = server.capabilities?.operation(op)?.params["strength"] { return r.min...r.max }
        return 0...1
    }

    /// What the enabled steps render.
    var signature: RecipeSignature { Recipe.signature(steps, takesStrength: takesStrength) }

    /// The preview on screen matches the current steps.
    var previewIsCurrent: Bool {
        guard let preview else { return false }
        return preview.signature == signature
    }

    // MARK: Steps

    func canAdd(_ op: PainiteKit.Operation) -> Bool {
        // An older server may not offer every operation (e.g. unblur).
        let offered = server.capabilities.map { $0.operation(op) != nil } ?? true
        return offered && Recipe.canAdd(op, to: steps) && !export.isRunning
    }

    func addStep(_ op: PainiteKit.Operation) {
        guard canAdd(op) else { return }
        let strength = server.capabilities?.operation(op)?.params["strength"]?.default ?? 0.5
        var new = steps
        // Keep an upscale last unless the photographer moves it: enhance detail at native size first.
        if op.scale == 1, let up = new.firstIndex(where: { $0.operation.scale > 1 }), up == new.count - 1 {
            new.insert(RecipeStep(operation: op, strength: strength), at: up)
        } else {
            new.append(RecipeStep(operation: op, strength: strength))
        }
        steps = new
    }

    func removeStep(_ id: UUID) {
        steps.removeAll { $0.id == id }
    }

    func moveStep(_ id: UUID, by offset: Int) {
        guard let i = steps.firstIndex(where: { $0.id == id }) else { return }
        let j = i + offset
        guard steps.indices.contains(j) else { return }
        steps.swapAt(i, j)
    }

    /// The model the server will use (or the preview used) for the i-th enabled step.
    func modelID(forEnabledStep i: Int) -> String? {
        let sig = signature
        guard sig.items.indices.contains(i) else { return nil }
        if let p = preview, p.signature == sig, p.models.indices.contains(i) { return p.models[i] }
        return server.capabilities?.operation(sig.items[i].operation)?.defaultModel
    }

    // MARK: Loading and upload

    private func load() async {
        let url = url
        do {
            let loaded = try await Task.detached(priority: .userInitiated) { try ImageLoader.load(url) }.value
            image = loaded.image
            imageSize = loaded.size
            sourceFormat = loaded.format
            bitsPerChannel = loaded.bitsPerChannel
            orientation = loaded.orientation
            storedSize = (loaded.storedWidth, loaded.storedHeight)
            exportFormat = loaded.format ?? .tiff
            if loaded.format == nil {
                loadError = "PAINITE works with JPEG and TIFF files. This file can be viewed but not enhanced."
                return
            }
        } catch {
            loadError = "Can't open this file: \(error.localizedDescription)"
            return
        }
        // Start the upload right away; opening the same photo again is free.
        _ = try? await ensureUploaded()
        schedulePreview(after: 0)
    }

    /// Uploads the original if the server does not have it yet. Returns its source ID.
    private func ensureUploaded() async throws -> String {
        if case .ready = upload, let sha { return "sha256:\(sha)" }
        if let uploadTask { return try await uploadTask.value }
        let task = Task { () throws -> String in
            let client = try server.requireClient()
            let caps = try await server.currentCapabilities()
            upload = .hashing
            let url = url
            let digest: String
            if let sha {
                digest = sha
            } else {
                digest = try await Task.detached(priority: .userInitiated) { try FileDigest.sha256(of: url) }.value
                sha = digest
            }
            let exists = try await client.retrying { try await client.sourceExists(sha256: digest) }
            if !exists {
                let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
                if size > caps.limits.maxUploadBytes {
                    throw APIError(code: "upload_too_large", message: "\(size) bytes; the server accepts up to \(caps.limits.maxUploadBytes)")
                }
                upload = .uploading(0)
                _ = try await client.retrying {
                    try await client.uploadSource(file: url, sha256: digest) { fraction in
                        Task { @MainActor [weak self] in
                            if case .uploading = self?.upload { self?.upload = .uploading(fraction) }
                        }
                    }
                }
            }
            upload = .ready
            return "sha256:\(digest)"
        }
        uploadTask = task
        defer { uploadTask = nil }
        do {
            return try await task.value
        } catch {
            if !(error is CancellationError) { upload = .failed(error.localizedDescription) }
            throw error
        }
    }

    /// The server deletes sources some time after their last job; upload again and retry once.
    private func withSource<T>(_ body: (String) async throws -> T) async throws -> T {
        let id = try await ensureUploaded()
        do {
            return try await body(id)
        } catch let e as APIError where e.code == "not_found" {
            upload = .idle
            return try await body(try await ensureUploaded())
        }
    }

    // MARK: Preview

    /// Called by the compare view whenever zoom or position changes. Only the
    /// first call picks an area (the middle of the photo); after that, zoom and
    /// pan just move the view.
    func viewportChanged(visible: CGRect, zoomPercent: Double) {
        self.zoomPercent = zoomPercent
        self.visible = visible
        if selection == nil, image != nil, imageSize.width > 0 {
            selectArea(CGRect(origin: .zero, size: imageSize))
        }
    }

    /// Sets the preview area (displayed source pixels) and renders it.
    func selectArea(_ rect: CGRect) {
        isSelectingArea = false
        guard image != nil, imageSize.width > 0,
              let r = PreviewGeometry.requestRegion(visible: rect, imageSize: imageSize, scale: Recipe.scale(signature))
        else { return }
        selection = CGRect(x: r.x, y: r.y, width: r.w, height: r.h)
        viewport?.setSelection(selection)
        updatePreviewBox()
        schedulePreview(after: 0)
    }

    func previewVisibleArea() {
        guard !visible.isEmpty else { return }
        selectArea(visible)
    }

    /// The selection, capped to what one preview may cover with the current steps.
    private func wantedRegion() -> Region? {
        guard let selection, image != nil else { return nil }
        return PreviewGeometry.requestRegion(visible: selection, imageSize: imageSize, scale: Recipe.scale(signature))
    }

    /// The box the next (or current) preview covers, exactly as the server will
    /// snap it to its tile grid.
    private func targetRegion() -> Region? {
        let sig = signature
        guard loadError == nil, !sig.items.isEmpty, let wanted = wantedRegion() else { return nil }
        if let p = preview, p.signature == sig, PreviewGeometry.covers(p.region, wanted) {
            return p.region
        }
        guard let tile = server.capabilities?.limits.tileSize, storedSize.width > 0 else { return wanted }
        return TileGrid.snap(wanted, storedWidth: storedSize.width, storedHeight: storedSize.height,
                             orientation: orientation, tile: tile)
    }

    private func updatePreviewBox() {
        let state: CompareView.BoxState
        switch previewState {
        case .rendering: state = .rendering
        case .failed: state = .failed
        default: state = targetRegion() == preview?.region && previewIsCurrent ? .ready : .pending
        }
        viewport?.setPreviewBox(targetRegion(), state: state)
    }

    private func settingsChanged() {
        if export.isRunning { return }
        if case .done = export { export = .idle }
        if case .failed = export { export = .idle }
        updatePreviewBox()
        schedulePreview(after: 0.25)
    }

    func retryPreview() {
        schedulePreview(after: 0)
    }

    func regeneratePreview() {
        cache.removeAll()
        cacheOrder.removeAll()
        preview = nil
        schedulePreview(after: 0)
    }

    private func schedulePreview(after delay: Double) {
        debounceTask?.cancel()
        debounceTask = Task { [weak self] in
            if delay > 0 { try? await Task.sleep(nanoseconds: UInt64(delay * 1e9)) }
            guard !Task.isCancelled else { return }
            self?.startPreview()
        }
    }

    private func startPreview() {
        let sig = signature
        guard image != nil, loadError == nil, !sig.items.isEmpty, let region = wantedRegion() else {
            if sig.items.isEmpty {
                previewTask?.cancel()
                cancelServerPreview()
                previewState = .idle
            }
            return
        }
        if let p = preview, p.signature == sig, PreviewGeometry.covers(p.region, region) {
            return
        }
        let key = CacheKey(signature: sig, region: region)
        if let hit = cache[key] {
            previewTask?.cancel()
            preview = hit
            previewState = .ready
            return
        }

        previewSeq += 1
        let seq = previewSeq
        previewTask?.cancel()
        cancelServerPreview()
        previewState = .rendering(0)
        previewTask = Task { [weak self] in
            guard let self else { return }
            do {
                let p = try await renderPreview(sig, region: region, seq: seq)
                guard seq == previewSeq else { return }
                remember(p, for: key)
                preview = p
                previewState = .ready
            } catch is CancellationError {
            } catch let e as APIError where e.code == "canceled" {
                // Superseded by a newer preview on the server.
            } catch {
                guard seq == previewSeq else { return }
                previewState = .failed(error.localizedDescription)
            }
        }
    }

    private func defaultModels(_ sig: RecipeSignature, _ caps: Capabilities) throws -> [String] {
        try sig.items.map { item in
            guard let m = caps.operation(item.operation)?.defaultModel else {
                throw APIError(code: "model_unavailable", message: "The server has no model for \(item.operation.title).")
            }
            return m
        }
    }

    private func renderPreview(_ sig: RecipeSignature, region: Region, seq: Int) async throws -> Preview {
        let client = try server.requireClient()
        let caps = try await server.currentCapabilities()
        let models = try defaultModels(sig, caps)
        let job = try await withSource { sourceID in
            let request = JobRequest(sourceId: sourceID, kind: .preview, steps: Recipe.jobSteps(sig, models: models),
                                     region: region, format: .tiff)
            let key = "mac-preview-\(UUID().uuidString)"
            return try await client.retrying { try await client.createJob(request, idempotencyKey: key) }
        }
        previewJobID = job.id
        let done = try await waitForJob(job, client: client, interval: 0.1, maxInterval: 0.5) { [weak self] p in
            guard let self, seq == previewSeq else { return }
            previewState = .rendering(p)
        }
        try Task.checkCancellation()
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("painite-preview-\(UUID().uuidString).tif")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try await client.retrying {
            try await client.downloadResult(jobID: done.id, sha256: done.result?.sha256, to: tmp)
        }
        let jobID = done.id
        Task.detached { try? await client.deleteResult(jobID: jobID) }
        let loaded = try await Task.detached(priority: .userInitiated) { try ImageLoader.load(tmp) }.value
        previewJobID = nil
        return Preview(image: loaded.image, region: done.region ?? region, signature: sig,
                       models: done.steps?.map(\.model) ?? [done.model], serverMs: done.timingsMs?["total"])
    }

    private func cancelServerPreview() {
        guard let id = previewJobID, let client = server.client else { return }
        previewJobID = nil
        Task.detached { _ = try? await client.cancelJob(id: id) }
    }

    private func remember(_ p: Preview, for key: CacheKey) {
        cache[key] = p
        cacheOrder.removeAll { $0 == key }
        cacheOrder.append(key)
        while cacheOrder.count > 6 {
            cache[cacheOrder.removeFirst()] = nil
        }
    }

    /// Polls until the job ends. Throws the job's error if it failed.
    private func waitForJob(_ job: Job, client: APIClient, interval: Double, maxInterval: Double,
                            progress: @escaping (Double) -> Void) async throws -> Job {
        var j = job
        var delay = interval
        while !j.state.isTerminal {
            try await Task.sleep(nanoseconds: UInt64(delay * 1e9))
            let id = j.id
            j = try await client.retrying { try await client.job(id: id) }
            progress(j.progress ?? 0)
            delay = min(delay * 1.5, maxInterval)
        }
        switch j.state {
        case .completed: return j
        case .canceled: throw APIError(code: "canceled", message: j.error?.message ?? "canceled")
        default: throw APIError(code: j.error?.code ?? "processing_failed", message: j.error?.message ?? "job failed")
        }
    }

    // MARK: Export

    var canSaveInPlace: Bool { handoff.canSaveInPlace && sourceFormat != nil }

    /// Lightroom handoff: replace the handed-over file, same format, then Finish in Lightroom.
    func saveInPlace() {
        guard canSaveInPlace, let format = sourceFormat else { return }
        startExport(to: url, format: format, inPlace: true)
    }

    func saveAs(to destination: URL, format: ImageFormat) {
        if SafeWrite.sameFile(destination, url) {
            if canSaveInPlace, format == sourceFormat {
                startExport(to: url, format: format, inPlace: true)
            } else {
                export = .failed("PAINITE never overwrites the original photo. Choose a different name.")
            }
            return
        }
        startExport(to: destination, format: format, inPlace: false)
    }

    func cancelExport() {
        exportTask?.cancel()
    }

    private func startExport(to destination: URL, format: ImageFormat, inPlace: Bool) {
        guard !export.isRunning, loadError == nil else { return }
        let sig = signature
        guard !sig.items.isEmpty else {
            export = .failed("Add at least one step before exporting.")
            return
        }
        // Pin the models the preview used, so export matches what was shown.
        let pinned = preview?.signature == sig ? preview?.models : nil
        export = .running("Preparing", nil)
        exportTask = Task { [weak self] in
            guard let self else { return }
            var jobID: String?
            let partial = SafeWrite.partialURL(for: destination)
            do {
                let client = try server.requireClient()
                let caps = try await server.currentCapabilities()
                let models = try pinned ?? defaultModels(sig, caps)
                if upload != .ready { export = .running("Uploading", nil) }
                let key = "mac-export-\(UUID().uuidString)"
                let job = try await withSource { sourceID in
                    let request = JobRequest(sourceId: sourceID, kind: .full, steps: Recipe.jobSteps(sig, models: models),
                                             region: nil, format: format)
                    return try await client.retrying { try await client.createJob(request, idempotencyKey: key) }
                }
                jobID = job.id
                export = .running("Rendering", 0)
                let done = try await waitForJob(job, client: client, interval: 0.3, maxInterval: 1.5) { [weak self] p in
                    self?.export = .running("Rendering", p)
                }
                export = .running("Downloading", 0)
                let id = done.id
                try await client.retrying {
                    try await client.downloadResult(jobID: id, sha256: done.result?.sha256, to: partial) { p in
                        Task { @MainActor [weak self] in
                            if case .running("Downloading", _) = self?.export { self?.export = .running("Downloading", p) }
                        }
                    }
                }
                try Task.checkCancellation()
                try SafeWrite.commit(partial, to: destination)
                Task.detached { try? await client.deleteResult(jobID: id) }
                export = .done(destination, inPlace: inPlace)
            } catch {
                try? FileManager.default.removeItem(at: partial)
                if let id = jobID, let client = server.client {
                    Task.detached { _ = try? await client.cancelJob(id: id) }
                }
                if error is CancellationError || (error as? APIError)?.code == "canceled" {
                    export = .failed("Export canceled. Nothing was written.")
                } else {
                    if (error as? APIError)?.code == "model_unavailable" {
                        preview = nil
                        previewState = .failed(error.localizedDescription)
                    }
                    export = .failed(error.localizedDescription)
                }
            }
        }
    }

    // MARK: Lifecycle

    func close() {
        debounceTask?.cancel()
        previewTask?.cancel()
        exportTask?.cancel()
        cancelServerPreview()
    }
}

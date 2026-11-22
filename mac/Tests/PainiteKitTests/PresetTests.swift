import Foundation
@testable import PainiteKit
import XCTest

final class PresetTests: XCTestCase {
    private func steps(_ ops: [PainiteKit.Operation]) -> [RecipeStep] { ops.map { RecipeStep(operation: $0) } }

    func testMovingMatchesSwiftUIOnMove() {
        let s = steps([.denoise, .sharpen, .face, .upscale2x])
        // Drag Face above Denoise.
        XCTAssertEqual(Recipe.moving(s, from: [2], to: 0).map(\.operation), [.face, .denoise, .sharpen, .upscale2x])
        // Drag Denoise below Face (destination is the index before removal).
        XCTAssertEqual(Recipe.moving(s, from: [0], to: 3).map(\.operation), [.sharpen, .face, .denoise, .upscale2x])
        // To the end.
        XCTAssertEqual(Recipe.moving(s, from: [1], to: 4).map(\.operation), [.denoise, .face, .upscale2x, .sharpen])
        // No-op.
        XCTAssertEqual(Recipe.moving(s, from: [1], to: 1).map(\.operation), s.map(\.operation))
    }

    func testPresetMatchingTracksCustom() {
        let high = Preset.named("High ISO")!
        var s = high.steps.map { $0.makeStep() }
        XCTAssertTrue(high.matches(s))
        s[0].strength = 0.701 // shows as 70 on the slider: still the preset
        XCTAssertTrue(high.matches(s))
        s[0].strength = 0.72
        XCTAssertFalse(high.matches(s))
        s = high.steps.map { $0.makeStep() }
        s[1].enabled = false
        XCTAssertFalse(high.matches(s))
        XCTAssertFalse(high.matches(Recipe.moving(high.steps.map { $0.makeStep() }, from: [1], to: 0)))
    }

    func testBuiltInPresetsAreValidRecipes() {
        for p in Preset.builtIn {
            XCTAssertLessThanOrEqual(p.steps.count, Recipe.maxSteps, p.name)
            XCTAssertLessThanOrEqual(p.steps.filter { $0.operation.scale > 1 }.count, 1, p.name)
        }
    }

    func testLastRecipeRoundTripAndDefault() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: "painite-tests-\(UUID().uuidString)"))
        XCTAssertEqual(LastRecipe.load(defaults).preset, "Clean Up")
        let r = LastRecipe(preset: nil, steps: [StoredStep(operation: .unblurFocus, strength: 0.3, enabled: false)])
        r.save(defaults)
        XCTAssertEqual(LastRecipe.load(defaults), r)
    }

    func testShortLabel() {
        let sig = Recipe.signature(
            [RecipeStep(operation: .denoise, strength: 0.6), RecipeStep(operation: .unblurMotion, strength: 0.4),
             RecipeStep(operation: .upscale2x)],
            takesStrength: { $0 != .upscale2x })
        XCTAssertEqual(Recipe.shortLabel(sig), "Denoise 60 · Deblur 40 · Upscale 2×")
    }
}

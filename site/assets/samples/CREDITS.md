# Sample photo credits

Every source photo here is public domain or CC0, so the samples can be published
without attribution requirements; the authors are credited anyway. All results
were produced by a PAINITE server (NVIDIA DGX Spark, GB10) with the `painite` CLI.

`before.jpg` and `after.jpg` in each folder show the same region at the same size.
Where the display scale is 2x, `before.jpg` is a bicubic 2x enlargement of the
input crop (for `upscale2x` and `pipeline` the result itself is 2x; for `face`
both crops are enlarged 2x with bicubic so the small face is easier to see).
`thumb.jpg` is the whole processed result, about 600 px wide.

| Op | Source | Author | License | Degradation | `--op` (strength) | Crop (input px → shown px) |
|---|---|---|---|---|---|---|
| denoise | [Strawberries ISO denoise test, ISO 12800 photo untouched JPEG from camera](https://commons.wikimedia.org/wiki/File:Strawberries_ISO_denoise_test,_ISO_12800_photo_untouched_JPEG_from_camera.jpg) | Alexis Jazz | CC0 | **Real**: ISO 12800 in-camera JPEG, full 5472×3648 | `denoise` (1.0) | 1200×800 at 100% → 1200×800 |
| sharpen | [Orange bell-peppers Rouffignac](https://commons.wikimedia.org/wiki/File:Orange_bell-peppers_Rouffignac.jpg) | Jebulon | CC0 | **Simulated**: resized to 2400 px wide, Gaussian blur σ = 1.6 px | `sharpen` (0.8) | 1200×800 at 100% → 1200×800 |
| upscale2x | [Ancient banana vendor](https://commons.wikimedia.org/wiki/File:Ancient_banana_vendor.jpg) | Wilfredor | CC0 | **Simulated**: downscaled (Lanczos) to 800×641 | `upscale2x` (n/a) | 450×380 → 900×760 (before: bicubic 2x) |
| face | [Mary Roberts Rinehart; photo by Harris & Ewing (The Woman Citizen, Feb. 1927)](https://commons.wikimedia.org/wiki/File:Mary_Roberts_Rinehart;_photo_by_Harris_%26_Ewing_(The_Woman_Citizen,_Feb._1927).png) | Harris & Ewing; published by The Woman Citizen Corporation | Public domain (US publication, 1927) | **Real**: 385×589 magazine scan, face about 130 px tall | `face` (1.0) | 270×300 → 540×600 (both bicubic 2x) |
| unblur_motion | [Road sign sundays Paris](https://commons.wikimedia.org/wiki/File:Road_sign_sundays_Paris.jpg) | Jebulon | CC0 | **Simulated**: resized to 1800 px wide, 21 px linear motion blur at 12° in linear light, σ = 1/255 Gaussian noise | `unblur_motion` (default) | 1200×800 at 100% → 1200×800 |
| unblur_focus | [Street sign in Daejeon](https://commons.wikimedia.org/wiki/File:Street_sign_in_Daejeon.jpg) | Bernard Gagnon | CC0 | **Simulated**: resized to 1800 px wide, disk (defocus) blur radius 7 px in linear light, σ = 1/255 Gaussian noise | `unblur_focus` (default) | 1200×700 at 100% → 1200×700 |
| pipeline | [Sweet Pepper Market](https://commons.wikimedia.org/wiki/File:Sweet_Pepper_Market.jpg) | Wilfredor | CC0 | **Simulated**: resized to 1000 px wide, Gaussian blur σ = 0.8 px, Poisson + Gaussian luminance and chroma noise, JPEG q90 | `denoise:0.6,sharpen:0.3,upscale2x` | 600×400 → 1200×800 (before: bicubic 2x) |

Models used (server defaults): denoise `scunet-real-gan@1.0.0`, sharpen
`realesrgan-x2-down@1.0.0`, upscale2x `nomosuni-esrgan-multijpg@1.0.0`, face
`gpen-bfr-1024@1.0.0`, unblur_motion `stripformer-realblurj@1.0.0`, unblur_focus
`lakdnet-defocus-lfdpdd-s@1.0.0`.

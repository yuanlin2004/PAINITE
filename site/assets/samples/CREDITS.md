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
| denoise | [Colorful façades, Rue de la Poissonnerie, Colmar, 2026](https://commons.wikimedia.org/wiki/File:Colorful_fa%C3%A7ades,_Rue_de_la_Poissonnerie,_Colmar,_2026.jpg) | DimiTalen | CC0 | **Simulated**: resized to 2400 px wide, Poisson + Gaussian luminance noise (full well 150 e⁻, read σ = 0.008, linear light) and chroma noise (σ = 0.025), like ISO 6400, JPEG q90 | `denoise` (1.0) | 1200×800 at 100% → 1200×800 |
| sharpen | [Brick Facade Odense Denmark](https://commons.wikimedia.org/wiki/File:Brick_Facade_Odense_Denmark.jpg) | Jebulon | CC0 | **Simulated**: resized to 2400 px wide, Gaussian blur σ = 1.0 px, JPEG q95 | `sharpen` (0.8) | 1200×800 at 100% → 1200×800 |
| upscale2x | [Ancient banana vendor](https://commons.wikimedia.org/wiki/File:Ancient_banana_vendor.jpg) | Wilfredor | CC0 | **Simulated**: downscaled (Lanczos) to 800×641 | `upscale2x` (n/a) | 450×380 → 900×760 (before: bicubic 2x) |
| face | [Mary Roberts Rinehart; photo by Harris & Ewing (The Woman Citizen, Feb. 1927)](https://commons.wikimedia.org/wiki/File:Mary_Roberts_Rinehart;_photo_by_Harris_%26_Ewing_(The_Woman_Citizen,_Feb._1927).png) | Harris & Ewing; published by The Woman Citizen Corporation | Public domain (US publication, 1927) | **Real**: 385×589 magazine scan, face about 130 px tall | `face` (1.0) | 270×300 → 540×600 (both bicubic 2x) |
| unblur_motion | [Road sign sundays Paris](https://commons.wikimedia.org/wiki/File:Road_sign_sundays_Paris.jpg) | Jebulon | CC0 | **Simulated**: resized to 1800 px wide, 21 px linear motion blur at 12° in linear light, σ = 1/255 Gaussian noise | `unblur_motion` (default) | 1200×800 at 100% → 1200×800 |
| unblur_focus | [Street sign in Daejeon](https://commons.wikimedia.org/wiki/File:Street_sign_in_Daejeon.jpg) | Bernard Gagnon | CC0 | **Simulated**: resized to 1800 px wide, disk (defocus) blur radius 7 px in linear light, σ = 1/255 Gaussian noise | `unblur_focus` (default) | 1200×700 at 100% → 1200×700 |
| pipeline | [Raja Chari, NASA portrait (2021) a](https://commons.wikimedia.org/wiki/File:Raja_Chari,_NASA_portrait_(2021)_a.jpg) | Josh Valcarcel, NASA | Public domain (US government work) | **Simulated**: head-and-shoulders crop at 2100×1400, Gaussian blur σ = 0.8 px, downscaled (Lanczos) to 1050×700, Poisson + Gaussian luminance noise (full well 300 e⁻) and chroma noise, JPEG q90; face about 230 px tall in the input | `denoise:0.8,upscale2x,face:0.8` | 600×400 → 1200×800 (before: bicubic 2x) |

Models used (server defaults): denoise `scunet-real-gan@1.0.0`, sharpen
`realesrgan-x2-down@1.0.0`, upscale2x `nomosuni-esrgan-multijpg@1.0.0`, face
`gpen-bfr-1024@1.0.0`, unblur_motion `stripformer-realblurj@1.0.0`, unblur_focus
`lakdnet-defocus-lfdpdd-s@1.0.0`.

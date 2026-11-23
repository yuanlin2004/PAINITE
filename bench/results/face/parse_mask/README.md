# Face paste mask: square (1.0.0) vs face parsing (1.1.0)

Before/after for the face-parsing paste mask of `face/gpen-bfr-1024@1.1.0`, made
with `bench/face/parse_mask_compare.py` through the server's own pipeline on
spark1 (GB10), 2026-11-23. Each sheet shows input | 1.0.0 square paste | 1.1.0
face parsing for one region, with change maps (|output - input| x8) underneath.
The sheets here are at face strength 1.0 so the difference is easy to see; the
numbers below are for both 1.0 and the default 0.5.

```
python face/parse_mask_compare.py --model-dir <data>/models/face/gpen-bfr-1024@1.1.0 \
    --strength 1.0 --out <out> migrant_mother.jpg@250,150,900,1000 mawlamyine_kiss.jpg \
    lange_brawley.jpg@500,80,1030,800 lang_baby.jpg half_selfie.jpg@1795,24,1623,1622
```

Inputs were the Commons files downscaled to 1900 px on the long side (old scans
at roughly the size of the Reddit report's ~1200 px scans); half_selfie is the
bench copy in `bench/face/real_photos.txt`.

| Sheet | What to look at |
|---|---|
| `migrant_mother.jpg` | 1.0.0 restores the fingers at her chin and the hair; 1.1.0 changes only the face (the two fingertips touching her lip are still inside the mask). |
| `mawlamyine_kiss.jpg` | 1.0.0 smears the sequins and weave of the headscarf, the baby's fleece and the baby's head; 1.1.0 leaves them alone. The baby's cheek pressed against the mother's face is labelled skin and is restored with her face. |
| `lange_brawley.jpg` | 1.0.0 draws a bright halo along hair and cheeks against the sky; 1.1.0 has none. |
| `lang_baby.jpg` | Hair, dress and background keep their scan texture; no seam at hairline or jaw. |
| `half_selfie.jpg` | Profile head in front of a building: 1.0.0 rewrites the windows and awning inside the square; 1.1.0 touches only cheek, ear and beard. |

## Mean change outside the face

Over the pixels the square paste changed (by at least 1 level) that face
parsing left alone (changed by at most 1 level), the mean absolute change in
8-bit levels, square vs parse. "Share" is the part of the square paste's
footprint that is outside the face. Parse is at most 1 there by construction;
the square column is what used to go wrong. Regions are the framed one or,
unframed, 3.2x each enhanced face's box.

| Photo | Region | Share | Strength 0.5: square / parse | Strength 1.0: square / parse |
|---|---|---|---|---|
| migrant_mother | framed face + hand | 74% / 63% | 1.63 / 0.36 | 2.69 / 0.23 |
| mawlamyine_kiss | mother (only face detected) | 85% / 78% | 3.20 / 0.16 | 6.03 / 0.10 |
| lange_brawley | framed, both faces | 82% / 72% | 2.33 / 0.41 | 3.49 / 0.31 |
| lang_baby | mother | 80% / 63% | 1.52 / 0.52 | 2.33 / 0.38 |
| lang_baby | baby | 77% / 59% | 1.22 / 0.60 | 1.68 / 0.44 |
| mother_daughter | mother | 65% / 57% | 2.12 / 0.31 | 3.75 / 0.20 |
| mother_daughter | baby | 64% / 55% | 1.95 / 0.34 | 3.38 / 0.21 |
| half_selfie | framed profile head | 89% / 86% | 2.77 / 0.08 | 4.84 / 0.06 |
| half_rice | girl | 66% / 52% | 1.66 / 0.39 | 2.92 / 0.20 |
| portrait_horridge | face | 58% / 44% | 2.11 / 0.45 | 4.06 / 0.25 |

(mother_daughter, half_rice and portrait_horridge were run as no-regression
checks; their sheets are not included.)

Face parsing (BiSeNet on the 1024 restored crop) adds about 12 ms per face:
a face job took 1474 ms/face with the square mask and 1486 ms/face with
parsing (5 photos, 21 faces, 3 runs each, end to end through the pipeline);
`_parse_mask` alone takes 12.4 ms on a 1024 crop.

## Known limits

- Skin touching the face (a hand at the cheek, a baby's cheek against a
  parent's) is connected to the face in the parse and is restored with it.
- Scan dust and scratches on hair, a bonnet or background next to the face
  are no longer cleaned up by the face op (1.0.0 cleaned them inside the
  square only).

## Sources

| Sheet | Commons file | Author | License |
|---|---|---|---|
| `migrant_mother.jpg` | [File:Lange-MigrantMother02.jpg](https://commons.wikimedia.org/wiki/File:Lange-MigrantMother02.jpg) (Migrant Mother, 1936, Library of Congress) | Dorothea Lange | Public domain |
| `mawlamyine_kiss.jpg` | [File:Mother Kisses Child - Mawlamyine (Moulmein) - Myanmar (Burma) (11955282476).jpg](https://commons.wikimedia.org/wiki/File:Mother_Kisses_Child_-_Mawlamyine_(Moulmein)_-_Myanmar_(Burma)_(11955282476).jpg) | Adam Jones | CC BY-SA 2.0 |
| `lange_brawley.jpg` | [File:Dorothea Lange, Homeless mother and child near Brawley, California, 1939.jpg](https://commons.wikimedia.org/wiki/File:Dorothea_Lange,_Homeless_mother_and_child_near_Brawley,_California,_1939.jpg) | Dorothea Lange | Public domain |
| `lang_baby.jpg` | [File:Mrs. William Lang and Baby - DPLA - 516e0ca609ad2887324395fe7e72a103.jpg](https://commons.wikimedia.org/wiki/File:Mrs._William_Lang_and_Baby_-_DPLA_-_516e0ca609ad2887324395fe7e72a103.jpg) | Zintsmaster | Public domain |
| `half_selfie.jpg` | [File:Couple selfie stick Konigssluis Amsterdam 2016-09-13-6621.jpg](https://commons.wikimedia.org/wiki/File:Couple_selfie_stick_Konigssluis_Amsterdam_2016-09-13-6621.jpg) | Slaunger | CC BY-SA 4.0 |
| (numbers only) mother_daughter | [File:Portrait of a mother and baby daughter (I0053028).tif](https://commons.wikimedia.org/wiki/File:Portrait_of_a_mother_and_baby_daughter_(I0053028).tif) | unknown | Public domain |

The sheets made from CC BY-SA photos (`mawlamyine_kiss.jpg`, `half_selfie.jpg`)
are adaptations and are shared under the same license as their source.

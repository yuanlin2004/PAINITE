# Upstream network definitions (not part of PAINITE)

These files are unchanged copies of third-party code, kept here only so the
server tests can build the unblur networks without a download. They are **not**
installed with `painite-server` or included in its packages, and they are not
covered by PAINITE's license.

At run time the server uses the copy that `painite-server models pull`
downloads from the pinned upstream commit (and checks against the SHA-256 in
the model's manifest) into the model's install directory.

| File | Upstream | License |
| --- | --- | --- |
| `Stripformer.py` | [pp00704831/Stripformer-ECCV-2022-](https://github.com/pp00704831/Stripformer-ECCV-2022-) @ `9efba889`, `models/Stripformer.py` | Non-commercial MIT-style, [`LICENSE-Stripformer`](LICENSE-Stripformer) |
| `LaKDNet.py` | [lingyanruan/LaKDNet](https://github.com/lingyanruan/LaKDNet) @ `0c6b262c`, `models/LaKDNet.py` | AGPL-3.0, [`LICENSE-LaKDNet`](LICENSE-LaKDNet) |

[`LICENSE-facexlib`](LICENSE-facexlib) is the MIT license of
[xinntao/facexlib](https://github.com/xinntao/facexlib). It covers the RetinaFace
detection and BiSeNet face-parsing weights of `face/gpen-bfr-1024` (downloaded
from facexlib's releases, and kept in the `models-v1` mirror) and the BiSeNet
network vendored in `painite_server/models/bisenet_arch.py`. It is uploaded to
the mirror release next to the weights.

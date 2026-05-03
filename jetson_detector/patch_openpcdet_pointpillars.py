#!/usr/bin/env python3
"""Patch an OpenPCDet checkout down to the PointPillars inference surface.

OpenPCDet imports many optional model families from package ``__init__`` files.
On Jetson this forces compilation/import of PointNet, ROI pooling, and sparse
convolution components that the KITTI PointPillars config does not use. This
patch keeps the deployed checkout focused on PointPillars.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


FILES: dict[str, str] = {
    "pcdet/models/detectors/__init__.py": """from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'PointPillar': PointPillar,
}


def build_detector(model_cfg, num_class, dataset):
    return __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
""",
    "pcdet/models/backbones_3d/__init__.py": """__all__ = {}
""",
    "pcdet/models/backbones_3d/pfe/__init__.py": """__all__ = {}
""",
    "pcdet/models/backbones_3d/vfe/__init__.py": """from .pillar_vfe import PillarVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'PillarVFE': PillarVFE,
}
""",
    "pcdet/models/backbones_2d/map_to_bev/__init__.py": """from .pointpillar_scatter import PointPillarScatter, PointPillarScatter3d

__all__ = {
    'PointPillarScatter': PointPillarScatter,
    'PointPillarScatter3d': PointPillarScatter3d,
}
""",
    "pcdet/models/dense_heads/__init__.py": """from .anchor_head_single import AnchorHeadSingle

__all__ = {
    'AnchorHeadSingle': AnchorHeadSingle,
}
""",
    "pcdet/models/roi_heads/__init__.py": """__all__ = {}
""",
}


DETECTOR_TEMPLATE_IMPORTS = """from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
"""

POINTPILLAR_TEMPLATE_IMPORTS = """from .. import backbones_2d, dense_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import vfe

backbones_3d = None
pfe = None
roi_heads = None
"""


def backup(path: Path, suffix: str) -> None:
    backup_path = path.with_name(path.name + suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def write_file(root: Path, rel: str, text: str, suffix: str) -> None:
    path = root / rel
    backup(path, suffix)
    path.write_text(text, encoding="utf-8")


def patch_detector_template(root: Path, suffix: str) -> None:
    path = root / "pcdet/models/detectors/detector3d_template.py"
    backup(path, suffix)
    text = path.read_text(encoding="utf-8")
    if POINTPILLAR_TEMPLATE_IMPORTS not in text:
        if DETECTOR_TEMPLATE_IMPORTS not in text:
            raise RuntimeError("detector3d_template imports did not match expected OpenPCDet layout")
        text = text.replace(DETECTOR_TEMPLATE_IMPORTS, POINTPILLAR_TEMPLATE_IMPORTS)
        path.write_text(text, encoding="utf-8")


def copy_built_extensions(root: Path) -> None:
    required = {
        "pcdet/ops/iou3d_nms": "iou3d_nms_cuda*.so",
        "pcdet/ops/roiaware_pool3d": "roiaware_pool3d_cuda*.so",
    }
    for rel, pattern in required.items():
        candidates = sorted((root / "build").glob(f"lib.*/{rel}/{pattern}"))
        dest_dir = root / rel
        if not candidates:
            if list(dest_dir.glob(pattern)):
                continue
            raise RuntimeError(f"{pattern} extension is not built yet")
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[-1], dest_dir / candidates[-1].name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpcdet-dir", default="~/OpenPCDet")
    args = parser.parse_args()

    root = Path(args.openpcdet_dir).expanduser()
    if not (root / "pcdet").is_dir():
        raise SystemExit(f"OpenPCDet checkout not found at {root}")

    suffix = f".pids-full-{int(time.time())}.bak"
    for rel, text in FILES.items():
        write_file(root, rel, text, suffix)
    patch_detector_template(root, suffix)
    copy_built_extensions(root)
    print(f"OpenPCDet PointPillars-only patch applied at {root}")


if __name__ == "__main__":
    main()

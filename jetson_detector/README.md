# Jetson PointPillars Detector

This folder is the Jetson half of Path B:

```text
Laptop Ouster XYZ+signal -> ZMQ tcp://10.1.63.30:5555 -> Jetson OpenPCDet PointPillars
Jetson detections -> laptop PointCloudWindow class-colored boxes
```

## Jetson setup

The Jetson already needs a working CUDA PyTorch install. Then run:

```bash
cd ~/pids/jetson_detector
./setup_openpcdet.sh
```

The script clones `open-mmlab/OpenPCDet`, installs this server's Python
dependencies, installs OpenPCDet editable, and downloads the official KITTI
PointPillar checkpoint from the OpenPCDet model zoo:

- config: `~/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml`
- checkpoint: `~/OpenPCDet/checkpoints/pointpillar_7728.pth`

`spconv` does not publish Jetson/aarch64 CUDA wheels. The server therefore
bundles a minimal PointPillars-only `spconv` shim that provides CPU voxelization
and import compatibility for OpenPCDet. It is not intended for SECOND, PV-RCNN,
VoxelNeXt, or other sparse-convolution models.

If Google Drive blocks the automated checkpoint download, download the
OpenPCDet KITTI `PointPillar` `model-18M` file manually and place it at the
checkpoint path above.

## Run the real server

```bash
python3 pointpillars_server.py \
  --host 0.0.0.0 \
  --port 5555 \
  --cfg ~/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  --ckpt ~/OpenPCDet/checkpoints/pointpillar_7728.pth
```

## Run mock mode

Use this before OpenPCDet is installed to validate the laptop UI, ZMQ, msgpack,
and box rendering path:

```bash
python3 pointpillars_server.py --mock --host 0.0.0.0 --port 5555
```

## Laptop UI

Open the desktop GUI, connect the Ouster, open `Open 3D View`, and set
`Object analysis` to:

- `pointpillars`: Jetson-only detections
- `auto`: Jetson detections with DBSCAN fallback if the Jetson request fails
- `dbscan`: local DBSCAN only
- `off`: no object analysis

The default endpoint is `tcp://10.1.63.30:5555`.
The laptop client currently sends an evenly sampled 30k-point subset per scan
to keep the Jetson CPU voxelization path responsive.

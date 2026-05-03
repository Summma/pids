# Headless Jetson Backend

This is the end-state runtime: the Jetson owns the sensors and serves the same
websocket contract the React dashboard already uses.

## Hardware Topology

- Ouster LiDAR Ethernet -> Jetson Ethernet interface.
- Visible camera USB -> Jetson, usually `/dev/video0`.
- Optional Boson thermal USB -> Jetson, usually another `/dev/videoN`.
- Laptop/browser connects over Wi-Fi/LAN to `ws://<jetson-ip>:9090`.

The laptop should not need the sensors physically connected. SSH is only for
starting, stopping, and inspecting the Jetson process.

## First-Time Setup On Jetson

```bash
cd ~/narya/pids
cp pids/backend/jetson.env.example pids/backend/jetson.env
nano pids/backend/jetson.env
pids/backend/setup_jetson_backend.sh
```

Then run it directly:

```bash
pids/backend/run_jetson_backend.sh
```

Or install as a user service:

```bash
mkdir -p ~/.config/systemd/user
cp pids/backend/narya-backend.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now narya-backend
journalctl --user -u narya-backend -f
```

## Dashboard Connection

If the frontend build has been deployed, open:

```text
http://<jetson-ip>:9090
```

Because the page is served from the Jetson backend, the default websocket host
will also be the Jetson.

If you run the Vite dev server on the laptop instead, open the dashboard,
click `Settings`, set:

```text
Jetson Host: <jetson-ip-or-hostname>
Port: 9090
```

Expected health check:

```bash
curl http://<jetson-ip>:9090/health
```

Expected routes:

- `/lidar`: binary point cloud frames plus `indoor_human` detections.
- `/camera`: real camera JPEG frames.
- `/thermal`: optional Boson thermal frames if `PIDS_THERMAL_DEVICE` is set.
- `/gemini/chat`: multimodal scene analyst endpoint.

Optional PointPillars support:

```bash
python3 jetson_detector/pointpillars_server.py --host 127.0.0.1 --port 5555
```

Then set these in `pids/backend/jetson.env`:

```bash
PIDS_DETECTION_MODE=auto
PIDS_POINTPILLARS_ENDPOINT=tcp://127.0.0.1:5555
```

In `auto` mode, the web backend keeps `indoor_human` as the seated/occupied
chair detector and uses PointPillars as a secondary support/proposal signal,
not as the final seated-person decision.

## Notes

- `PIDS_LIDAR_HOST` should be the Ouster address as seen from the Jetson. If
  the LiDAR is directly attached over link-local Ethernet, `169.254.62.165` is
  the current known value.
- If the Jetson has multiple `/dev/video*` devices, run
  `v4l2-ctl --list-devices` and update `PIDS_CAMERA_DEVICE` /
  `PIDS_THERMAL_DEVICE`.
- The PySide Boson GUI is now just a local calibration/debug tool. It should
  not be required for the production dashboard path.

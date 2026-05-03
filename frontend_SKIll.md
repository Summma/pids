# Frontend Skill - Narya
## Expert UI System For Real-Time Perimeter Intelligence

Narya is not a dashboard skin. Narya is an operator-grade, real-time sensor-fusion interface for thermal, lidar, RF, calibration, mapping, and threat assessment. Every frontend decision must make the operator faster, calmer, and more certain.

Use this skill whenever building, modifying, or reviewing the Narya frontend.

---

## Product Principle

Narya turns raw sensor streams into spatial understanding.

The frontend must answer four questions at a glance:

1. What sensors are alive?
2. What is the environment doing right now?
3. What objects or signals matter?
4. What should the operator inspect next?

The UI should feel like an instrument panel, not a website. It should be dense, quiet, precise, and alive.

---

## Current Stack

Use the existing React/Vite architecture.

```text
pids-frontend/
  src/
    App.jsx
    hooks/
    panels/
    components/
    styles/
    utils/
```

Approved core tools:

- React for panel composition and state ownership
- Vite for local development and production builds
- Three.js for 3D lidar, point clouds, sensor replicas, and spatial scenes
- Canvas 2D for lightweight spectrum, thermal, and map drawing
- CSS modules for panel-local styling
- Shared tokens in `src/styles/tokens.css`

Do not revert to the old single-file `index.html` approach. That was a prototype. Narya is now a modular application.

---

## Architecture Rules

### Stream Ownership

Open each WebSocket stream once at the application level.

`App.jsx` owns:

```text
useThermal()
useLidar()
useRF()
useThreats()
useFusion()
```

Panels receive stream objects as props. Do not create duplicate WebSocket connections inside panels.

Correct:

```jsx
const lidar = useLidar()
<LidarPanel lidarData={lidar} />
```

Wrong:

```jsx
function LidarPanel() {
  const lidar = useLidar()
}
```

### Backend Does The Thinking

The Jetson/backend should perform:

- thermal normalization when possible
- lidar scan conversion to XYZ
- lidar decimation/capping
- clustering and tracking
- RF FFT and peak detection
- fusion map generation
- threat scoring
- calibration projection math

The frontend should:

- decode stream payloads
- update typed arrays
- draw the latest state
- support inspection and control
- avoid doing expensive perception work

The frontend is allowed to render. It is not allowed to become the perception pipeline.

---

## Visual Identity

Narya should feel technical, restrained, and immediate.

### Tone

- Calm under pressure
- High contrast but not loud
- Scan-first, not decorative
- Spatial, instrument-like, and exact

### Avoid

- Marketing hero layouts
- Decorative gradients as backgrounds
- Floating cards nested inside cards
- Oversized typography in panels
- One-note blue/purple monotony
- UI text that explains obvious controls
- Anything that makes the app feel like a mockup rather than a tool

---

## Color System

Use existing tokens first.

```css
:root {
  --bg:        #080c10;
  --surface:   #0d1520;
  --surface-2: #131e2e;
  --surface-3: #192438;

  --border:    #1e3a5f;
  --border-2:  #2a4f7a;

  --accent:    #00d4ff;
  --accent-lo: rgba(0, 212, 255, 0.08);
  --accent-md: rgba(0, 212, 255, 0.16);

  --live:    #00d4ff;
  --warn:    #ff8c42;
  --danger:  #ff2244;
  --success: #00e676;

  --thermal-color: #ff6b35;
  --lidar-color:   #00d4ff;
  --rf-color:      #c77dff;
}
```

### Semantic Use

- Cyan: live state, lidar, active outlines
- Orange: thermal identity, warnings, hot objects
- Purple: RF identity
- Green: clear/safe/passive success
- Red: critical threat, failure, destructive state
- Yellow: selected track, peak marker, medium urgency

### Thermal Ramp

Thermal data may use a blue-to-red ramp. UI chrome should not copy the full thermal ramp unless representing actual thermal values.

Preferred thermal ramp:

```text
cold    #0044ff
cool    #00aaff
middle  #00ff88
warm    #ffcc00
hot     #ff2200
```

---

## Typography

Use `JetBrains Mono` for the current Narya interface unless the design system is intentionally changed.

Panel text should be compact:

```text
9px  - tiny state, axis labels
10px - buttons, metadata, legends
11px - panel titles
13px - body/default
18px - exceptional emphasis only
```

Rules:

- Letter spacing may be positive for labels, never negative.
- Do not scale fonts with viewport width.
- Long labels must truncate or use shorter labels.
- Panel headers must never overlap controls.

---

## Layout Model

Narya uses a fixed operator shell:

```text
┌──────────────────────────────────────────────────────────┐
│ TitleBar: system state, alerts, settings, calibration    │
├──────────────────┬──────────────────┬──────────────────┤
│ Thermal          │ Lidar 3D          │ Fusion Map       │
│                  │                  │                  │
├──────────────────┴───────┬──────────┴──────────────────┤
│ RF Spectrum              │ Threat Fusion                │
└──────────────────────────┴──────────────────────────────┘
```

Current top row:

```text
ThermalPanel | LidarPanel | FusionMapPanel
```

Current bottom row:

```text
RFPanel | ThreatPanel
```

### Panel Rules

- Use `PanelShell` for all primary panels.
- Header height must remain compact.
- Header controls must fit in their lane.
- Use short button labels plus `title` tooltips.
- Body content should fill available space.
- No panel should require page scroll.

---

## Panel Responsibilities

### Thermal Panel

Purpose: show live heat context and basic radiometric readout.

Inputs:

```text
/thermal
```

Expected behavior:

- Render latest thermal frame.
- Show min/max thermal scale.
- Support palette selection.
- Support FFC command.
- Support snapshot.
- Crosshair readout should be quick and non-blocking.

Performance:

- Render only on new frame or palette change.
- Do not recolor at display refresh rate.
- Prefer backend-colored frames if CPU becomes an issue.

Future ideal:

- Backend sends RGB/RGBA or compressed image frame.
- Frontend only blits pixels.

### Lidar Panel

Purpose: show an embedded 3D replica of the lidar sensor and its live point cloud.

Inputs:

```text
/lidar
```

Current design:

- Three.js scene inside the dashboard panel.
- Small lidar body at origin.
- Ground grid.
- Range rings.
- Live point cloud.
- Color modes:
  - intensity
  - height
- Pop-out full 3D view available.

Rules:

- Do not use 2D bird's-eye as the primary lidar panel.
- The main lidar panel should feel like a physical sensor view.
- Points should be GPU-rendered through `BufferGeometry`.
- Mutate preallocated buffers where possible.
- Keep dashboard 3D lighter than the full pop-out view.

### Fusion Map Panel

Purpose: render the fused understanding of the area.

Inputs:

```text
/fusion
/lidar fallback
/threats fallback
```

It may show:

- occupancy grid
- fused tracks
- zones
- sensor pose
- lidar fallback points
- threat fallback tracks

Rules:

- Fusion data wins over raw fallback data.
- Occupancy and zones should read as world structure.
- Tracks should be inspectable.
- The map should not pretend certainty; use visual confidence.

### RF Panel

Purpose: show spectrum state and RF emitters.

Inputs:

```text
/rf
```

Expected behavior:

- Spectrum line/fill.
- Noise floor.
- Peak labels.
- Optional waterfall.
- Band selection commands.

Performance:

- Draw only on new FFT frame.
- FFT computation belongs on backend.
- Peak detection belongs on backend.

### Threat Panel

Purpose: turn fused detections into an operator queue.

Inputs:

```text
/threats
```

Expected behavior:

- Sort by confidence, range, or speed.
- Show modalities contributing to the threat.
- Use compact, card-based repeated items.
- Critical threats should be obvious without making the entire app frantic.

---

## Data Contracts

### `/thermal`

JSON WebSocket frame:

```json
{
  "type": "frame",
  "w": 640,
  "h": 512,
  "data": "base64_8bit_pixels",
  "t_min": 18.2,
  "t_max": 42.7,
  "radiometric": true,
  "seq": 123,
  "ts": 1714700000.123
}
```

### `/lidar`

JSON envelope with base64 binary payload:

```json
{
  "type": "frame",
  "n": 60000,
  "data": "base64_float32_xyz_intensity",
  "clusters": [],
  "seq": 123,
  "ts": 1714700000.123
}
```

Binary point layout, little-endian:

```text
offset 0   float32 x meters
offset 4   float32 y meters
offset 8   float32 z meters
offset 12  float32 intensity normalized 0..1
```

### `/fusion`

```json
{
  "type": "frame",
  "map": {
    "width": 120,
    "height": 120,
    "resolution": 0.5,
    "origin": { "x": -30, "z": -30 },
    "cells": [0, 0.2, 0.8]
  },
  "tracks": [
    {
      "id": 1,
      "position": { "x": 8.2, "z": 14.5 },
      "velocity": { "x": 0.1, "z": -0.3 },
      "confidence": 0.88,
      "speed": 0.32
    }
  ],
  "zones": [],
  "pose": { "x": 0, "z": 0 },
  "seq": 123,
  "ts": 1714700000.123
}
```

### `/rf`

```json
{
  "type": "frame",
  "n_fft": 1024,
  "data": "base64_float32_fft_bins",
  "center_freq": 433920000,
  "sample_rate": 2400000,
  "noise_floor": -85,
  "peaks": [
    { "freq": 433920000, "power": -42.1, "label": "ISM" }
  ],
  "ts": 1714700000.123
}
```

### `/threats`

```json
{
  "type": "frame",
  "threats": [
    {
      "id": 1,
      "confidence": 0.91,
      "range": 18.4,
      "bearing": 32,
      "speed": 1.2,
      "age_frames": 12,
      "modalities": ["lidar", "thermal"]
    }
  ],
  "ts": 1714700000.123
}
```

---

## Performance Rules

Narya should prefer backend computation and frontend rendering.

### Hard Rules

- Do not run expensive perception logic in React render paths.
- Do not create WebSocket connections per component.
- Do not allocate large typed arrays inside animation loops.
- Do not redraw canvas panels at 60 FPS unless interaction requires it.
- Do not colorize every point on the CPU when a shader can do it.
- Do not send unlimited lidar points to the browser.
- Cap point cloud size in backend and frontend.

### Expected Frontend CPU Work

Allowed:

- WebSocket JSON parse
- base64 decode for current protocol
- typed-array copy into preallocated buffers
- canvas draw of current frame
- Three.js render of visible 3D scenes

Not allowed:

- DBSCAN
- tracking
- thermal-to-lidar projection
- FFT
- threat scoring
- map accumulation
- mesh reconstruction

### Rendering Strategy

Thermal:

- draw on `meta.seq`
- avoid continuous RAF

Lidar dashboard:

- update geometry on `meta.seq`
- render scene after data update or user orbit

Fusion:

- draw on fusion frame or fallback lidar/threat update

RF:

- draw on RF timestamp update

3D pop-out:

- may use RAF while open because orbit controls and full scene interaction need it

---

## Three.js Guidance

Use Three.js for all 3D elements.

For point clouds:

```js
const positions = new Float32Array(MAX_POINTS * 3)
const intensities = new Float32Array(MAX_POINTS)

const geometry = new THREE.BufferGeometry()
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
geometry.setAttribute('intensity', new THREE.BufferAttribute(intensities, 1))
geometry.setDrawRange(0, 0)
```

Update existing buffers:

```js
positions.set(frame.positions.subarray(0, n * 3))
intensities.set(frame.intensities.subarray(0, n))
geometry.attributes.position.needsUpdate = true
geometry.attributes.intensity.needsUpdate = true
geometry.setDrawRange(0, n)
```

Never recreate geometry per frame.

### Shader Color

Use shaders for point coloring where possible.

Intensity mode:

```glsl
vColor = ramp(intensity);
```

Height mode:

```glsl
vColor = ramp((position.y + 2.0) / 4.0);
```

Thermal mode, future:

```glsl
vColor = thermalRamp(temp);
```

---

## Interaction Design

Controls should be short and physical.

Use:

- `INT` for intensity
- `HT` for height
- `FUSE` for fusion layer
- `OCC` for occupancy
- `TRK` for tracks
- `FOL` for follow
- `POP` for pop-out

Add `title` attributes for clarity.

Do not put long explanatory text in panel headers. The operator learns the instrument; the UI should not chatter.

---

## Connection States

Every panel should show a connection state through `PanelShell`.

States:

```text
connecting
live
stale
offline
```

Meaning:

- `connecting`: socket opening or reconnecting
- `live`: receiving frames
- `stale`: socket exists but frames stopped
- `offline`: socket closed or failed

Future improvement: hooks should mark stale based on last frame timestamp.

---

## Backend Contract

The backend runs on Jetson and binds to:

```bash
python3 server.py --host 0.0.0.0 --port 9090
```

Thermal:

```bash
--device 0 --fps 2
```

Lidar:

```bash
--lidar-host 169.254.62.165 --lidar-fps 3 --lidar-max-points 60000
```

The frontend default connects to:

```text
ws://<jetson-ip>:9090
```

The UI settings panel must let the operator set host and port.

---

## Expert Review Checklist

Before shipping a Narya frontend change, verify:

- Build passes with `npm run build`.
- No panel header overlaps at narrow widths.
- No duplicate WebSocket streams are opened.
- No unnecessary `requestAnimationFrame` loops were added.
- The frontend remains useful when one sensor is offline.
- Connection state is visible.
- Main information is visible without scrolling.
- 3D scenes are not blank.
- Lidar point count is capped.
- Backend owns sensor processing.
- Controls are compact and understandable.

---

## Design North Star

Narya should feel like the operator is holding a live model of the perimeter.

The best version of Narya is not flashy. It is lucid.

It lets someone glance at the screen and know:

```text
the system is alive
the map is forming
the sensor is oriented
the threat queue is credible
the next action is obvious
```

Build toward that.

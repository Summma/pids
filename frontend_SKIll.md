# Frontend Skill — Real-Time Thermal-Fused 3D World Viewer
## Specification v2.0

---

## Framework Decision

**Three.js r165 via importmap (ES modules). Single `index.html`. No build step.**

```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"
  }
}
</script>
<script type="module">
  import * as THREE from 'three'
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
  // ... all app code here
</script>
```

### Why Three.js wins

| Framework | Verdict | Reason |
|---|---|---|
| **Three.js + importmap** | ✅ USE THIS | OrbitControls built-in, BufferGeometry GPU path optimal, ShaderMaterial = free GPU thermal coloring |
| React / Vue | ❌ | vDOM reconciler fights the 10fps buffer hot-path. `needsUpdate = true` must be zero-overhead |
| Babylon.js | ❌ | 2.5× bundle, built for meshes/physics, no benefit for point clouds |
| Deck.gl | ❌ | Geospatial-first, requires React, wrong domain |
| Raw WebGL | ❌ | 3× code, OrbitControls alone = hours |
| Potree | ❌ | Offline LAS/LAZ files only, not real-time WebSocket |

---

## Design Identity

### Color Tokens

```css
:root {
  --bg:         #080c10;   /* canvas background */
  --surface:    #0d1520;   /* HUD panels */
  --surface-2:  #131e2e;   /* secondary surfaces, hover */
  --border:     #1e3a5f;   /* panel borders */
  --accent:     #00d4ff;   /* LIVE indicator, highlights */
  --accent-dim: #00446688;
  --warn:       #ff8c42;   /* STALE, warnings */
  --danger:     #ff2244;   /* OFFLINE, critical */
  --text-pri:   #e8f4f8;
  --text-sec:   #6a8fa8;
}
```

**Rule: Blue→red hues are reserved for the thermal ramp only. Never use them for UI chrome.**

### Thermal Ramp (5-stop)

```
0.00 → #0044ff  cold
0.25 → #00aaff
0.50 → #00ff88
0.75 → #ffcc00
1.00 → #ff2200  hot
```

Implemented in GLSL — NOT in JS per-point. See shader section.

### Typography

- Font: `JetBrains Mono` (Google Fonts CDN)
- Sizes: 11px labels · 13px HUD values · 15px panel titles · 22px large stats
- Weights: 400 body · 600 values · 700 headers

---

## File Structure

```
index.html   ← entire app, self-contained
```

All CSS in `<style>`. All JS in `<script type="module">`. CDN imports only via importmap.

Additional CDN:
```html
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
```

---

## Layout

```
┌──────────────────────────────────────────────────────────┐
│  TITLE BAR 40px  —  ◈ NARYA · STREAM NAME · ● LIVE      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│              THREE.JS CANVAS (full viewport)             │
│                                                          │
│  [axes 120×120 inset, bottom-left corner of canvas]      │
│                                                          │
│  ┌───────────────┐              ┌─────────────────────┐  │
│  │  STATS HUD    │              │  THERMAL LEGEND     │  │
│  └───────────────┘              └─────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

- Title bar: `position: fixed; top: 0; width: 100%; height: 40px; z-index: 100`
- Canvas: `position: fixed; top: 40px; left: 0; right: 0; bottom: 0`
- Stats HUD: `position: fixed; bottom: 20px; left: 20px; pointer-events: none`
- Thermal legend: `position: fixed; bottom: 20px; right: 20px; pointer-events: auto`

---

## GLSL Vertex Shader (thermal coloring on GPU)

This replaces ALL per-point JS color computation. Zero CPU cost.

```glsl
// Vertex shader — stored as JS template literal VERT_SHADER
uniform float uTmin;
uniform float uTmax;
attribute float temp;
varying vec3 vColor;

vec3 thermalRamp(float t) {
  vec3 c0 = vec3(0.000, 0.267, 1.000);  // #0044ff  cold
  vec3 c1 = vec3(0.000, 0.667, 1.000);  // #00aaff
  vec3 c2 = vec3(0.000, 1.000, 0.533);  // #00ff88
  vec3 c3 = vec3(1.000, 0.800, 0.000);  // #ffcc00
  vec3 c4 = vec3(1.000, 0.133, 0.000);  // #ff2200  hot

  float u = clamp((t - uTmin) / (uTmax - uTmin + 0.001), 0.0, 1.0);

  if (u < 0.25) return mix(c0, c1, u / 0.25);
  if (u < 0.50) return mix(c1, c2, (u - 0.25) / 0.25);
  if (u < 0.75) return mix(c2, c3, (u - 0.50) / 0.25);
               return mix(c3, c4, (u - 0.75) / 0.25);
}

void main() {
  vColor = thermalRamp(temp);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = 2.5;
}
```

```glsl
// Fragment shader — round points (discard corners)
varying vec3 vColor;
void main() {
  float d = length(gl_PointCoord - vec2(0.5));
  if (d > 0.5) discard;
  gl_FragColor = vec4(vColor, 0.92);
}
```

Use `THREE.ShaderMaterial`:
```js
const material = new THREE.ShaderMaterial({
  uniforms: {
    uTmin: { value: 0.0 },
    uTmax: { value: 100.0 },
  },
  vertexShader: VERT_SHADER,
  fragmentShader: FRAG_SHADER,
})
```

Update uniforms per frame (not per point):
```js
material.uniforms.uTmin.value = envelope.t_min
material.uniforms.uTmax.value = envelope.t_max
```

---

## Three.js Scene Setup

```js
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

const MAX_POINTS = 8192

// Renderer
const renderer = new THREE.WebGLRenderer({ antialias: false })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setSize(window.innerWidth, window.innerHeight - 40)
document.getElementById('canvas-container').appendChild(renderer.domElement)

// Scene
const scene = new THREE.Scene()
scene.background = new THREE.Color(0x080c10)

// Camera
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / (window.innerHeight - 40), 0.1, 500)
camera.position.set(0, 8, 20)

// Controls
const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true
controls.dampingFactor = 0.05

// Ground grid
const grid = new THREE.GridHelper(40, 40, 0x1e3a5f, 0x0d1a2e)
scene.add(grid)

// Pre-allocated buffers — NEVER allocate inside message handler
const positions = new Float32Array(MAX_POINTS * 3)
const temps     = new Float32Array(MAX_POINTS)

const geometry = new THREE.BufferGeometry()
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
geometry.setAttribute('temp',     new THREE.BufferAttribute(temps, 1))
geometry.setDrawRange(0, 0)

const points = new THREE.Points(geometry, material)
scene.add(points)

// Axes inset (separate scene, renders in 120×120 corner)
const axesScene  = new THREE.Scene()
const axesCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 10)
axesCamera.position.set(2, 2, 2)
axesScene.add(new THREE.AxesHelper(1))

// Render loop
function animate() {
  requestAnimationFrame(animate)
  controls.update()

  // Main scene
  renderer.setViewport(0, 0, window.innerWidth, window.innerHeight - 40)
  renderer.setScissorTest(false)
  renderer.render(scene, camera)

  // Axes inset — 120×120 bottom-left
  const S = 120
  renderer.setViewport(12, 12, S, S)
  renderer.setScissor(12, 12, S, S)
  renderer.setScissorTest(true)
  axesCamera.quaternion.copy(camera.quaternion)
  renderer.render(axesScene, axesCamera)
  renderer.setScissorTest(false)
}
animate()
```

---

## Binary Parse — Zero Allocation

**NEVER use `Uint8Array.from(atob(...), c => c.charCodeAt(0))` — the callback allocates.**

Pre-allocate the raw decode buffer alongside the point buffers:

```js
const rawBuffer = new Uint8Array(MAX_POINTS * 16)  // 131,072 bytes, allocated once
```

```js
function handleFrame(msg) {
  const envelope = JSON.parse(msg.data)
  if (envelope.type !== 'frame') return

  const binaryStr = atob(envelope.data)
  const len = Math.min(binaryStr.length, rawBuffer.length)

  // Manual loop — fastest charCode path in V8, no callback allocation
  for (let i = 0; i < len; i++) rawBuffer[i] = binaryStr.charCodeAt(i)

  const view = new DataView(rawBuffer.buffer, 0, len)
  const n = Math.min(envelope.n, MAX_POINTS)

  for (let i = 0; i < n; i++) {
    const b = i * 16
    positions[i * 3]     = view.getFloat32(b,      true)  // x
    positions[i * 3 + 1] = view.getFloat32(b + 4,  true)  // y
    positions[i * 3 + 2] = view.getFloat32(b + 8,  true)  // z
    temps[i]             = view.getFloat32(b + 12, true)  // temp °C
  }

  // GPU upload
  geometry.attributes.position.needsUpdate = true
  geometry.attributes.temp.needsUpdate = true
  geometry.setDrawRange(0, n)

  // Update uniforms (not per-point — just twice per frame)
  material.uniforms.uTmin.value = envelope.t_min
  material.uniforms.uTmax.value = envelope.t_max

  // Update HUD
  updateStats(envelope)
  lastFrameTs = Date.now()
  setState('LIVE')
}
```

---

## WebSocket + Auto-Reconnect

```js
let retryDelay = 1000
let ws = null

function connect(url) {
  setState('CONNECTING')
  ws = new WebSocket(url)
  ws.onopen  = () => { retryDelay = 1000 }
  ws.onclose = () => {
    setState('OFFLINE')
    setTimeout(() => connect(url), retryDelay)
    retryDelay = Math.min(retryDelay * 2, 16000)
  }
  ws.onerror = () => setState('OFFLINE')
  ws.onmessage = handleFrame
}

// Auto-start sim if no connection after 3s
setTimeout(() => {
  if (state !== 'LIVE') startSimMode()
}, 3000)
```

---

## Demo / Simulation Mode

Activated by `D` key or auto after 3s with no Jetson connection.
Shows `⬡ SIM` amber pill. Full UI is functional.

```js
let simInterval = null

function startSimMode() {
  setState('SIM')
  const N = 4096
  let t = 0
  simInterval = setInterval(() => {
    t += 0.05
    for (let i = 0; i < N; i++) {
      const angle = (i / N) * Math.PI * 2 + t
      const r = 3 + Math.sin(i * 0.1 + t) * 2
      positions[i * 3]     = Math.cos(angle) * r
      positions[i * 3 + 1] = Math.sin(i * 0.05 + t) * 1.5 + (Math.random() - 0.5) * 0.05
      positions[i * 3 + 2] = Math.sin(angle) * r
      temps[i] = 20 + 40 * (0.5 + 0.5 * Math.sin(i * 0.03 + t * 2))
    }
    geometry.attributes.position.needsUpdate = true
    geometry.attributes.temp.needsUpdate = true
    geometry.setDrawRange(0, N)
    material.uniforms.uTmin.value = 20
    material.uniforms.uTmax.value = 60
    updateStats({ n: N, t_min: 20, t_max: 60, seq: simSeq++, ts: Date.now() / 1000 })
  }, 100)
}

function stopSimMode() {
  clearInterval(simInterval)
  simInterval = null
}
```

---

## Connection State Machine

| State | Pill | Color | Trigger |
|---|---|---|---|
| `CONNECTING` | `◌ CONNECTING` | amber pulse | Socket opening |
| `LIVE` | `● LIVE` | `--accent` pulse | Frame received |
| `STALE` | `⚠ STALE` | `--warn` | No frame > 500ms |
| `OFFLINE` | `✕ OFFLINE` | `--danger` | Socket closed/error |
| `SIM` | `⬡ SIM` | amber steady | Demo mode |

```js
let state = 'CONNECTING'
let lastFrameTs = 0

// Stale detection
setInterval(() => {
  if (state === 'LIVE' && Date.now() - lastFrameTs > 500) setState('STALE')
}, 100)

function setState(s) {
  state = s
  // update pill DOM
}
```

---

## Stats HUD (bottom-left)

```
STREAM
─────────────────────────────
Points          8 192
Frame rate      10.0 fps
Latency         14 ms
Frames rx       1 247
─────────────────────────────
T min           18.4 °C      ← colored #0044ff
T max           67.2 °C      ← colored #ff2200
T mean          31.6 °C      ← colored mid-ramp
T spread        48.8 °C
```

- `pointer-events: none`
- Temperature values set `style.color` via JS `thermalHex(t, tMin, tMax)` — interpolated 5-stop ramp returning CSS rgb()

---

## Thermal Legend (bottom-right)

```
TEMPERATURE  °C
█████████████████████████████
COLD 18.4 °C          67.2 °C HOT

[ AUTO RANGE ]   [ LOCK ]
```

- Gradient bar: `linear-gradient(to right, #0044ff, #00aaff, #00ff88, #ffcc00, #ff2200)`
- Min/max labels update every frame
- LOCK: sets `frozen = true` — stops buffer writes, render loop continues
- `pointer-events: auto`

---

## Settings Panel (S key)

```
SETTINGS                                    ✕
──────────────────────────────────────────────
WebSocket URL    ws://[ 192.168.1.42 ]:9090
Show grid        [✓]
Show axes        [✓]
Demo mode        [ ]
──────────────────────────────────────────────
                              [ APPLY ]
```

Safe localStorage wrapper (survives sandboxed iframes + private mode):
```js
const store = {
  get: (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d } catch { return d } },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)) } catch {} }
}
```

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Lock / unlock frame |
| `R` | Reset camera to default |
| `N` | Toggle auto-range normalize |
| `F` | Fullscreen |
| `S` | Open / close settings |
| `D` | Toggle demo / sim mode |
| `?` | Keybindings overlay |
| `Esc` | Close overlays |

---

## No-Signal State

When `state === 'OFFLINE'` and no frames ever received:

```
        ◈
  AWAITING STREAM
  ws://192.168.1.42:9090
  Reconnecting in 4s...
```

Centered over canvas. Grid still renders beneath it.

---

## Performance Budget

| Step | Budget | Method |
|---|---|---|
| atob + parse loop | < 2ms | Pre-allocated Uint8Array, manual charCode loop |
| Thermal coloring | 0ms | GLSL vertex shader on GPU |
| GPU buffer upload | < 2ms | `needsUpdate = true` pattern |
| Three.js render | < 6ms | antialias: false, pixelRatio capped at 2 |
| **Total @ 10fps** | **< 10ms** | 90ms headroom per frame |

### Hard rules
- Never `new Float32Array()` inside `onmessage`
- Pre-allocate `rawBuffer` at init — never inside the message handler
- Never recreate `BufferGeometry` — mutate attributes in place
- Always `geometry.setDrawRange(0, n)` for variable point counts
- Update uniforms once per frame, not per point

---

## Data Contract

### Frame envelope (server → client, 10 fps)

```json
{
  "type": "frame",
  "seq": 1247,
  "ts": 1714500000.123,
  "n": 8192,
  "t_min": 18.4,
  "t_max": 67.2,
  "data": "<base64-encoded binary blob>"
}
```

### Binary blob layout (per point, 16 bytes, little-endian float32)

| Offset | Field | Type | Unit |
|---|---|---|---|
| 0 | x | float32 | metres |
| 4 | y | float32 | metres |
| 8 | z | float32 | metres |
| 12 | temp | float32 | °C |

---

## Changes from v1.0

| # | Change | Why |
|---|---|---|
| 1 | importmap instead of bare CDN script tags | Cleaner, no relative path issues, native Chrome support |
| 2 | GLSL vertex shader for thermal coloring | Moves 81,920 color computations/sec from CPU to GPU |
| 3 | Fragment shader round point discard | Points render as circles, not squares — much more readable |
| 4 | Pre-allocated rawBuffer + manual charCode loop | Fastest V8 path, zero allocation in hot path |
| 5 | Demo / sim mode | Full UI without Jetson — essential for development/demo |
| 6 | SIM state added to state machine | Operator always knows what they're looking at |
| 7 | localStorage safe wrapper | Survives sandboxed iframes and private browsing |

---

## Deliverable

Single `index.html`. Open in Chrome. No install. No build.

- **With Jetson:** Point to `ws://<jetson-ip>:9090` → world appears in < 2s
- **Without Jetson:** Press `D` or wait 3s → sim mode → full demo ready

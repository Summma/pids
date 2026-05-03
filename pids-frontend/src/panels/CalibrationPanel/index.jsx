import { useEffect, useMemo, useRef, useState } from 'react'
import { PALETTES } from '@/utils/palettes'
import styles from './CalibrationPanel.module.css'

const LENS_PRESETS = {
  'Boson 8.7mm (~95deg)': 95.0,
  'Boson 13.1mm (~32.7deg)': 32.66,
  'Boson 14mm (~50deg)': 50.0,
  'Boson 18mm (~40deg)': 40.0,
  'Boson 24mm (~24deg)': 24.0,
  'Boson 36mm (~16deg)': 16.0,
  'Boson 60mm (~10deg)': 10.0,
}

const CHANNELS = [
  { key: 'intensity', label: 'Reflectivity / signal' },
  { key: 'range', label: 'Range' },
]

const WEBCAM_HFOV_DEG = 55

const DEFAULT_CALIBRATION = {
  intr: { width: 640, height: 512, fx: 686, fy: 686, cx: 320, cy: 256 },
  extr: { tx: 0, ty: 0, tz: 0, rollDeg: 0, pitchDeg: 0, yawDeg: 0 },
  source: 'gui_default',
}

const R_LIDAR_TO_CAM_BASE = [
  [0, -1, 0],
  [0, 0, -1],
  [1, 0, 0],
]

export default function CalibrationPanel({
  lidarData,
  thermalData,
  cameraData,
  calibrationOverride = null,
  onCalibrationChange,
  onClose,
}) {
  const canvasRef = useRef(null)
  const thermalCanvasRef = useRef(null)
  const cameraImageRef = useRef(null)
  const [cameraReadySeq, setCameraReadySeq] = useState(0)
  const fileRef = useRef(null)

  const liveCalibration = useMemo(
    () => normalizeThermalCalibration(thermalData?.meta?.calibration),
    [thermalData?.meta?.calibration],
  )
  const activeCalibration = useMemo(
    () => calibrationOverride
      ? normalizeThermalCalibration(calibrationOverride)
      : liveCalibration,
    [calibrationOverride, liveCalibration],
  )
  const [cal, setCal] = useState(activeCalibration)
  const [channel, setChannel] = useState('intensity')
  const [lens, setLens] = useState('Boson 14mm (~50deg)')
  const [saved, setSaved] = useState(false)
  const [stats, setStats] = useState({ projected: 0, points: 0 })
  const [webcamRotation, setWebcamRotation] = useState({ rollDeg: 0, pitchDeg: 0, yawDeg: 0 })

  useEffect(() => {
    setCal(activeCalibration)
  }, [activeCalibration])

  // Time-sync to the slowest stream. When the lidar ticks, pull the
  // camera and thermal frames closest in ts so the three panes show the
  // same captured moment instead of "latest of each".
  const lidarTs = lidarData?.meta?.ts ?? 0
  const cameraSync = cameraData?.frameAt ? cameraData.frameAt(lidarTs) : cameraData?.frame
  const thermalSync = thermalData?.frameAt ? thermalData.frameAt(lidarTs) : thermalData?.frameRef?.current
  const cameraSyncSrc = cameraSync?.src ?? ''
  const cameraSyncTs = cameraSync?.ts ?? 0
  const thermalSyncTs = thermalSync?.ts ?? 0

  useEffect(() => {
    if (!cameraSyncSrc) {
      cameraImageRef.current = null
      return
    }
    const img = new Image()
    img.onload = () => {
      cameraImageRef.current = img
      setCameraReadySeq(seq => seq + 1)
    }
    img.src = cameraSyncSrc
  }, [cameraSyncSrc])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rendered = renderCalibrationCanvas({
      canvas,
      thermalCanvasRef,
      cameraImage: cameraImageRef.current,
      lidarFrame: lidarData?.frameRef?.current,
      thermalFrame: thermalSync,
      cameraFrame: cameraSync,
      calibration: cal,
      channel,
      webcamRotation,
    })
    setStats(rendered)
  }, [cal, channel, lidarData?.meta?.seq, cameraReadySeq, cameraSyncTs, thermalSyncTs, lidarData?.frameRef, webcamRotation])

  function commitCalibration(nextCalibration) {
    setCal(nextCalibration)
    onCalibrationChange?.(nextCalibration)
  }

  function updateExtr(key, value) {
    commitCalibration({
      ...cal,
      extr: {
        ...cal.extr,
        [key]: Number(value),
      },
      source: 'web_override',
    })
  }

  function updateLens(name) {
    setLens(name)
    commitCalibration({
      ...cal,
      intr: intrinsicsFromHfov(cal.intr.width, cal.intr.height, LENS_PRESETS[name]),
      source: 'web_override',
    })
  }

  function resetExtrinsics() {
    commitCalibration({
      ...cal,
      extr: { ...DEFAULT_CALIBRATION.extr },
      source: 'web_override',
    })
  }

  function useLiveCalibration() {
    setCal(liveCalibration)
    onCalibrationChange?.(null)
  }

  function saveJSON() {
    const blob = new Blob([JSON.stringify(calibrationToJson(cal), null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'calibration.json'
    a.click()
    URL.revokeObjectURL(a.href)
    setSaved(true)
    window.setTimeout(() => setSaved(false), 1600)
  }

  function loadJSON(event) {
    const file = event.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        commitCalibration({
          ...normalizeThermalCalibration(JSON.parse(String(ev.target?.result ?? '{}'))),
          source: 'loaded_json',
        })
      } catch {}
    }
    reader.readAsText(file)
    event.target.value = ''
  }

  const lidarState = lidarData?.connState ?? 'connecting'
  const thermalState = thermalData?.connState ?? 'connecting'
  const cameraState = cameraData?.connState ?? 'connecting'
  const pointText = `${stats.projected.toLocaleString()} / ${stats.points.toLocaleString()} pts`
  const sourceText = cal.source || 'gui_default'
  const syncCameraMs = lidarTs > 0 && cameraSyncTs > 0 ? Math.round((cameraSyncTs - lidarTs) * 1000) : null
  const syncThermalMs = lidarTs > 0 && thermalSyncTs > 0 ? Math.round((thermalSyncTs - lidarTs) * 1000) : null
  const syncText = (syncCameraMs !== null || syncThermalMs !== null)
    ? `sync Δcam ${formatSync(syncCameraMs)} Δtherm ${formatSync(syncThermalMs)}`
    : ''

  return (
    <div className={styles.overlay}>
      <div className={styles.header}>
        <span className={styles.title}>Thermal-Lidar Calibration</span>
        <div className={styles.headerRight}>
          <span className={styles.status}>
            Lidar {lidarState} | Thermal {thermalState} | Camera {cameraState} | {pointText}
            {syncText ? ` | ${syncText}` : ''}
          </span>
          <button className="btn" onClick={onClose}>Close</button>
        </div>
      </div>

      <div className={styles.body}>
        <div className={styles.viewport}>
          <canvas ref={canvasRef} className={styles.canvas} />
        </div>

        <div className={styles.controls}>
          <section className={styles.section}>
            <div className={styles.sectionTitle}>View</div>
            <div className={styles.fieldRow}>
              <label className={styles.fieldLabel}>Channel</label>
              <select
                className={styles.select}
                value={channel}
                onChange={event => setChannel(event.target.value)}
              >
                {CHANNELS.map(item => (
                  <option key={item.key} value={item.key}>{item.label}</option>
                ))}
              </select>
            </div>
            <div className={styles.readout}>
              {cal.intr.width}x{cal.intr.height} | fx {cal.intr.fx.toFixed(1)} | fy {cal.intr.fy.toFixed(1)}
            </div>
          </section>

          <section className={styles.section}>
            <div className={styles.sectionTitle}>Extrinsics</div>
            <div className={styles.fieldRow}>
              <label className={styles.fieldLabel}>Lens</label>
              <select className={styles.select} value={lens} onChange={event => updateLens(event.target.value)}>
                {Object.keys(LENS_PRESETS).map(name => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </div>
            {[
              ['tx', 'tx (m)', 0.005],
              ['ty', 'ty (m)', 0.005],
              ['tz', 'tz (m)', 0.005],
              ['rollDeg', 'roll (deg)', 0.5],
              ['pitchDeg', 'pitch (deg)', 0.5],
              ['yawDeg', 'yaw (deg)', 0.5],
            ].map(([key, label, step]) => (
              <div key={key} className={styles.fieldRow}>
                <label className={styles.fieldLabel}>{label}</label>
                <input
                  type="number"
                  step={step}
                  className={styles.fieldInput}
                  value={cal.extr[key]}
                  onChange={event => updateExtr(key, event.target.value)}
                />
              </div>
            ))}
          </section>

          <section className={styles.section}>
            <div className={styles.sectionTitle}>Webcam rotation</div>
            {[
              ['rollDeg', 'roll (deg)', 0.5],
              ['pitchDeg', 'pitch (deg)', 0.5],
              ['yawDeg', 'yaw (deg)', 0.5],
            ].map(([key, label, step]) => (
              <div key={key} className={styles.fieldRow}>
                <label className={styles.fieldLabel}>{label}</label>
                <input
                  type="number"
                  step={step}
                  className={styles.fieldInput}
                  value={webcamRotation[key]}
                  onChange={event => setWebcamRotation(prev => ({ ...prev, [key]: Number(event.target.value) }))}
                />
              </div>
            ))}
            <div className={styles.readout}>
              fov {WEBCAM_HFOV_DEG.toFixed(0)}deg · small-angle approx (yaw/pitch shift, roll rotation)
            </div>
            <div className={styles.fileActions}>
              <button
                className="btn"
                onClick={() => setWebcamRotation({ rollDeg: 0, pitchDeg: 0, yawDeg: 0 })}
              >
                Reset webcam
              </button>
            </div>
          </section>

          <section className={styles.section}>
            <div className={styles.sectionTitle}>Calibration</div>
            <div className={styles.readout}>Source: {sourceText}</div>
            <div className={styles.fileActions}>
              <button className="btn" onClick={useLiveCalibration}>Use live</button>
              <button className="btn" onClick={resetExtrinsics}>Reset</button>
              <button className={`btn ${saved ? 'active' : ''}`} onClick={saveJSON}>
                {saved ? 'Saved' : 'Save JSON'}
              </button>
              <button className="btn" onClick={() => fileRef.current?.click()}>Load JSON</button>
              <input ref={fileRef} type="file" accept=".json" hidden onChange={loadJSON} />
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}

function renderCalibrationCanvas({ canvas, thermalCanvasRef, cameraImage, lidarFrame, thermalFrame, cameraFrame, calibration, channel, webcamRotation }) {
  const intr = calibration.intr
  const width = intr.width
  const height = intr.height
  const totalWidth = width * 3
  if (canvas.width !== totalWidth || canvas.height !== height) {
    canvas.width = totalWidth
    canvas.height = height
  }

  const ctx = canvas.getContext('2d')
  if (!ctx) return { projected: 0, points: lidarFrame?.n ?? 0 }

  ctx.imageSmoothingEnabled = true
  const lidar = synthesizeLidarView(lidarFrame, calibration, channel)
  if (lidar.image) {
    ctx.putImageData(lidar.image, 0, 0)
  } else {
    fillBlank(ctx, 0, 0, width, height)
  }

  drawThermalFrame(ctx, thermalCanvasRef, thermalFrame, width, height, width, 0)
  drawCameraFrame(ctx, cameraImage, width, height, width * 2, 0, calibration, webcamRotation)
  drawPersonOverlay(ctx, cameraFrame, calibration, width, height)
  drawPanelLabels(ctx, width)
  return { projected: lidar.projected, points: lidar.points }
}

function drawPersonOverlay(ctx, cameraFrame, calibration, paneWidth, paneHeight) {
  const persons = cameraFrame?.persons
  if (!Array.isArray(persons) || persons.length === 0) return
  const srcW = cameraFrame?.frameW || 0
  const srcH = cameraFrame?.frameH || 0
  if (srcW <= 0 || srcH <= 0) return

  const intr = calibration.intr
  const thermalHfov = 2 * Math.atan(intr.width / (2 * intr.fx))
  const thermalVfov = 2 * Math.atan(intr.height / (2 * intr.fy))
  const webcamHfov = degToRad(WEBCAM_HFOV_DEG)
  const webcamFocalPx = (srcW / 2) / Math.tan(webcamHfov / 2)
  const cropW = Math.min(srcW, 2 * webcamFocalPx * Math.tan(thermalHfov / 2))
  const cropH = Math.min(srcH, 2 * webcamFocalPx * Math.tan(thermalVfov / 2))
  const sx = (srcW - cropW) / 2
  const sy = (srcH - cropH) / 2

  // Lidar pane (dx=0) and Thermal pane (dx=paneWidth) both fill the pane with
  // content subtending the thermal FOV, so the cropped webcam region maps
  // directly to (dx, 0) -> (dx + paneWidth, paneHeight).
  ctx.lineWidth = 2.5
  ctx.font = '700 13px sans-serif'
  ctx.textBaseline = 'top'

  for (const person of persons) {
    const box = person?.bbox_xyxy
    if (!Array.isArray(box) || box.length < 4) continue
    const conf = Number(person.confidence ?? 0)
    const label = `person ${conf.toFixed(2)}`

    // Project box from webcam pixel coords -> cropped region -> pane fraction.
    const fx1 = (box[0] - sx) / cropW
    const fy1 = (box[1] - sy) / cropH
    const fx2 = (box[2] - sx) / cropW
    const fy2 = (box[3] - sy) / cropH
    if (fx2 <= 0 || fy2 <= 0 || fx1 >= 1 || fy1 >= 1) continue

    for (const dx of [0, paneWidth]) {
      const x1 = dx + Math.max(0, fx1) * paneWidth
      const y1 = Math.max(0, fy1) * paneHeight
      const x2 = dx + Math.min(1, fx2) * paneWidth
      const y2 = Math.min(1, fy2) * paneHeight
      ctx.strokeStyle = 'rgba(77, 191, 255, 0.95)'
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
      const labelW = ctx.measureText(label).width + 8
      const labelH = 16
      const labelY = Math.max(0, y1 - labelH)
      ctx.fillRect(x1, labelY, labelW, labelH)
      ctx.fillStyle = '#ffffff'
      ctx.fillText(label, x1 + 4, labelY + 1)
    }
  }
}

function synthesizeLidarView(frame, calibration, channel) {
  const intr = calibration.intr
  const width = intr.width
  const height = intr.height
  const points = frame?.n ?? 0
  if (!frame?.positions || points <= 0) return { image: null, projected: 0, points }

  const r = lidarToCameraRotation(calibration)
  const { extr } = calibration
  const hits = []

  for (let i = 0; i < points; i++) {
    const src = i * 3
    const x = frame.positions[src]
    const y = frame.positions[src + 1]
    const z = frame.positions[src + 2]
    const range = Math.hypot(x, y, z)
    if (!Number.isFinite(range) || range <= 0.3) continue

    const camX = r[0][0] * x + r[0][1] * y + r[0][2] * z + extr.tx
    const camY = r[1][0] * x + r[1][1] * y + r[1][2] * z + extr.ty
    const camZ = r[2][0] * x + r[2][1] * y + r[2][2] * z + extr.tz
    if (camZ <= 0.05) continue

    const u = Math.trunc(intr.fx * camX / camZ + intr.cx)
    const v = Math.trunc(intr.fy * camY / camZ + intr.cy)
    if (u < 0 || u >= width || v < 0 || v >= height) continue

    hits.push({
      pixel: v * width + u,
      depth: camZ,
      value: channel === 'range' ? range : Number(frame.intensities?.[i] ?? 0),
    })
  }

  if (!hits.length) return { image: null, projected: 0, points }

  hits.sort((a, b) => b.depth - a.depth)
  const scalars = new Float32Array(width * height)
  const mask = new Uint8Array(width * height)
  for (const hit of hits) {
    scalars[hit.pixel] = hit.value
    mask[hit.pixel] = 1
  }

  const covered = []
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) covered.push(scalars[i])
  }

  let lo = percentile(covered, 2)
  let hi = percentile(covered, 98)
  if (!Number.isFinite(lo)) lo = 0
  if (!Number.isFinite(hi) || hi - lo < 1e-6) hi = lo + 1

  const gray = new Uint8Array(width * height)
  const scale = 255 / (hi - lo)
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) gray[i] = clampByte((scalars[i] - lo) * scale)
  }

  const dilated = dilate3x3(gray, width, height)
  const rgba = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < dilated.length; i++) {
    const [r, g, b] = turboRgb(dilated[i] / 255)
    const dst = i * 4
    rgba[dst] = r
    rgba[dst + 1] = g
    rgba[dst + 2] = b
    rgba[dst + 3] = 255
  }

  return {
    image: new ImageData(rgba, width, height),
    projected: covered.length,
    points,
  }
}

function drawThermalFrame(ctx, thermalCanvasRef, frame, width, height, dx, dy) {
  if (!frame?.data?.length || !frame.w || !frame.h) {
    fillBlank(ctx, dx, dy, width, height)
    return
  }

  if (!thermalCanvasRef.current) thermalCanvasRef.current = document.createElement('canvas')
  const source = thermalCanvasRef.current
  source.width = frame.w
  source.height = frame.h
  const sourceCtx = source.getContext('2d')
  if (!sourceCtx) {
    fillBlank(ctx, dx, dy, width, height)
    return
  }

  const image = sourceCtx.createImageData(frame.w, frame.h)
  const lut = PALETTES.IRONBOW
  const n = Math.min(frame.data.length, frame.w * frame.h)
  for (let i = 0; i < n; i++) {
    const [r, g, b] = lut[frame.data[i]]
    const dst = i * 4
    image.data[dst] = r
    image.data[dst + 1] = g
    image.data[dst + 2] = b
    image.data[dst + 3] = 255
  }
  sourceCtx.putImageData(image, 0, 0)
  ctx.drawImage(source, dx, dy, width, height)
}

function drawPanelLabels(ctx, paneWidth) {
  ctx.font = '700 18px sans-serif'
  ctx.textBaseline = 'top'
  ctx.lineWidth = 4
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.72)'
  ctx.fillStyle = '#ffffff'
  ctx.strokeText('LIDAR', 10, 10)
  ctx.fillText('LIDAR', 10, 10)
  ctx.strokeText('THERMAL', paneWidth + 10, 10)
  ctx.fillText('THERMAL', paneWidth + 10, 10)
  ctx.strokeText('CAMERA', paneWidth * 2 + 10, 10)
  ctx.fillText('CAMERA', paneWidth * 2 + 10, 10)
}

function drawCameraFrame(ctx, cameraImage, paneWidth, paneHeight, dx, dy, calibration, webcamRotation) {
  fillBlank(ctx, dx, dy, paneWidth, paneHeight)
  if (!cameraImage || !cameraImage.naturalWidth || !cameraImage.naturalHeight) return
  const srcW = cameraImage.naturalWidth
  const srcH = cameraImage.naturalHeight

  const intr = calibration.intr
  const thermalHfov = 2 * Math.atan(intr.width / (2 * intr.fx))
  const thermalVfov = 2 * Math.atan(intr.height / (2 * intr.fy))
  const webcamHfov = degToRad(WEBCAM_HFOV_DEG)
  const webcamFocalPx = (srcW / 2) / Math.tan(webcamHfov / 2)

  // Crop a centered rect of the webcam frame that subtends the thermal FOV.
  // Clamp to source size in case the thermal lens is wider than the webcam.
  const cropW = Math.min(srcW, 2 * webcamFocalPx * Math.tan(thermalHfov / 2))
  const cropH = Math.min(srcH, 2 * webcamFocalPx * Math.tan(thermalVfov / 2))

  // Yaw/pitch: shift the crop center inside the source frame using the
  // small-angle pinhole approximation shift = focal_px * tan(angle).
  // This is exact for translations, and a good approximation for visual
  // alignment of small rotations (<~15deg) without doing a full homography.
  const rot = webcamRotation || { rollDeg: 0, pitchDeg: 0, yawDeg: 0 }
  const yawShiftPx = webcamFocalPx * Math.tan(degToRad(rot.yawDeg || 0))
  const pitchShiftPx = webcamFocalPx * Math.tan(degToRad(rot.pitchDeg || 0))
  let sx = (srcW - cropW) / 2 + yawShiftPx
  let sy = (srcH - cropH) / 2 + pitchShiftPx
  sx = Math.max(0, Math.min(srcW - cropW, sx))
  sy = Math.max(0, Math.min(srcH - cropH, sy))

  const fitScale = Math.min(paneWidth / cropW, paneHeight / cropH)
  // Roll: rotate the canvas around the pane center, clipped so we never
  // draw outside the camera pane. Overscale the drawn rect so the rotated
  // image fully covers the upright pane (no tilted black wedges at the
  // corners) — required half-extents come from projecting the pane
  // corners onto the rotated frame.
  const rollDeg = rot.rollDeg || 0
  const rollRad = degToRad(rollDeg)
  const cosA = Math.abs(Math.cos(rollRad))
  const sinA = Math.abs(Math.sin(rollRad))
  let scale = fitScale
  if (rollDeg !== 0) {
    const minDrawW = paneWidth * cosA + paneHeight * sinA
    const minDrawH = paneWidth * sinA + paneHeight * cosA
    const coverScale = Math.max(minDrawW / cropW, minDrawH / cropH)
    scale = Math.max(fitScale, coverScale)
  }
  const drawW = cropW * scale
  const drawH = cropH * scale
  const offsetX = dx + (paneWidth - drawW) / 2
  const offsetY = dy + (paneHeight - drawH) / 2
  ctx.save()
  ctx.beginPath()
  ctx.rect(dx, dy, paneWidth, paneHeight)
  ctx.clip()
  if (rollDeg) {
    const cxPane = dx + paneWidth / 2
    const cyPane = dy + paneHeight / 2
    ctx.translate(cxPane, cyPane)
    ctx.rotate(degToRad(rollDeg))
    ctx.translate(-cxPane, -cyPane)
  }
  ctx.drawImage(cameraImage, sx, sy, cropW, cropH, offsetX, offsetY, drawW, drawH)
  ctx.restore()
}

function fillBlank(ctx, x, y, width, height) {
  ctx.fillStyle = '#000000'
  ctx.fillRect(x, y, width, height)
}

function dilate3x3(gray, width, height) {
  const out = new Uint8Array(gray.length)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let maxValue = 0
      for (let yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy++) {
        const row = yy * width
        for (let xx = Math.max(0, x - 1); xx <= Math.min(width - 1, x + 1); xx++) {
          const value = gray[row + xx]
          if (value > maxValue) maxValue = value
        }
      }
      out[y * width + x] = maxValue
    }
  }
  return out
}

function percentile(values, pct) {
  if (!values.length) return 0
  values.sort((a, b) => a - b)
  const rank = (values.length - 1) * pct / 100
  const lo = Math.floor(rank)
  const hi = Math.ceil(rank)
  if (lo === hi) return values[lo]
  return values[lo] + (values[hi] - values[lo]) * (rank - lo)
}

function intrinsicsFromHfov(width, height, hfovDeg) {
  const f = width / (2 * Math.tan(degToRad(hfovDeg) / 2))
  return { width, height, fx: f, fy: f, cx: width / 2, cy: height / 2 }
}

function normalizeThermalCalibration(value) {
  const intrValue = value?.intrinsics ?? value?.intr ?? {}
  const extrValue = value?.extrinsics ?? value?.extr ?? {}
  return {
    intr: {
      width: intOr(intrValue.width, DEFAULT_CALIBRATION.intr.width, 1),
      height: intOr(intrValue.height, DEFAULT_CALIBRATION.intr.height, 1),
      fx: numberOr(intrValue.fx, DEFAULT_CALIBRATION.intr.fx),
      fy: numberOr(intrValue.fy, DEFAULT_CALIBRATION.intr.fy),
      cx: numberOr(intrValue.cx, DEFAULT_CALIBRATION.intr.cx),
      cy: numberOr(intrValue.cy, DEFAULT_CALIBRATION.intr.cy),
    },
    extr: {
      tx: numberOr(extrValue.tx, DEFAULT_CALIBRATION.extr.tx),
      ty: numberOr(extrValue.ty, DEFAULT_CALIBRATION.extr.ty),
      tz: numberOr(extrValue.tz, DEFAULT_CALIBRATION.extr.tz),
      rollDeg: numberOr(extrValue.rollDeg ?? extrValue.roll_deg, DEFAULT_CALIBRATION.extr.rollDeg),
      pitchDeg: numberOr(extrValue.pitchDeg ?? extrValue.pitch_deg, DEFAULT_CALIBRATION.extr.pitchDeg),
      yawDeg: numberOr(extrValue.yawDeg ?? extrValue.yaw_deg, DEFAULT_CALIBRATION.extr.yawDeg),
    },
    source: String(value?.source ?? DEFAULT_CALIBRATION.source),
  }
}

function calibrationToJson(calibration) {
  return {
    intrinsics: {
      width: calibration.intr.width,
      height: calibration.intr.height,
      fx: calibration.intr.fx,
      fy: calibration.intr.fy,
      cx: calibration.intr.cx,
      cy: calibration.intr.cy,
    },
    extrinsics: {
      tx: calibration.extr.tx,
      ty: calibration.extr.ty,
      tz: calibration.extr.tz,
      roll_deg: calibration.extr.rollDeg,
      pitch_deg: calibration.extr.pitchDeg,
      yaw_deg: calibration.extr.yawDeg,
    },
  }
}

function lidarToCameraRotation(calibration) {
  const { extr } = calibration
  return matMul3(
    eulerToR(degToRad(extr.rollDeg), degToRad(extr.pitchDeg), degToRad(extr.yawDeg)),
    R_LIDAR_TO_CAM_BASE,
  )
}

function eulerToR(roll, pitch, yaw) {
  const cr = Math.cos(roll), sr = Math.sin(roll)
  const cp = Math.cos(pitch), sp = Math.sin(pitch)
  const cy = Math.cos(yaw), sy = Math.sin(yaw)
  const rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
  const ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
  const rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
  return matMul3(matMul3(rz, ry), rx)
}

function matMul3(a, b) {
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
    }
  }
  return out
}

function turboRgb(t) {
  const x = Math.max(0, Math.min(1, t))
  const x2 = x * x
  const x3 = x2 * x
  const x4 = x3 * x
  const x5 = x4 * x
  return [
    clampByte((0.13572138 + 4.61539260 * x - 42.66032258 * x2 + 132.13108234 * x3 - 152.94239396 * x4 + 59.28637943 * x5) * 255),
    clampByte((0.09140261 + 2.19418839 * x + 4.84296658 * x2 - 14.18503333 * x3 + 4.27729857 * x4 + 2.82956604 * x5) * 255),
    clampByte((0.10667330 + 12.64194608 * x - 60.58204836 * x2 + 110.36276771 * x3 - 89.90310912 * x4 + 27.34824973 * x5) * 255),
  ]
}

function numberOr(value, fallback) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function intOr(value, fallback, minValue) {
  return Math.max(minValue, Math.round(numberOr(value, fallback)))
}

function degToRad(value) {
  return value * Math.PI / 180
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)))
}

function formatSync(ms) {
  if (ms === null) return '—'
  const sign = ms > 0 ? '+' : ''
  return `${sign}${ms}ms`
}

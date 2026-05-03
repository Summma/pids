import { useRef, useEffect, useState, useCallback } from 'react'
import PanelShell from '@/components/PanelShell'
import { PALETTES, PALETTE_NAMES } from '@/utils/palettes'
import { thermalCss } from '@/utils/thermalColor'
import styles from './ThermalPanel.module.css'

export default function ThermalPanel({ thermalData }) {
  const { connState, frameRef, meta, sendControl } = thermalData
  const canvasRef  = useRef(null)
  const imgDataRef = useRef(null)
  const [palette,   setPalette]   = useState('IRONBOW')
  const [recording, setRecording] = useState(false)
  const [crosshair, setCrosshair] = useState(null)   // { x, y, temp }

  // Render only when a new backend frame arrives or the palette changes.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx    = canvas.getContext('2d')
    const frame = frameRef.current
    if (!frame) return

    const { data, w, h } = frame
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width  = w
      canvas.height = h
      imgDataRef.current = null
    }

    if (!imgDataRef.current) {
      imgDataRef.current = ctx.createImageData(w, h)
    }

    const lut = PALETTES[palette]
    const px  = imgDataRef.current.data

    for (let i = 0; i < data.length; i++) {
      const c = lut[data[i]]
      px[i * 4]     = c[0]
      px[i * 4 + 1] = c[1]
      px[i * 4 + 2] = c[2]
      px[i * 4 + 3] = 255
    }

    ctx.putImageData(imgDataRef.current, 0, 0)
  }, [palette, frameRef, meta.seq])

  const onMouseMove = useCallback((e) => {
    const canvas = canvasRef.current
    const frame  = frameRef.current
    if (!frame) return
    const rect = canvas.getBoundingClientRect()
    const scaleX = frame.w / rect.width
    const scaleY = frame.h / rect.height
    const px = Math.floor((e.clientX - rect.left) * scaleX)
    const py = Math.floor((e.clientY - rect.top)  * scaleY)
    const idx = py * frame.w + px
    if (idx >= 0 && idx < frame.data.length) {
      const rawVal = frame.data[idx] / 255
      const temp   = meta.tMin + rawVal * (meta.tMax - meta.tMin)
      setCrosshair({ x: e.clientX - rect.left, y: e.clientY - rect.top, temp })
    }
  }, [frameRef, meta])

  const snapshot = useCallback(() => {
    const a = document.createElement('a')
    a.download = `thermal_${Date.now()}.png`
    a.href = canvasRef.current.toDataURL()
    a.click()
  }, [])

  const triggerFFC = () => sendControl({ type: 'ffc' })

  const controls = (
    <div className={styles.controls}>
      <select
        className="ctrl-select"
        value={palette}
        onChange={e => { setPalette(e.target.value); imgDataRef.current = null }}
      >
        {PALETTE_NAMES.map(p => <option key={p} value={p}>{p.replace('_', ' ')}</option>)}
      </select>
      <button className="btn" onClick={triggerFFC}>FFC</button>
      <button className="btn" onClick={snapshot}>Snap</button>
      <button
        className={`btn ${recording ? 'danger' : ''}`}
        onClick={() => setRecording(r => !r)}
      >
        {recording ? '⏹ Stop' : '⏺ Record'}
      </button>
    </div>
  )

  return (
    <PanelShell title="Thermal" modality="thermal" connState={connState} controls={controls}>
      <div
        className={styles.viewport}
        onMouseMove={onMouseMove}
        onMouseLeave={() => setCrosshair(null)}
      >
        <canvas ref={canvasRef} className={styles.canvas} />

        {crosshair && (
          <div
            className={styles.crosshairTip}
            style={{ left: crosshair.x + 10, top: crosshair.y - 10 }}
          >
            <span style={{ color: thermalCss(crosshair.temp, meta.tMin, meta.tMax) }}>
              {crosshair.temp.toFixed(1)} °C
            </span>
          </div>
        )}

        <div className={styles.tempBar}>
          <span className={styles.tempMin}>{meta.tMin.toFixed(1)} °C</span>
          <div className={styles.rampBar} />
          <span className={styles.tempMax}>{meta.tMax.toFixed(1)} °C</span>
        </div>

        {recording && <div className={styles.recIndicator}>⏺ Recording</div>}
      </div>
    </PanelShell>
  )
}

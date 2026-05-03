import { useState, useRef, useCallback } from 'react'
import styles from './CalibrationPanel.module.css'

const DEFAULT_CAL = {
  intrinsics: { fx: 500, fy: 500, cx: 320, cy: 256 },
  extrinsics: {
    R: [[1,0,0],[0,1,0],[0,0,1]],
    t: [0, 0, 0],
  },
}

export default function CalibrationPanel({ onClose }) {
  const [cal, setCal]       = useState(DEFAULT_CAL)
  const [alpha, setAlpha]   = useState(0.5)
  const [edgeOn, setEdgeOn] = useState(false)
  const [saved, setSaved]   = useState(false)
  const fileRef = useRef(null)

  const updateIntrinsic = (key, val) => {
    setCal(c => ({ ...c, intrinsics: { ...c.intrinsics, [key]: Number(val) } }))
  }

  const updateT = (i, val) => {
    setCal(c => {
      const t = [...c.extrinsics.t]
      t[i] = Number(val)
      return { ...c, extrinsics: { ...c.extrinsics, t } }
    })
  }

  const updateR = (row, col, val) => {
    setCal(c => {
      const R = c.extrinsics.R.map(r => [...r])
      R[row][col] = Number(val)
      return { ...c, extrinsics: { ...c.extrinsics, R } }
    })
  }

  const saveJSON = () => {
    const blob = new Blob([JSON.stringify(cal, null, 2)], { type: 'application/json' })
    const a    = document.createElement('a')
    a.href     = URL.createObjectURL(blob)
    a.download = 'calibration.json'
    a.click()
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const loadJSON = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      try { setCal(JSON.parse(ev.target.result)) } catch {}
    }
    reader.readAsText(file)
  }

  return (
    <div className={styles.overlay}>
      <div className={styles.header}>
        <span className={styles.title}>⊕ CALIBRATION</span>
        <div className={styles.headerRight}>
          <span className={styles.hint}>Align lidar edges to thermal edges</span>
          <button className="btn" onClick={onClose}>✕ CLOSE</button>
        </div>
      </div>

      <div className={styles.body}>
        {/* Viewport */}
        <div className={styles.viewport}>
          <div className={styles.viewportPlaceholder}>
            <span>SYNTHESIZED LIDAR-ON-THERMAL VIEW</span>
            <span className={styles.viewHint}>Connect sensors to enable live alignment</span>
          </div>
          <div className={styles.viewControls}>
            <label className={styles.sliderLabel}>
              BLEND
              <input
                type="range" min={0} max={1} step={0.01}
                value={alpha}
                onChange={e => setAlpha(Number(e.target.value))}
                className={styles.slider}
              />
              {(alpha * 100).toFixed(0)}%
            </label>
            <button
              className={`btn ${edgeOn ? 'active' : ''}`}
              onClick={() => setEdgeOn(e => !e)}
            >
              CANNY EDGES
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className={styles.controls}>
          <section className={styles.section}>
            <div className={styles.sectionTitle}>CAMERA INTRINSICS</div>
            {['fx', 'fy', 'cx', 'cy'].map(k => (
              <div key={k} className={styles.fieldRow}>
                <label className={styles.fieldLabel}>{k}</label>
                <input
                  type="number"
                  className={styles.fieldInput}
                  value={cal.intrinsics[k]}
                  onChange={e => updateIntrinsic(k, e.target.value)}
                />
              </div>
            ))}
          </section>

          <section className={styles.section}>
            <div className={styles.sectionTitle}>TRANSLATION (m)</div>
            {['X', 'Y', 'Z'].map((axis, i) => (
              <div key={axis} className={styles.fieldRow}>
                <label className={styles.fieldLabel}>t{axis}</label>
                <input
                  type="number" step={0.001}
                  className={styles.fieldInput}
                  value={cal.extrinsics.t[i]}
                  onChange={e => updateT(i, e.target.value)}
                />
              </div>
            ))}
          </section>

          <section className={styles.section}>
            <div className={styles.sectionTitle}>ROTATION MATRIX</div>
            <div className={styles.matrix}>
              {cal.extrinsics.R.map((row, ri) =>
                row.map((val, ci) => (
                  <input
                    key={`${ri}${ci}`}
                    type="number" step={0.001}
                    className={styles.matInput}
                    value={val}
                    onChange={e => updateR(ri, ci, e.target.value)}
                  />
                ))
              )}
            </div>
          </section>

          <div className={styles.fileActions}>
            <button className={`btn ${saved ? 'active' : ''}`} onClick={saveJSON}>
              {saved ? '✓ SAVED' : '↓ SAVE JSON'}
            </button>
            <button className="btn" onClick={() => fileRef.current.click()}>
              ↑ LOAD JSON
            </button>
            <input ref={fileRef} type="file" accept=".json" hidden onChange={loadJSON} />
          </div>
        </div>
      </div>
    </div>
  )
}

import { useRef, useEffect, useState } from 'react'
import PanelShell from '@/components/PanelShell'
import styles from './LidarPanel.module.css'

const CHANNELS   = [
  { key: 'range', label: 'RNG' },
  { key: 'signal', label: 'SIG' },
  { key: 'reflectivity', label: 'REFL' },
  { key: 'near-ir', label: 'NIR' },
]
const RANGE_MAX  = 50   // metres
const RING_COUNT = 4

function rangeColor(intensity, channel) {
  const v = Math.max(0, Math.min(1, intensity))
  if (channel === 'range') {
    const r = Math.round(v * 255)
    return `rgb(${r}, ${Math.round((1 - v) * 120)}, ${Math.round((1 - v) * 255)})`
  }
  if (channel === 'signal' || channel === 'near-ir') {
    const g = Math.round(v * 255)
    return `rgb(0, ${g}, ${Math.round((1 - v) * 100)})`
  }
  // reflectivity
  const hot = Math.round(v * 255)
  return `rgb(${hot}, ${Math.round(v * 140)}, 0)`
}

export default function LidarPanel({ lidarData, onPopOut3D }) {
  const { connState, frameRef, meta, clusters } = lidarData
  const canvasRef = useRef(null)
  const rafRef    = useRef(null)
  const [channel, setChannel] = useState('range')
  const [selectedCluster, setSelectedCluster] = useState(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')

    const render = () => {
      rafRef.current = requestAnimationFrame(render)
      const frame = frameRef.current
      const w = canvas.offsetWidth
      const h = canvas.offsetHeight
      canvas.width  = w
      canvas.height = h

      ctx.fillStyle = '#080c10'
      ctx.fillRect(0, 0, w, h)

      const cx = w / 2
      const cy = h / 2
      const scale = Math.min(w, h) / 2 / RANGE_MAX

      // Range rings
      ctx.strokeStyle = '#1e3a5f'
      ctx.lineWidth   = 1
      ctx.setLineDash([4, 4])
      for (let r = 1; r <= RING_COUNT; r++) {
        const pxr = (RANGE_MAX / RING_COUNT) * r * scale
        ctx.beginPath()
        ctx.arc(cx, cy, pxr, 0, Math.PI * 2)
        ctx.stroke()
        ctx.fillStyle = '#3d5a6e'
        ctx.font = '9px JetBrains Mono'
        ctx.fillText(`${(RANGE_MAX / RING_COUNT) * r}m`, cx + pxr + 3, cy - 3)
      }
      ctx.setLineDash([])

      // Cardinal lines
      ctx.strokeStyle = '#1e3a5f'
      ctx.lineWidth = 1
      ;[0, Math.PI / 2, Math.PI, Math.PI * 1.5].forEach(a => {
        ctx.beginPath()
        ctx.moveTo(cx, cy)
        ctx.lineTo(cx + Math.cos(a) * RANGE_MAX * scale, cy + Math.sin(a) * RANGE_MAX * scale)
        ctx.stroke()
      })

      // Origin
      ctx.fillStyle = '#00d4ff'
      ctx.beginPath()
      ctx.arc(cx, cy, 3, 0, Math.PI * 2)
      ctx.fill()

      if (!frame || frame.n === 0) return

      // Points — project XZ plane (bird's-eye)
      const { positions, intensities, n } = frame
      for (let i = 0; i < n; i++) {
        const x = positions[i * 3]
        const z = positions[i * 3 + 2]
        const v = intensities[i]
        const px = cx + x * scale
        const py = cy - z * scale
        if (px < 0 || px > w || py < 0 || py > h) continue
        ctx.fillStyle = rangeColor(v, channel)
        ctx.fillRect(px, py, 1.5, 1.5)
      }

      // Cluster overlays
      clusters.forEach(cl => {
        const px = cx + cl.centroid.x * scale
        const py = cy - cl.centroid.z * scale
        const isSelected = selectedCluster?.id === cl.id
        ctx.strokeStyle = isSelected ? '#ffcc00' : '#00d4ff'
        ctx.lineWidth   = isSelected ? 2 : 1
        const r = Math.max(6, cl.radius * scale)
        ctx.beginPath()
        ctx.arc(px, py, r, 0, Math.PI * 2)
        ctx.stroke()
        ctx.fillStyle = isSelected ? '#ffcc00' : '#00d4ff'
        ctx.font = '9px JetBrains Mono'
        ctx.fillText(`T${cl.id}`, px + r + 2, py)
      })
    }

    render()
    return () => cancelAnimationFrame(rafRef.current)
  }, [channel, frameRef, clusters, selectedCluster])

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current
    const rect   = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const cx = canvas.width  / 2
    const cy = canvas.height / 2
    const scale = Math.min(canvas.width, canvas.height) / 2 / RANGE_MAX

    const hit = clusters.find(cl => {
      const px = cx + cl.centroid.x * scale
      const py = cy - cl.centroid.z * scale
      return Math.hypot(mx - px, my - py) < 20
    })
    setSelectedCluster(hit ?? null)
  }

  const controls = (
    <div className={styles.controls}>
      {CHANNELS.map(ch => (
        <button
          key={ch.key}
          className={`btn ${channel === ch.key ? 'active' : ''}`}
          onClick={() => setChannel(ch.key)}
          title={ch.key}
        >
          {ch.label}
        </button>
      ))}
      {onPopOut3D && (
        <button className="btn" onClick={onPopOut3D} title="Open 3D point cloud">3D</button>
      )}
    </div>
  )

  return (
    <PanelShell title="LIDAR" subtitle="BIRD'S EYE" modality="lidar" connState={connState} controls={controls}>
      <div className={styles.viewport}>
        <canvas
          ref={canvasRef}
          className={styles.canvas}
          onClick={handleCanvasClick}
        />
        {selectedCluster && (
          <ClusterInspector cluster={selectedCluster} onClose={() => setSelectedCluster(null)} />
        )}
        <div className={styles.ptCount}>{meta.n.toLocaleString()} pts</div>
      </div>
    </PanelShell>
  )
}

function ClusterInspector({ cluster, onClose }) {
  return (
    <div className={styles.inspector}>
      <div className={styles.inspectorHeader}>
        <span>CLUSTER T{cluster.id}</span>
        <button onClick={onClose} className={styles.inspectorClose}>✕</button>
      </div>
      <Row label="Range"     value={`${cluster.range?.toFixed(1)} m`} />
      <Row label="Footprint" value={`${cluster.footprint?.toFixed(2)} m²`} />
      <Row label="Speed"     value={`${cluster.speed?.toFixed(2)} m/s`} />
      <Row label="Points"    value={cluster.points} />
      <Row label="Age"       value={`${cluster.age_frames} frames`} />
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className={styles.inspectorRow}>
      <span className={styles.inspectorLabel}>{label}</span>
      <span className={styles.inspectorVal}>{value}</span>
    </div>
  )
}

import { useEffect, useMemo, useRef, useState } from 'react'
import PanelShell from '@/components/PanelShell'
import styles from './FusionMapPanel.module.css'

const WORLD_M = 60
const GRID_STEP = 10
const MAX_DRAW_PTS = 18000
const LAYERS = [
  { key: 'fusion', label: 'FUSE' },
  { key: 'occupancy', label: 'OCC' },
  { key: 'tracks', label: 'TRK' },
]

export default function FusionMapPanel({ fusionData, lidarData, threatData }) {
  const { connState, frame, sendControl } = fusionData
  const { frameRef, clusters } = lidarData
  const { threats } = threatData
  const canvasRef = useRef(null)
  const rafRef = useRef(null)
  const [layer, setLayer] = useState('fusion')
  const [follow, setFollow] = useState(true)
  const [selected, setSelected] = useState(null)

  const tracks = useMemo(() => {
    const fused = frame.tracks?.length ? frame.tracks : threats
    return normalizeTracks(fused)
  }, [frame.tracks, threats])

  const subtitle = frame.map
    ? `${frame.stats.mapped.toLocaleString()} CELLS`
    : `${tracks.length} TRACK${tracks.length === 1 ? '' : 'S'}`

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    const render = () => {
      rafRef.current = requestAnimationFrame(render)
      const rect = canvas.getBoundingClientRect()
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const w = Math.max(1, Math.floor(rect.width * dpr))
      const h = Math.max(1, Math.floor(rect.height * dpr))
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      drawMap(ctx, rect.width, rect.height, {
        frame,
        lidarFrame: frameRef.current,
        clusters,
        tracks,
        layer,
        follow,
        selected,
      })
    }

    render()
    return () => cancelAnimationFrame(rafRef.current)
  }, [frame, frameRef, clusters, tracks, layer, follow, selected])

  const handleClick = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const hit = pickTrack(
      e.clientX - rect.left,
      e.clientY - rect.top,
      rect.width,
      rect.height,
      tracks,
      frame.pose,
      follow,
    )
    setSelected(hit)
  }

  const resetMap = () => sendControl({ type: 'reset_map' })

  const controls = (
    <div className={styles.controls}>
      {LAYERS.map(item => (
        <button
          key={item.key}
          className={`btn ${layer === item.key ? 'active' : ''}`}
          onClick={() => setLayer(item.key)}
          title={item.key}
        >
          {item.label}
        </button>
      ))}
      <button className={`btn ${follow ? 'active' : ''}`} onClick={() => setFollow(f => !f)}>
        FOL
      </button>
      <button className="btn" onClick={resetMap}>RESET</button>
    </div>
  )

  return (
    <PanelShell
      title="FUSION MAP"
      subtitle={subtitle}
      connState={connState}
      controls={controls}
    >
      <div className={styles.viewport}>
        <canvas ref={canvasRef} className={styles.canvas} onClick={handleClick} />
        <div className={styles.legend}>
          <span><i className={styles.lidarSwatch} /> LIDAR</span>
          <span><i className={styles.thermalSwatch} /> HEAT</span>
          <span><i className={styles.threatSwatch} /> THREAT</span>
        </div>
        {selected && <TrackReadout track={selected} onClose={() => setSelected(null)} />}
      </div>
    </PanelShell>
  )
}

function drawMap(ctx, w, h, opts) {
  ctx.clearRect(0, 0, w, h)
  ctx.fillStyle = '#080c10'
  ctx.fillRect(0, 0, w, h)

  const view = makeView(w, h, opts.frame.pose, opts.follow)
  drawGrid(ctx, w, h, view)
  drawZones(ctx, opts.frame.zones, view)

  if ((opts.layer === 'fusion' || opts.layer === 'occupancy') && opts.frame.map) {
    drawOccupancy(ctx, opts.frame.map, view)
  }

  if ((opts.layer === 'fusion' || opts.layer === 'tracks') && opts.lidarFrame?.n > 0) {
    drawLidarPoints(ctx, opts.lidarFrame, view)
  }

  drawClusters(ctx, opts.clusters, view)
  drawTracks(ctx, opts.tracks, view, opts.selected)
  drawEgo(ctx, view)
}

function makeView(w, h, pose, follow) {
  const scale = Math.min(w, h) / WORLD_M
  const cx = w / 2
  const cy = h / 2
  const ox = follow ? pose?.x ?? 0 : 0
  const oz = follow ? pose?.z ?? pose?.y ?? 0 : 0
  return {
    scale,
    toScreen(x, z) {
      return [cx + (x - ox) * scale, cy - (z - oz) * scale]
    },
  }
}

function drawGrid(ctx, w, h, view) {
  ctx.strokeStyle = '#132238'
  ctx.lineWidth = 1
  ctx.font = '10px JetBrains Mono'
  ctx.fillStyle = '#3d5a6e'

  for (let v = -WORLD_M; v <= WORLD_M; v += GRID_STEP) {
    const [x0, y0] = view.toScreen(v, -WORLD_M)
    const [x1, y1] = view.toScreen(v, WORLD_M)
    ctx.beginPath()
    ctx.moveTo(x0, y0)
    ctx.lineTo(x1, y1)
    ctx.stroke()

    const [lx, ly] = view.toScreen(-WORLD_M / 2, v)
    const [rx, ry] = view.toScreen(WORLD_M / 2, v)
    ctx.beginPath()
    ctx.moveTo(lx, ly)
    ctx.lineTo(rx, ry)
    ctx.stroke()
  }

  ctx.strokeStyle = '#1e3a5f'
  ctx.beginPath()
  ctx.moveTo(w / 2, 0)
  ctx.lineTo(w / 2, h)
  ctx.moveTo(0, h / 2)
  ctx.lineTo(w, h / 2)
  ctx.stroke()
}

function drawOccupancy(ctx, map, view) {
  const width = map.width ?? map.w ?? 0
  const height = map.height ?? map.h ?? 0
  const cells = decodeCells(map)
  if (!width || !height || !cells?.length) return

  const res = map.resolution ?? map.res ?? 0.5
  const origin = map.origin ?? {}
  const ox = origin.x ?? map.origin_x ?? -(width * res) / 2
  const oz = origin.z ?? origin.y ?? map.origin_z ?? -(height * res) / 2
  const size = Math.max(1, res * view.scale + 0.5)

  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const cell = cells[row * width + col]
      const occ = cellValue(cell)
      if (occ <= 0.05) continue
      const temp = cellTemp(cell)
      const conf = cellConfidence(cell)
      const [x, y] = view.toScreen(ox + col * res, oz + row * res)
      ctx.fillStyle = cellColor(occ, temp, conf)
      ctx.fillRect(x, y, size, size)
    }
  }
}

function drawLidarPoints(ctx, frame, view) {
  const { positions, intensities, n } = frame
  const stride = Math.max(1, Math.floor(n / MAX_DRAW_PTS))
  ctx.globalAlpha = 0.55
  for (let i = 0; i < n; i += stride) {
    const x = positions[i * 3]
    const z = positions[i * 3 + 2]
    const v = Math.max(0, Math.min(1, intensities[i] ?? 0.45))
    const [px, py] = view.toScreen(x, z)
    ctx.fillStyle = `rgba(0, ${Math.round(140 + v * 90)}, 255, 0.55)`
    ctx.fillRect(px, py, 1.4, 1.4)
  }
  ctx.globalAlpha = 1
}

function drawClusters(ctx, clusters, view) {
  clusters.forEach(cl => {
    const c = normalizePoint(cl.centroid)
    if (!c) return
    const radius = Math.max(5, (cl.radius ?? cl.range_radius ?? 0.7) * view.scale)
    const [x, y] = view.toScreen(c.x, c.z)
    ctx.strokeStyle = '#00d4ff'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.stroke()
  })
}

function drawTracks(ctx, tracks, view, selected) {
  tracks.forEach(track => {
    const [x, y] = view.toScreen(track.x, track.z)
    const r = selected?.id === track.id ? 9 : 6
    const color = confidenceColor(track.confidence)

    ctx.strokeStyle = color
    ctx.fillStyle = color
    ctx.lineWidth = selected?.id === track.id ? 2 : 1
    ctx.beginPath()
    ctx.arc(x, y, r, 0, Math.PI * 2)
    ctx.stroke()

    const vx = track.vx ?? Math.sin((track.bearing ?? 0) * Math.PI / 180) * track.speed
    const vz = track.vz ?? Math.cos((track.bearing ?? 0) * Math.PI / 180) * track.speed
    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineTo(x + vx * view.scale * 1.4, y - vz * view.scale * 1.4)
    ctx.stroke()

    ctx.font = '10px JetBrains Mono'
    ctx.fillText(`T${track.id}`, x + 8, y - 8)
  })
}

function drawZones(ctx, zones, view) {
  zones.forEach(zone => {
    const pts = zone.points ?? zone.vertices
    if (!pts?.length) return
    ctx.strokeStyle = zone.color ?? '#ff8c42'
    ctx.fillStyle = 'rgba(255, 140, 66, 0.06)'
    ctx.lineWidth = 1
    ctx.beginPath()
    pts.forEach((p, i) => {
      const q = normalizePoint(p)
      if (!q) return
      const [x, y] = view.toScreen(q.x, q.z)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
  })
}

function drawEgo(ctx, view) {
  const [x, y] = view.toScreen(0, 0)
  ctx.fillStyle = '#e8f4f8'
  ctx.strokeStyle = '#00d4ff'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(x, y - 9)
  ctx.lineTo(x - 7, y + 7)
  ctx.lineTo(x + 7, y + 7)
  ctx.closePath()
  ctx.fill()
  ctx.stroke()
}

function pickTrack(mx, my, w, h, tracks, pose, follow) {
  const view = makeView(w, h, pose, follow)
  return tracks.find(track => {
    const [x, y] = view.toScreen(track.x, track.z)
    return Math.hypot(mx - x, my - y) <= 16
  }) ?? null
}

function TrackReadout({ track, onClose }) {
  return (
    <div className={styles.readout}>
      <div className={styles.readoutHeader}>
        <span>TRACK T{track.id}</span>
        <button onClick={onClose}>x</button>
      </div>
      <Row label="Confidence" value={`${Math.round(track.confidence * 100)}%`} />
      <Row label="Range" value={`${track.range.toFixed(1)} m`} />
      <Row label="Speed" value={`${track.speed.toFixed(1)} m/s`} />
      <Row label="Bearing" value={`${track.bearing.toFixed(0)} deg`} />
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className={styles.row}>
      <span>{label}</span>
      <b>{value}</b>
    </div>
  )
}

function normalizeTracks(items) {
  return items.map((item, i) => {
    const pos = normalizePoint(item.position ?? item.centroid ?? item)
    const vel = normalizePoint(item.velocity ?? {})
    const x = pos?.x ?? Math.sin((item.bearing ?? 0) * Math.PI / 180) * (item.range ?? 0)
    const z = pos?.z ?? Math.cos((item.bearing ?? 0) * Math.PI / 180) * (item.range ?? 0)
    const range = item.range ?? Math.hypot(x, z)
    const bearing = item.bearing ?? ((Math.atan2(x, z) * 180 / Math.PI) + 360) % 360
    return {
      id: item.id ?? item.track_id ?? i + 1,
      x,
      z,
      vx: vel?.x,
      vz: vel?.z,
      confidence: clamp01(item.confidence ?? item.score ?? 0.45),
      range,
      bearing,
      speed: item.speed ?? Math.hypot(vel?.x ?? 0, vel?.z ?? 0),
    }
  })
}

function normalizePoint(p) {
  if (!p) return null
  if (Array.isArray(p)) return { x: Number(p[0] ?? 0), z: Number(p[2] ?? p[1] ?? 0) }
  return {
    x: Number(p.x ?? 0),
    z: Number(p.z ?? p.y ?? 0),
  }
}

function decodeCells(map) {
  if (Array.isArray(map.cells)) return map.cells
  if (!map.data) return null
  try {
    const raw = atob(map.data)
    return Array.from(raw, ch => ch.charCodeAt(0) / 255)
  } catch {
    return null
  }
}

function cellValue(cell) {
  if (typeof cell === 'number') return cell
  return cell?.occupancy ?? cell?.occ ?? cell?.value ?? 0
}

function cellTemp(cell) {
  if (typeof cell === 'number') return null
  return cell?.temperature ?? cell?.temp ?? null
}

function cellConfidence(cell) {
  if (typeof cell === 'number') return null
  return cell?.confidence ?? cell?.conf ?? null
}

function cellColor(occ, temp, conf) {
  if (conf != null && conf > 0.65) return `rgba(255, 34, 68, ${0.25 + occ * 0.65})`
  if (temp != null) {
    const hot = clamp01((temp - 15) / 45)
    const r = Math.round(70 + hot * 185)
    const g = Math.round(190 - hot * 80)
    return `rgba(${r}, ${g}, 40, ${0.22 + occ * 0.62})`
  }
  return `rgba(0, 212, 255, ${0.12 + occ * 0.6})`
}

function confidenceColor(confidence) {
  if (confidence >= 0.8) return '#ff2244'
  if (confidence >= 0.6) return '#ff8c42'
  if (confidence >= 0.4) return '#ffcc00'
  return '#00e676'
}

function clamp01(v) {
  return Math.max(0, Math.min(1, Number(v) || 0))
}

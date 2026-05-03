import PanelShell from '@/components/PanelShell'
import styles from './ObjectListPanel.module.css'

export default function ObjectListPanel({ lidarData, selectedObjectKey = '', onSelectObject }) {
  const { connState, detections = [], detectionMeta = {} } = lidarData
  const objects = normalizeDetections(detections)
    .sort((a, b) => b.score - a.score)
    .slice(0, 12)

  const subtitle = objects.length
    ? `${objects.length} object${objects.length === 1 ? '' : 's'}`
    : detectionMeta.status || 'Waiting'

  return (
    <PanelShell title="Objects" subtitle={subtitle} connState={connState} showStatusDot={false}>
      <div className={styles.body}>
        {objects.length ? (
          <div className={styles.list}>
            {objects.map(object => {
              const selected = object.key === selectedObjectKey
              return (
                <button
                  type="button"
                  className={`${styles.row} ${selected ? styles.rowActive : ''}`}
                  key={object.key}
                  aria-pressed={selected}
                  onClick={() => onSelectObject?.(object.key)}
                >
                  <div className={styles.main}>
                    <span className={styles.label}>{object.label}</span>
                    <span className={styles.source}>
                      {object.source}
                      {object.fusionNote ? ` · ${object.fusionNote.replaceAll('_', ' ')}` : ''}
                    </span>
                    <div className={styles.geometry}>
                      <span>Range {meters(object.range)}</span>
                      <span>L {meters(object.length)}</span>
                      <span>W {meters(object.width)}</span>
                      <span>H {meters(object.height)}</span>
                      <span>Center {object.center.map(metersCompact).join(', ')}</span>
                      <span>Yaw {degrees(object.yaw)}</span>
                      {Number.isFinite(object.supportPoints) && <span>Pts {object.supportPoints}</span>}
                      {Number.isFinite(object.supportZSpan) && object.supportZSpan > 0 && <span>Z span {meters(object.supportZSpan)}</span>}
                    </div>
                    <div className={styles.evidence}>
                      {object.thermalUnavailable ? (
                        <span>Thermal unavailable</span>
                      ) : (
                        <>
                          <span>Thermal {percent(object.thermalScore)}</span>
                          <span>Cov {percent(object.thermalCoverage)}</span>
                          {object.thermalHotFraction > 0 && <span>Hot {percent(object.thermalHotFraction)}</span>}
                          {object.thermalMax > 0 && <span>Tmax {percent(object.thermalMax)}</span>}
                          {Number.isFinite(object.thermalTempC) && <span>Temp {object.thermalTempC.toFixed(1)}°C</span>}
                        </>
                      )}
                      {object.pointpillarsSupport > 0 && <span>PP {percent(object.pointpillarsSupport)}</span>}
                    </div>
                  </div>
                  <div className={styles.metrics}>
                    <span>Fusion {percent(object.fusionScore)}</span>
                    <span>Model {percent(object.modelScore)}</span>
                  </div>
                </button>
              )
            })}
          </div>
        ) : (
          <div className={styles.empty}>
            {detectionMeta.status || 'Waiting for detections'}
          </div>
        )}
      </div>
    </PanelShell>
  )
}

function normalizeDetections(items) {
  if (!Array.isArray(items)) return []
  return items.map((item, i) => {
    const center = vec3(item.center ?? item.centroid ?? item.position, [0, 0, 0])
    const size = vec3(item.size, null)
      ?? bboxSize(item.bbox_min ?? item.min, item.bbox_max ?? item.max)
      ?? [0.35, 0.35, 0.8]
    const label = String(item.class_name ?? item.label ?? item.kind ?? 'object').replaceAll('_', ' ')
    const id = item.track_id ?? item.id ?? i + 1
    const source = item.source ?? 'detector'
    return {
      id,
      key: objectKey(source, id),
      source,
      center,
      size,
      label,
      score: clamp01(item.score ?? item.confidence ?? item.model_score ?? 0),
      fusionScore: clamp01(item.fusion_score ?? item.score ?? item.confidence ?? item.model_score ?? 0),
      modelScore: clamp01(item.model_score ?? item.score ?? item.confidence ?? 0),
      thermalScore: clamp01(item.thermal_score ?? 0),
      thermalCoverage: clamp01(item.thermal_coverage ?? 0),
      thermalMax: clamp01(item.thermal_max ?? 0),
      thermalHotFraction: clamp01(item.thermal_hot_fraction ?? 0),
      thermalTempC: finiteOrNull(item.thermal_temp_c),
      pointpillarsSupport: clamp01(item.pointpillars_support ?? 0),
      supportPoints: finiteOrNull(item.support_points),
      supportZSpan: finiteOrNull(item.support_z_span),
      fusionNote: String(item.fusion_note ?? ''),
      thermalUnavailable: String(item.fusion_note ?? '').includes('thermal_unavailable'),
      range: Math.hypot(center[0], center[1], center[2]),
      length: Math.abs(size[0]),
      width: Math.abs(size[1]),
      height: Math.abs(size[2]),
      yaw: Number(item.yaw ?? item.heading ?? 0) || 0,
    }
  }).filter(item => item.size.every(Number.isFinite) && item.center.every(Number.isFinite))
}

function objectKey(source, id) {
  return `${String(source)}:${String(id)}`
}

function vec3(value, fallback) {
  if (Array.isArray(value) && value.length >= 3) return value.slice(0, 3).map(Number)
  if (value && typeof value === 'object') {
    return [Number(value.x), Number(value.y), Number(value.z ?? value.height ?? 0)]
  }
  return fallback
}

function bboxSize(minValue, maxValue) {
  const mn = vec3(minValue, null)
  const mx = vec3(maxValue, null)
  if (!mn || !mx) return null
  return [Math.abs(mx[0] - mn[0]), Math.abs(mx[1] - mn[1]), Math.abs(mx[2] - mn[2])]
}

function clamp01(v) {
  return Math.max(0, Math.min(1, Number(v) || 0))
}

function finiteOrNull(v) {
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

function percent(v) {
  return `${Math.round(clamp01(v) * 100)}%`
}

function meters(value) {
  return `${Number(value).toFixed(1)} m`
}

function metersCompact(value) {
  return `${Number(value).toFixed(1)}`
}

function degrees(value) {
  return `${Math.round(value * 180 / Math.PI)}deg`
}

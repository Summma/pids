import PanelShell from '@/components/PanelShell'
import styles from './ObjectListPanel.module.css'

export default function ObjectListPanel({ lidarData }) {
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
            {objects.map(object => (
              <div className={styles.row} key={`${object.source}-${object.id}`}>
                <div className={styles.main}>
                  <span className={styles.label}>{object.label}</span>
                  <span className={styles.source}>{object.source}</span>
                </div>
                <div className={styles.metrics}>
                  <span>{Math.round(object.score * 100)}%</span>
                  <span>{object.range.toFixed(1)} m</span>
                </div>
              </div>
            ))}
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
    return {
      id: item.track_id ?? item.id ?? i + 1,
      source: item.source ?? 'detector',
      center,
      size,
      label,
      score: clamp01(item.score ?? item.confidence ?? item.model_score ?? 0),
      range: Math.hypot(center[0], center[1], center[2]),
    }
  }).filter(item => item.size.every(Number.isFinite) && item.center.every(Number.isFinite))
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

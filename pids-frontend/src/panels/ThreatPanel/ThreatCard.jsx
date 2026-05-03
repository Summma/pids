import styles from './ThreatPanel.module.css'

const MODALITY_ICONS = {
  thermal: { icon: '♨', label: 'Thermal', cls: 'thermal' },
  lidar:   { icon: '◈', label: 'Lidar',   cls: 'lidar'   },
  rf:      { icon: '⫿', label: 'RF',    cls: 'rf'      },
}

function confidenceColor(c) {
  if (c >= 0.8) return 'var(--threat-crit)'
  if (c >= 0.6) return 'var(--threat-high)'
  if (c >= 0.4) return 'var(--threat-medium)'
  return 'var(--threat-low)'
}

function confidenceLabel(c) {
  if (c >= 0.8) return 'Critical'
  if (c >= 0.6) return 'High'
  if (c >= 0.4) return 'Medium'
  return 'Low'
}

export default function ThreatCard({ threat, selected, onClick }) {
  const { id, confidence, modalities = [], position = {}, velocity = {},
          speed = 0, range = 0, bearing = 0, age_frames = 0 } = threat

  const color = confidenceColor(confidence)

  return (
    <div
      className={`${styles.card} ${selected ? styles.cardSelected : ''}`}
      onClick={onClick}
      style={{ '--threat-color': color }}
    >
      <div className={styles.cardTop}>
        <div className={styles.trackId}>T{String(id).padStart(3, '0')}</div>
        <div className={styles.modalityIcons}>
          {Object.entries(MODALITY_ICONS).map(([key, cfg]) => (
            <span
              key={key}
              className={`${styles.modalIcon} ${modalities.includes(key) ? styles[cfg.cls] : styles.modalInactive}`}
              title={cfg.label}
            >
              {cfg.icon}
            </span>
          ))}
        </div>
        <div className={styles.confLabel} style={{ color }}>
          {confidenceLabel(confidence)}
        </div>
      </div>

      <div className={styles.confBarWrap}>
        <div
          className={styles.confBar}
          style={{ width: `${confidence * 100}%`, background: color }}
        />
      </div>
      <div className={styles.confPct} style={{ color }}>
        {(confidence * 100).toFixed(0)}%
      </div>

      <div className={styles.cardStats}>
        <Stat label="Range" value={`${range.toFixed(1)} m`} />
        <Stat label="Bearing" value={`${bearing.toFixed(0)}°`} />
        <Stat label="Speed" value={`${speed.toFixed(1)} m/s`} />
        <Stat label="Age" value={`${age_frames}f`} />
      </div>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statLabel}>{label}</span>
      <span className={styles.statVal}>{value}</span>
    </div>
  )
}

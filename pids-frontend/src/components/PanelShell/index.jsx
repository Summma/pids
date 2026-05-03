import styles from './PanelShell.module.css'

const STATE_LABEL = {
  live:       { cls: styles.live,       label: 'Live' },
  connecting: { cls: styles.offline,    label: 'Connecting' },
  stale:      { cls: styles.offline,    label: 'Stale' },
  offline:    { cls: styles.offline,    label: 'Offline' },
}

export default function PanelShell({
  title,
  subtitle,
  connState = 'connecting',
  modality,          // 'thermal' | 'lidar' | 'rf' | undefined
  controls,          // ReactNode — right-side header controls
  onPopOut,          // fn → show pop-out button
  children,
  className = '',
  bare = false,
  showStatusDot = true,
}) {
  const sc = STATE_LABEL[connState] ?? STATE_LABEL.connecting

  return (
    <div className={`${styles.shell} ${bare ? styles.bare : ''} ${className}`}>
      {bare ? (
        <span
          className={`${styles.bareStateDot} ${sc.cls}`}
          title={sc.label}
          aria-label={sc.label}
        />
      ) : (
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <span className={styles.title}>{title}</span>
            {subtitle && <span className={styles.subtitle}>{subtitle}</span>}
          </div>
          <div className={styles.headerRight}>
            {controls}
            {showStatusDot && (
              <span
                className={`${styles.stateDot} ${sc.cls}`}
                title={sc.label}
                aria-label={sc.label}
              />
            )}
            {onPopOut && (
              <button className={styles.popOutBtn} onClick={onPopOut} title="Pop out">
                ⤢
              </button>
            )}
          </div>
        </div>
      )}
      <div className={styles.body}>
        {children}
      </div>
    </div>
  )
}

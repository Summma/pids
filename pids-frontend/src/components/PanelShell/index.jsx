import styles from './PanelShell.module.css'

const STATE_LABEL = {
  live:       { cls: styles.live,       dot: true  },
  connecting: { cls: styles.connecting, dot: true  },
  stale:      { cls: styles.stale,      dot: true  },
  offline:    { cls: styles.offline,    dot: false },
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
}) {
  const sc = STATE_LABEL[connState] ?? STATE_LABEL.connecting

  return (
    <div className={`${styles.shell} ${className}`}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          {modality && <span className={`${styles.modalityDot} ${styles[modality]}`} />}
          <span className={styles.title}>{title}</span>
          {subtitle && <span className={styles.subtitle}>{subtitle}</span>}
        </div>
        <div className={styles.headerRight}>
          {controls}
          <div className={`${styles.statePill} ${sc.cls}`}>
            {sc.dot && <span className={styles.dot} />}
            {connState.toUpperCase()}
          </div>
          {onPopOut && (
            <button className={styles.popOutBtn} onClick={onPopOut} title="Pop out">
              ⤢
            </button>
          )}
        </div>
      </div>
      <div className={styles.body}>
        {children}
      </div>
    </div>
  )
}

import { useState } from 'react'
import { saveConfig, HOST, PORT } from '@/utils/wsConfig'
import styles from './TitleBar.module.css'

export default function TitleBar({
  threatStats = { critical: 0 },
  calibrationActive = false,
  onGoHome,
  onOpenCalibration,
}) {
  const [showSettings, setShowSettings] = useState(false)
  const [host, setHost] = useState(HOST)
  const [port, setPort] = useState(PORT)

  const goHome = () => {
    setShowSettings(false)
    onGoHome?.()
  }

  const toggleCalibration = () => {
    setShowSettings(false)
    onOpenCalibration?.()
  }

  const applySettings = () => {
    saveConfig(host, Number(port))
    setShowSettings(false)
    window.location.reload()
  }

  return (
    <>
      <header className={styles.bar}>
        <div className={styles.left}>
          <button className={styles.logoButton} onClick={goHome} title="Back to main page" aria-label="Back to main page">
            Narya
          </button>
        </div>

        <div className={styles.center}>
          {threatStats.critical > 0 && (
            <div className={styles.threatAlert}>
              {threatStats.critical} critical threat{threatStats.critical > 1 ? 's' : ''}
            </div>
          )}
        </div>

        <div className={styles.right}>
          <button
            className={`${styles.iconBtn} ${calibrationActive ? styles.iconBtnActive : ''}`}
            onClick={toggleCalibration}
            title={calibrationActive ? 'Close calibration' : 'Calibration'}
            aria-label={calibrationActive ? 'Close calibration' : 'Open calibration'}
          >
            <CalibrationIcon />
          </button>
          <button
            className={`${styles.iconBtn} ${showSettings ? styles.iconBtnActive : ''}`}
            onClick={() => setShowSettings(s => !s)}
            title="Settings"
            aria-label="Open settings"
          >
            <GearIcon />
          </button>
        </div>
      </header>

      {showSettings && (
        <div className={styles.settingsDropdown}>
          <div className={styles.settingsTitle}>Connection</div>
          <div className={styles.settingsRow}>
            <label className={styles.settingsLabel}>Jetson Host</label>
            <input
              className={styles.settingsInput}
              value={host}
              onChange={e => setHost(e.target.value)}
              placeholder="192.168.1.42"
            />
          </div>
          <div className={styles.settingsRow}>
            <label className={styles.settingsLabel}>Port</label>
            <input
              className={styles.settingsInput}
              value={port}
              onChange={e => setPort(e.target.value)}
              placeholder="9090"
              style={{ width: 70 }}
            />
          </div>
          <button className={styles.settingsApply} onClick={applySettings}>
            Apply and reconnect
          </button>
        </div>
      )}
    </>
  )
}

function CalibrationIcon() {
  return (
    <svg className={styles.icon} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="6.5" />
      <path d="M12 3.5v3M12 17.5v3M3.5 12h3M17.5 12h3" />
      <circle cx="12" cy="12" r="1.8" />
    </svg>
  )
}

function GearIcon() {
  return (
    <svg className={styles.icon} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="3.4" />
      <path d="M12 2.8l1.1 2.2 2.4.6 2-1.3 2.2 2.2-1.3 2 .6 2.4 2.2 1.1-1.1 3-2.5-.2-1.8 1.8.2 2.5-3 1.1-1.1-2.2-2.4-.6-2 1.3-2.2-2.2 1.3-2-.6-2.4-2.2-1.1 1.1-3 2.5.2 1.8-1.8-.2-2.5z" />
    </svg>
  )
}

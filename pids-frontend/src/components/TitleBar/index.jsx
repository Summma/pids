import { useState } from 'react'
import { saveConfig, HOST, PORT } from '@/utils/wsConfig'
import styles from './TitleBar.module.css'

export default function TitleBar({ systemState, threatStats = { critical: 0 }, onOpenCalibration }) {
  const [showSettings, setShowSettings] = useState(false)
  const [host, setHost] = useState(HOST)
  const [port, setPort] = useState(PORT)

  const applySettings = () => {
    saveConfig(host, Number(port))
    setShowSettings(false)
    window.location.reload()
  }

  return (
    <>
      <header className={styles.bar}>
        <div className={styles.left}>
          <span className={styles.logo}>◈ Narya</span>
          
          <div className={styles.divider} />
          <span className={styles.version}>v1.0</span>
        </div>

        <div className={styles.center}>
          {threatStats.critical > 0 && (
            <div className={styles.threatAlert}>
              ⚠ {threatStats.critical} CRITICAL THREAT{threatStats.critical > 1 ? 'S' : ''}
            </div>
          )}
        </div>

        <div className={styles.right}>
          <div className={styles.sensorPills}>
            <SensorPill label="THERM" state={systemState.thermal} color="thermal" />
            <SensorPill label="LIDAR" state={systemState.lidar}   color="lidar"   />
          </div>
          <button className={styles.iconBtn} onClick={onOpenCalibration} title="Calibration">⊕</button>
          <button className={styles.iconBtn} onClick={() => setShowSettings(s => !s)} title="Settings">⚙</button>
        </div>
      </header>

      {showSettings && (
        <div className={styles.settingsDropdown}>
          <div className={styles.settingsTitle}>SETTINGS</div>
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
            APPLY &amp; RECONNECT
          </button>
        </div>
      )}
    </>
  )
}

function SensorPill({ label, state, color }) {
  const isLive = state === 'live'
  return (
    <div className={`${styles.pill} ${isLive ? styles[color] : styles.pillOffline}`}>
      <span className={`${styles.pillDot} ${isLive ? styles.pillDotLive : ''}`} />
      {label}
    </div>
  )
}

import { useEffect, useState } from 'react'
import {
  HOST,
  PORT,
  DETECTION_MODE,
  DETECTION_MODE_OPTIONS,
  LIDAR_MAX_POINTS,
  LIDAR_POINT_OPTIONS,
  THERMAL_PALETTE,
  THERMAL_PALETTE_OPTIONS,
  SHOW_THERMAL_FOV,
} from '@/utils/wsConfig'
import styles from './TitleBar.module.css'

export default function TitleBar({
  threatStats = { critical: 0 },
  calibrationActive = false,
  streamConfig = {
    host: HOST,
    port: PORT,
    detectionMode: DETECTION_MODE,
    lidarMaxPoints: LIDAR_MAX_POINTS,
  },
  displayConfig = {
    thermalPalette: THERMAL_PALETTE,
    showThermalFov: SHOW_THERMAL_FOV,
  },
  lidarBackground = null,
  backgroundBusy = false,
  backgroundMessage = '',
  onStreamSettingsChange,
  onDisplaySettingsChange,
  onCaptureLidarBackground,
  onClearLidarBackground,
  onGoHome,
  onOpenCalibration,
}) {
  const [showSettings, setShowSettings] = useState(false)
  const [host, setHost] = useState(streamConfig.host)
  const [port, setPort] = useState(streamConfig.port)
  const [detectionMode, setDetectionMode] = useState(streamConfig.detectionMode)
  const [lidarMaxPoints, setLidarMaxPoints] = useState(streamConfig.lidarMaxPoints)
  const [thermalPalette, setThermalPalette] = useState(displayConfig.thermalPalette)
  const [showThermalFov, setShowThermalFov] = useState(displayConfig.showThermalFov)

  useEffect(() => {
    setHost(streamConfig.host)
    setPort(streamConfig.port)
    setDetectionMode(streamConfig.detectionMode)
    setLidarMaxPoints(streamConfig.lidarMaxPoints)
  }, [streamConfig.host, streamConfig.port, streamConfig.detectionMode, streamConfig.lidarMaxPoints])

  useEffect(() => {
    setThermalPalette(displayConfig.thermalPalette)
    setShowThermalFov(displayConfig.showThermalFov)
  }, [displayConfig.thermalPalette, displayConfig.showThermalFov])

  const streamChanged =
    String(host).trim() !== String(streamConfig.host)
    || Number(port) !== Number(streamConfig.port)
    || detectionMode !== streamConfig.detectionMode
    || Number(lidarMaxPoints) !== Number(streamConfig.lidarMaxPoints)

  const goHome = () => {
    setShowSettings(false)
    onGoHome?.()
  }

  const toggleCalibration = () => {
    setShowSettings(false)
    onOpenCalibration?.()
  }

  const reconnectStreams = () => {
    onStreamSettingsChange?.({
      host: String(host).trim() || HOST,
      port: Number(port),
      detectionMode,
      lidarMaxPoints: Number(lidarMaxPoints),
    })
    setShowSettings(false)
  }

  const changeThermalPalette = (value) => {
    setThermalPalette(value)
    onDisplaySettingsChange?.({ thermalPalette: value, showThermalFov })
  }

  const changeShowThermalFov = (value) => {
    setShowThermalFov(value)
    onDisplaySettingsChange?.({ thermalPalette, showThermalFov: value })
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
          <div className={styles.settingsTitle}>Detector</div>
          <div className={styles.settingsRow}>
            <label className={styles.settingsLabel}>Model</label>
            <select
              className={styles.settingsSelect}
              value={detectionMode}
              onChange={e => setDetectionMode(e.target.value)}
              title={DETECTION_MODE_OPTIONS.find(option => option.value === detectionMode)?.title}
            >
              {DETECTION_MODE_OPTIONS.map(option => (
                <option key={option.value} value={option.value} title={option.title}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.settingsRow}>
            <label className={styles.settingsLabel}>Points</label>
            <select
              className={styles.settingsSelect}
              value={lidarMaxPoints}
              onChange={e => setLidarMaxPoints(Number(e.target.value))}
              title="Maximum lidar points sent to this browser per frame"
            >
              {LIDAR_POINT_OPTIONS.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.settingsTitle}>Background</div>
          <div className={styles.settingsButtonRow}>
            <button
              type="button"
              className={styles.settingsSecondary}
              onClick={onCaptureLidarBackground}
              disabled={backgroundBusy}
              title="Capture the current lidar room state and subtract it before detector inference"
            >
              Capture room
            </button>
            <button
              type="button"
              className={styles.settingsSecondary}
              onClick={onClearLidarBackground}
              disabled={backgroundBusy || !lidarBackground?.hasSnapshot}
              title="Disable background subtraction and clear the stored room snapshot"
            >
              Clear
            </button>
          </div>
          <div className={styles.settingsNote}>
            {backgroundStatusText(lidarBackground, backgroundMessage)}
          </div>
          <div className={styles.settingsTitle}>Display</div>
          <div className={styles.settingsRow}>
            <label className={styles.settingsLabel}>Thermal palette</label>
            <select
              className={styles.settingsSelect}
              value={thermalPalette}
              onChange={e => changeThermalPalette(e.target.value)}
              title="Palette used for projected thermal pixels in the 3D view"
            >
              {THERMAL_PALETTE_OPTIONS.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <label className={styles.settingsCheckRow}>
            <span className={styles.settingsLabel}>Show thermal FOV</span>
            <input
              type="checkbox"
              checked={showThermalFov}
              onChange={e => changeShowThermalFov(e.target.checked)}
            />
          </label>
          <div className={styles.settingsNote}>Display changes apply immediately.</div>
          <button
            className={styles.settingsApply}
            onClick={reconnectStreams}
            disabled={!streamChanged}
          >
            Reconnect streams
          </button>
        </div>
      )}
    </>
  )
}

function backgroundStatusText(background, message) {
  if (background?.pendingCapture) return 'Waiting for the next lidar frame to capture.'
  if (!background?.hasSnapshot) return message || 'No room snapshot. Detector uses the full point cloud.'
  const foreground = Math.round(background.lastForegroundPoints || 0).toLocaleString()
  const original = Math.round(background.lastOriginalPoints || 0).toLocaleString()
  const removedPct = Math.round((background.lastRemovedFraction || 0) * 100)
  const mode = background.enabled ? 'enabled' : 'stored, disabled'
  return `Snapshot ${mode}. Detector foreground ${foreground}/${original}; ${removedPct}% removed.`
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

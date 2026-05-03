import { useMemo, useRef, useState } from 'react'
import TitleBar        from '@/components/TitleBar'
import GeminiSidebar   from '@/components/GeminiSidebar'
import WorldMapPanel   from '@/panels/WorldMapPanel'
import CameraPanel     from '@/panels/CameraPanel'
import ObjectListPanel from '@/panels/ObjectListPanel'
import CalibrationPanel from '@/panels/CalibrationPanel'
import { useThermal }  from '@/hooks/useThermal'
import { useLidar }    from '@/hooks/useLidar'
import { useCamera }   from '@/hooks/useCamera'
import {
  apiUrl,
  saveConfig,
  HOST,
  PORT,
  DETECTION_MODE,
  LIDAR_MAX_POINTS,
  THERMAL_PALETTE,
  SHOW_THERMAL_FOV,
} from '@/utils/wsConfig'
import { synthesizePersonDetections } from '@/utils/yoloDetections'
import styles from './App.module.css'

const SENSOR_VIEWS = [
  { key: 'fusion', label: 'Thermal + Lidar' },
  { key: 'camera', label: 'Camera' },
]

const INITIAL_STREAM_CONFIG = {
  host: HOST,
  port: PORT,
  detectionMode: DETECTION_MODE,
  lidarMaxPoints: LIDAR_MAX_POINTS,
}

const INITIAL_DISPLAY_CONFIG = {
  thermalPalette: THERMAL_PALETTE,
  showThermalFov: SHOW_THERMAL_FOV,
}

const MISSION_BRIEF_PROMPT = `Generate a concise operator mission brief from the current live scene.

Use this format:
Current scene:
Objects of interest:
Thermal evidence:
Recommended next action:

Be direct and evidence-based. Call out uncertainty. Do not invent objects, locations, identities, weapons, or intent.`

export default function App() {
  const [showCal,  setShowCal]  = useState(false)
  const [activeSensorView, setActiveSensorView] = useState('fusion')
  const [selectedObjectKey, setSelectedObjectKey] = useState('')
  const [thermalCalibrationOverride, setThermalCalibrationOverride] = useState(null)
  const [streamConfig, setStreamConfig] = useState(INITIAL_STREAM_CONFIG)
  const [displayConfig, setDisplayConfig] = useState(INITIAL_DISPLAY_CONFIG)
  const [analystMessages, setAnalystMessages] = useState([])
  const [analystBusy, setAnalystBusy] = useState(false)
  const [backgroundBusy, setBackgroundBusy] = useState(false)
  const [backgroundMessage, setBackgroundMessage] = useState('')
  const worldRef = useRef(null)
  const cameraRef = useRef(null)

  const thermal = useThermal(streamConfig)
  const lidar   = useLidar(streamConfig)
  const camera  = useCamera(streamConfig)

  // Anchor sensor sync on the slowest stream (lidar): pull camera + thermal
  // frames whose ts is closest to the lidar's ts so the YOLO bbox, thermal
  // sample, and 3D lidar position all describe the same captured moment.
  const yoloDetections = useMemo(() => {
    const lidarTs = lidar.meta?.ts ?? 0
    const cameraSync = camera.frameAt ? camera.frameAt(lidarTs) : camera.frame
    const thermalSync = thermal.frameAt ? thermal.frameAt(lidarTs) : thermal.frameRef?.current
    return synthesizePersonDetections({
      persons: cameraSync?.persons ?? [],
      frameW: cameraSync?.frameW ?? 0,
      frameH: cameraSync?.frameH ?? 0,
      thermalFrame: thermalSync,
      thermalMin: thermal.meta?.tMin,
      thermalMax: thermal.meta?.tMax,
      thermalCalibration: thermal.meta?.calibration,
      lidarFrame: lidar.frameRef?.current,
    })
  }, [
    camera.frame?.seq,
    thermal.meta?.seq,
    lidar.meta?.seq,
    thermal.meta?.calibration,
  ])

  // YOLO is the sole object detector — drop the backend's 3D detections
  // before they reach the world view, the object list, or the scene analyst.
  const lidarWithYolo = useMemo(() => ({
    ...lidar,
    detections: yoloDetections,
  }), [lidar, yoloDetections])

  // Lidar is the slowest stream — use it as the time anchor so the thermal
  // overlay and camera feed don't visibly run ahead of the 3D points.
  const lidarTs = lidar.meta?.ts ?? 0
  const lidarSeq = lidar.meta?.seq ?? 0

  const syncedThermalData = useMemo(() => {
    const matched = thermal.frameAt ? thermal.frameAt(lidarTs) : thermal.frameRef?.current
    return {
      connState: thermal.connState,
      frameAt: thermal.frameAt,
      frameRef: { current: matched ?? null },
      meta: {
        ...thermal.meta,
        seq: lidarSeq,
        ts: matched?.ts ?? 0,
      },
      error: thermal.error,
      sendControl: thermal.sendControl,
    }
  }, [thermal.connState, thermal.frameAt, thermal.frameRef, thermal.meta, thermal.error, thermal.sendControl, lidarTs, lidarSeq])

  const syncedCameraData = useMemo(() => {
    const matched = camera.frameAt ? camera.frameAt(lidarTs) : camera.frame
    return {
      connState: camera.connState,
      frameAt: camera.frameAt,
      frame: matched ?? camera.frame,
      error: camera.error,
      sendControl: camera.sendControl,
    }
  }, [camera.connState, camera.frameAt, camera.frame, camera.error, camera.sendControl, lidarTs])

  function selectObject(objectKey) {
    setSelectedObjectKey(objectKey || '')
    if (objectKey) setActiveSensorView('fusion')
  }

  function sendMissionBrief() {
    askSceneAnalyst(MISSION_BRIEF_PROMPT, 'Mission brief')
  }

  function applyStreamSettings(nextStreamConfig) {
    const saved = saveConfig(
      nextStreamConfig.host,
      Number(nextStreamConfig.port),
      nextStreamConfig.detectionMode,
      nextStreamConfig.lidarMaxPoints,
      displayConfig.thermalPalette,
      displayConfig.showThermalFov,
    )
    setStreamConfig(streamFromSaved(saved))
    setDisplayConfig(displayFromSaved(saved))
  }

  function applyDisplaySettings(nextDisplayConfig) {
    const saved = saveConfig(
      streamConfig.host,
      streamConfig.port,
      streamConfig.detectionMode,
      streamConfig.lidarMaxPoints,
      nextDisplayConfig.thermalPalette,
      nextDisplayConfig.showThermalFov,
    )
    setStreamConfig(streamFromSaved(saved))
    setDisplayConfig(displayFromSaved(saved))
  }

  async function updateLidarBackground(action) {
    if (backgroundBusy) return
    setBackgroundBusy(true)
    setBackgroundMessage('')
    try {
      const response = await fetch(apiUrl('/lidar/background', streamConfig), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok || !data.ok) {
        throw new Error(data.error || `background request failed (${response.status})`)
      }
      setBackgroundMessage(data.message || '')
    } catch (err) {
      setBackgroundMessage(err?.message || 'background request failed')
    } finally {
      setBackgroundBusy(false)
    }
  }

  async function askSceneAnalyst(message, displayText = message) {
    const text = message.trim()
    if (!text || analystBusy) return

    const userMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      text: displayText,
    }
    setAnalystMessages(prev => [...prev, userMessage])
    setAnalystBusy(true)

    let timeout = null
    try {
      const payload = {
        message: text,
        images: captureSceneImages(worldRef, cameraRef),
        scene: sceneSnapshot(lidarWithYolo, thermal, camera),
      }

      const controller = new AbortController()
      timeout = window.setTimeout(() => controller.abort(), 90000)
      const response = await fetch(apiUrl('/gemini/chat', streamConfig), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })

      const data = await response.json().catch(() => ({}))
      if (!response.ok || !data.ok) {
        throw new Error(cleanAnalystError(data.error) || `Scene analyst request failed (${response.status})`)
      }

      setAnalystMessages(prev => [
        ...prev,
        {
          id: `${Date.now()}-analyst`,
          role: 'assistant',
          text: data.text || 'No response text returned.',
        },
      ])
    } catch (err) {
      setAnalystMessages(prev => [
        ...prev,
        {
          id: `${Date.now()}-error`,
          role: 'assistant',
          text: err?.name === 'AbortError'
            ? 'Scene analyst took too long to respond.'
            : `Scene analyst is unavailable: ${err?.message || 'unknown error'}`,
          error: true,
        },
      ])
    } finally {
      if (timeout) window.clearTimeout(timeout)
      setAnalystBusy(false)
    }
  }

  return (
    <div className={styles.root}>
      <TitleBar
        calibrationActive={showCal}
        streamConfig={streamConfig}
        displayConfig={displayConfig}
        onStreamSettingsChange={applyStreamSettings}
        onDisplaySettingsChange={applyDisplaySettings}
        lidarBackground={lidar.backgroundMeta}
        backgroundBusy={backgroundBusy}
        backgroundMessage={backgroundMessage}
        onCaptureLidarBackground={() => updateLidarBackground('capture')}
        onClearLidarBackground={() => updateLidarBackground('clear')}
        onGoHome={() => setShowCal(false)}
        onOpenCalibration={() => setShowCal(open => !open)}
      />

      <main className={styles.main}>
        <div className={styles.workspace}>
          <div className={styles.liveGrid}>
            <div className={styles.sensorStage}>
              <div className={styles.viewSwitch} aria-label="Sensor view">
                {SENSOR_VIEWS.map(view => {
                  const active = activeSensorView === view.key
                  return (
                    <button
                      key={view.key}
                      type="button"
                      className={`${styles.viewSwitchButton} ${active ? styles.viewSwitchButtonActive : ''}`}
                      aria-pressed={active}
                      onClick={() => setActiveSensorView(view.key)}
                    >
                      {view.label}
                    </button>
                  )
                })}
              </div>

              <div
                className={`${styles.sensorLayer} ${activeSensorView === 'fusion' ? styles.sensorLayerActive : ''}`}
                aria-hidden={activeSensorView !== 'fusion'}
              >
                <WorldMapPanel
                  ref={worldRef}
                  lidarData={lidarWithYolo}
                  thermalData={syncedThermalData}
                  selectedObjectKey={selectedObjectKey}
                  onSelectObject={selectObject}
                  thermalCalibrationOverride={thermalCalibrationOverride}
                  thermalPalette={displayConfig.thermalPalette}
                  showThermalFov={displayConfig.showThermalFov}
                />
              </div>

              <div
                className={`${styles.sensorLayer} ${activeSensorView === 'camera' ? styles.sensorLayerActive : ''}`}
                aria-hidden={activeSensorView !== 'camera'}
              >
                <CameraPanel ref={cameraRef} cameraData={syncedCameraData} />
              </div>
            </div>
          </div>
          <div className={styles.sideStack}>
            <GeminiSidebar
              messages={analystMessages}
              busy={analystBusy}
              onSend={askSceneAnalyst}
              onMissionBrief={sendMissionBrief}
            />
            <ObjectListPanel
              lidarData={lidarWithYolo}
              selectedObjectKey={selectedObjectKey}
              onSelectObject={selectObject}
            />
          </div>
        </div>
      </main>

      {showCal && (
        <CalibrationPanel
          lidarData={lidar}
          thermalData={thermal}
          cameraData={camera}
          calibrationOverride={thermalCalibrationOverride}
          onCalibrationChange={setThermalCalibrationOverride}
          onClose={() => setShowCal(false)}
        />
      )}
    </div>
  )
}

function captureSceneImages(worldRef, cameraRef) {
  const images = []
  const world = worldRef.current?.capture?.()
  if (world) {
    images.push({
      label: '3D lidar and thermal render',
      dataUrl: world,
    })
  }

  const camera = cameraRef.current?.capture?.()
  if (camera) {
    images.push({
      label: 'Visible camera frame',
      dataUrl: camera,
    })
  }

  return images
}

function cleanAnalystError(error) {
  if (!error) return ''
  return String(error)
    .replace(/\bGemini\b/g, 'Scene analyst')
    .replace(/\bgemini\b/g, 'scene analyst')
}

function sceneSnapshot(lidar, thermal, camera) {
  return {
    lidar: {
      state: lidar.connState,
      points: lidar.meta?.n ?? 0,
      seq: lidar.meta?.seq ?? 0,
      ts: lidar.meta?.ts ?? 0,
      detection_meta: lidar.detectionMeta ?? {},
      background: lidar.backgroundMeta ?? {},
      detections: Array.isArray(lidar.detections) ? lidar.detections.slice(0, 32) : [],
    },
    thermal: {
      state: thermal.connState,
      seq: thermal.meta?.seq ?? 0,
      ts: thermal.meta?.ts ?? 0,
      min_c: thermal.meta?.tMin ?? null,
      max_c: thermal.meta?.tMax ?? null,
      width: thermal.meta?.w ?? null,
      height: thermal.meta?.h ?? null,
      error: thermal.error ?? '',
    },
    camera: {
      state: camera.connState,
      seq: camera.frame?.seq ?? 0,
      ts: camera.frame?.ts ?? 0,
      width: camera.frame?.w ?? null,
      height: camera.frame?.h ?? null,
    },
  }
}

function streamFromSaved(saved) {
  return {
    host: saved.host,
    port: saved.port,
    detectionMode: saved.detectionMode,
    lidarMaxPoints: saved.lidarMaxPoints,
  }
}

function displayFromSaved(saved) {
  return {
    thermalPalette: saved.thermalPalette,
    showThermalFov: saved.showThermalFov,
  }
}

import { useRef, useState } from 'react'
import TitleBar        from '@/components/TitleBar'
import GeminiSidebar   from '@/components/GeminiSidebar'
import WorldMapPanel   from '@/panels/WorldMapPanel'
import CameraPanel     from '@/panels/CameraPanel'
import ObjectListPanel from '@/panels/ObjectListPanel'
import CalibrationPanel from '@/panels/CalibrationPanel'
import { useThermal }  from '@/hooks/useThermal'
import { useLidar }    from '@/hooks/useLidar'
import { useCamera }   from '@/hooks/useCamera'
import { apiUrl } from '@/utils/wsConfig'
import styles from './App.module.css'

export default function App() {
  const [showCal,  setShowCal]  = useState(false)
  const [analystMessages, setAnalystMessages] = useState([])
  const [analystBusy, setAnalystBusy] = useState(false)
  const worldRef = useRef(null)
  const cameraRef = useRef(null)

  const thermal = useThermal()
  const lidar   = useLidar()
  const camera  = useCamera()

  async function askSceneAnalyst(message) {
    const text = message.trim()
    if (!text || analystBusy) return

    const userMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      text,
    }
    setAnalystMessages(prev => [...prev, userMessage])
    setAnalystBusy(true)

    let timeout = null
    try {
      const payload = {
        message: text,
        images: captureSceneImages(worldRef, cameraRef),
        scene: sceneSnapshot(lidar, thermal, camera),
      }

      const controller = new AbortController()
      timeout = window.setTimeout(() => controller.abort(), 90000)
      const response = await fetch(apiUrl('/gemini/chat'), {
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
        onGoHome={() => setShowCal(false)}
        onOpenCalibration={() => setShowCal(open => !open)}
      />

      <main className={styles.main}>
        <div className={styles.workspace}>
          <div className={styles.liveGrid}>
            <WorldMapPanel
              ref={worldRef}
              lidarData={lidar}
              thermalData={thermal}
            />
            <CameraPanel ref={cameraRef} cameraData={camera} />
          </div>
          <div className={styles.sideStack}>
            <GeminiSidebar
              messages={analystMessages}
              busy={analystBusy}
              onSend={askSceneAnalyst}
            />
            <ObjectListPanel lidarData={lidar} />
          </div>
        </div>
      </main>

      {showCal && <CalibrationPanel onClose={() => setShowCal(false)} />}
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

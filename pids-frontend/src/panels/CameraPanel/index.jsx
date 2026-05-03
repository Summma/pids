import { forwardRef, useImperativeHandle, useRef } from 'react'
import PanelShell from '@/components/PanelShell'
import styles from './CameraPanel.module.css'

const CameraPanel = forwardRef(function CameraPanel({ cameraData }, ref) {
  const { connState, frame, error } = cameraData
  const imgRef = useRef(null)
  useImperativeHandle(ref, () => ({
    capture() {
      if (frame.src) return frame.src
      if (!imgRef.current) return null
      return captureImageElement(imgRef.current)
    },
  }), [frame.src])

  return (
    <PanelShell title="Live Camera" modality="camera" connState={connState} bare>
      <div className={styles.viewport}>
        {frame.src ? (
          <img ref={imgRef} className={styles.image} src={frame.src} alt="Live visible camera stream" />
        ) : (
          <div className={styles.empty}>
            <div className={styles.emptyTitle}>No camera frame</div>
            <div className={styles.emptyText}>
              {error || 'Waiting for a real camera stream from /camera.'}
            </div>
          </div>
        )}

        <div className={styles.readout}>
          <span>{frame.seq ? `Seq ${frame.seq}` : 'No signal'}</span>
          {error && <span className={styles.warn}>{error}</span>}
        </div>
      </div>
    </PanelShell>
  )
})

export default CameraPanel

function captureImageElement(img) {
  const width = img.naturalWidth || img.clientWidth
  const height = img.naturalHeight || img.clientHeight
  if (!width || !height) return null

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.drawImage(img, 0, 0, width, height)
  return canvas.toDataURL('image/jpeg', 0.86)
}

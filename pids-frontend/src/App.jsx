import { useState } from 'react'
import TitleBar        from '@/components/TitleBar'
import ThermalPanel    from '@/panels/ThermalPanel'
import WorldMapPanel   from '@/panels/WorldMapPanel'
import PointCloudPanel from '@/panels/PointCloudPanel'
import CalibrationPanel from '@/panels/CalibrationPanel'
import { useThermal }  from '@/hooks/useThermal'
import { useLidar }    from '@/hooks/useLidar'
import { useFusion }   from '@/hooks/useFusion'
import styles from './App.module.css'

export default function App() {
  const [show3D,   setShow3D]   = useState(false)
  const [showCal,  setShowCal]  = useState(false)

  const thermal = useThermal()
  const lidar   = useLidar()
  const fusion  = useFusion()

  const systemState = {
    thermal: thermal.connState,
    lidar:   lidar.connState,
  }

  return (
    <div className={styles.root}>
      <TitleBar
        systemState={systemState}
        onOpenCalibration={() => setShowCal(true)}
      />

      <main className={styles.main}>
        <div className={styles.liveGrid}>
          <ThermalPanel thermalData={thermal} />
          <WorldMapPanel
            lidarData={lidar}
            fusionData={fusion}
            thermalData={thermal}
            onPopOut3D={() => setShow3D(true)}
          />
        </div>
      </main>

      {show3D  && <PointCloudPanel lidarData={lidar} onClose={() => setShow3D(false)}  />}
      {showCal && <CalibrationPanel onClose={() => setShowCal(false)} />}
    </div>
  )
}

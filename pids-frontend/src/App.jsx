import { useState } from 'react'
import TitleBar        from '@/components/TitleBar'
import ThermalPanel    from '@/panels/ThermalPanel'
import LidarPanel      from '@/panels/LidarPanel'
import RFPanel         from '@/panels/RFPanel'
import ThreatPanel     from '@/panels/ThreatPanel'
import FusionMapPanel  from '@/panels/FusionMapPanel'
import PointCloudPanel from '@/panels/PointCloudPanel'
import CalibrationPanel from '@/panels/CalibrationPanel'
import { useThreats }  from '@/hooks/useThreats'
import { useThermal }  from '@/hooks/useThermal'
import { useLidar }    from '@/hooks/useLidar'
import { useRF }       from '@/hooks/useRF'
import { useFusion }   from '@/hooks/useFusion'
import styles from './App.module.css'

export default function App() {
  const [show3D,   setShow3D]   = useState(false)
  const [showCal,  setShowCal]  = useState(false)

  const thermal = useThermal()
  const lidar   = useLidar()
  const rf      = useRF()
  const threats = useThreats()
  const fusion  = useFusion()

  const systemState = {
    thermal: thermal.connState,
    lidar:   lidar.connState,
    rf:      rf.connState,
  }

  return (
    <div className={styles.root}>
      <TitleBar
        systemState={systemState}
        threatStats={threats.stats}
        onOpenCalibration={() => setShowCal(true)}
      />

      <main className={styles.main}>
        <div className={styles.sensorRow}>
          <ThermalPanel thermalData={thermal} />
          <LidarPanel lidarData={lidar} onPopOut3D={() => setShow3D(true)} />
          <FusionMapPanel fusionData={fusion} lidarData={lidar} threatData={threats} />
        </div>

        <div className={styles.bottomRow}>
          <RFPanel rfData={rf} />
          <ThreatPanel threatData={threats} />
        </div>
      </main>

      {show3D  && <PointCloudPanel lidarData={lidar} onClose={() => setShow3D(false)}  />}
      {showCal && <CalibrationPanel onClose={() => setShowCal(false)} />}
    </div>
  )
}

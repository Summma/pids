import { useState } from 'react'
import PanelShell from '@/components/PanelShell'
import { RF_BANDS } from '@/hooks/useRF'
import Spectrum from './Spectrum'
import Waterfall from './Waterfall'
import styles from './RFPanel.module.css'

export default function RFPanel({ rfData }) {
  const { connState, fftRef, meta, peaks, setBand } = rfData
  const [activeBand, setActiveBand] = useState('ISM 433')
  const [showWaterfall, setShowWaterfall] = useState(true)

  const handleBand = (band) => {
    setActiveBand(band)
    setBand(band)
  }

  const controls = (
    <div className={styles.controls}>
      <select
        className="ctrl-select"
        value={activeBand}
        onChange={e => handleBand(e.target.value)}
      >
        {Object.keys(RF_BANDS).map(b => (
          <option key={b} value={b}>{b}</option>
        ))}
      </select>
      <button
        className={`btn ${showWaterfall ? 'active' : ''}`}
        onClick={() => setShowWaterfall(w => !w)}
      >
        FALL
      </button>
    </div>
  )

  const peakCount = peaks.length
  const subtitle  = peakCount > 0 ? `${peakCount} EMITTER${peakCount > 1 ? 'S' : ''}` : undefined

  return (
    <PanelShell
      title="RF SPECTRUM"
      subtitle={subtitle}
      modality="rf"
      connState={connState}
      controls={controls}
    >
      <div className={styles.rfBody}>
        <div className={styles.specWrap}>
          <Spectrum fftRef={fftRef} meta={meta} peaks={peaks} />
        </div>
        {showWaterfall && (
          <div className={styles.fallWrap}>
            <Waterfall fftRef={fftRef} meta={meta} />
          </div>
        )}
        {peaks.length > 0 && (
          <div className={styles.peakList}>
            {peaks.map((pk, i) => (
              <div key={i} className={styles.peakItem}>
                <span className={styles.peakFreq}>{(pk.freq / 1e6).toFixed(3)} MHz</span>
                <span className={styles.peakPwr}>{pk.power.toFixed(1)} dBm</span>
                {pk.label && <span className={styles.peakLabel}>{pk.label}</span>}
              </div>
            ))}
          </div>
        )}
      </div>
    </PanelShell>
  )
}

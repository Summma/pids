import { useState } from 'react'
import PanelShell from '@/components/PanelShell'
import ThreatCard from './ThreatCard'
import styles from './ThreatPanel.module.css'

export default function ThreatPanel({ threatData }) {
  const { connState, threats } = threatData
  const [selectedId, setSelectedId] = useState(null)
  const [sortBy, setSortBy] = useState('confidence')

  const sorted = [...threats].sort((a, b) => {
    if (sortBy === 'confidence') return b.confidence - a.confidence
    if (sortBy === 'range')      return a.range - b.range
    if (sortBy === 'speed')      return b.speed - a.speed
    return 0
  })

  const subtitle = threats.length > 0
    ? `${threats.length} TRACK${threats.length > 1 ? 'S' : ''}`
    : 'CLEAR'

  const controls = (
    <div className={styles.controls}>
      <span className={styles.sortLabel}>SORT</span>
      {['confidence', 'range', 'speed'].map(s => (
        <button
          key={s}
          className={`btn ${sortBy === s ? 'active' : ''}`}
          onClick={() => setSortBy(s)}
        >
          {s.toUpperCase()}
        </button>
      ))}
    </div>
  )

  return (
    <PanelShell
      title="THREAT FUSION"
      subtitle={subtitle}
      connState={connState}
      controls={controls}
    >
      <div className={styles.body}>
        {threats.length === 0 ? (
          <div className={styles.clear}>
            <span className={styles.clearIcon}>✓</span>
            <span>PERIMETER CLEAR</span>
          </div>
        ) : (
          <div className={styles.cardList}>
            {sorted.map(t => (
              <ThreatCard
                key={t.id}
                threat={t}
                selected={selectedId === t.id}
                onClick={() => setSelectedId(id => id === t.id ? null : t.id)}
              />
            ))}
          </div>
        )}
      </div>
    </PanelShell>
  )
}

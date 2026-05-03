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
    ? `${threats.length} track${threats.length > 1 ? 's' : ''}`
    : 'Clear'

  const controls = (
    <div className={styles.controls}>
      <span className={styles.sortLabel}>Sort</span>
      {['confidence', 'range', 'speed'].map(s => (
        <button
          key={s}
          className={`btn ${sortBy === s ? 'active' : ''}`}
          onClick={() => setSortBy(s)}
        >
          {s}
        </button>
      ))}
    </div>
  )

  return (
    <PanelShell
      title="Threat fusion"
      subtitle={subtitle}
      connState={connState}
      controls={controls}
    >
      <div className={styles.body}>
        {threats.length === 0 ? (
          <div className={styles.clear}>
            <span className={styles.clearIcon}>✓</span>
            <span>Perimeter clear</span>
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

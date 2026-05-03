import { useRef, useEffect } from 'react'
import styles from './RFPanel.module.css'

const WATERFALL_ROWS = 120

export default function Waterfall({ fftRef, meta }) {
  const canvasRef  = useRef(null)
  const bufferRef  = useRef([])   // ring buffer of rows
  const rafRef     = useRef(null)
  const lastNFft   = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')

    const render = () => {
      rafRef.current = requestAnimationFrame(render)
      const w = canvas.offsetWidth
      const h = canvas.offsetHeight
      canvas.width  = w
      canvas.height = h

      const { nFft, noiseFloor } = meta
      if (!nFft) return

      const fft    = fftRef.current
      const dbMin  = noiseFloor - 5
      const dbMax  = noiseFloor + 55

      // Append new row
      if (nFft !== lastNFft.current) {
        bufferRef.current = []
        lastNFft.current  = nFft
      }

      const row = new Float32Array(nFft)
      for (let i = 0; i < nFft; i++) row[i] = fft[i]
      bufferRef.current.push(row)
      if (bufferRef.current.length > WATERFALL_ROWS) bufferRef.current.shift()

      const rowH  = h / WATERFALL_ROWS
      const buf   = bufferRef.current
      const start = Math.max(0, WATERFALL_ROWS - buf.length)

      for (let r = 0; r < buf.length; r++) {
        const rowData = buf[buf.length - 1 - r]
        const y       = (start + r) * rowH

        for (let i = 0; i < nFft; i++) {
          const x  = (i / nFft) * w
          const bw = Math.max(1, w / nFft)
          const u  = Math.max(0, Math.min(1, (rowData[i] - dbMin) / (dbMax - dbMin)))
          ctx.fillStyle = waterfallColor(u)
          ctx.fillRect(x, y, bw + 0.5, rowH + 0.5)
        }
      }
    }

    render()
    return () => cancelAnimationFrame(rafRef.current)
  }, [fftRef, meta])

  return <canvas ref={canvasRef} className={styles.waterfallCanvas} />
}

function waterfallColor(u) {
  // Black → purple → blue → cyan → green → yellow → red
  const stops = [
    [0.00, [0,   0,   0  ]],
    [0.20, [40,  0,  100 ]],
    [0.40, [0,   50, 200 ]],
    [0.60, [0,  200, 200 ]],
    [0.75, [0,  220,  60 ]],
    [0.88, [220,200,   0 ]],
    [1.00, [255, 50,   0 ]],
  ]
  for (let i = 1; i < stops.length; i++) {
    if (u <= stops[i][0]) {
      const [u0, c0] = stops[i - 1]
      const [u1, c1] = stops[i]
      const f = (u - u0) / (u1 - u0)
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * f)
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * f)
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * f)
      return `rgb(${r},${g},${b})`
    }
  }
  return 'rgb(255,50,0)'
}

import { useRef, useEffect } from 'react'
import styles from './RFPanel.module.css'

export default function Spectrum({ fftRef, meta, peaks }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx    = canvas.getContext('2d')

    const w = canvas.offsetWidth
    const h = canvas.offsetHeight
    canvas.width  = w
    canvas.height = h

    ctx.fillStyle = '#080c10'
    ctx.fillRect(0, 0, w, h)

    const { noiseFloor, nFft } = meta
    if (!nFft) return

    const fft        = fftRef.current
    const dbMin      = noiseFloor - 10
    const dbMax      = noiseFloor + 60
    const dbRange    = dbMax - dbMin
    const padTop     = 8
    const padBottom  = 24
    const plotH      = h - padTop - padBottom

    // Noise floor line
    const nfY = padTop + plotH * (1 - (noiseFloor - dbMin) / dbRange)
    ctx.strokeStyle = '#3d5a6e'
    ctx.lineWidth   = 1
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(0, nfY)
    ctx.lineTo(w, nfY)
    ctx.stroke()
    ctx.setLineDash([])

      // dB axis labels
      ctx.fillStyle = '#3d5a6e'
      ctx.font      = '9px JetBrains Mono'
      for (let db = Math.ceil(dbMin / 10) * 10; db <= dbMax; db += 20) {
        const y = padTop + plotH * (1 - (db - dbMin) / dbRange)
        ctx.fillText(`${db}`, 2, y + 3)
      }

      // Spectrum fill
      const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH)
      grad.addColorStop(0,   'rgba(199,125,255,0.9)')
      grad.addColorStop(0.5, 'rgba(100,60,160,0.5)')
      grad.addColorStop(1,   'rgba(30,0,60,0.1)')

      ctx.beginPath()
      ctx.moveTo(0, padTop + plotH)
      for (let i = 0; i < nFft; i++) {
        const x = (i / (nFft - 1)) * w
        const y = padTop + plotH * (1 - (fft[i] - dbMin) / dbRange)
        if (i === 0) ctx.lineTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.lineTo(w, padTop + plotH)
      ctx.closePath()
      ctx.fillStyle = grad
      ctx.fill()

      // Spectrum line
      ctx.strokeStyle = '#c77dff'
      ctx.lineWidth   = 1.5
      ctx.beginPath()
      for (let i = 0; i < nFft; i++) {
        const x = (i / (nFft - 1)) * w
        const y = padTop + plotH * (1 - (fft[i] - dbMin) / dbRange)
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.stroke()

      // Peak markers
      const halfBW = meta.sampleRate / 2
      const freqToX = (f) => ((f - (meta.centerFreq - halfBW)) / meta.sampleRate) * w

      peaks.forEach(pk => {
        const x   = freqToX(pk.freq)
        const y   = padTop + plotH * (1 - (pk.power - dbMin) / dbRange)
        const mhz = (pk.freq / 1e6).toFixed(3)

        ctx.fillStyle   = '#ffcc00'
        ctx.strokeStyle = '#ffcc00'
        ctx.lineWidth   = 1

        ctx.beginPath()
        ctx.moveTo(x, y - 4)
        ctx.lineTo(x, padTop + plotH)
        ctx.setLineDash([2, 3])
        ctx.stroke()
        ctx.setLineDash([])

        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)
        ctx.fill()

        ctx.font      = '9px JetBrains Mono'
        ctx.fillStyle = '#ffcc00'
        ctx.fillText(`${mhz} MHz`, x + 5, y - 2)
      })

      // Freq axis
      ctx.fillStyle = '#3d5a6e'
      ctx.font      = '9px JetBrains Mono'
      const startMHz = ((meta.centerFreq - halfBW) / 1e6).toFixed(1)
      const endMHz   = ((meta.centerFreq + halfBW) / 1e6).toFixed(1)
      const centMHz  = (meta.centerFreq / 1e6).toFixed(1)
      ctx.fillText(`${startMHz}`, 2, h - 6)
      ctx.fillText(`${centMHz} MHz`, w / 2 - 25, h - 6)
    ctx.fillText(`${endMHz}`, w - 35, h - 6)
  }, [fftRef, meta, meta.ts, peaks])

  return <canvas ref={canvasRef} className={styles.specCanvas} />
}

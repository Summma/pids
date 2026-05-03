const STOPS = [
  [0.00, [0,   68,  255]],
  [0.25, [0,   170, 255]],
  [0.50, [0,   255, 136]],
  [0.75, [255, 204, 0  ]],
  [1.00, [255, 34,  0  ]],
]

export function thermalRgb(t, tMin, tMax) {
  const u = Math.max(0, Math.min(1, (t - tMin) / Math.max(tMax - tMin, 0.001)))
  for (let i = 1; i < STOPS.length; i++) {
    if (u <= STOPS[i][0]) {
      const [u0, c0] = STOPS[i - 1]
      const [u1, c1] = STOPS[i]
      const f = (u - u0) / (u1 - u0)
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ]
    }
  }
  return [255, 34, 0]
}

export function thermalCss(t, tMin, tMax) {
  const [r, g, b] = thermalRgb(t, tMin, tMax)
  return `rgb(${r},${g},${b})`
}

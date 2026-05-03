function buildLUT(stops) {
  const lut = new Array(256)
  for (let i = 0; i < 256; i++) {
    const u = i / 255
    let s = stops[0], e = stops[stops.length - 1]
    for (let j = 1; j < stops.length; j++) {
      if (u <= stops[j][0]) { s = stops[j - 1]; e = stops[j]; break }
    }
    const f = (u - s[0]) / Math.max(e[0] - s[0], 0.001)
    lut[i] = [
      Math.round(s[1] + (e[1] - s[1]) * f),
      Math.round(s[2] + (e[2] - s[2]) * f),
      Math.round(s[3] + (e[3] - s[3]) * f),
    ]
  }
  return lut
}

export const PALETTES = {
  WHITE_HOT: buildLUT([
    [0.0, 0,   0,   0  ],
    [1.0, 255, 255, 255],
  ]),

  BLACK_HOT: buildLUT([
    [0.0, 255, 255, 255],
    [1.0, 0,   0,   0  ],
  ]),

  IRONBOW: buildLUT([
    [0.00,   0,   0,   0],
    [0.10,  30,   0,  80],
    [0.25,  90,   0, 140],
    [0.40, 160,  20,  60],
    [0.55, 210,  70,   0],
    [0.70, 240, 150,   0],
    [0.85, 255, 220, 100],
    [1.00, 255, 255, 255],
  ]),

  RAINBOW: buildLUT([
    [0.00,   0,   0, 255],
    [0.25,   0, 255, 255],
    [0.50,   0, 255,   0],
    [0.75, 255, 255,   0],
    [1.00, 255,   0,   0],
  ]),

  LAVA: buildLUT([
    [0.00,   0,   0,   0],
    [0.33,  80,   0,   0],
    [0.66, 220,  80,   0],
    [0.85, 255, 200,  50],
    [1.00, 255, 255, 200],
  ]),
}

export const PALETTE_NAMES = Object.keys(PALETTES)

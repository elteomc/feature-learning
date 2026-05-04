const families = {
  isotropic: {
    label: "Isotropic Gaussian",
    summary: "Baseline teacher-student setting with isotropic inputs",
    weightedResidual: 3.579e-3,
    boundRatio: 0.147,
    betaOverMean: 0.993,
    betaOverMeanStd: 0.042,
    betaOverMeanMin: 0.847,
    betaOverMeanMax: 1.059,
    pushedPair: 5.59e-5,
    aPair: 260.947,
    symmetricRelative: 0.288
  },
  anisotropic: {
    label: "Anisotropic Gaussian",
    summary: "Gaussian inputs with a nontrivial covariance spectrum",
    weightedResidual: 1.896e-3,
    boundRatio: 0.0896,
    betaOverMean: 1.002,
    betaOverMeanStd: 0.015,
    betaOverMeanMin: 0.968,
    betaOverMeanMax: 1.050,
    pushedPair: 1.11e-5,
    aPair: 76.103,
    symmetricRelative: 0.273
  },
  low_rank_signal: {
    label: "Low-rank signal",
    summary: "Inputs with a planted signal subspace plus isotropic noise",
    weightedResidual: 6.356e-4,
    boundRatio: 0.0987,
    betaOverMean: 1.015,
    betaOverMeanStd: 0.050,
    betaOverMeanMin: 0.893,
    betaOverMeanMax: 1.181,
    pushedPair: 2.51e-6,
    aPair: 97.771,
    symmetricRelative: 0.340
  }
}

const sweeps = {
  regime: {
    label: "Residual energy",
    title: "Test of raw sensitivity versus residual energy",
    description: "The x-axis is centered on the Residual energy slider. Stationarity defect, pair defect, bad-direction gain, and beta correlation controls move the curves.",
    xLabel: "residual energy",
    logScale: true,
    tableNote: "Each row is a hypothetical checkpoint at a different residual energy. Stationarity defect and pair compression are held fixed, so raw sensitivity varies purely because beta fit scales with residual energy.",
    series: [
      { key: "weightedError", label: "weighted proxy", cssVar: "--series-a" },
      { key: "rawSensitivity", label: "raw sensitivity", cssVar: "--series-b" }
    ],
    columns: [
      ["residualEnergy", "residual energy"],
      ["betaFit", "beta fit"],
      ["rawSensitivity", "raw sensitivity"],
      ["regime", "regime"],
      ["isCurrent", "selected"]
    ]
  },
  beta: {
    label: "Beta correlation",
    title: "Beta identity sweep",
    description: "The x-axis is centered on the Correlation slider. Leverage CV and Residual CV change the height of the beta-error curves.",
    xLabel: "correlation",
    logScale: false,
    tableNote: "Each row is a possible leverage-residual correlation. Leverage CV and residual CV are fixed by the sidebar, so the table focuses on the beta error implied by changing correlation.",
    series: [
      { key: "signedBetaError", label: "signed beta error", cssVar: "--series-a" },
      { key: "absoluteBetaError", label: "absolute beta error", cssVar: "--series-c" }
    ],
    columns: [
      ["corr", "correlation"],
      ["signedBetaError", "signed error"],
      ["absoluteBetaError", "absolute error"],
      ["isCurrent", "selected"]
    ]
  },
  pair: {
    label: "Pair gain",
    title: "Pair defect versus stationarity gain",
    description: "The x-axis is centered on the Bad-direction gain slider. Pair defect and defect-gain correlation change the risk scale.",
    xLabel: "bad-direction gain",
    logScale: true,
    tableNote: "Each row changes the stationarity gain on a bad pair direction. The pair defect is held fixed so the table shows how gain turns a worst-direction defect into actual risk.",
    series: [
      { key: "supportDefect", label: "support defect", cssVar: "--series-c" },
      { key: "gainWeightedContribution", label: "gain-weighted contribution", cssVar: "--series-b" }
    ],
    columns: [
      ["gain", "gain"],
      ["gainWeightedContribution", "defect times gain squared"],
      ["diagnosticRisk", "diagnostic risk"],
      ["warning", "warning"],
      ["isCurrent", "selected"]
    ]
  }
}

const figureBase = "../../paper/figures/"

const reportedFigures = [
  {
    group: "Main evidence",
    label: "Weighted residual by family",
    file: "weighted_residual_by_family.png",
    caption: "Weighted-law residual at best-stationarity checkpoints. Lower is better; all three families land well below 0.01.",
    detailHtml: "This is the most direct evidence plot for the late-training law. Lower bars mean the learned feature matrix satisfies <span class=\"math\">H<sup>2</sup> &approx; &kappa;<sub>eff</sub> G&#771;</span> more closely at the best-stationarity checkpoint."
  },
  {
    group: "Main evidence",
    label: "Theorem bound ratio by family",
    file: "theorem_bound_ratio_by_family.png",
    caption: "Observed weighted residual divided by the deterministic bound. Values below 1 mean the theorem covers the observed error.",
    detailHtml: "The deterministic theorem gives an upper-bound template involving pair error and stationarity defect. Values below one mean the observed residual is covered by that bound. The bound is expected to be conservative across all three families."
  },
  {
    group: "Main evidence",
    label: "Beta over residual energy",
    file: "beta_over_residual_energy_by_family.png",
    caption: "Beta fit tracks mean residual squared across all checkpoints.",
    detailHtml: "The beta bridge says <span class=\"math\">&beta;<sub>fit</sub></span> is a leverage-weighted residual average. Values near one mean that hidden-gradient leverage is not strongly biasing the residual average away from ordinary mean residual energy."
  },
  {
    group: "Main evidence",
    label: "Pushed pair error",
    file: "pushed_pair_error_by_family.png",
    caption: "Pair error after pushing through the learned feature map.",
    detailHtml: "This plot is about the pair error that actually enters the weighted law. It can be small even when the support-normalized <span class=\"math\">A<sub>pair</sub></span> diagnostic is large."
  },
  {
    group: "Trajectories",
    label: "Isotropic trajectory",
    file: "isotropic_two_regime_trajectory.png",
    caption: "Two-regime trajectory for isotropic seed 0.",
    detailHtml: "This trajectory shows how the raw-conditioned relation and the weighted relation appear at different phases of training in the baseline isotropic family."
  },
  {
    group: "Trajectories",
    label: "Anisotropic trajectory",
    file: "anisotropic_two_regime_trajectory.png",
    caption: "Two-regime trajectory for anisotropic seed 0.",
    detailHtml: "This checks the same two-regime story when the input covariance has a nontrivial spectrum, where the effective-dimension correction matters."
  },
  {
    group: "Trajectories",
    label: "Low-rank trajectory",
    file: "low_rank_signal_two_regime_trajectory.png",
    caption: "Two-regime trajectory for low-rank signal seed 0.",
    detailHtml: "This checks the trajectory story in a structured signal family rather than pure Gaussian noise."
  },
  {
    group: "Beta collapse",
    label: "Isotropic beta collapse",
    file: "isotropic_beta_collapse.png",
    caption: "Beta collapse for the isotropic representative run.",
    detailHtml: "This shows <span class=\"math\">&beta;<sub>fit</sub></span> shrinking as interpolation is approached. That collapse explains why converting the weighted law into a raw AGOP law becomes numerically delicate late in training."
  },
  {
    group: "Beta collapse",
    label: "Anisotropic beta collapse",
    file: "anisotropic_beta_collapse.png",
    caption: "Beta collapse for the anisotropic representative run.",
    detailHtml: "This checks whether the beta-collapse mechanism remains visible under anisotropic input geometry."
  },
  {
    group: "Beta collapse",
    label: "Low-rank beta collapse",
    file: "low_rank_signal_beta_collapse.png",
    caption: "Beta collapse for the low-rank representative run.",
    detailHtml: "This checks whether the beta bridge remains interpretable when the signal lives mostly in a low-dimensional subspace."
  },
  {
    group: "Secondary diagnostics",
    label: "Symmetric relative error",
    file: "symmetric_relative_error_by_family.png",
    caption: "Secondary relative weighted-law diagnostic.",
    detailHtml: "This is a scale-normalized version of the weighted-law residual. It is useful for honest reporting, but it is not the central theorem diagnostic."
  },
  {
    group: "Failure-mode taxonomy",
    label: "Failure regime map",
    file: "failure_modes/regime_map.png",
    caption: "Algebraic taxonomy of failure regimes in the (beta-axis, pair-axis) plane.",
    detailHtml: "This is a deterministic algebraic picture, not a trained-network plot. It separates regimes where beta-bridge collapse versus pair-compression failure dominate. The three empirical families land in the benign quadrant (top-left)."
  },
  {
    group: "Failure-mode taxonomy",
    label: "Beta failure — toy example",
    file: "failure_modes/beta_failure_toy.png",
    caption: "Algebraic toy example of beta over- and under-estimation.",
    detailHtml: "A single high-leverage sample pulls <span class=\"math\">&beta;<sub>fit</sub></span> above <span class=\"math\">mean(r<sup>2</sup>)</span> when it has high residual, and below it when it has low residual. This is what the beta bridge controls."
  },
  {
    group: "Failure-mode taxonomy",
    label: "Pair failure — toy example",
    file: "failure_modes/pair_failure_toy.png",
    caption: "Algebraic toy example of high-gain vs low-gain pair geometry.",
    detailHtml: "The same global <span class=\"math\">A<sub>pair</sub></span> value can produce large pushed error in one case and small pushed error in another, depending on whether the bad direction is also a high-gain direction."
  }
]

function resolveColor(cssVar) {
  return getComputedStyle(document.body).getPropertyValue(cssVar).trim()
}

function option(value, label) {
  const item = document.createElement("option")
  item.value = value
  item.textContent = label
  return item
}

function numberValue(id) {
  return Number(document.getElementById(id).value)
}

function pow10(id) {
  return Math.pow(10, numberValue(id))
}

function formatSci(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return String(value)
  }
  return value.toExponential(2)
}

function formatCompact(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return String(value)
  }
  const abs = Math.abs(value)
  if (abs > 0 && (abs < 0.01 || abs >= 1000)) {
    return value.toExponential(2)
  }
  return value.toFixed(3)
}

function setOutput(id, value, fixed = false) {
  const output = document.getElementById(id)
  output.textContent = fixed ? value.toFixed(2) : formatSci(value)
}

function seededMultiplier(seed, familyKey) {
  let hash = seed + 17
  for (let i = 0; i < familyKey.length; i += 1) {
    hash = (hash * 31 + familyKey.charCodeAt(i)) % 9973
  }
  return 0.92 + (hash % 17) * 0.01
}

function currentState() {
  const familyKey = document.getElementById("family-select").value
  const family = families[familyKey]
  const seed = Math.max(0, Math.min(99, Math.round(numberValue("seed-input"))))
  const seedScale = seededMultiplier(seed, familyKey)
  const familyWeightedScale = family.weightedResidual / families.isotropic.weightedResidual
  const familyPairScale = family.pushedPair / families.isotropic.pushedPair
  const residualEnergy = pow10("residual-slider") * family.betaOverMean * seedScale
  const stationarityDefect = pow10("stationarity-slider") * (0.75 + familyWeightedScale) * seedScale
  const leverageCv = pow10("leverage-cv-slider")
  const residSqCv = pow10("resid-cv-slider")
  const corr = numberValue("corr-slider")
  const pairDefect = pow10("pair-defect-slider")
  const pairGain = pow10("pair-gain-slider")
  const gainCorr = numberValue("gain-corr-slider")
  const pushedPairProxy = pairDefect * pairGain * pairGain * (0.5 + familyPairScale) * seedScale
  const weightedError = stationarityDefect + pushedPairProxy
  const betaError = corr * leverageCv * residSqCv
  const betaDistortion = Math.max(0.05, 1 + betaError)
  const betaFit = residualEnergy * betaDistortion
  const rawSensitivity = weightedError / Math.max(betaFit, 1e-12)
  const diagnosticRisk = Math.abs(gainCorr) * pairDefect * Math.max(pairGain, 1e-8)

  return {
    residualEnergy,
    stationarityDefect,
    leverageCv,
    residSqCv,
    corr,
    pairDefect,
    pairGain,
    gainCorr,
    pushedPairProxy,
    weightedError,
    betaFit,
    rawSensitivity,
    betaError,
    betaDistortion,
    diagnosticRisk
  }
}

function buildRegimeRows(state) {
  const rows = []
  const center = Math.log10(Math.max(state.residualEnergy, 1e-8))
  for (let i = 0; i <= 10; i += 1) {
    const logResidual = center - 2.5 + i * 0.5
    const residualEnergy = Math.pow(10, logResidual)
    const betaFit = residualEnergy * state.betaDistortion
    const rawSensitivity = state.weightedError / Math.max(betaFit, 1e-12)
    rows.push({
      x: residualEnergy,
      residualEnergy,
      betaFit,
      weightedError: state.weightedError,
      rawSensitivity,
      regime: betaFit < 1e-3 ? "late weighted" : "raw-conditioned",
      isCurrent: i === 5 ? "current" : ""
    })
  }
  return rows
}

function buildBetaRows(state) {
  const rows = []
  // Use the largest symmetric window around state.corr that still fits in [-1, 1].
  // This guarantees that i === 5 (the midpoint) lands exactly on state.corr.
  const half = Math.min(1 + state.corr, 1 - state.corr)
  const start = state.corr - half
  const end = state.corr + half
  for (let i = 0; i <= 10; i += 1) {
    const corr = start + (end - start) * i / 10
    const signedBetaError = corr * state.leverageCv * state.residSqCv
    rows.push({
      x: corr,
      corr,
      leverageCv: state.leverageCv,
      residSqCv: state.residSqCv,
      signedBetaError,
      absoluteBetaError: Math.abs(signedBetaError),
      isCurrent: i === 5 ? "current" : ""
    })
  }
  return rows
}

function buildPairRows(state) {
  const rows = []
  const center = Math.log10(Math.max(state.pairGain, 1e-8))
  for (let i = 0; i <= 10; i += 1) {
    const logGain = center - 3 + i * 0.6
    const gain = Math.pow(10, logGain)
    const gainWeightedContribution = state.pairDefect * gain * gain
    const diagnosticRisk = Math.abs(state.gainCorr) * state.pairDefect * gain
    rows.push({
      x: gain,
      gain,
      supportDefect: state.pairDefect,
      gainWeightedContribution,
      diagnosticRisk,
      warning: diagnosticRisk > 1 ? "watch" : "harmless",
      isCurrent: i === 5 ? "current" : ""
    })
  }
  return rows
}

function buildRows(sweepKey, state) {
  if (sweepKey === "beta") {
    return buildBetaRows(state)
  }
  if (sweepKey === "pair") {
    return buildPairRows(state)
  }
  return buildRegimeRows(state)
}

function plotScale(values, minPixel, maxPixel, logScale = false) {
  const finite = values.filter((value) => Number.isFinite(value) && (!logScale || value > 0))
  const safeValues = finite.length > 0 ? finite : [1]
  let minValue = Math.min(...safeValues)
  let maxValue = Math.max(...safeValues)

  if (minValue === maxValue) {
    const padding = logScale ? Math.max(minValue * 0.5, 1e-12) : Math.max(Math.abs(minValue) * 0.2, 1)
    minValue -= padding
    maxValue += padding
  }

  const transform = (value) => logScale ? Math.log10(Math.max(value, 1e-12)) : value
  const minTransformed = transform(minValue)
  const maxTransformed = transform(maxValue)
  const span = Math.max(maxTransformed - minTransformed, 1e-9)

  return (value) => {
    const ratio = (transform(value) - minTransformed) / span
    return minPixel + ratio * (maxPixel - minPixel)
  }
}

function plotDomain(values, logScale = false) {
  const finite = values.filter((value) => Number.isFinite(value) && (!logScale || value > 0))
  const safeValues = finite.length > 0 ? finite : [1]
  let minValue = Math.min(...safeValues)
  let maxValue = Math.max(...safeValues)

  if (minValue === maxValue) {
    const padding = logScale ? Math.max(minValue * 0.5, 1e-12) : Math.max(Math.abs(minValue) * 0.2, 1)
    minValue -= padding
    maxValue += padding
  }

  return { minValue, maxValue }
}

function tickValues(values, count, logScale = false) {
  const { minValue, maxValue } = plotDomain(values, logScale)
  if (logScale) {
    const minPower = Math.floor(Math.log10(Math.max(minValue, 1e-12)))
    const maxPower = Math.ceil(Math.log10(Math.max(maxValue, 1e-12)))
    const ticks = []
    for (let power = minPower; power <= maxPower; power += 1) {
      ticks.push(Math.pow(10, power))
    }
    return ticks.length > 0 ? ticks : [minValue, maxValue]
  }

  const ticks = []
  const span = Math.max(maxValue - minValue, 1e-9)
  for (let i = 0; i < count; i += 1) {
    ticks.push(minValue + span * i / Math.max(count - 1, 1))
  }
  return ticks
}

function linePath(rows, key, xScale, yScale) {
  return rows.map((row, index) => {
    const command = index === 0 ? "M" : "L"
    return command + " " + xScale(row.x).toFixed(2) + " " + yScale(row[key]).toFixed(2)
  }).join(" ")
}

function svgElement(name, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name)
  Object.entries(attrs).forEach(([key, value]) => {
    element.setAttribute(key, value)
  })
  return element
}

function renderPlot(rows, sweep) {
  const svg = document.getElementById("live-plot")
  svg.innerHTML = ""

  const width = 760
  const height = 360
  const left = 64
  const right = 28
  const top = 24
  const bottom = 56
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom

  // Use the explicit logScale flag from the sweep config (improvement B).
  const useLog = sweep.logScale

  const xValues = rows.map((row) => row.x)
  const yValues = sweep.series.flatMap((series) =>
    rows.map((row) => useLog ? Math.abs(row[series.key]) : row[series.key])
  )

  const xScale = plotScale(xValues, left, left + plotWidth, useLog)
  const yScaleRaw = plotScale(yValues, 0, plotHeight, useLog)
  const yScale = (value) => {
    const plotted = useLog ? Math.abs(value) : value
    return top + plotHeight - yScaleRaw(plotted)
  }

  // Plot frame
  svg.appendChild(svgElement("rect", {
    x: left, y: top, width: plotWidth, height: plotHeight, rx: 14, class: "plot-frame"
  }))

  // Grid lines and y-axis labels share the same tick positions so they align (bug 3).
  const yTicks = tickValues(yValues, 5, useLog)
  yTicks.forEach((tick) => {
    const y = yScale(tick)
    svg.appendChild(svgElement("line", {
      x1: left, x2: left + plotWidth, y1: y, y2: y,
      "stroke-width": 1, class: "grid-line"
    }))
    const label = svgElement("text", {
      x: left - 12, y: y + 4,
      "text-anchor": "end", fill: "currentColor", "font-size": 12, class: "axis-label"
    })
    label.textContent = formatCompact(tick)
    svg.appendChild(label)
  })

  // X-axis ticks and labels
  tickValues(xValues, 5, useLog).forEach((tick) => {
    const x = xScale(tick)
    svg.appendChild(svgElement("line", {
      x1: x, x2: x, y1: top + plotHeight, y2: top + plotHeight + 6,
      stroke: "currentColor", "stroke-width": 1, class: "axis-tick"
    }))
    const label = svgElement("text", {
      x, y: top + plotHeight + 23,
      "text-anchor": "middle", fill: "currentColor", "font-size": 12, class: "axis-label"
    })
    label.textContent = formatCompact(tick)
    svg.appendChild(label)
  })

  // Lines — draw all series first so dots always sit on top
  sweep.series.forEach((series) => {
    const color = resolveColor(series.cssVar)
    svg.appendChild(svgElement("path", {
      d: linePath(rows, series.key, xScale, yScale),
      fill: "none", stroke: color, "stroke-width": 4,
      "stroke-linecap": "round", "stroke-linejoin": "round"
    }))
  })

  // Non-current dots
  const panelColor = resolveColor("--panel")
  sweep.series.forEach((series) => {
    const color = resolveColor(series.cssVar)
    rows.filter((row) => !row.isCurrent).forEach((row) => {
      svg.appendChild(svgElement("circle", {
        cx: xScale(row.x), cy: yScale(row[series.key]), r: 4, fill: color
      }))
    })
  })

  // Current dots drawn last so they appear on top of every line and dot (improvement A).
  sweep.series.forEach((series) => {
    const color = resolveColor(series.cssVar)
    rows.filter((row) => row.isCurrent).forEach((row) => {
      svg.appendChild(svgElement("circle", {
        cx: xScale(row.x), cy: yScale(row[series.key]),
        r: 7, fill: color, stroke: panelColor, "stroke-width": 2.5
      }))
    })
  })

  // X-axis label
  const xLabel = svgElement("text", {
    x: left + plotWidth / 2, y: height - 16,
    "text-anchor": "middle", fill: "currentColor", "font-size": 14
  })
  xLabel.textContent = sweep.xLabel
  svg.appendChild(xLabel)

  // Y-axis label built from the series names so it names what is plotted (improvement C).
  const yLabelText = sweep.series.map((s) => s.label).join(" · ")
  const yLabel = svgElement("text", {
    x: 14,
    y: top + plotHeight / 2,
    "text-anchor": "middle",
    fill: "currentColor",
    "font-size": 11,
    transform: "rotate(-90 14 " + (top + plotHeight / 2) + ")"
  })
  yLabel.textContent = yLabelText
  svg.appendChild(yLabel)
}

function renderLegend(sweep) {
  const legend = document.getElementById("plot-legend")
  legend.innerHTML = ""
  sweep.series.forEach((series) => {
    const item = document.createElement("span")
    const swatch = document.createElement("span")
    swatch.className = "legend-swatch"
    swatch.style.background = "var(" + series.cssVar + ")"
    item.appendChild(swatch)
    item.append(series.label)
    legend.appendChild(item)
  })
}

function renderTable(rows, sweep) {
  const head = document.getElementById("sweep-table-head")
  const body = document.getElementById("sweep-table-body")
  const note = document.getElementById("sweep-table-note")
  head.innerHTML = ""
  body.innerHTML = ""
  note.textContent = sweep.tableNote

  const headerRow = document.createElement("tr")
  sweep.columns.forEach(([, label]) => {
    const cell = document.createElement("th")
    cell.textContent = label
    headerRow.appendChild(cell)
  })
  head.appendChild(headerRow)

  rows.forEach((row) => {
    const tableRow = document.createElement("tr")
    if (row.isCurrent) {
      tableRow.classList.add("current-row")
    }
    sweep.columns.forEach(([key]) => {
      const cell = document.createElement("td")
      cell.textContent = typeof row[key] === "number" ? formatCompact(row[key]) : row[key]
      tableRow.appendChild(cell)
    })
    body.appendChild(tableRow)
  })
}

function insightCard(label, value, text) {
  const template = document.getElementById("insight-template")
  const card = template.content.firstElementChild.cloneNode(true)
  card.querySelector(".insight-label").textContent = label
  card.querySelector(".insight-value").textContent = value
  card.querySelector(".insight-text").textContent = text
  return card
}

function renderInsights(state) {
  const target = document.getElementById("insight-cards")
  target.innerHTML = ""
  target.appendChild(insightCard(
    "beta bridge",
    formatSci(state.betaFit),
    "The live model ties beta fit to residual energy, matching the empirical bridge."
  ))
  target.appendChild(insightCard(
    "weighted proxy",
    formatSci(state.weightedError),
    "Stationarity defect plus gain-weighted pair contribution controls the weighted law."
  ))
  target.appendChild(insightCard(
    "beta identity error",
    formatCompact(state.betaError),
    "Correlation times leverage CV times residual CV gives the signed beta error."
  ))
  target.appendChild(insightCard(
    "pair risk",
    formatSci(state.diagnosticRisk),
    "Large defects become dangerous when they align with high stationarity gain."
  ))
}

function renderSummaryTable() {
  const tbody = document.getElementById("summary-table")
  tbody.innerHTML = ""
  Object.values(families).forEach((family) => {
    const row = document.createElement("tr")
    const cells = [
      family.label,
      formatSci(family.weightedResidual),
      family.betaOverMean.toFixed(3),
      formatSci(family.pushedPair)
    ]

    cells.forEach((text, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td")
      cell.textContent = text
      row.appendChild(cell)
    })
    tbody.appendChild(row)
  })
}

function renderFamilyText() {
  const familyKey = document.getElementById("family-select").value
  const family = families[familyKey]
  document.getElementById("family-title").textContent = family.label
  document.getElementById("family-summary").textContent = family.summary
}

function renderLive() {
  const state = currentState()
  const sweepKey = document.getElementById("sweep-select").value
  const sweep = sweeps[sweepKey]
  const rows = buildRows(sweepKey, state)

  setOutput("residual-output", state.residualEnergy)
  setOutput("stationarity-output", state.stationarityDefect)
  setOutput("leverage-cv-output", state.leverageCv)
  setOutput("resid-cv-output", state.residSqCv)
  setOutput("corr-output", state.corr, true)
  setOutput("pair-defect-output", state.pairDefect)
  setOutput("pair-gain-output", state.pairGain)
  setOutput("gain-corr-output", state.gainCorr, true)

  renderFamilyText()
  document.getElementById("sweep-title").textContent = sweep.title
  document.getElementById("sweep-description").textContent = sweep.description
  renderLegend(sweep)
  renderPlot(rows, sweep)
  renderTable(rows, sweep)
  renderInsights(state)
}

function renderReportedFigure() {
  const index = Number(document.getElementById("figure-select").value)
  const figure = reportedFigures[index]
  const image = document.getElementById("figure-image")
  const caption = document.getElementById("figure-caption")
  const detail = document.getElementById("figure-detail")
  image.alt = figure.caption
  image.src = figureBase + figure.file
  caption.textContent = figure.caption
  detail.innerHTML = figure.detailHtml || ""
  image.onerror = () => {
    caption.textContent = "Figure not found at the current relative path: " + figure.file
  }
}

function setTheme(theme) {
  document.body.dataset.theme = theme
  const button = document.getElementById("theme-button")
  button.textContent = theme === "dark" ? "Use light theme" : "Use dark theme"
  localStorage.setItem("weightedAgopTheme", theme)
}

function toggleTheme() {
  const nextTheme = document.body.dataset.theme === "dark" ? "light" : "dark"
  setTheme(nextTheme)
  renderLive()
}

function setView(view) {
  document.getElementById("live-view").hidden = view !== "live"
  document.getElementById("reported-view").hidden = view !== "reported"
  document.querySelector(".control-sidebar").dataset.activeView = view

  document.querySelectorAll(".view-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view)
  })
}

function renderSelectors() {
  const familySelect = document.getElementById("family-select")
  const sweepSelect = document.getElementById("sweep-select")
  const figureSelect = document.getElementById("figure-select")

  Object.entries(families).forEach(([key, family]) => {
    familySelect.appendChild(option(key, family.label))
  })

  Object.entries(sweeps).forEach(([key, sweep]) => {
    sweepSelect.appendChild(option(key, sweep.label))
  })

  const groups = new Map()
  reportedFigures.forEach((figure, index) => {
    const groupName = figure.group || "Other"
    if (!groups.has(groupName)) {
      const group = document.createElement("optgroup")
      group.label = groupName
      groups.set(groupName, group)
      figureSelect.appendChild(group)
    }
    groups.get(groupName).appendChild(option(String(index), figure.label))
  })
}

function reset() {
  document.getElementById("family-select").value = "isotropic"
  document.getElementById("seed-input").value = "0"
  document.getElementById("sweep-select").value = "regime"
  document.getElementById("residual-slider").value = "-3"
  document.getElementById("stationarity-slider").value = "-4"
  document.getElementById("leverage-cv-slider").value = "-1"
  document.getElementById("resid-cv-slider").value = "0"
  document.getElementById("corr-slider").value = "0.05"
  document.getElementById("pair-defect-slider").value = "2"
  document.getElementById("pair-gain-slider").value = "-2"
  document.getElementById("gain-corr-slider").value = "0.05"
  document.getElementById("figure-select").value = "0"
  renderLive()
  renderReportedFigure()
}

function bindEvents() {
  [
    "family-select",
    "sweep-select",
    "residual-slider",
    "stationarity-slider",
    "leverage-cv-slider",
    "resid-cv-slider",
    "corr-slider",
    "pair-defect-slider",
    "pair-gain-slider",
    "gain-corr-slider"
  ].forEach((id) => {
    const control = document.getElementById(id)
    control.addEventListener("input", renderLive)
    control.addEventListener("change", renderLive)
  })

  // Seed gets its own handler so we can clamp the displayed value on commit (improvement F).
  const seedInput = document.getElementById("seed-input")
  seedInput.addEventListener("input", renderLive)
  seedInput.addEventListener("change", () => {
    seedInput.value = Math.max(0, Math.min(99, Math.round(Number(seedInput.value) || 0)))
    renderLive()
  })

  document.getElementById("figure-select").addEventListener("change", renderReportedFigure)
  document.getElementById("reset-button").addEventListener("click", reset)
  document.getElementById("theme-button").addEventListener("click", toggleTheme)

  document.querySelectorAll(".view-button").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view))
  })
}

function main() {
  renderSelectors()
  renderSummaryTable()
  bindEvents()
  setTheme(localStorage.getItem("weightedAgopTheme") || "dark")
  reset()
  setView("live")
}

main()

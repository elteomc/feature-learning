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
    description: "The x-axis is centered on the Residual energy slider. Raw sensitivity equals weighted error over beta fit, which spikes once beta fit collapses below the weighted error.",
    xLabel: "residual energy",
    logScale: true,
    tableNote: "Each row is a hypothetical checkpoint at a different residual energy. Stationarity defect and pair compression are held fixed, so raw sensitivity varies purely because beta fit scales with residual energy.",
    activeGroups: ["weighted", "pair"],
    activeText: "Active sliders: Residual energy, Stationarity defect, Pair defect, Bad-direction gain.",
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
    description: "The x-axis is centered on the Correlation slider. Beta error equals corr times leverage CV times residual CV. Increase Leverage CV or Residual CV to grow the height of the curves.",
    xLabel: "correlation",
    logScale: false,
    tableNote: "Each row is a possible leverage-residual correlation. Leverage CV and residual CV are fixed by the sidebar, so the table focuses on the beta error implied by changing correlation.",
    activeGroups: ["beta"],
    activeText: "Active sliders: Leverage CV, Residual CV, Correlation. (Try moving Leverage CV up to see the curves grow.)",
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
    description: "The x-axis is centered on the Bad-direction gain slider. Even a large support-normalized pair defect is harmless if the bad direction has small stationarity gain.",
    xLabel: "bad-direction gain",
    logScale: true,
    tableNote: "Each row changes the stationarity gain on a bad pair direction. The pair defect is held fixed so the table shows how gain turns a worst-direction defect into actual risk.",
    activeGroups: ["pair"],
    activeText: "Active sliders: Pair defect, Bad-direction gain, Defect-gain correlation.",
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

const trajectoryMetrics = {
  r2: {
    label: "R-squared (1 - resid_mean_sq / var_y)",
    description: "The fraction of label variance the model explains. Computed live from var(y) of each family.",
    logScale: false,
    cssVar: "--series-a",
    derived: true
  },
  gamma_tilde_eff_rel_h2: {
    label: "Weighted-law relative residual",
    description: "Weighted residual normalized by the spectral norm of H squared. The theorem says this should shrink late in training.",
    logScale: true,
    cssVar: "--series-a"
  },
  beta_fit: {
    label: "Beta fit",
    description: "Hidden-leverage weighted average of squared residuals. Should track mean residual squared.",
    logScale: true,
    cssVar: "--series-b"
  },
  theorem_bound_ratio: {
    label: "Theorem bound ratio",
    description: "Observed weighted residual divided by the deterministic theorem bound. Values below 1 mean the theorem covers the observed error.",
    logScale: false,
    cssVar: "--series-c"
  },
  pair_push_scaled_op: {
    label: "Pushed pair error",
    description: "The pair error after pushing through the learned feature map. This is what the weighted law actually sees.",
    logScale: true,
    cssVar: "--series-c"
  }
}

let trajectoryData = null
let matrixSnapshots = null
let currentSnapshotKey = "late"
let scrubberStepIndex = 0
let scrubberPinnedToBest = true

function trajectoryRowMetric(row, metricKey, varY) {
  if (metricKey === "r2") {
    if (!Number.isFinite(varY) || varY <= 0) return null
    if (!Number.isFinite(row.resid_mean_sq)) return null
    return 1 - row.resid_mean_sq / varY
  }
  const value = row[metricKey]
  return Number.isFinite(value) ? value : null
}

const figureBase = "../../paper/figures/"

const reportedFigures = [
  {
    group: "Main evidence",
    label: "Weighted residual by family",
    file: "weighted_residual_by_family.png",
    caption: "Weighted-law residual at best-stationarity checkpoints. Lower is better, and all three families land well below 0.01.",
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
    label: "Beta failure, toy example",
    file: "failure_modes/beta_failure_toy.png",
    caption: "Algebraic toy example of beta over- and under-estimation.",
    detailHtml: "A single high-leverage sample pulls <span class=\"math\">&beta;<sub>fit</sub></span> above <span class=\"math\">mean(r<sup>2</sup>)</span> when it has high residual, and below it when it has low residual. This is what the beta bridge controls."
  },
  {
    group: "Failure-mode taxonomy",
    label: "Pair failure, toy example",
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
  // Defect-gain correlation amplifies pushed pair contribution: a positive
  // correlation means worst-direction pair defects also have high stationarity
  // gain, so the bad direction matters. A negative correlation suppresses it.
  const gainCorrAmplification = 1 + gainCorr
  const pushedPairProxy = pairDefect * pairGain * pairGain * (0.5 + familyPairScale) * seedScale * gainCorrAmplification
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

  // Lines, draw all series first so dots always sit on top
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

function highlightActiveControls(sweep) {
  document.querySelectorAll(".control-group").forEach((group) => {
    const groupKey = group.dataset.group
    const isActive = sweep.activeGroups && sweep.activeGroups.includes(groupKey)
    group.classList.toggle("is-inactive", groupKey && !isActive)
  })
  const activeControlsLabel = document.getElementById("sweep-active-controls")
  if (activeControlsLabel) {
    activeControlsLabel.textContent = sweep.activeText || ""
  }
}

function renderTrajectoryLegend(metricKey) {
  const legend = document.getElementById("trajectory-legend")
  legend.innerHTML = ""
  const lossSpan = document.createElement("span")
  const lossSwatch = document.createElement("span")
  lossSwatch.className = "legend-swatch"
  lossSwatch.style.background = "var(--muted)"
  lossSpan.appendChild(lossSwatch)
  lossSpan.append("loss_total (left axis)")
  legend.appendChild(lossSpan)

  const overlay = trajectoryMetrics[metricKey]
  if (overlay) {
    const span = document.createElement("span")
    const swatch = document.createElement("span")
    swatch.className = "legend-swatch"
    swatch.style.background = "var(" + overlay.cssVar + ")"
    span.appendChild(swatch)
    span.append(overlay.label + " (right axis)")
    legend.appendChild(span)
  }
}

function activeFamilyData() {
  if (!trajectoryData) return null
  const family = document.getElementById("family-select").value
  return trajectoryData.families ? trajectoryData.families[family] : null
}

function configureScrubber(familyData) {
  const scrubber = document.getElementById("trajectory-scrubber")
  if (!scrubber || !familyData) return
  const history = familyData.history || []
  scrubber.min = "0"
  scrubber.max = String(Math.max(0, history.length - 1))
  scrubber.step = "1"
  if (scrubberPinnedToBest && Number.isFinite(familyData.best_step)) {
    let bestIndex = history.findIndex((row) => row.step === familyData.best_step)
    if (bestIndex < 0) bestIndex = history.length - 1
    scrubberStepIndex = bestIndex
  }
  scrubberStepIndex = Math.max(0, Math.min(history.length - 1, scrubberStepIndex))
  scrubber.value = String(scrubberStepIndex)
}

function renderTrajectoryPlot(family, metricKey) {
  const svg = document.getElementById("trajectory-plot")
  svg.innerHTML = ""
  const configSpan = document.getElementById("trajectory-config")
  if (!trajectoryData) {
    const text = svgElement("text", {
      x: 380, y: 160, "text-anchor": "middle", fill: "currentColor", "font-size": 14
    })
    text.textContent = "Loading trajectory data..."
    svg.appendChild(text)
    return
  }
  const familyData = trajectoryData.families[family]
  if (!familyData) {
    const text = svgElement("text", {
      x: 380, y: 160, "text-anchor": "middle", fill: "currentColor", "font-size": 14
    })
    text.textContent = "No trajectory available for this family."
    svg.appendChild(text)
    return
  }
  if (configSpan) {
    const c = familyData.config
    configSpan.textContent = "n=" + c.n + ", d=" + c.d + ", m_teacher=" + c.m_teacher + ", m_student=" + c.m_student + ", seed=" + familyData.seed
  }

  const overlayInfo = trajectoryMetrics[metricKey]
  const varY = familyData.var_y

  const width = 760
  const height = 320
  const left = 64
  const right = 70
  const top = 28
  const bottom = 56
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom

  svg.appendChild(svgElement("rect", {
    x: left, y: top, width: plotWidth, height: plotHeight, rx: 14, class: "plot-frame"
  }))

  const history = familyData.history.filter((row) => row.step !== null && row.step !== undefined)
  if (history.length === 0) return
  const steps = history.map((row) => row.step)
  const lossValues = history.map((row) => row.loss_total).filter((value) => Number.isFinite(value) && value > 0)
  const overlayLog = overlayInfo && overlayInfo.logScale
  const overlayPairs = history
    .map((row) => ({ step: row.step, value: trajectoryRowMetric(row, metricKey, varY) }))
    .filter((entry) => entry.value !== null && Number.isFinite(entry.value) && (!overlayLog || entry.value > 0))
  const overlayValues = overlayPairs.map((entry) => entry.value)

  const xScale = plotScale(steps, left, left + plotWidth, false)
  const lossLogScale = plotScale(lossValues, 0, plotHeight, true)
  const lossY = (value) => top + plotHeight - lossLogScale(Math.max(value, 1e-30))
  const overlayScaleRaw = plotScale(overlayValues.length ? overlayValues : [0, 1], 0, plotHeight, !!overlayLog)
  const overlayY = (value) => {
    const v = overlayLog ? Math.max(value, 1e-30) : value
    return top + plotHeight - overlayScaleRaw(v)
  }

  tickValues(lossValues, 5, true).forEach((tick) => {
    const y = lossY(tick)
    svg.appendChild(svgElement("line", {
      x1: left, x2: left + plotWidth, y1: y, y2: y, class: "grid-line", "stroke-width": 1
    }))
    const label = svgElement("text", {
      x: left - 10, y: y + 4, "text-anchor": "end", fill: "currentColor", "font-size": 11, class: "axis-label"
    })
    label.textContent = formatCompact(tick)
    svg.appendChild(label)
  })

  if (overlayValues.length > 0) {
    tickValues(overlayValues, 4, !!overlayLog).forEach((tick) => {
      const y = overlayY(tick)
      const label = svgElement("text", {
        x: left + plotWidth + 10, y: y + 4, "text-anchor": "start", fill: "currentColor", "font-size": 11, class: "axis-label"
      })
      label.textContent = formatCompact(tick)
      svg.appendChild(label)
    })
  }

  tickValues(steps, 6, false).forEach((tick) => {
    const x = xScale(tick)
    svg.appendChild(svgElement("line", {
      x1: x, x2: x, y1: top + plotHeight, y2: top + plotHeight + 6,
      stroke: "currentColor", "stroke-width": 1, class: "axis-tick"
    }))
    const label = svgElement("text", {
      x, y: top + plotHeight + 23, "text-anchor": "middle", fill: "currentColor", "font-size": 11, class: "axis-label"
    })
    label.textContent = Math.round(tick).toString()
    svg.appendChild(label)
  })

  const lossPath = history
    .filter((row) => Number.isFinite(row.loss_total) && row.loss_total > 0)
    .map((row, index) => (index === 0 ? "M" : "L") + " " + xScale(row.step).toFixed(2) + " " + lossY(row.loss_total).toFixed(2))
    .join(" ")
  svg.appendChild(svgElement("path", {
    d: lossPath, fill: "none", stroke: resolveColor("--muted"), "stroke-width": 3, "stroke-linecap": "round", "stroke-linejoin": "round"
  }))

  if (overlayInfo && overlayPairs.length > 0) {
    const overlayPath = overlayPairs
      .map((entry, index) => (index === 0 ? "M" : "L") + " " + xScale(entry.step).toFixed(2) + " " + overlayY(entry.value).toFixed(2))
      .join(" ")
    svg.appendChild(svgElement("path", {
      d: overlayPath, fill: "none", stroke: resolveColor(overlayInfo.cssVar), "stroke-width": 3, "stroke-linecap": "round", "stroke-linejoin": "round"
    }))
  }

  if (Number.isFinite(familyData.best_step)) {
    const x = xScale(familyData.best_step)
    svg.appendChild(svgElement("line", {
      x1: x, x2: x, y1: top, y2: top + plotHeight,
      stroke: resolveColor("--olive-bright"), "stroke-width": 1.5, "stroke-dasharray": "4 4"
    }))
    const label = svgElement("text", {
      x: x + 6, y: top + 14, fill: resolveColor("--olive-bright"), "font-size": 11, "font-weight": 700
    })
    label.textContent = "best stationarity (step " + familyData.best_step + ")"
    svg.appendChild(label)
  }

  const scrubRow = history[Math.max(0, Math.min(history.length - 1, scrubberStepIndex))]
  if (scrubRow && Number.isFinite(scrubRow.step)) {
    const xs = xScale(scrubRow.step)
    svg.appendChild(svgElement("line", {
      x1: xs, x2: xs, y1: top, y2: top + plotHeight,
      stroke: resolveColor("--rust"), "stroke-width": 2
    }))
    if (Number.isFinite(scrubRow.loss_total) && scrubRow.loss_total > 0) {
      svg.appendChild(svgElement("circle", {
        cx: xs, cy: lossY(scrubRow.loss_total), r: 5,
        fill: resolveColor("--muted"), stroke: resolveColor("--panel"), "stroke-width": 2
      }))
    }
    const overlayValueAtScrub = trajectoryRowMetric(scrubRow, metricKey, varY)
    if (overlayInfo && overlayValueAtScrub !== null && Number.isFinite(overlayValueAtScrub) && (!overlayLog || overlayValueAtScrub > 0)) {
      svg.appendChild(svgElement("circle", {
        cx: xs, cy: overlayY(overlayValueAtScrub), r: 5,
        fill: resolveColor(overlayInfo.cssVar), stroke: resolveColor("--panel"), "stroke-width": 2
      }))
    }
  }

  const xLabel = svgElement("text", {
    x: left + plotWidth / 2, y: height - 14, "text-anchor": "middle", fill: "currentColor", "font-size": 13
  })
  xLabel.textContent = "training step"
  svg.appendChild(xLabel)

  const yLabelLeft = svgElement("text", {
    x: 14, y: top + plotHeight / 2, "text-anchor": "middle", fill: "currentColor", "font-size": 11,
    transform: "rotate(-90 14 " + (top + plotHeight / 2) + ")"
  })
  yLabelLeft.textContent = "loss_total (log)"
  svg.appendChild(yLabelLeft)

  if (overlayInfo) {
    const yLabelRight = svgElement("text", {
      x: width - 14, y: top + plotHeight / 2, "text-anchor": "middle", fill: "currentColor", "font-size": 11,
      transform: "rotate(90 " + (width - 14) + " " + (top + plotHeight / 2) + ")"
    })
    yLabelRight.textContent = overlayInfo.label
    svg.appendChild(yLabelRight)
  }
}

function renderTrajectoryReadout(familyData, metricKey) {
  const readout = document.getElementById("trajectory-readout")
  const output = document.getElementById("trajectory-scrubber-output")
  if (!readout) return
  readout.innerHTML = ""
  if (!familyData) {
    if (output) output.textContent = ""
    return
  }
  const history = familyData.history || []
  if (history.length === 0) {
    if (output) output.textContent = ""
    return
  }
  const index = Math.max(0, Math.min(history.length - 1, scrubberStepIndex))
  const row = history[index]
  if (output) {
    const phase = row.phase ? " (" + row.phase + ")" : ""
    output.textContent = "step " + row.step + phase
  }

  const cards = [
    ["loss_total", row.loss_total, formatSci],
    ["resid_mean_sq", row.resid_mean_sq, formatSci],
    ["beta_fit", row.beta_fit, formatSci],
    ["gamma_tilde / H^2", row.gamma_tilde_eff_rel_h2, formatCompact],
    ["theorem bound ratio", row.theorem_bound_ratio, formatCompact],
    ["pushed pair", row.pair_push_scaled_op, formatSci]
  ]
  const r2 = trajectoryRowMetric(row, "r2", familyData.var_y)
  if (r2 !== null) cards.unshift(["R^2", r2, formatCompact])

  cards.forEach(([label, value, formatter]) => {
    const card = document.createElement("div")
    card.className = "readout-card"
    const labelEl = document.createElement("p")
    labelEl.className = "readout-label"
    labelEl.textContent = label
    const valueEl = document.createElement("p")
    valueEl.className = "readout-value"
    valueEl.textContent = Number.isFinite(value) ? formatter(value) : "-"
    card.appendChild(labelEl)
    card.appendChild(valueEl)
    readout.appendChild(card)
  })
}

function getSnapshot(familyKey, snapshotKey) {
  if (!matrixSnapshots) return null
  const fam = matrixSnapshots.families && matrixSnapshots.families[familyKey]
  if (!fam) return null
  return (fam.snapshots || []).find((s) => s.label === snapshotKey) || null
}

function colorForCell(value, vmax) {
  if (!Number.isFinite(value)) return "rgba(0,0,0,0)"
  if (vmax <= 0) return "rgb(20, 20, 20)"
  const t = Math.min(1, Math.max(0, value / vmax))
  const stops = [
    [13, 21, 32],
    [41, 64, 87],
    [111, 127, 18],
    [215, 242, 91],
    [255, 240, 200]
  ]
  const idx = t * (stops.length - 1)
  const i = Math.floor(idx)
  const frac = idx - i
  const a = stops[i]
  const b = stops[Math.min(i + 1, stops.length - 1)]
  const r = Math.round(a[0] + (b[0] - a[0]) * frac)
  const g = Math.round(a[1] + (b[1] - a[1]) * frac)
  const c = Math.round(a[2] + (b[2] - a[2]) * frac)
  return "rgb(" + r + "," + g + "," + c + ")"
}

function renderAlignmentHeatmap() {
  const svg = document.getElementById("alignment-heatmap")
  if (!svg) return
  svg.innerHTML = ""
  const meta = document.getElementById("alignment-meta")
  const familyKey = document.getElementById("family-select").value
  const snapshot = getSnapshot(familyKey, currentSnapshotKey)
  if (!snapshot) {
    const text = svgElement("text", { x: 180, y: 180, "text-anchor": "middle", fill: "currentColor", "font-size": 14 })
    text.textContent = matrixSnapshots ? "No snapshot for this family." : "Loading matrix snapshots..."
    svg.appendChild(text)
    if (meta) meta.textContent = ""
    return
  }
  const matrix = snapshot.alignment_abs
  const rows = matrix.length
  const cols = matrix[0].length
  const padTop = 38
  const padBottom = 26
  const padLeft = 40
  const padRight = 22
  const width = 360
  const height = 360
  const cellW = (width - padLeft - padRight) / cols
  const cellH = (height - padTop - padBottom) / rows

  let vmax = 0
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      if (Number.isFinite(matrix[i][j])) vmax = Math.max(vmax, matrix[i][j])
    }
  }

  const heading = svgElement("text", {
    x: width / 2, y: 22, "text-anchor": "middle",
    fill: "currentColor", "font-size": 12, "font-weight": 700
  })
  heading.textContent = "|B B*^T| at " + snapshot.label + ", step " + snapshot.step
  svg.appendChild(heading)

  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      const x = padLeft + j * cellW
      const y = padTop + i * cellH
      svg.appendChild(svgElement("rect", {
        x, y, width: cellW + 0.5, height: cellH + 0.5,
        fill: colorForCell(matrix[i][j], vmax)
      }))
    }
  }

  for (let j = 0; j < cols; j += 1) {
    const label = svgElement("text", {
      x: padLeft + (j + 0.5) * cellW, y: padTop - 6,
      "text-anchor": "middle", fill: "currentColor", "font-size": 10
    })
    label.textContent = "T" + j
    svg.appendChild(label)
  }
  for (let i = 0; i < rows; i += 4) {
    const label = svgElement("text", {
      x: padLeft - 6, y: padTop + (i + 0.7) * cellH,
      "text-anchor": "end", fill: "currentColor", "font-size": 10
    })
    label.textContent = "S" + i
    svg.appendChild(label)
  }
  const xAxisLabel = svgElement("text", {
    x: padLeft + (cols * cellW) / 2, y: height - 6,
    "text-anchor": "middle", fill: "currentColor", "font-size": 11
  })
  xAxisLabel.textContent = "teacher units"
  svg.appendChild(xAxisLabel)
  const yAxisLabel = svgElement("text", {
    x: 14, y: padTop + (rows * cellH) / 2,
    "text-anchor": "middle", fill: "currentColor", "font-size": 11,
    transform: "rotate(-90 14 " + (padTop + (rows * cellH) / 2) + ")"
  })
  yAxisLabel.textContent = "student units"
  svg.appendChild(yAxisLabel)

  if (meta) {
    const r2 = (snapshot.resid_mean_sq !== null && Number.isFinite(snapshot.resid_mean_sq) && matrixSnapshots.families[familyKey].var_y > 0)
      ? (1 - snapshot.resid_mean_sq / matrixSnapshots.families[familyKey].var_y)
      : null
    const r2Text = r2 === null ? "n/a" : formatCompact(r2)
    meta.textContent = "max |B B*^T| = " + formatCompact(vmax) + " | R^2 = " + r2Text + " | loss = " + formatSci(snapshot.loss_total)
  }
}

function renderSpectrumPlot() {
  const svg = document.getElementById("spectrum-plot")
  const legend = document.getElementById("spectrum-legend")
  if (!svg) return
  svg.innerHTML = ""
  if (legend) legend.innerHTML = ""

  const familyKey = document.getElementById("family-select").value
  const snapshot = getSnapshot(familyKey, currentSnapshotKey)
  if (!snapshot) {
    const text = svgElement("text", { x: 190, y: 140, "text-anchor": "middle", fill: "currentColor", "font-size": 14 })
    text.textContent = matrixSnapshots ? "No snapshot for this family." : "Loading matrix snapshots..."
    svg.appendChild(text)
    return
  }
  const h2 = (snapshot.h2_eigs || []).map((value) => Math.max(value, 0))
  const kg = (snapshot.kappa_g_tilde_eigs || []).map((value) => Math.max(value, 0))

  const padTop = 28
  const padBottom = 36
  const padLeft = 60
  const padRight = 16
  const width = 380
  const height = 280
  const plotWidth = width - padLeft - padRight
  const plotHeight = height - padTop - padBottom
  const k = Math.max(h2.length, kg.length)

  svg.appendChild(svgElement("rect", {
    x: padLeft, y: padTop, width: plotWidth, height: plotHeight, rx: 12, class: "plot-frame"
  }))

  const all = h2.concat(kg).filter((value) => value > 0)
  const fallback = [1e-6, 1]
  const yScale = plotScale(all.length ? all : fallback, 0, plotHeight, true)
  const yPos = (value) => padTop + plotHeight - yScale(Math.max(value, 1e-30))

  tickValues(all.length ? all : fallback, 5, true).forEach((tick) => {
    const y = yPos(tick)
    svg.appendChild(svgElement("line", {
      x1: padLeft, x2: padLeft + plotWidth, y1: y, y2: y, class: "grid-line", "stroke-width": 1
    }))
    const label = svgElement("text", {
      x: padLeft - 8, y: y + 4, "text-anchor": "end", fill: "currentColor", "font-size": 10, class: "axis-label"
    })
    label.textContent = formatCompact(tick)
    svg.appendChild(label)
  })

  const groupWidth = plotWidth / Math.max(k, 1)
  const barWidth = Math.max(2, groupWidth * 0.4)

  for (let i = 0; i < k; i += 1) {
    const xCenter = padLeft + (i + 0.5) * groupWidth
    if (h2[i] !== undefined && h2[i] > 0) {
      const y = yPos(h2[i])
      svg.appendChild(svgElement("rect", {
        x: xCenter - barWidth - 1, y, width: barWidth, height: padTop + plotHeight - y,
        fill: resolveColor("--series-a"), rx: 2
      }))
    }
    if (kg[i] !== undefined && kg[i] > 0) {
      const y = yPos(kg[i])
      svg.appendChild(svgElement("rect", {
        x: xCenter + 1, y, width: barWidth, height: padTop + plotHeight - y,
        fill: resolveColor("--series-c"), rx: 2
      }))
    }
    if (i % 2 === 0) {
      const tick = svgElement("text", {
        x: xCenter, y: padTop + plotHeight + 14, "text-anchor": "middle", fill: "currentColor", "font-size": 10
      })
      tick.textContent = String(i + 1)
      svg.appendChild(tick)
    }
  }

  const xLabel = svgElement("text", {
    x: padLeft + plotWidth / 2, y: height - 6, "text-anchor": "middle", fill: "currentColor", "font-size": 11
  })
  xLabel.textContent = "eigenvalue rank (descending)"
  svg.appendChild(xLabel)

  const yLabel = svgElement("text", {
    x: 14, y: padTop + plotHeight / 2, "text-anchor": "middle", fill: "currentColor", "font-size": 11,
    transform: "rotate(-90 14 " + (padTop + plotHeight / 2) + ")"
  })
  yLabel.textContent = "eigenvalue (log)"
  svg.appendChild(yLabel)

  if (legend) {
    const a = document.createElement("span")
    const aSwatch = document.createElement("span")
    aSwatch.className = "legend-swatch"
    aSwatch.style.background = "var(--series-a)"
    a.appendChild(aSwatch)
    a.append("eig(H squared)")
    legend.appendChild(a)
    const c = document.createElement("span")
    const cSwatch = document.createElement("span")
    cSwatch.className = "legend-swatch"
    cSwatch.style.background = "var(--series-c)"
    c.appendChild(cSwatch)
    c.append("eig(kappa_eff times G_tilde)")
    legend.appendChild(c)
  }
}

function classifyRegime(state) {
  // High beta error => beta-bridge collapse risk.
  // High gain-weighted pair contribution => pair-compression failure risk.
  const betaBad = Math.abs(state.betaError) > 0.25
  const pairBad = state.pairDefect * state.pairGain * state.pairGain > 1
  if (betaBad && pairBad) return { name: "Compound failure", color: "#b75c34", text: "Both bridges are broken: weighted law and raw law are both unreliable." }
  if (betaBad) return { name: "Beta-bridge collapse", color: "#f2a16f", text: "β_fit is mis-tracking residual energy. Raw AGOP conversion is fragile." }
  if (pairBad) return { name: "Pair-compression failure", color: "#5a96c8", text: "A bad direction overlaps with high stationarity gain. Pushed pair error blows up." }
  return { name: "Benign quadrant", color: "#6f7f12", text: "Both bridges hold. Weighted law is well-conditioned in this regime." }
}

function renderRegimeLocator(state) {
  const svg = document.getElementById("regime-plot")
  const status = document.getElementById("regime-status")
  svg.innerHTML = ""
  const width = 360
  const height = 320
  const left = 48
  const right = 16
  const top = 28
  const bottom = 48
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom

  // Quadrant rectangles. x-axis = beta error (signed), y-axis = pushed pair contribution.
  // Left half = benign beta, right half = beta collapse.
  // Bottom = harmless pair, top = pair failure.
  const xMid = left + plotWidth / 2
  const yMid = top + plotHeight / 2
  const benign = "rgba(111, 127, 18, 0.18)"
  const orange = "rgba(247, 161, 111, 0.22)"
  const blue = "rgba(90, 150, 200, 0.22)"
  const rust = "rgba(183, 92, 52, 0.22)"

  svg.appendChild(svgElement("rect", { x: left, y: yMid, width: plotWidth / 2, height: plotHeight / 2, fill: benign }))
  svg.appendChild(svgElement("rect", { x: xMid, y: yMid, width: plotWidth / 2, height: plotHeight / 2, fill: orange }))
  svg.appendChild(svgElement("rect", { x: left, y: top, width: plotWidth / 2, height: plotHeight / 2, fill: blue }))
  svg.appendChild(svgElement("rect", { x: xMid, y: top, width: plotWidth / 2, height: plotHeight / 2, fill: rust }))
  svg.appendChild(svgElement("rect", { x: left, y: top, width: plotWidth, height: plotHeight, rx: 12, class: "plot-frame", fill: "none" }))

  // Quadrant labels
  const labels = [
    { x: left + plotWidth * 0.25, y: yMid + plotHeight * 0.32, text: "Benign" },
    { x: xMid + plotWidth * 0.25, y: yMid + plotHeight * 0.32, text: "Beta collapse" },
    { x: left + plotWidth * 0.25, y: top + plotHeight * 0.18, text: "Pair failure" },
    { x: xMid + plotWidth * 0.25, y: top + plotHeight * 0.18, text: "Compound" }
  ]
  labels.forEach((l) => {
    const text = svgElement("text", {
      x: l.x, y: l.y, "text-anchor": "middle", class: "regime-quadrant-label"
    })
    text.textContent = l.text
    svg.appendChild(text)
  })

  // Crosshair axes
  svg.appendChild(svgElement("line", { x1: xMid, x2: xMid, y1: top, y2: top + plotHeight, class: "grid-line" }))
  svg.appendChild(svgElement("line", { x1: left, x2: left + plotWidth, y1: yMid, y2: yMid, class: "grid-line" }))

  // Map state -> coordinates.
  // x: clamp signed betaError to [-1, 1] then map to [left, left+plotWidth]
  const xVal = Math.max(-1, Math.min(1, state.betaError))
  const xPos = left + (xVal + 1) / 2 * plotWidth
  // y: log10 of pushedPairProxy normalized to a range. larger = more bad = closer to top.
  const logProxy = Math.log10(Math.max(state.pushedPairProxy, 1e-12))
  const yNorm = Math.max(0, Math.min(1, (logProxy + 6) / 8)) // -6..2 visible range
  const yPos = top + (1 - yNorm) * plotHeight

  // Reference dots for the three real families (benign quadrant, near origin)
  const familyKeys = ["isotropic", "anisotropic", "low_rank_signal"]
  familyKeys.forEach((key, index) => {
    const fam = families[key]
    const fx = left + plotWidth * (0.18 + index * 0.06)
    const logFam = Math.log10(Math.max(fam.pushedPair, 1e-12))
    const fyNorm = Math.max(0, Math.min(1, (logFam + 6) / 8))
    const fy = top + (1 - fyNorm) * plotHeight
    svg.appendChild(svgElement("circle", {
      cx: fx, cy: fy, r: 5, fill: "rgba(111, 127, 18, 0.55)", stroke: resolveColor("--olive"), "stroke-width": 1.5
    }))
    const tag = svgElement("text", {
      x: fx, y: fy - 8, "text-anchor": "middle", "font-size": 9, fill: resolveColor("--muted")
    })
    tag.textContent = key.replace("_signal", "").replace("_", " ").slice(0, 10)
    svg.appendChild(tag)
  })

  // Live position
  const regime = classifyRegime(state)
  svg.appendChild(svgElement("circle", {
    cx: xPos, cy: yPos, r: 9, fill: regime.color, stroke: resolveColor("--panel"), "stroke-width": 2.5
  }))

  // Axis labels
  const xLabel = svgElement("text", {
    x: left + plotWidth / 2, y: height - 16, "text-anchor": "middle", fill: "currentColor", "font-size": 11
  })
  xLabel.textContent = "← benign beta · beta error · beta collapse →"
  svg.appendChild(xLabel)

  const yLabel = svgElement("text", {
    x: 14, y: top + plotHeight / 2, "text-anchor": "middle", fill: "currentColor", "font-size": 11,
    transform: "rotate(-90 14 " + (top + plotHeight / 2) + ")"
  })
  yLabel.textContent = "pushed pair (log)"
  svg.appendChild(yLabel)

  if (status) {
    status.textContent = regime.name + ". " + regime.text
    status.style.borderLeft = "4px solid " + regime.color
    status.style.paddingLeft = "12px"
  }
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
  highlightActiveControls(sweep)
  renderLegend(sweep)
  renderPlot(rows, sweep)
  renderTable(rows, sweep)
  renderInsights(state)
  renderRegimeLocator(state)
  const family = document.getElementById("family-select").value
  const metricKey = document.getElementById("trajectory-metric-select").value
  const familyData = activeFamilyData()
  configureScrubber(familyData)
  renderTrajectoryPlot(family, metricKey)
  renderTrajectoryLegend(metricKey)
  renderTrajectoryReadout(familyData, metricKey)
  renderAlignmentHeatmap()
  renderSpectrumPlot()
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

function openFiguresDialog() {
  const dialog = document.getElementById("figures-dialog")
  if (!dialog) return
  if (typeof dialog.showModal === "function") {
    if (!dialog.open) dialog.showModal()
  } else {
    dialog.setAttribute("open", "")
  }
  renderReportedFigure()
}

function renderSelectors() {
  const familySelect = document.getElementById("family-select")
  const sweepSelect = document.getElementById("sweep-select")
  const figureSelect = document.getElementById("figure-select")
  const trajectoryMetricSelect = document.getElementById("trajectory-metric-select")

  Object.entries(families).forEach(([key, family]) => {
    familySelect.appendChild(option(key, family.label))
  })

  Object.entries(sweeps).forEach(([key, sweep]) => {
    sweepSelect.appendChild(option(key, sweep.label))
  })

  Object.entries(trajectoryMetrics).forEach(([key, metric]) => {
    trajectoryMetricSelect.appendChild(option(key, metric.label))
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
  document.getElementById("trajectory-metric-select").value = "r2"
  scrubberPinnedToBest = true
  currentSnapshotKey = "late"
  document.querySelectorAll(".snapshot-button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.snapshot === "late")
  })
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
  document.getElementById("trajectory-metric-select").addEventListener("change", renderLive)

  const scrubber = document.getElementById("trajectory-scrubber")
  if (scrubber) {
    scrubber.addEventListener("input", () => {
      scrubberStepIndex = Number(scrubber.value)
      scrubberPinnedToBest = false
      const familyData = activeFamilyData()
      const metricKey = document.getElementById("trajectory-metric-select").value
      renderTrajectoryPlot(document.getElementById("family-select").value, metricKey)
      renderTrajectoryReadout(familyData, metricKey)
    })
  }

  document.querySelectorAll(".snapshot-button").forEach((button) => {
    button.addEventListener("click", () => {
      const snapshotKey = button.dataset.snapshot
      if (!snapshotKey) return
      currentSnapshotKey = snapshotKey
      document.querySelectorAll(".snapshot-button").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.snapshot === snapshotKey)
      })
      renderAlignmentHeatmap()
      renderSpectrumPlot()
    })
  })

  const openFigures = document.getElementById("open-figures")
  if (openFigures) {
    openFigures.addEventListener("click", openFiguresDialog)
  }
  const dialog = document.getElementById("figures-dialog")
  if (dialog) {
    const closeBtn = dialog.querySelector(".dialog-close")
    if (closeBtn) {
      closeBtn.addEventListener("click", () => {
        if (typeof dialog.close === "function") dialog.close()
        else dialog.removeAttribute("open")
      })
    }
  }
}

function loadDatasets() {
  const trajectoryPromise = fetch("data/trajectories.json", { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error("trajectories.json HTTP " + response.status)
      return response.json()
    })
    .then((payload) => { trajectoryData = payload })
    .catch((err) => {
      console.warn("trajectories.json unavailable:", err)
      trajectoryData = { families: {} }
    })

  const matrixPromise = fetch("data/matrix_snapshots.json", { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error("matrix_snapshots.json HTTP " + response.status)
      return response.json()
    })
    .then((payload) => { matrixSnapshots = payload })
    .catch((err) => {
      console.warn("matrix_snapshots.json unavailable:", err)
      matrixSnapshots = { families: {} }
    })

  return Promise.all([trajectoryPromise, matrixPromise]).then(() => renderLive())
}

function main() {
  renderSelectors()
  renderSummaryTable()
  bindEvents()
  setTheme(localStorage.getItem("weightedAgopTheme") || "dark")
  reset()
  loadDatasets()
}

main()

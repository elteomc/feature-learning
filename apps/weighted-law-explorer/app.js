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

const metrics = {
  weightedResidual: {
    label: "Weighted residual",
    description: "Mean operator norm residual at best-stationarity checkpoints",
    format: "sci"
  },
  boundRatio: {
    label: "Theorem bound ratio",
    description: "Observed weighted residual divided by the deterministic bound",
    format: "fixed"
  },
  betaOverMean: {
    label: "Beta over residual energy",
    description: "Mean beta_fit divided by mean residual squared across all checkpoints",
    format: "fixed"
  },
  pushedPair: {
    label: "Pushed pair error",
    description: "Pair error after pushing through the learned feature map",
    format: "sci"
  },
  symmetricRelative: {
    label: "Symmetric relative error",
    description: "Secondary relative weighted-law diagnostic",
    format: "fixed"
  },
  aPair: {
    label: "A_pair",
    description: "Conservative support-normalized worst-direction diagnostic",
    format: "fixed"
  }
}

const claims = [
  {
    title: "Late training is weighted",
    text: "The stable convergence-side relation is H^2 approx kappa_eff G_tilde."
  },
  {
    title: "Raw AGOP is intermediate",
    text: "The raw bridge becomes ill-conditioned as beta_fit collapses near interpolation."
  },
  {
    title: "Beta tracks residual energy",
    text: "Across checkpoints, beta_fit / mean(r^2) stays close to one in all three families."
  },
  {
    title: "Pushed pair error matters",
    text: "A_pair can be large, but the pair error after pushing through B is small."
  },
  {
    title: "Bad directions need gain",
    text: "The new pair diagnostics check whether large pair defects occur in directions with small stationarity-induced gain."
  }
]

const figureBase = "../../paper/figures/"

const figures = [
  {
    label: "Weighted residual by family",
    file: "weighted_residual_by_family.png",
    caption: "Weighted-law residual at best-stationarity checkpoints"
  },
  {
    label: "Theorem bound ratio by family",
    file: "theorem_bound_ratio_by_family.png",
    caption: "Observed weighted residual divided by the deterministic bound"
  },
  {
    label: "Beta over residual energy",
    file: "beta_over_residual_energy_by_family.png",
    caption: "Beta_fit tracks mean residual squared across all checkpoints"
  },
  {
    label: "Pushed pair error",
    file: "pushed_pair_error_by_family.png",
    caption: "Pair error after pushing through the learned feature map"
  },
  {
    label: "Symmetric relative error",
    file: "symmetric_relative_error_by_family.png",
    caption: "Secondary relative weighted-law diagnostic"
  },
  {
    label: "Bridge operator error",
    file: "bridge_operator_error_by_family.png",
    caption: "Relative Frobenius error of M_tilde minus beta_fit M"
  },
  {
    label: "Pair gain contribution",
    file: "pair_gain_weighted_contribution_by_family.png",
    caption: "Largest gain-weighted pair contribution at best-stationarity checkpoints"
  },
  {
    label: "Pair gain correlation",
    file: "pair_gain_correlation_by_family.png",
    caption: "Absolute correlation between pair defect size and stationarity gain"
  },
  {
    label: "Isotropic trajectory",
    file: "isotropic_two_regime_trajectory.png",
    caption: "Two-regime trajectory for isotropic seed 0"
  },
  {
    label: "Anisotropic trajectory",
    file: "anisotropic_two_regime_trajectory.png",
    caption: "Two-regime trajectory for anisotropic seed 0"
  },
  {
    label: "Low-rank trajectory",
    file: "low_rank_signal_two_regime_trajectory.png",
    caption: "Two-regime trajectory for low-rank signal seed 0"
  },
  {
    label: "Isotropic phase diagram",
    file: "isotropic_phase_diagram.png",
    caption: "Raw quality compared with weighted quality along training"
  },
  {
    label: "Isotropic pair gain diagnostics",
    file: "isotropic_pair_gain_diagnostics.png",
    caption: "Worst pair defect compared with stationarity-gain-weighted contributions"
  }
]

function formatValue(value, kind) {
  if (kind === "sci") {
    return value.toExponential(3)
  }
  return value.toFixed(3)
}

function option(value, label) {
  const item = document.createElement("option")
  item.value = value
  item.textContent = label
  return item
}

function renderSelectors() {
  const familySelect = document.getElementById("family-select")
  const metricSelect = document.getElementById("metric-select")
  const figureSelect = document.getElementById("figure-select")

  Object.entries(families).forEach(([key, family]) => {
    familySelect.appendChild(option(key, family.label))
  })

  Object.entries(metrics).forEach(([key, metric]) => {
    metricSelect.appendChild(option(key, metric.label))
  })

  figures.forEach((figure, index) => {
    figureSelect.appendChild(option(String(index), figure.label))
  })
}

function renderMetricCard() {
  const familyKey = document.getElementById("family-select").value
  const metricKey = document.getElementById("metric-select").value
  const family = families[familyKey]
  const metric = metrics[metricKey]
  const value = family[metricKey]
  const card = document.getElementById("metric-card")

  card.innerHTML = ""
  const title = document.createElement("h3")
  title.textContent = family.label + ": " + metric.label

  const body = document.createElement("p")
  body.textContent = metric.description

  const valueNode = document.createElement("p")
  valueNode.textContent = formatValue(value, metric.format)

  const note = document.createElement("p")
  note.textContent = family.summary

  card.appendChild(title)
  card.appendChild(valueNode)
  card.appendChild(body)
  card.appendChild(note)

  if (metricKey === "betaOverMean") {
    const range = document.createElement("p")
    range.textContent = "Range across checkpoints: " +
      family.betaOverMeanMin.toFixed(3) +
      " to " +
      family.betaOverMeanMax.toFixed(3)
    card.appendChild(range)
  }

  renderMetricBars()
}

function renderClaims() {
  const target = document.getElementById("claim-cards")
  target.innerHTML = ""
  claims.forEach((claim) => {
    const template = document.getElementById("claim-template")
    const details = template.content.firstElementChild.cloneNode(true)
    const summary = details.querySelector("summary")
    const text = details.querySelector("p")
    summary.textContent = claim.title
    text.textContent = claim.text
    details.appendChild(summary)
    details.appendChild(text)
    target.appendChild(details)
  })
}

function metricScale(metricKey, value) {
  const values = Object.values(families).map((family) => family[metricKey])
  const maxValue = Math.max(...values)
  if (!Number.isFinite(maxValue) || maxValue <= 0) {
    return 0
  }
  return Math.max(4, 100 * value / maxValue)
}

function renderMetricBars() {
  const metricKey = document.getElementById("metric-select").value
  const metric = metrics[metricKey]
  const target = document.getElementById("metric-bars")
  target.innerHTML = ""

  Object.values(families).forEach((family) => {
    const template = document.getElementById("bar-template")
    const bar = template.content.firstElementChild.cloneNode(true)
    const value = family[metricKey]
    bar.querySelector(".bar-label").textContent = family.label
    bar.querySelector(".bar-value").textContent = formatValue(value, metric.format)
    bar.querySelector(".bar-fill").style.width = metricScale(metricKey, value).toFixed(1) + "%"
    target.appendChild(bar)
  })
}

function renderTable() {
  const tbody = document.getElementById("summary-table")
  tbody.innerHTML = ""
  Object.entries(families).forEach(([key, family]) => {
    const row = document.createElement("tr")
    const cells = [
      family.label,
      family.weightedResidual.toExponential(2),
      family.boundRatio.toFixed(3),
      family.betaOverMean.toFixed(3),
      family.pushedPair.toExponential(2)
    ]

    cells.forEach((text, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td")
      cell.textContent = text
      row.appendChild(cell)
    })
    tbody.appendChild(row)
  })
}

function renderFigure() {
  const index = Number(document.getElementById("figure-select").value)
  const figure = figures[index]
  const image = document.getElementById("figure-image")
  const caption = document.getElementById("figure-caption")
  image.src = figureBase + figure.file
  caption.textContent = figure.caption
}

function pow10FromSlider(id) {
  return Math.pow(10, Number(document.getElementById(id).value))
}

function setOutput(id, value) {
  document.getElementById(id).textContent = value.toExponential(2)
}

function setFixedOutput(id, value) {
  document.getElementById(id).textContent = value.toFixed(2)
}

function simCard(label, value, text) {
  const template = document.getElementById("simulator-card-template")
  const card = template.content.firstElementChild.cloneNode(true)
  card.querySelector(".sim-label").textContent = label
  card.querySelector(".sim-value").textContent = value
  card.querySelector(".sim-text").textContent = text
  return card
}

function renderSimulator() {
  const residualEnergy = pow10FromSlider("residual-slider")
  const stationarityDefect = pow10FromSlider("stationarity-slider")
  const pushedPairError = pow10FromSlider("pair-slider")
  const betaFit = residualEnergy
  const weightedError = stationarityDefect + pushedPairError
  const rawAmplification = 1 / Math.max(betaFit, 1e-12)
  const rawSensitivity = weightedError * rawAmplification
  const regime = betaFit < 1e-3 ? "late weighted regime" : "intermediate raw-conditioned regime"

  setOutput("residual-output", residualEnergy)
  setOutput("stationarity-output", stationarityDefect)
  setOutput("pair-output", pushedPairError)

  const target = document.getElementById("simulator-cards")
  target.innerHTML = ""
  target.appendChild(simCard(
    "beta bridge",
    betaFit.toExponential(2),
    "In the experiments, beta_fit tracks residual energy."
  ))
  target.appendChild(simCard(
    "weighted residual proxy",
    weightedError.toExponential(2),
    "Stationarity defect plus pushed pair error controls the weighted law."
  ))
  target.appendChild(simCard(
    "raw sensitivity proxy",
    rawSensitivity.toExponential(2),
    "The raw conversion amplifies weighted-law error by about one over beta."
  ))
  target.appendChild(simCard(
    "regime",
    regime,
    "The transition is qualitative, not a theorem threshold."
  ))
}

function renderBetaDiagnostic() {
  const leverageCv = pow10FromSlider("leverage-cv-slider")
  const residSqCv = pow10FromSlider("resid-cv-slider")
  const corr = Number(document.getElementById("corr-slider").value)
  const deterministicBound = leverageCv * residSqCv
  const signedRelativeError = corr * deterministicBound

  setOutput("leverage-cv-output", leverageCv)
  setOutput("resid-cv-output", residSqCv)
  setFixedOutput("corr-output", corr)

  const target = document.getElementById("beta-diagnostic-cards")
  target.innerHTML = ""
  target.appendChild(simCard(
    "deterministic bound",
    deterministicBound.toExponential(2),
    "Cauchy-Schwarz gives absolute beta relative error at most CV(s) times CV(r^2)."
  ))
  target.appendChild(simCard(
    "signed error identity",
    signedRelativeError.toExponential(2),
    "The exact centered identity is correlation times the two CV factors."
  ))
  target.appendChild(simCard(
    "diagnostic question",
    Math.abs(signedRelativeError).toExponential(2),
    "Small values mean beta_fit should track mean residual squared."
  ))
  target.appendChild(simCard(
    "next experiment",
    "logged per checkpoint",
    "The branch records leverage CV, residual CV, correlation, beta error, and the CV bound."
  ))
}

function renderPairDiagnostic() {
  const worstDefect = pow10FromSlider("pair-defect-slider")
  const badDirectionGain = pow10FromSlider("pair-gain-slider")
  const gainCorr = Number(document.getElementById("gain-corr-slider").value)
  const gainWeightedContribution = worstDefect * badDirectionGain * badDirectionGain
  const correlatedRisk = Math.abs(gainCorr) * worstDefect

  setOutput("pair-defect-output", worstDefect)
  setOutput("pair-gain-output", badDirectionGain)
  setFixedOutput("gain-corr-output", gainCorr)

  const target = document.getElementById("pair-diagnostic-cards")
  target.innerHTML = ""
  target.appendChild(simCard(
    "support defect",
    worstDefect.toExponential(2),
    "This is the conservative worst-direction pair diagnostic."
  ))
  target.appendChild(simCard(
    "gain on bad direction",
    badDirectionGain.toExponential(2),
    "Small stationarity-induced gain can make a large defect mostly invisible."
  ))
  target.appendChild(simCard(
    "gain-weighted contribution",
    gainWeightedContribution.toExponential(2),
    "The experiment logs this kind of contribution by pair eigendirection."
  ))
  target.appendChild(simCard(
    "diagnostic risk",
    correlatedRisk.toExponential(2),
    "High positive coupling between defect size and gain is the failure warning."
  ))
}

function reset() {
  document.getElementById("family-select").value = "isotropic"
  document.getElementById("metric-select").value = "weightedResidual"
  document.getElementById("figure-select").value = "0"
  document.getElementById("residual-slider").value = "-3"
  document.getElementById("stationarity-slider").value = "-4"
  document.getElementById("pair-slider").value = "-5"
  document.getElementById("leverage-cv-slider").value = "-1"
  document.getElementById("resid-cv-slider").value = "0"
  document.getElementById("corr-slider").value = "0.05"
  document.getElementById("pair-defect-slider").value = "2"
  document.getElementById("pair-gain-slider").value = "-2"
  document.getElementById("gain-corr-slider").value = "0.05"
  renderMetricCard()
  renderFigure()
  renderSimulator()
  renderBetaDiagnostic()
  renderPairDiagnostic()
}

function bindEvents() {
  document.getElementById("family-select").addEventListener("change", renderMetricCard)
  document.getElementById("metric-select").addEventListener("change", renderMetricCard)
  document.getElementById("figure-select").addEventListener("change", renderFigure)
  document.getElementById("reset-button").addEventListener("click", reset)
  document.getElementById("residual-slider").addEventListener("input", renderSimulator)
  document.getElementById("stationarity-slider").addEventListener("input", renderSimulator)
  document.getElementById("pair-slider").addEventListener("input", renderSimulator)
  document.getElementById("leverage-cv-slider").addEventListener("input", renderBetaDiagnostic)
  document.getElementById("resid-cv-slider").addEventListener("input", renderBetaDiagnostic)
  document.getElementById("corr-slider").addEventListener("input", renderBetaDiagnostic)
  document.getElementById("pair-defect-slider").addEventListener("input", renderPairDiagnostic)
  document.getElementById("pair-gain-slider").addEventListener("input", renderPairDiagnostic)
  document.getElementById("gain-corr-slider").addEventListener("input", renderPairDiagnostic)
}

function main() {
  renderSelectors()
  renderClaims()
  renderTable()
  bindEvents()
  reset()
}

main()

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
}

function renderClaims() {
  const target = document.getElementById("claim-cards")
  target.innerHTML = ""
  claims.forEach((claim) => {
    const details = document.createElement("details")
    details.open = true
    const summary = document.createElement("summary")
    summary.textContent = claim.title
    const text = document.createElement("p")
    text.textContent = claim.text
    details.appendChild(summary)
    details.appendChild(text)
    target.appendChild(details)
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

function reset() {
  document.getElementById("family-select").value = "isotropic"
  document.getElementById("metric-select").value = "weightedResidual"
  document.getElementById("figure-select").value = "0"
  renderMetricCard()
  renderFigure()
}

function bindEvents() {
  document.getElementById("family-select").addEventListener("change", renderMetricCard)
  document.getElementById("metric-select").addEventListener("change", renderMetricCard)
  document.getElementById("figure-select").addEventListener("change", renderFigure)
  document.getElementById("reset-button").addEventListener("click", reset)
}

function main() {
  renderSelectors()
  renderClaims()
  renderTable()
  bindEvents()
  reset()
}

main()

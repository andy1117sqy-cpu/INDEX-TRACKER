# Meridian Index Lab

A dark institutional quantitative terminal for tracking, curve-fitting, and forecasting five flagship equity indexes across the US and East Asia:

| Region | Index | Symbol | Exchange | Currency |
| :--- | :--- | :--- | :--- | :--- |
| United States | NASDAQ Composite | `^IXIC` | NASDAQ | USD |
| China | Shanghai Composite | `000001.SS` | SSE | CNY |
| South Korea | KOSPI Composite | `^KS11` | KRX | KRW |
| Japan | Nikkei 225 | `^N225` | TSE | JPY |
| Hong Kong | Hang Seng Index | `^HSI` | HKEX | HKD |

## Key Capabilities

- **Live Market Feed with Curated Fallback**: Fetches 1-year daily OHLCV series through an allowlist-validated Express proxy (`/api/markets`) with a 15-minute in-memory cache, automatically falling back to a deterministic multi-regime historical archive if upstream quotes are unreachable.
- **Multi-Model Curve-Fitting Arena**: Benchmarks six preset models independently per market on a strict 25-session out-of-sample holdout window:
  - **Neural Networks ("Neuro Web" Variants)**:
    - *Neural Net · Compact* — 1 hidden layer (`[16]`, `tanh`)
    - *Neural Net · Deep 2L* — 2 hidden layers (`[24→12]`, `tanh`)
    - *Neural Net · Wide Residual* — 2 hidden layers (`[32→16]`, `silu`, linear residual skip)
  - **Statistical & Econometric Challengers**:
    - *Holt-Winters Damped* — grid-tuned double exponential smoothing
    - *Ridge Cubic + AR(3)* — regularized polynomial log-price trend with autoregressive residual correction
    - *Autoregressive AR(8)* — ridge-regularized multi-lag return model
- **Dynamic Champion Selection & Full Result Retention**: Automatically crowns each market's best model by **RMSE**, **MAPE**, or **Directional Accuracy**. All runner-up models, error metrics, and timestamped benchmark snapshots are retained so you can overlay competing curves on the forecast chart and track champion transitions as market regimes shift.
- **Neural Net Lab**: Configure custom feedforward MLPs with 1–3 hidden layers, 4–64 units per layer, `tanh` / `relu` / `silu` activations, optional linear residual skip connections, lookback window (`5–30` sessions), learning rate, L2 weight decay, and epochs. Inspect the live network topology diagram and epoch-by-epoch train/validation MSE convergence chart, then save custom models directly into the market's persistent leaderboard.
- **Interactive Forecast Visualization & Export**: Switch historical zoom ranges (`3M`, `6M`, `1Y`) and forecast horizons (`7D`, `14D`, `30D`), inspect 80% and 95% empirical confidence cones with crosshair tooltips, compare all five markets on a base-100 normalized chart in **Global Overview**, and export benchmark tables as **CSV** or **JSON**.

## Running Locally on Your PC

### Prerequisites

- **Node.js** `20.x` or newer (includes `npm`)

### 1. Download the Project

Download the project files to your computer using the **Download / Export** option in your workspace toolbar (or clone from your repository if exported to GitHub), then unzip and open a terminal in the project folder.

### 2. Install Dependencies

```bash
npm install
```

### 3. Start the Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser. No API keys are required to run market tracking, neural network training, or forecasting locally.

### 4. Production Build & Verification Commands

```bash
# Type-check with TypeScript
npm run lint

# Build the production frontend bundle into dist/
npm run build

# Run deterministic unit tests for math, feature pipeline, neural net, and benchmarks
npx tsx --test src/engine/engine.test.ts

# Serve the production build (after npm run build)
NODE_ENV=production npx tsx server.ts
```

## Project Structure

```text
├── server.ts                         # Express server, allowlist-validated Yahoo Finance proxy, 15m cache, Vite middleware
├── index.html                        # Entry HTML & typography links (Plus Jakarta Sans, Public Sans, JetBrains Mono)
├── src/
│   ├── App.tsx                       # Main workspace shell, benchmark orchestration, and state lifecycle
│   ├── types/market.ts               # Core TypeScript interfaces for markets, models, metrics, and persistence
│   ├── data/marketsCatalog.ts        # 5-market catalog and deterministic multi-regime fallback generator
│   ├── engine/
│   │   ├── math.ts                   # Ridge linear solver, accuracy metrics, trading calendar, confidence cones
│   │   ├── features.ts               # Log-return, momentum, volatility, EMA features & leakage-free scaler
│   │   ├── neuralNet.ts              # Configurable 1–3 layer MLP with Adam, L2 decay, skip path, early stopping
│   │   ├── classicalModels.ts        # Holt-Winters, Ridge Cubic + AR(3), and Autoregressive AR(8)
│   │   ├── benchmark.ts              # Multi-model suite runner, champion ranking, and CSV export builder
│   │   └── engine.test.ts            # Node unit test suite (node:test)
│   ├── store/persistence.ts          # localStorage persistence for custom models, overlays, and run history
│   ├── utils/format.ts               # Tabular price, percentage, date, and file download helpers
│   └── components/
│       ├── Sidebar.tsx               # Workspace navigation and 5-market sparkline rail
│       ├── ForecastChart.tsx         # Interactive SVG forecast canvas with holdout split & confidence cones
│       ├── ForecastTerminalView.tsx  # Primary KPI strip, chart, horizon controls, and overlay selector
│       ├── ModelArenaView.tsx        # Sortable model leaderboard, metric selector, export, and run history
│       ├── NeuralNetLabView.tsx      # Layer/unit builder, topology SVG, loss curve, and custom archive
│       └── GlobalOverviewView.tsx    # Normalized 5-market comparative chart and regional champion matrix
```

## Modeling Methodology

1. **Stationary Feature Space**: All neural networks and autoregressive models train on normalized daily log-returns alongside 5-day momentum, 15-day momentum, 10-day rolling volatility, and fast-vs-slow EMA spread—keeping inputs scale-invariant across indexes ranging from ~2,600 (KOSPI) to ~50,000+ (Nikkei 225).
2. **Leakage-Free Validation**: Feature and target z-score scalers are fit strictly on the training split. Neural networks reserve the final 16% of the training window as an internal validation set and restore the lowest-validation-loss weight checkpoint before scoring on the untouched 25-session holdout test set.
3. **Empirical Confidence Bands**: 80% and 95% forward prediction cones are derived from out-of-sample holdout log-return residual volatility scaled across the forecast horizon.

## Server API Endpoints

| Endpoint | Description |
| :--- | :--- |
| `GET /api/health` | Returns server health status and the five supported market IDs. |
| `GET /api/markets` | Returns daily price series for all five markets (`?refresh=1` bypasses the 15-minute cache). |
| `GET /api/markets/:marketId` | Returns a single market series for `nasdaq`, `sse`, `kospi`, `nikkei`, or `hsi`. |

## Disclaimer

All forecasts, model rankings, and confidence bands are generated for quantitative research and educational analysis only and do not constitute financial or investment advice.

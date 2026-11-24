import { Play, Plus, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { formatArchitectureLabel } from '../engine/neuralNet';
import type {
  ActivationName,
  EpochLossPoint,
  MarketSeries,
  ModelRunResult,
  NeuralConfig,
} from '../types/market';
import { formatIndexPrice, formatSignedPercent } from '../utils/format';

interface NeuralNetLabViewProps {
  series: MarketSeries;
  champion: ModelRunResult;
  customModels: readonly ModelRunResult[];
  isTraining: boolean;
  onTrainCustomModel: (config: NeuralConfig) => void;
  onDeleteCustomModel: (modelId: string) => void;
  onInspectInTerminal: (modelId: string) => void;
}

const QUICK_TEMPLATES: ReadonlyArray<{ label: string; config: Omit<NeuralConfig, 'id'> }> = [
  {
    label: 'Shallow Wide [32]',
    config: {
      name: 'MLP Shallow Wide',
      hiddenLayers: [32],
      activation: 'tanh',
      lookback: 12,
      learningRate: 0.005,
      epochs: 70,
      l2: 0.015,
      residualSkip: false,
    },
  },
  {
    label: 'Deep 3-Layer [28→16→8]',
    config: {
      name: 'MLP Deep 3-Layer',
      hiddenLayers: [28, 16, 8],
      activation: 'silu',
      lookback: 14,
      learningRate: 0.005,
      epochs: 80,
      l2: 0.025,
      residualSkip: true,
    },
  },
  {
    label: 'Short-Momentum [16→8]',
    config: {
      name: 'MLP Fast Momentum',
      hiddenLayers: [16, 8],
      activation: 'relu',
      lookback: 7,
      learningRate: 0.008,
      epochs: 65,
      l2: 0.015,
      residualSkip: true,
    },
  },
];

function NetworkTopologyDiagram({
  inputCount,
  hiddenLayers,
  residualSkip,
}: {
  inputCount: number;
  hiddenLayers: readonly number[];
  residualSkip: boolean;
}) {
  const columns = [
    { label: `Input (${inputCount})`, count: Math.min(8, inputCount), raw: inputCount, color: '#94A3B8' },
    ...hiddenLayers.map((units, idx) => ({
      label: `Hidden ${idx + 1} (${units})`,
      count: Math.min(8, Math.max(3, Math.round(units / 6))),
      raw: units,
      color: '#38BDF8',
    })),
    { label: 'Output (1)', count: 1, raw: 1, color: '#10B981' },
  ];

  const width = 560;
  const height = 210;
  const padX = 58;
  const padTop = 26;
  const usableH = height - 62;

  const colX = (cIdx: number) =>
    padX + (cIdx / Math.max(1, columns.length - 1)) * (width - padX * 2);

  const nodeY = (nIdx: number, total: number) => {
    if (total === 1) return padTop + usableH / 2;
    return padTop + (nIdx / (total - 1)) * usableH;
  };

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-auto block"
      role="img"
      aria-label={`Neural network topology with hidden layers ${hiddenLayers.join(', ')}`}
    >
      {/* Dense connections */}
      {columns.slice(0, -1).map((col, cIdx) => {
        const nextCol = columns[cIdx + 1];
        const x1 = colX(cIdx);
        const x2 = colX(cIdx + 1);
        return Array.from({ length: col.count }).map((_, i) =>
          Array.from({ length: nextCol.count }).map((__, j) => (
            <line
              key={`${cIdx}-${i}-${j}`}
              x1={x1}
              y1={nodeY(i, col.count)}
              x2={x2}
              y2={nodeY(j, nextCol.count)}
              stroke="rgba(148, 163, 184, 0.14)"
              strokeWidth="1"
            />
          )),
        );
      })}

      {/* Residual skip arc */}
      {residualSkip && (
        <g>
          <path
            d={`M ${colX(0)} ${padTop - 4} Q ${width / 2} 2 ${colX(columns.length - 1)} ${nodeY(0, 1) - 10}`}
            fill="none"
            stroke="#F59E0B"
            strokeWidth="1.5"
            strokeDasharray="4 3"
          />
          <text
            x={width / 2}
            y={14}
            textAnchor="middle"
            fill="#F59E0B"
            fontSize="10"
            fontFamily="JetBrains Mono, monospace"
          >
            Linear Residual Skip
          </text>
        </g>
      )}

      {/* Nodes */}
      {columns.map((col, cIdx) => {
        const x = colX(cIdx);
        return (
          <g key={col.label}>
            {Array.from({ length: col.count }).map((_, nIdx) => (
              <circle
                key={nIdx}
                cx={x}
                cy={nodeY(nIdx, col.count)}
                r={cIdx === columns.length - 1 ? 6 : 4.5}
                fill="#0E1422"
                stroke={col.color}
                strokeWidth="1.6"
              />
            ))}
            <text
              x={x}
              y={height - 10}
              textAnchor="middle"
              fill="#CBD5E1"
              fontSize="10.5"
              fontFamily="JetBrains Mono, monospace"
            >
              {col.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function LossCurveChart({ history }: { history: readonly EpochLossPoint[] }) {
  if (history.length < 2) return null;
  const width = 560;
  const height = 190;
  const pad = { top: 18, right: 16, bottom: 28, left: 46 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const allVals = history.flatMap((p) => [p.trainLoss, p.valLoss]);
  const minL = Math.min(...allVals);
  const maxL = Math.max(...allVals);
  const span = maxL - minL || 0.1;

  const xAt = (idx: number) => pad.left + (idx / (history.length - 1)) * innerW;
  const yAt = (val: number) => pad.top + ((maxL - val) / span) * innerH;

  const trainPath = history
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'}${xAt(idx).toFixed(1)},${yAt(p.trainLoss).toFixed(1)}`)
    .join(' ');
  const valPath = history
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'}${xAt(idx).toFixed(1)},${yAt(p.valLoss).toFixed(1)}`)
    .join(' ');

  const ticks = [minL, (minL + maxL) / 2, maxL];

  return (
    <div data-testid="loss-curve-chart">
      <div className="mb-2 flex items-center gap-5 text-xs text-slate-300">
        <span className="inline-flex items-center gap-2">
          <span className="h-0.5 w-4 bg-sky-400 inline-block" />
          Train MSE ({history[history.length - 1].trainLoss.toFixed(3)})
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-0.5 w-4 bg-amber-400 inline-block" />
          Validation MSE ({history[history.length - 1].valLoss.toFixed(3)})
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto block">
        {ticks.map((t, idx) => {
          const y = yAt(t);
          return (
            <g key={idx}>
              <line
                x1={pad.left}
                x2={pad.left + innerW}
                y1={y}
                y2={y}
                stroke="rgba(148, 163, 184, 0.1)"
              />
              <text
                x={pad.left - 8}
                y={y + 3.5}
                textAnchor="end"
                fill="#94A3B8"
                fontSize="10"
                fontFamily="JetBrains Mono, monospace"
              >
                {t.toFixed(2)}
              </text>
            </g>
          );
        })}

        <path d={trainPath} fill="none" stroke="#38BDF8" strokeWidth="1.9" />
        <path d={valPath} fill="none" stroke="#F59E0B" strokeWidth="1.7" strokeDasharray="4 2.5" />

        <text
          x={pad.left}
          y={height - 8}
          fill="#94A3B8"
          fontSize="10"
          fontFamily="JetBrains Mono, monospace"
        >
          Epoch 1
        </text>
        <text
          x={pad.left + innerW}
          y={height - 8}
          textAnchor="end"
          fill="#94A3B8"
          fontSize="10"
          fontFamily="JetBrains Mono, monospace"
        >
          Epoch {history[history.length - 1].epoch}
        </text>
      </svg>
    </div>
  );
}

export function NeuralNetLabView({
  series,
  champion,
  customModels,
  isTraining,
  onTrainCustomModel,
  onDeleteCustomModel,
  onInspectInTerminal,
}: NeuralNetLabViewProps) {
  const [name, setName] = useState('Custom MLP 2L');
  const [hiddenLayers, setHiddenLayers] = useState<number[]>([24, 12]);
  const [activation, setActivation] = useState<ActivationName>('tanh');
  const [lookback, setLookback] = useState(12);
  const [learningRate, setLearningRate] = useState(0.005);
  const [epochs, setEpochs] = useState(75);
  const [l2, setL2] = useState(0.015);
  const [residualSkip, setResidualSkip] = useState(true);
  const [formError, setFormError] = useState<string | null>(null);

  const latestClose = series.bars[series.bars.length - 1]?.close ?? 0;
  const activeDiagnosticModel =
    customModels[0] ?? (champion.family === 'neural' ? champion : undefined);

  const totalParams = (() => {
    const dims = [lookback + 4, ...hiddenLayers, 1];
    let count = 0;
    for (let i = 0; i < dims.length - 1; i++) {
      count += dims[i] * dims[i + 1] + dims[i + 1];
    }
    if (residualSkip) count += lookback + 4;
    return count;
  })();

  const handleAddLayer = () => {
    if (hiddenLayers.length >= 3) return;
    const prev = hiddenLayers[hiddenLayers.length - 1] ?? 16;
    setHiddenLayers([...hiddenLayers, Math.max(4, Math.round(prev / 2))]);
  };

  const handleRemoveLayer = (idx: number) => {
    if (hiddenLayers.length <= 1) return;
    setHiddenLayers(hiddenLayers.filter((_, i) => i !== idx));
  };

  const handleUpdateUnits = (idx: number, nextUnits: number) => {
    const safe = Math.max(4, Math.min(64, Math.round(nextUnits || 4)));
    setHiddenLayers(hiddenLayers.map((u, i) => (i === idx ? safe : u)));
  };

  const handleApplyTemplate = (tpl: Omit<NeuralConfig, 'id'>) => {
    setName(tpl.name);
    setHiddenLayers([...tpl.hiddenLayers]);
    setActivation(tpl.activation);
    setLookback(tpl.lookback);
    setLearningRate(tpl.learningRate);
    setEpochs(tpl.epochs);
    setL2(tpl.l2);
    setResidualSkip(tpl.residualSkip);
    setFormError(null);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) {
      setFormError('Please provide an architecture name before training.');
      return;
    }
    setFormError(null);
    const config: NeuralConfig = {
      id: `custom-${Date.now()}`,
      name: trimmed,
      hiddenLayers,
      activation,
      lookback,
      learningRate,
      epochs,
      l2,
      residualSkip,
      seed: Math.floor(Date.now() % 100000),
      isCustom: true,
    };
    onTrainCustomModel(config);
  };

  const previewConfig: NeuralConfig = {
    id: 'preview',
    name,
    hiddenLayers,
    activation,
    lookback,
    learningRate,
    epochs,
    l2,
    residualSkip,
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-xs text-slate-400">
            {series.meta.name} ({series.meta.symbol}) · Current Champion: {champion.modelName}
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-50">
            Neural Network Architecture Lab
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {QUICK_TEMPLATES.map((tpl) => (
            <button
              key={tpl.label}
              type="button"
              onClick={() => handleApplyTemplate(tpl.config)}
              className="rounded-md border border-slate-800 bg-[#111827] px-2.5 py-1.5 text-xs text-slate-300 hover:border-slate-700 hover:text-slate-100 transition-colors whitespace-nowrap shrink-0"
            >
              {tpl.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
        {/* Left: Architecture & Hyperparameter Builder */}
        <form
          onSubmit={handleSubmit}
          className="xl:col-span-5 rounded-lg border border-slate-800/80 bg-[#111827] p-5 space-y-5"
        >
          <div>
            <h2 className="text-base font-semibold text-slate-100">Layer &amp; Unit Configuration</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Trained on stationary log-returns, momentum, and volatility features with Adam optimization.
            </p>
          </div>

          <div>
            <label htmlFor="nn-name" className="block text-xs font-medium text-slate-300 mb-1.5">
              Architecture Name
            </label>
            <input
              id="nn-name"
              type="text"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                if (formError) setFormError(null);
              }}
              maxLength={48}
              placeholder="e.g. Deep Quant 3L"
              className="w-full rounded-md border border-slate-700 bg-[#0B101B] px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-sky-400 focus:outline-none"
            />
            {formError && <p className="mt-1.5 text-xs text-rose-400">{formError}</p>}
          </div>

          {/* Dynamic Hidden Layer Stack (1-3 layers) */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-300">
                Hidden Layers ({hiddenLayers.length} / 3)
              </span>
              <button
                type="button"
                onClick={handleAddLayer}
                disabled={hiddenLayers.length >= 3}
                className="inline-flex items-center gap-1 rounded-sm border border-slate-700 bg-[#0B101B] px-2.5 py-1 text-xs font-medium text-sky-300 hover:border-sky-500/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
              >
                <Plus className="h-3.5 w-3.5" />
                Add Layer
              </button>
            </div>

            <div className="space-y-2.5">
              {hiddenLayers.map((units, idx) => (
                <div
                  key={idx}
                  className="rounded-md border border-slate-800 bg-[#0E1422] px-3 py-2.5"
                >
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <label
                      htmlFor={`layer-units-${idx}`}
                      className="text-xs font-medium text-slate-300"
                    >
                      Hidden Layer {idx + 1}
                    </label>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs tabular-nums text-sky-300">
                        {units} units
                      </span>
                      {hiddenLayers.length > 1 && (
                        <button
                          type="button"
                          onClick={() => handleRemoveLayer(idx)}
                          aria-label={`Remove Hidden Layer ${idx + 1}`}
                          className="text-slate-400 hover:text-rose-400 p-0.5"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                  <input
                    id={`layer-units-${idx}`}
                    type="range"
                    min={4}
                    max={64}
                    step={2}
                    value={units}
                    onChange={(e) => handleUpdateUnits(idx, Number(e.target.value))}
                    className="w-full accent-sky-400 cursor-pointer"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Activation & Skip Connection */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <span className="block text-xs font-medium text-slate-300 mb-1.5">Activation</span>
              <div
                className="grid grid-cols-3 rounded-md border border-slate-800 bg-[#0B101B] p-0.5"
                role="group"
                aria-label="Activation function"
              >
                {(['tanh', 'relu', 'silu'] as const).map((fn) => (
                  <button
                    key={fn}
                    type="button"
                    onClick={() => setActivation(fn)}
                    className={`py-1.5 text-xs font-mono rounded-sm transition-colors ${
                      activation === fn
                        ? 'bg-sky-500/20 text-sky-300'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {fn}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-col justify-between">
              <span className="block text-xs font-medium text-slate-300 mb-1.5">
                Residual Skip Path
              </span>
              <label className="inline-flex h-[34px] items-center gap-2 rounded-md border border-slate-800 bg-[#0B101B] px-3 text-xs text-slate-200 cursor-pointer whitespace-nowrap">
                <input
                  type="checkbox"
                  checked={residualSkip}
                  onChange={(e) => setResidualSkip(e.target.checked)}
                  className="accent-sky-400 h-3.5 w-3.5 shrink-0"
                />
                <span>Enable linear skip</span>
              </label>
            </div>
          </div>

          {/* Numeric Training Hyperparameters */}
          <div className="grid grid-cols-2 gap-3.5">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <label htmlFor="nn-lookback" className="text-slate-300">
                  Lookback Window
                </label>
                <span className="font-mono tabular-nums text-slate-200">{lookback}d</span>
              </div>
              <input
                id="nn-lookback"
                type="range"
                min={5}
                max={30}
                value={lookback}
                onChange={(e) => setLookback(Number(e.target.value))}
                className="w-full accent-sky-400 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <label htmlFor="nn-epochs" className="text-slate-300">
                  Training Epochs
                </label>
                <span className="font-mono tabular-nums text-slate-200">{epochs}</span>
              </div>
              <input
                id="nn-epochs"
                type="range"
                min={30}
                max={180}
                step={10}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full accent-sky-400 cursor-pointer"
              />
            </div>

            <div>
              <label htmlFor="nn-lr" className="block text-xs text-slate-300 mb-1">
                Learning Rate
              </label>
              <select
                id="nn-lr"
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-[#0B101B] px-2.5 py-1.5 font-mono text-xs text-slate-100 focus:border-sky-400 focus:outline-none"
              >
                <option value={0.002}>0.0020 (Conservative)</option>
                <option value={0.005}>0.0050 (Balanced)</option>
                <option value={0.008}>0.0080 (Standard)</option>
                <option value={0.015}>0.0150 (Fast)</option>
              </select>
            </div>

            <div>
              <label htmlFor="nn-l2" className="block text-xs text-slate-300 mb-1">
                L2 Weight Decay
              </label>
              <select
                id="nn-l2"
                value={l2}
                onChange={(e) => setL2(Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-[#0B101B] px-2.5 py-1.5 font-mono text-xs text-slate-100 focus:border-sky-400 focus:outline-none"
              >
                <option value={0.008}>0.0080 (Moderate)</option>
                <option value={0.015}>0.0150 (Standard)</option>
                <option value={0.025}>0.0250 (Strong)</option>
                <option value={0.04}>0.0400 (High Decay)</option>
              </select>
            </div>
          </div>

          <div className="flex items-center justify-between border-t border-slate-800/80 pt-4">
            <div className="text-xs text-slate-400 font-mono tabular-nums">
              {totalParams.toLocaleString()} trainable weights
            </div>
            <button
              type="submit"
              disabled={isTraining}
              className="inline-flex items-center gap-2 rounded-md bg-sky-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-sky-400 disabled:opacity-60 transition-colors whitespace-nowrap shrink-0"
            >
              <Play className="h-3.5 w-3.5 fill-current" />
              {isTraining ? 'Training…' : 'Train & Evaluate'}
            </button>
          </div>
        </form>

        {/* Right: Live Topology Diagram & Convergence Diagnostics */}
        <div className="xl:col-span-7 space-y-5">
          <section className="rounded-lg border border-slate-800/80 bg-[#111827] p-5">
            <div className="flex flex-wrap items-baseline justify-between gap-2 mb-3">
              <h2 className="text-base font-semibold text-slate-100">Live Network Topology</h2>
              <span className="font-mono text-xs text-slate-400">
                {formatArchitectureLabel(previewConfig)}
              </span>
            </div>
            <NetworkTopologyDiagram
              inputCount={lookback + 4}
              hiddenLayers={hiddenLayers}
              residualSkip={residualSkip}
            />
          </section>

          <section className="rounded-lg border border-slate-800/80 bg-[#111827] p-5">
            <div className="flex flex-wrap items-baseline justify-between gap-2 mb-3">
              <div>
                <h2 className="text-base font-semibold text-slate-100">
                  Training &amp; Validation Loss Convergence
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  {activeDiagnosticModel
                    ? `Showing ${activeDiagnosticModel.modelName} (${activeDiagnosticModel.architectureLabel}) · trained in ${activeDiagnosticModel.trainingMs}ms`
                    : 'Train a custom architecture to inspect its convergence curve.'}
                </p>
              </div>
              {activeDiagnosticModel && (
                <span className="font-mono text-xs tabular-nums text-slate-300">
                  RMSE {formatIndexPrice(activeDiagnosticModel.metrics.rmse)} · MAPE{' '}
                  {activeDiagnosticModel.metrics.mape.toFixed(2)}%
                </span>
              )}
            </div>

            {activeDiagnosticModel?.lossHistory ? (
              <LossCurveChart history={activeDiagnosticModel.lossHistory} />
            ) : (
              <p className="py-8 text-xs text-slate-400">
                No neural loss history available for the current selection.
              </p>
            )}
          </section>

          {/* Saved Custom Neural Architectures for this Market */}
          <section className="rounded-lg border border-slate-800/80 bg-[#111827] p-5">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-base font-semibold text-slate-100">
                  Saved Custom Neural Nets ({series.meta.shortName})
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Every trained experiment enters the leaderboard and persists across sessions.
                </p>
              </div>
            </div>

            {customModels.length === 0 ? (
              <p className="py-4 text-xs text-slate-400">
                No custom neural architectures saved for {series.meta.shortName} yet. Adjust layers or units on the left and click Train &amp; Evaluate.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-2 pr-3 font-medium">Name</th>
                      <th className="py-2 pr-3 font-medium">Topology</th>
                      <th className="py-2 pr-3 font-medium text-right">RMSE</th>
                      <th className="py-2 pr-3 font-medium text-right">MAPE</th>
                      <th className="py-2 pr-3 font-medium text-right">30D Target</th>
                      <th className="py-2 font-medium text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {customModels.map((m) => {
                      const t30 = m.forecastValues[m.forecastValues.length - 1] ?? latestClose;
                      const pct = latestClose > 0 ? ((t30 - latestClose) / latestClose) * 100 : 0;
                      const isChamp = m.modelId === champion.modelId;

                      return (
                        <tr key={m.modelId} className="hover:bg-slate-800/35">
                          <td className="py-2.5 pr-3 font-medium text-slate-100 whitespace-nowrap">
                            {m.modelName}
                            {isChamp && (
                              <span className="ml-2 text-[11px] text-emerald-400">Champion</span>
                            )}
                          </td>
                          <td className="py-2.5 pr-3 font-mono text-slate-400 whitespace-nowrap">
                            {m.architectureLabel}
                          </td>
                          <td className="py-2.5 pr-3 font-mono tabular-nums text-right text-slate-200">
                            {formatIndexPrice(m.metrics.rmse)}
                          </td>
                          <td className="py-2.5 pr-3 font-mono tabular-nums text-right text-slate-200">
                            {m.metrics.mape.toFixed(2)}%
                          </td>
                          <td className="py-2.5 pr-3 font-mono tabular-nums text-right whitespace-nowrap">
                            <span className="text-slate-100">{formatIndexPrice(t30)}</span>{' '}
                            <span className={pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                              ({formatSignedPercent(pct, 1)})
                            </span>
                          </td>
                          <td className="py-2.5 text-right whitespace-nowrap">
                            <div className="inline-flex items-center gap-2">
                              <button
                                type="button"
                                onClick={() => onInspectInTerminal(m.modelId)}
                                className="text-sky-300 hover:text-sky-200 font-medium"
                              >
                                Plot
                              </button>
                              <button
                                type="button"
                                onClick={() => onDeleteCustomModel(m.modelId)}
                                aria-label={`Delete ${m.modelName}`}
                                className="text-slate-400 hover:text-rose-400"
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

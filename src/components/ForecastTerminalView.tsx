import { ArrowRight, Check } from 'lucide-react';
import type {
  ChartRange,
  ForecastHorizon,
  MarketSeries,
  ModelRunResult,
  RankingMetric,
} from '../types/market';
import {
  formatIndexPrice,
  formatShortDate,
  formatSignedPercent,
  formatSignedPoints,
} from '../utils/format';
import { ForecastChart } from './ForecastChart';

interface ForecastTerminalViewProps {
  series: MarketSeries;
  sortedResults: readonly ModelRunResult[];
  champion: ModelRunResult;
  pinnedOverlayIds: readonly string[];
  horizon: ForecastHorizon;
  range: ChartRange;
  rankingMetric: RankingMetric;
  onChangeHorizon: (h: ForecastHorizon) => void;
  onChangeRange: (r: ChartRange) => void;
  onToggleOverlay: (modelId: string) => void;
  onOpenLab: () => void;
  onOpenArena: () => void;
}

const HORIZONS: readonly ForecastHorizon[] = [7, 14, 30];
const RANGES: readonly ChartRange[] = ['3M', '6M', '1Y'];

export function ForecastTerminalView({
  series,
  sortedResults,
  champion,
  pinnedOverlayIds,
  horizon,
  range,
  rankingMetric,
  onChangeHorizon,
  onChangeRange,
  onToggleOverlay,
  onOpenLab,
  onOpenArena,
}: ForecastTerminalViewProps) {
  const { meta, bars } = series;
  const latestBar = bars[bars.length - 1];
  const prevBar = bars[bars.length - 2];

  const dayDeltaPts = latestBar && prevBar ? latestBar.close - prevBar.close : 0;
  const dayDeltaPct =
    latestBar && prevBar && prevBar.close > 0 ? (dayDeltaPts / prevBar.close) * 100 : 0;

  const targetIdx = Math.min(horizon, champion.forecastValues.length) - 1;
  const targetPrice = champion.forecastValues[targetIdx] ?? latestBar.close;
  const targetDate = champion.forecastDates[targetIdx] ?? '';
  const expectedReturnPct =
    latestBar.close > 0 ? ((targetPrice - latestBar.close) / latestBar.close) * 100 : 0;
  const bandLow95 = champion.band95Low[targetIdx] ?? targetPrice;
  const bandHigh95 = champion.band95High[targetIdx] ?? targetPrice;

  const challengers = sortedResults.filter(
    (r) => r.modelId !== champion.modelId && pinnedOverlayIds.includes(r.modelId),
  );
  const runnerUps = sortedResults.filter((r) => r.modelId !== champion.modelId);

  const sourceLabel =
    series.source === 'live'
      ? 'Live Market Feed'
      : series.source === 'cached'
        ? 'Cached Live Feed'
        : 'Historical Benchmark Archive';

  const metricLabel =
    rankingMetric === 'rmse'
      ? 'Lowest RMSE'
      : rankingMetric === 'mape'
        ? 'Lowest MAPE'
        : 'Highest Directional Acc.';

  return (
    <div className="space-y-6">
      {/* Market Title & Unboxed Provenance Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
            <span>{meta.region}</span>
            <span aria-hidden="true">·</span>
            <span className="font-mono text-slate-300">{meta.symbol}</span>
            <span aria-hidden="true">·</span>
            <span>{meta.exchange}</span>
            <span aria-hidden="true">·</span>
            <span>{sourceLabel}</span>
            <span aria-hidden="true">·</span>
            <span className="font-mono">Close {latestBar.date}</span>
          </div>
          <h1 className="mt-1.5 text-2xl md:text-3xl font-semibold tracking-tight text-slate-50">
            {meta.name}
          </h1>
        </div>

        {/* Chart Range & Forecast Horizon Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <div
            className="inline-flex items-center rounded-md border border-slate-800 bg-[#111827] p-0.5"
            role="group"
            aria-label="Historical zoom range"
          >
            {RANGES.map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => onChangeRange(r)}
                className={`px-3 py-1.5 text-xs font-mono font-medium rounded-sm transition-colors whitespace-nowrap shrink-0 ${
                  range === r
                    ? 'bg-slate-800 text-slate-100'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {r}
              </button>
            ))}
          </div>

          <div
            className="inline-flex items-center rounded-md border border-slate-800 bg-[#111827] p-0.5"
            role="group"
            aria-label="Forecast horizon"
          >
            {HORIZONS.map((h) => (
              <button
                key={h}
                type="button"
                onClick={() => onChangeHorizon(h)}
                className={`px-3 py-1.5 text-xs font-mono font-medium rounded-sm transition-colors whitespace-nowrap shrink-0 ${
                  horizon === h
                    ? 'bg-sky-500/20 text-sky-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {h}D Forecast
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Fallback notice (non-blocking, crisp flat border) */}
      {series.source === 'fallback' && (
        <div className="rounded-md border border-amber-500/25 bg-amber-500/5 px-4 py-2.5 text-xs text-amber-200/90">
          Live upstream quotes are currently unreachable from this network; running models on the curated historical benchmark archive.
        </div>
      )}

      {/* 4-Column Flat KPI Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="rounded-lg border border-slate-800/80 bg-[#111827] p-4">
          <p className="text-xs text-slate-400">Latest Close ({meta.currency})</p>
          <p className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-slate-50">
            {formatIndexPrice(latestBar.close)}
          </p>
          <p
            className={`mt-1 font-mono text-xs tabular-nums ${
              dayDeltaPts >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {formatSignedPoints(dayDeltaPts)} ({formatSignedPercent(dayDeltaPct)}) 1D
          </p>
        </div>

        <div className="rounded-lg border border-slate-800/80 bg-[#111827] p-4">
          <p className="text-xs text-slate-400">
            {horizon}D Projected Target ({formatShortDate(targetDate)})
          </p>
          <p className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-sky-300">
            {formatIndexPrice(targetPrice)}
          </p>
          <p
            className={`mt-1 font-mono text-xs tabular-nums ${
              expectedReturnPct >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {formatSignedPercent(expectedReturnPct)} expected ·{' '}
            {expectedReturnPct >= 0 ? 'Bullish bias' : 'Bearish bias'}
          </p>
        </div>

        <div className="rounded-lg border border-slate-800/80 bg-[#111827] p-4">
          <p className="text-xs text-slate-400">Active Champion ({metricLabel})</p>
          <p className="mt-1.5 text-lg font-semibold text-slate-100 truncate">
            {champion.modelName}
          </p>
          <p className="mt-1 font-mono text-xs tabular-nums text-slate-400 truncate">
            MAPE {champion.metrics.mape.toFixed(2)}% · Dir {champion.metrics.directionalAccuracy.toFixed(1)}%
          </p>
        </div>

        <div className="rounded-lg border border-slate-800/80 bg-[#111827] p-4">
          <p className="text-xs text-slate-400">{horizon}D 95% Confidence Range</p>
          <p className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-slate-100">
            {formatIndexPrice(bandLow95)} – {formatIndexPrice(bandHigh95)}
          </p>
          <p className="mt-1 font-mono text-xs tabular-nums text-slate-400">
            RMSE {formatIndexPrice(champion.metrics.rmse)} · R² {champion.metrics.r2.toFixed(3)}
          </p>
        </div>
      </div>

      {/* Primary Forecast Chart Panel */}
      <section className="rounded-lg border border-slate-800/80 bg-[#111827] p-5">
        <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-slate-800/70 pb-3.5">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Curve Fit &amp; Forward Forecast Trajectory
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Champion: {champion.modelName} ({champion.architectureLabel}) · Evaluated on 25-session out-of-sample holdout
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <button
              type="button"
              onClick={onOpenLab}
              className="inline-flex items-center gap-1.5 text-sky-300 hover:text-sky-200 font-medium whitespace-nowrap shrink-0"
            >
              Tune Neural Layers
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        <ForecastChart
          bars={bars}
          champion={champion}
          challengers={challengers}
          horizon={horizon}
          range={range}
          currency={meta.currency}
        />
      </section>

      {/* Retained Challenger Curve Toggles */}
      <section className="rounded-lg border border-slate-800/80 bg-[#111827] p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between mb-4">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Retained Model Comparison
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Toggle runner-up models to overlay their backtest and forecast curves directly on the chart above.
            </p>
          </div>
          <button
            type="button"
            onClick={onOpenArena}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-sky-300 hover:text-sky-200 whitespace-nowrap shrink-0"
          >
            Open Full Arena ({sortedResults.length} models)
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 pr-4 font-medium">Overlay</th>
                <th className="py-2.5 pr-4 font-medium">Model</th>
                <th className="py-2.5 pr-4 font-medium">Architecture</th>
                <th className="py-2.5 pr-4 font-medium text-right">RMSE</th>
                <th className="py-2.5 pr-4 font-medium text-right">MAPE</th>
                <th className="py-2.5 pr-4 font-medium text-right">Dir. Acc.</th>
                <th className="py-2.5 font-medium text-right">{horizon}D Target</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {runnerUps.map((m) => {
                const isPinned = pinnedOverlayIds.includes(m.modelId);
                const tVal = m.forecastValues[Math.min(horizon, m.forecastValues.length) - 1] ?? 0;
                const tPct =
                  latestBar.close > 0 ? ((tVal - latestBar.close) / latestBar.close) * 100 : 0;

                return (
                  <tr key={m.modelId} className="hover:bg-slate-800/35 transition-colors">
                    <td className="py-2.5 pr-4">
                      <button
                        type="button"
                        onClick={() => onToggleOverlay(m.modelId)}
                        aria-pressed={isPinned}
                        aria-label={`Overlay ${m.modelName} on chart`}
                        className={`inline-flex items-center gap-1.5 rounded-sm border px-2.5 py-1 text-xs font-medium transition-colors whitespace-nowrap ${
                          isPinned
                            ? 'border-sky-500/50 bg-sky-500/15 text-sky-300'
                            : 'border-slate-700 bg-[#0E1422] text-slate-300 hover:border-slate-600'
                        }`}
                      >
                        {isPinned && <Check className="h-3 w-3" />}
                        {isPinned ? 'Shown' : 'Compare'}
                      </button>
                    </td>
                    <td className="py-2.5 pr-4 font-medium text-slate-100 whitespace-nowrap">
                      {m.modelName}
                      {m.isCustom && (
                        <span className="ml-2 text-[11px] font-normal text-sky-400">Custom</span>
                      )}
                    </td>
                    <td className="py-2.5 pr-4 font-mono text-slate-400 whitespace-nowrap">
                      {m.architectureLabel}
                    </td>
                    <td className="py-2.5 pr-4 font-mono tabular-nums text-right text-slate-200">
                      {formatIndexPrice(m.metrics.rmse)}
                    </td>
                    <td className="py-2.5 pr-4 font-mono tabular-nums text-right text-slate-200">
                      {m.metrics.mape.toFixed(2)}%
                    </td>
                    <td className="py-2.5 pr-4 font-mono tabular-nums text-right text-slate-200">
                      {m.metrics.directionalAccuracy.toFixed(1)}%
                    </td>
                    <td className="py-2.5 font-mono tabular-nums text-right whitespace-nowrap">
                      <span className="text-slate-100">{formatIndexPrice(tVal)}</span>{' '}
                      <span className={tPct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                        ({formatSignedPercent(tPct, 1)})
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

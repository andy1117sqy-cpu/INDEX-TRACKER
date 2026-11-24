import { ArrowUpRight } from 'lucide-react';
import { MARKET_CATALOG } from '../data/marketsCatalog';
import type {
  ForecastHorizon,
  MarketId,
  MarketSeries,
  ModelRunResult,
  RankingMetric,
} from '../types/market';
import { formatIndexPrice, formatSignedPercent } from '../utils/format';

interface GlobalOverviewViewProps {
  markets: Partial<Record<MarketId, MarketSeries>>;
  champions: Partial<Record<MarketId, ModelRunResult | null>>;
  allResultsByMarket: Partial<Record<MarketId, ModelRunResult[]>>;
  horizon: ForecastHorizon;
  rankingMetric: RankingMetric;
  onSelectMarket: (id: MarketId) => void;
}

export function GlobalOverviewView({
  markets,
  champions,
  allResultsByMarket,
  horizon,
  rankingMetric,
  onSelectMarket,
}: GlobalOverviewViewProps) {
  const historyWindow = 60;
  const totalSteps = historyWindow + horizon;

  // Build normalized indexed-to-100 trajectories across all 5 markets
  const normalizedTracks = MARKET_CATALOG.map((meta) => {
    const series = markets[meta.id];
    const champ = champions[meta.id];
    if (!series || !champ || series.bars.length < 10) return null;

    const histSlice = series.bars.slice(-historyWindow);
    const baseClose = histSlice[0].close || 1;
    const histNorm = histSlice.map((b) => (b.close / baseClose) * 100);
    const forecastNorm = champ.forecastValues
      .slice(0, horizon)
      .map((v) => (v / baseClose) * 100);

    return {
      meta,
      histNorm,
      forecastNorm,
      endValue: forecastNorm[forecastNorm.length - 1] ?? histNorm[histNorm.length - 1],
    };
  }).filter((t): t is NonNullable<typeof t> => t !== null);

  const allY = normalizedTracks.flatMap((t) => [...t.histNorm, ...t.forecastNorm]);
  const yMin = allY.length ? Math.min(...allY) - 2 : 90;
  const yMax = allY.length ? Math.max(...allY) + 2 : 110;

  const width = 920;
  const height = 310;
  const pad = { top: 20, right: 64, bottom: 28, left: 16 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const xAt = (idx: number) => pad.left + (idx / Math.max(1, totalSteps - 1)) * innerW;
  const yAt = (val: number) => pad.top + ((yMax - val) / (yMax - yMin || 1)) * innerH;
  const splitX = xAt(historyWindow - 1);

  const yTicks = [yMin, yMin + (yMax - yMin) * 0.33, yMin + (yMax - yMin) * 0.66, yMax];

  const neuralChampionCount = MARKET_CATALOG.filter(
    (m) => champions[m.id]?.family === 'neural',
  ).length;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-xs text-slate-400">
            US · China · South Korea · Japan · Hong Kong · Ranked by{' '}
            {rankingMetric === 'rmse'
              ? 'RMSE'
              : rankingMetric === 'mape'
                ? 'MAPE'
                : 'Directional Accuracy'}
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-50">
            Five-Market Comparative Overview
          </h1>
        </div>
        <p className="font-mono text-xs tabular-nums text-slate-300">
          Neural champions: {neuralChampionCount} / 5 markets
        </p>
      </div>

      {/* Normalized Multi-Market Trajectory Chart */}
      <section className="rounded-lg border border-slate-800/80 bg-[#111827] p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Indexed Performance &amp; {horizon}D Champion Projections (Base = 100)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Solid lines show trailing 60-session normalized closes; dashed continuations show each market&apos;s active champion forecast.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-xs">
            {normalizedTracks.map((t) => (
              <span key={t.meta.id} className="inline-flex items-center gap-1.5 text-slate-300">
                <span
                  className="h-0.5 w-3.5 inline-block"
                  style={{ backgroundColor: t.meta.accentColor }}
                />
                {t.meta.shortName}
              </span>
            ))}
          </div>
        </div>

        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto block"
          role="img"
          aria-label="Normalized 5-market comparative historical and forecast trajectories"
        >
          <rect
            x={splitX}
            y={pad.top}
            width={Math.max(0, pad.left + innerW - splitX)}
            height={innerH}
            fill="rgba(56, 189, 248, 0.04)"
          />

          {yTicks.map((val, idx) => {
            const y = yAt(val);
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
                  x={pad.left + innerW + 8}
                  y={y + 4}
                  fill="#94A3B8"
                  fontSize="11"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {val.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Baseline 100 reference */}
          {100 >= yMin && 100 <= yMax && (
            <line
              x1={pad.left}
              x2={pad.left + innerW}
              y1={yAt(100)}
              y2={yAt(100)}
              stroke="rgba(148, 163, 184, 0.28)"
              strokeDasharray="2 3"
            />
          )}

          {/* Split line */}
          <line
            x1={splitX}
            x2={splitX}
            y1={pad.top}
            y2={pad.top + innerH}
            stroke="rgba(56, 189, 248, 0.45)"
            strokeDasharray="4 3"
          />
          <text
            x={splitX + 6}
            y={pad.top + 12}
            fill="#7DD3FC"
            fontSize="10"
            fontFamily="Public Sans, sans-serif"
          >
            {horizon}D Forecast Zone
          </text>

          {normalizedTracks.map((track) => {
            const offset = historyWindow - track.histNorm.length;
            const histPath = track.histNorm
              .map(
                (v, i) =>
                  `${i === 0 ? 'M' : 'L'}${xAt(offset + i).toFixed(1)},${yAt(v).toFixed(1)}`,
              )
              .join(' ');

            const anchorY = track.histNorm[track.histNorm.length - 1];
            const forecastPts = [anchorY, ...track.forecastNorm];
            const fPath = forecastPts
              .map(
                (v, i) =>
                  `${i === 0 ? 'M' : 'L'}${xAt(historyWindow - 1 + i).toFixed(1)},${yAt(v).toFixed(1)}`,
              )
              .join(' ');

            return (
              <g key={track.meta.id}>
                <path
                  d={histPath}
                  fill="none"
                  stroke={track.meta.accentColor}
                  strokeWidth="1.75"
                  strokeLinejoin="round"
                />
                <path
                  d={fPath}
                  fill="none"
                  stroke={track.meta.accentColor}
                  strokeWidth="2.1"
                  strokeDasharray="5 3"
                  strokeLinejoin="round"
                />
              </g>
            );
          })}

          <text
            x={pad.left}
            y={height - 8}
            fill="#94A3B8"
            fontSize="10.5"
            fontFamily="JetBrains Mono, monospace"
          >
            T-60 Sessions
          </text>
          <text
            x={splitX}
            y={height - 8}
            textAnchor="middle"
            fill="#94A3B8"
            fontSize="10.5"
            fontFamily="JetBrains Mono, monospace"
          >
            Latest Close
          </text>
          <text
            x={pad.left + innerW}
            y={height - 8}
            textAnchor="end"
            fill="#7DD3FC"
            fontSize="10.5"
            fontFamily="JetBrains Mono, monospace"
          >
            +{horizon}D Horizon
          </text>
        </svg>
      </section>

      {/* Regional Champion Summary Table */}
      <section className="rounded-lg border border-slate-800/80 bg-[#111827] p-5">
        <div className="mb-3.5">
          <h2 className="text-base font-semibold text-slate-100">
            Regional Champion Matrix
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Select any index row to inspect full curve fits, confidence cones, and retained challengers.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 pr-4 font-medium">Market</th>
                <th className="py-2.5 pr-4 font-medium text-right">Latest Close</th>
                <th className="py-2.5 pr-4 font-medium text-right">1D Move</th>
                <th className="py-2.5 pr-4 font-medium">Active Champion</th>
                <th className="py-2.5 pr-4 font-medium text-right">MAPE</th>
                <th className="py-2.5 pr-4 font-medium text-right">Dir. Acc.</th>
                <th className="py-2.5 pr-4 font-medium text-right">{horizon}D Target</th>
                <th className="py-2.5 font-medium text-right">Open</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {MARKET_CATALOG.map((meta) => {
                const series = markets[meta.id];
                const champ = champions[meta.id];
                const totalModels = allResultsByMarket[meta.id]?.length ?? 0;
                const bars = series?.bars ?? [];
                const latest = bars[bars.length - 1];
                const prev = bars[bars.length - 2];
                const dayPct =
                  latest && prev && prev.close > 0
                    ? ((latest.close - prev.close) / prev.close) * 100
                    : 0;
                const targetVal =
                  champ?.forecastValues[Math.min(horizon, champ.forecastValues.length) - 1] ?? 0;
                const targetPct =
                  latest && latest.close > 0 ? ((targetVal - latest.close) / latest.close) * 100 : 0;

                return (
                  <tr key={meta.id} className="hover:bg-slate-800/35 transition-colors">
                    <td className="py-3 pr-4 whitespace-nowrap">
                      <div className="font-medium text-slate-100">{meta.name}</div>
                      <div className="text-[11px] text-slate-400">
                        {meta.region} · <span className="font-mono">{meta.symbol}</span> ·{' '}
                        {totalModels} models
                      </div>
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-right text-slate-100">
                      {latest ? formatIndexPrice(latest.close) : '—'}
                    </td>
                    <td
                      className={`py-3 pr-4 font-mono tabular-nums text-right ${
                        dayPct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {latest ? formatSignedPercent(dayPct) : '—'}
                    </td>
                    <td className="py-3 pr-4 whitespace-nowrap">
                      <div className="font-medium text-slate-100">
                        {champ?.modelName ?? 'Evaluating…'}
                      </div>
                      {champ && (
                        <div className="font-mono text-[11px] text-slate-400">
                          {champ.architectureLabel}
                        </div>
                      )}
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-right text-slate-200">
                      {champ ? `${champ.metrics.mape.toFixed(2)}%` : '—'}
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-right text-slate-200">
                      {champ ? `${champ.metrics.directionalAccuracy.toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-right whitespace-nowrap">
                      {champ ? (
                        <>
                          <span className="text-slate-100">{formatIndexPrice(targetVal)}</span>{' '}
                          <span className={targetPct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                            ({formatSignedPercent(targetPct, 1)})
                          </span>
                        </>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="py-3 text-right">
                      <button
                        type="button"
                        onClick={() => onSelectMarket(meta.id)}
                        aria-label={`Inspect ${meta.name} forecast`}
                        className="inline-flex items-center gap-1 rounded-sm border border-slate-700 bg-[#0B101B] px-2.5 py-1 text-xs font-medium text-sky-300 hover:border-sky-500/50 transition-colors whitespace-nowrap"
                      >
                        Inspect
                        <ArrowUpRight className="h-3.5 w-3.5" />
                      </button>
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

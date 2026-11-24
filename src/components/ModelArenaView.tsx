import { Check, Download, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { buildBenchmarkCsv } from '../engine/benchmark';
import type {
  BenchmarkSnapshot,
  ForecastHorizon,
  MarketSeries,
  ModelRunResult,
  RankingMetric,
} from '../types/market';
import {
  formatIndexPrice,
  formatSignedPercent,
  formatTimestamp,
  triggerFileDownload,
} from '../utils/format';

interface ModelArenaViewProps {
  series: MarketSeries;
  sortedResults: readonly ModelRunResult[];
  champion: ModelRunResult;
  pinnedOverlayIds: readonly string[];
  rankingMetric: RankingMetric;
  horizon: ForecastHorizon;
  marketSnapshots: readonly BenchmarkSnapshot[];
  onChangeRankingMetric: (metric: RankingMetric) => void;
  onToggleOverlay: (modelId: string) => void;
  onDeleteCustomModel: (modelId: string) => void;
  onOpenLab: () => void;
}

const RANK_OPTIONS: ReadonlyArray<{ id: RankingMetric; label: string }> = [
  { id: 'rmse', label: 'Lowest RMSE' },
  { id: 'mape', label: 'Lowest MAPE' },
  { id: 'directionalAccuracy', label: 'Directional Acc.' },
];

export function ModelArenaView({
  series,
  sortedResults,
  champion,
  pinnedOverlayIds,
  rankingMetric,
  horizon,
  marketSnapshots,
  onChangeRankingMetric,
  onToggleOverlay,
  onDeleteCustomModel,
  onOpenLab,
}: ModelArenaViewProps) {
  const [familyFilter, setFamilyFilter] = useState<'all' | 'neural' | 'statistical'>('all');
  const latestClose = series.bars[series.bars.length - 1]?.close ?? 0;

  const filteredResults = sortedResults.filter((r) =>
    familyFilter === 'all' ? true : r.family === familyFilter,
  );

  const handleExportCsv = () => {
    const csv = buildBenchmarkCsv(series.meta.name, series.meta.symbol, sortedResults, horizon);
    triggerFileDownload(
      `${series.meta.id}-benchmark-${new Date().toISOString().slice(0, 10)}.csv`,
      csv,
      'text/csv;charset=utf-8',
    );
  };

  const handleExportJson = () => {
    const payload = {
      market: series.meta,
      source: series.source,
      rankingMetric,
      exportedAt: new Date().toISOString(),
      championId: champion.modelId,
      models: sortedResults.map((r) => ({
        modelId: r.modelId,
        modelName: r.modelName,
        family: r.family,
        architectureLabel: r.architectureLabel,
        metrics: r.metrics,
        forecastDates: r.forecastDates,
        forecastValues: r.forecastValues,
      })),
    };
    triggerFileDownload(
      `${series.meta.id}-benchmark-${new Date().toISOString().slice(0, 10)}.json`,
      JSON.stringify(payload, null, 2),
      'application/json',
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-xs text-slate-400">
            {series.meta.region} · {series.meta.symbol} · {sortedResults.length} retained models
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-50">
            Model Arena — {series.meta.name}
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={handleExportCsv}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-[#111827] px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-600 hover:bg-slate-800/70 transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="h-3.5 w-3.5" />
            Export CSV
          </button>
          <button
            type="button"
            onClick={handleExportJson}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-[#111827] px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-600 hover:bg-slate-800/70 transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="h-3.5 w-3.5" />
            Export JSON
          </button>
        </div>
      </div>

      {/* Ranking Criterion & Family Filter Bar */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between rounded-lg border border-slate-800/80 bg-[#111827] px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-medium text-slate-400">Crown Champion By</span>
          <div
            className="inline-flex items-center rounded-md border border-slate-800 bg-[#0B101B] p-0.5"
            role="group"
            aria-label="Champion ranking metric"
          >
            {RANK_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => onChangeRankingMetric(opt.id)}
                className={`px-3 py-1 text-xs font-medium rounded-sm transition-colors whitespace-nowrap shrink-0 ${
                  rankingMetric === opt.id
                    ? 'bg-sky-500/20 text-sky-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div
          className="inline-flex items-center rounded-md border border-slate-800 bg-[#0B101B] p-0.5 self-start sm:self-auto"
          role="group"
          aria-label="Filter model family"
        >
          {(['all', 'neural', 'statistical'] as const).map((fam) => (
            <button
              key={fam}
              type="button"
              onClick={() => setFamilyFilter(fam)}
              className={`px-3 py-1 text-xs font-medium rounded-sm capitalize transition-colors whitespace-nowrap shrink-0 ${
                familyFilter === fam
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {fam === 'all' ? 'All Models' : fam === 'neural' ? 'Neural Nets' : 'Statistical'}
            </button>
          ))}
        </div>
      </div>

      {/* Main Model Leaderboard Table */}
      <section className="rounded-lg border border-slate-800/80 bg-[#111827] p-5">
        <div className="mb-3.5 flex items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              All Evaluated Architectures &amp; Baselines
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Every competing model is preserved so regime shifts can dynamically promote a new champion.
            </p>
          </div>
          <button
            type="button"
            onClick={onOpenLab}
            className="text-xs font-medium text-sky-300 hover:text-sky-200 whitespace-nowrap shrink-0"
          >
            + Add Custom Neural Net
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 pr-3 font-medium">Rank</th>
                <th className="py-2.5 pr-4 font-medium">Model</th>
                <th className="py-2.5 pr-4 font-medium">Family / Topology</th>
                <th className="py-2.5 pr-4 font-medium text-right">RMSE</th>
                <th className="py-2.5 pr-4 font-medium text-right">MAE</th>
                <th className="py-2.5 pr-4 font-medium text-right">MAPE</th>
                <th className="py-2.5 pr-4 font-medium text-right">R²</th>
                <th className="py-2.5 pr-4 font-medium text-right">Dir. Acc.</th>
                <th className="py-2.5 pr-4 font-medium text-right">{horizon}D Target</th>
                <th className="py-2.5 font-medium text-right">Chart</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredResults.map((m) => {
                const overallRank = sortedResults.findIndex((r) => r.modelId === m.modelId) + 1;
                const isChamp = m.modelId === champion.modelId;
                const isPinned = pinnedOverlayIds.includes(m.modelId);
                const targetVal =
                  m.forecastValues[Math.min(horizon, m.forecastValues.length) - 1] ?? latestClose;
                const retPct =
                  latestClose > 0 ? ((targetVal - latestClose) / latestClose) * 100 : 0;

                return (
                  <tr
                    key={m.modelId}
                    className={`transition-colors ${
                      isChamp ? 'bg-sky-500/5 hover:bg-sky-500/8' : 'hover:bg-slate-800/35'
                    }`}
                  >
                    <td className="py-3 pr-3 font-mono tabular-nums text-slate-400">
                      #{overallRank}
                    </td>
                    <td className="py-3 pr-4 whitespace-nowrap">
                      <div className="font-medium text-slate-100">{m.modelName}</div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {isChamp ? (
                          <span className="text-emerald-400 font-medium">Active Champion</span>
                        ) : (
                          <span>Challenger</span>
                        )}
                        {m.isCustom && <span> · Custom Architecture</span>}
                      </div>
                    </td>
                    <td className="py-3 pr-4 whitespace-nowrap">
                      <span className="text-slate-300 capitalize">{m.family}</span>
                      <span className="mx-1.5 text-slate-600">/</span>
                      <span className="font-mono text-slate-400">{m.architectureLabel}</span>
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-right text-slate-100">
                      {formatIndexPrice(m.metrics.rmse)}
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-right text-slate-300">
                      {formatIndexPrice(m.metrics.mae)}
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-right text-slate-200">
                      {m.metrics.mape.toFixed(2)}%
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-right text-slate-300">
                      {m.metrics.r2.toFixed(3)}
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-right text-slate-200">
                      {m.metrics.directionalAccuracy.toFixed(1)}%
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-right whitespace-nowrap">
                      <span className="text-slate-100">{formatIndexPrice(targetVal)}</span>{' '}
                      <span className={retPct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                        ({formatSignedPercent(retPct, 1)})
                      </span>
                    </td>
                    <td className="py-3 text-right whitespace-nowrap">
                      <div className="inline-flex items-center gap-1.5 justify-end">
                        {isChamp ? (
                          <span className="text-[11px] font-medium text-sky-300 px-2">Primary</span>
                        ) : (
                          <button
                            type="button"
                            onClick={() => onToggleOverlay(m.modelId)}
                            aria-pressed={isPinned}
                            aria-label={`Toggle ${m.modelName} overlay`}
                            className={`inline-flex items-center gap-1 rounded-sm border px-2 py-1 text-[11px] font-medium transition-colors ${
                              isPinned
                                ? 'border-sky-500/50 bg-sky-500/15 text-sky-300'
                                : 'border-slate-700 bg-[#0B101B] text-slate-300 hover:border-slate-600'
                            }`}
                          >
                            {isPinned && <Check className="h-3 w-3" />}
                            {isPinned ? 'Overlay On' : 'Overlay'}
                          </button>
                        )}
                        {m.isCustom && (
                          <button
                            type="button"
                            onClick={() => onDeleteCustomModel(m.modelId)}
                            aria-label={`Remove custom model ${m.modelName}`}
                            className="inline-flex h-7 w-7 items-center justify-center rounded-sm border border-slate-800 text-slate-400 hover:border-rose-500/40 hover:text-rose-300 transition-colors"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Saved Benchmark Runs & Champion Transition Log */}
      <section className="rounded-lg border border-slate-800/80 bg-[#111827] p-5">
        <div className="mb-3.5">
          <h2 className="text-base font-semibold text-slate-100">
            Saved Benchmark History ({series.meta.shortName})
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Chronological log of benchmark runs recording which model led under each market regime and metric.
          </p>
        </div>

        {marketSnapshots.length === 0 ? (
          <p className="py-6 text-xs text-slate-400">
            No saved benchmark snapshots yet. Click Run Benchmark in the top bar or train a custom neural network to record history.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 pr-4 font-medium">Run Timestamp</th>
                  <th className="py-2 pr-4 font-medium">Data Feed</th>
                  <th className="py-2 pr-4 font-medium">Ranking Metric</th>
                  <th className="py-2 pr-4 font-medium">Crowned Champion</th>
                  <th className="py-2 pr-4 font-medium text-right">RMSE</th>
                  <th className="py-2 pr-4 font-medium text-right">MAPE</th>
                  <th className="py-2 font-medium text-right">30D Target</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {marketSnapshots.map((snap) => (
                  <tr key={snap.id} className="hover:bg-slate-800/30">
                    <td className="py-2.5 pr-4 font-mono tabular-nums text-slate-300 whitespace-nowrap">
                      {formatTimestamp(snap.timestamp)}
                    </td>
                    <td className="py-2.5 pr-4 text-slate-400 capitalize whitespace-nowrap">
                      {snap.dataSource} · {snap.modelsEvaluated} models
                    </td>
                    <td className="py-2.5 pr-4 text-slate-300 whitespace-nowrap">
                      {snap.rankingMetric === 'rmse'
                        ? 'RMSE'
                        : snap.rankingMetric === 'mape'
                          ? 'MAPE'
                          : 'Directional Acc.'}
                    </td>
                    <td className="py-2.5 pr-4 font-medium text-slate-100 whitespace-nowrap">
                      {snap.championModelName}
                    </td>
                    <td className="py-2.5 pr-4 font-mono tabular-nums text-right text-slate-200">
                      {formatIndexPrice(snap.championRmse)}
                    </td>
                    <td className="py-2.5 pr-4 font-mono tabular-nums text-right text-slate-200">
                      {snap.championMape.toFixed(2)}%
                    </td>
                    <td className="py-2.5 font-mono tabular-nums text-right text-sky-300">
                      {formatIndexPrice(snap.championTarget30d)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

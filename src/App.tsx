import { Menu, Play, RefreshCw } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { ForecastTerminalView } from './components/ForecastTerminalView';
import { GlobalOverviewView } from './components/GlobalOverviewView';
import { ModelArenaView } from './components/ModelArenaView';
import { NeuralNetLabView } from './components/NeuralNetLabView';
import { Sidebar } from './components/Sidebar';
import {
  MARKET_IDS,
  MARKET_META_MAP,
  buildFallbackMarketSeries,
} from './data/marketsCatalog';
import {
  runMarketBenchmarkSuite,
  selectChampionModel,
  sortModelResults,
} from './engine/benchmark';
import { runNeuralModel, sanitizeNeuralConfig } from './engine/neuralNet';
import {
  appendSnapshot,
  loadPersistedState,
  savePersistedState,
} from './store/persistence';
import type {
  BenchmarkSnapshot,
  ChartRange,
  ForecastHorizon,
  MarketId,
  MarketSeries,
  ModelRunResult,
  NeuralConfig,
  PersistedWorkspaceState,
  RankingMetric,
  WorkspaceView,
} from './types/market';

const VIEW_TITLES: Record<WorkspaceView, string> = {
  forecast: 'Forecast Terminal',
  arena: 'Model Arena',
  lab: 'Neural Net Lab',
  overview: 'Global Overview',
};

function buildInitialFallbackMarkets(): Record<MarketId, MarketSeries> {
  return {
    nasdaq: buildFallbackMarketSeries('nasdaq'),
    sse: buildFallbackMarketSeries('sse'),
    kospi: buildFallbackMarketSeries('kospi'),
    nikkei: buildFallbackMarketSeries('nikkei'),
    hsi: buildFallbackMarketSeries('hsi'),
  };
}

function makeSnapshot(
  series: MarketSeries,
  results: readonly ModelRunResult[],
  rankingMetric: RankingMetric,
): BenchmarkSnapshot | null {
  const champ = selectChampionModel(results, rankingMetric);
  if (!champ || series.bars.length === 0) return null;
  const latest = series.bars[series.bars.length - 1];
  const target30 = champ.forecastValues[champ.forecastValues.length - 1] ?? latest.close;

  return {
    id: `snap-${series.meta.id}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    timestamp: new Date().toISOString(),
    marketId: series.meta.id,
    dataSource: series.source,
    latestClose: latest.close,
    latestDate: latest.date,
    rankingMetric,
    championModelId: champ.modelId,
    championModelName: champ.modelName,
    championRmse: champ.metrics.rmse,
    championMape: champ.metrics.mape,
    championTarget30d: target30,
    modelsEvaluated: results.length,
  };
}

export default function App() {
  const [persisted, setPersisted] = useState<PersistedWorkspaceState>(() => loadPersistedState());
  const [activeMarketId, setActiveMarketId] = useState<MarketId>('nasdaq');
  const [activeView, setActiveView] = useState<WorkspaceView>('forecast');
  const [horizon, setHorizon] = useState<ForecastHorizon>(30);
  const [range, setRange] = useState<ChartRange>('6M');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const [markets, setMarkets] = useState<Record<MarketId, MarketSeries>>(() =>
    buildInitialFallbackMarkets(),
  );
  const [resultsByMarket, setResultsByMarket] = useState<
    Partial<Record<MarketId, ModelRunResult[]>>
  >({});
  const [isLoadingFeed, setIsLoadingFeed] = useState(true);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [isTrainingCustom, setIsTrainingCustom] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  useEffect(() => {
    savePersistedState(persisted);
  }, [persisted]);

  const evaluateAllMarkets = useCallback(
    (
      nextMarkets: Record<MarketId, MarketSeries>,
      customMap: Record<MarketId, NeuralConfig[]>,
      rankingMetric: RankingMetric,
      recordSnapshotFor?: MarketId | 'all',
    ) => {
      const nextResults: Partial<Record<MarketId, ModelRunResult[]>> = {};
      const newSnapshots: BenchmarkSnapshot[] = [];

      for (const id of MARKET_IDS) {
        const series = nextMarkets[id];
        const evaluated = runMarketBenchmarkSuite(series.bars, customMap[id] ?? [], 25, 30);
        nextResults[id] = evaluated;

        if (recordSnapshotFor === 'all' || recordSnapshotFor === id) {
          const snap = makeSnapshot(series, evaluated, rankingMetric);
          if (snap) newSnapshots.push(snap);
        }
      }

      setResultsByMarket(nextResults);

      if (newSnapshots.length > 0) {
        setPersisted((prev) => {
          let nextHistory = prev.snapshotHistory;
          for (const snap of newSnapshots) {
            nextHistory = appendSnapshot(nextHistory, snap);
          }
          return { ...prev, snapshotHistory: nextHistory };
        });
      }
    },
    [],
  );

  // Initial fetch of live market series (falling back seamlessly to curated dataset)
  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      setIsLoadingFeed(true);
      let loadedMarkets = buildInitialFallbackMarkets();

      try {
        const resp = await fetch('/api/markets');
        if (resp.ok) {
          const payload = (await resp.json()) as {
            markets?: Partial<Record<MarketId, MarketSeries>>;
          };
          if (payload.markets) {
            for (const id of MARKET_IDS) {
              const s = payload.markets[id];
              if (s && Array.isArray(s.bars) && s.bars.length >= 60) {
                loadedMarkets[id] = s;
              }
            }
          }
        }
      } catch {
        // Fallback already initialized cleanly
      }

      if (cancelled) return;
      setMarkets(loadedMarkets);
      evaluateAllMarkets(
        loadedMarkets,
        persisted.customConfigsByMarket,
        persisted.rankingMetric,
        persisted.snapshotHistory.length === 0 ? 'nasdaq' : undefined,
      );
      setIsLoadingFeed(false);
    }

    bootstrap();
    return () => {
      cancelled = true;
    };
    // Run once on initial mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const showToast = useCallback((msg: string) => {
    setStatusMessage(msg);
    window.setTimeout(() => {
      setStatusMessage((curr) => (curr === msg ? null : curr));
    }, 3500);
  }, []);

  const handleRefreshFeed = useCallback(async () => {
    setIsLoadingFeed(true);
    try {
      const resp = await fetch('/api/markets?refresh=1');
      if (resp.ok) {
        const payload = (await resp.json()) as {
          markets?: Partial<Record<MarketId, MarketSeries>>;
        };
        if (payload.markets) {
          const updated = { ...markets };
          for (const id of MARKET_IDS) {
            const s = payload.markets[id];
            if (s && Array.isArray(s.bars) && s.bars.length >= 60) {
              updated[id] = s;
            }
          }
          setMarkets(updated);
          evaluateAllMarkets(
            updated,
            persisted.customConfigsByMarket,
            persisted.rankingMetric,
            activeMarketId,
          );
          showToast('Market index curves refreshed and re-evaluated.');
        }
      }
    } catch {
      showToast('Live feed unreachable; using retained benchmark archive.');
    } finally {
      setIsLoadingFeed(false);
    }
  }, [activeMarketId, evaluateAllMarkets, markets, persisted, showToast]);

  const handleRunBenchmark = useCallback(() => {
    setIsBenchmarking(true);
    window.setTimeout(() => {
      const targetSeries = markets[activeMarketId];
      const evaluated = runMarketBenchmarkSuite(
        targetSeries.bars,
        persisted.customConfigsByMarket[activeMarketId] ?? [],
        25,
        30,
      );
      setResultsByMarket((prev) => ({ ...prev, [activeMarketId]: evaluated }));
      const snap = makeSnapshot(targetSeries, evaluated, persisted.rankingMetric);
      if (snap) {
        setPersisted((prev) => ({
          ...prev,
          snapshotHistory: appendSnapshot(prev.snapshotHistory, snap),
        }));
      }
      setIsBenchmarking(false);
      showToast(
        `Benchmark completed for ${targetSeries.meta.shortName} (${evaluated.length} models scored).`,
      );
    }, 40);
  }, [activeMarketId, markets, persisted, showToast]);

  const handleTrainCustomModel = useCallback(
    (rawConfig: NeuralConfig) => {
      setIsTrainingCustom(true);
      const cleanConfig = sanitizeNeuralConfig({ ...rawConfig, isCustom: true });
      const targetSeries = markets[activeMarketId];

      window.setTimeout(() => {
        const customResult = runNeuralModel(targetSeries.bars, cleanConfig, 25, 30);

        setResultsByMarket((prev) => {
          const existing = prev[activeMarketId] ?? [];
          const next = [customResult, ...existing.filter((r) => r.modelId !== cleanConfig.id)];
          const snap = makeSnapshot(targetSeries, next, persisted.rankingMetric);
          if (snap) {
            setPersisted((p) => ({
              ...p,
              customConfigsByMarket: {
                ...p.customConfigsByMarket,
                [activeMarketId]: [
                  cleanConfig,
                  ...(p.customConfigsByMarket[activeMarketId] ?? []).filter(
                    (c) => c.id !== cleanConfig.id,
                  ),
                ],
              },
              pinnedOverlaysByMarket: {
                ...p.pinnedOverlaysByMarket,
                [activeMarketId]: Array.from(
                  new Set([cleanConfig.id, ...(p.pinnedOverlaysByMarket[activeMarketId] ?? [])]),
                ).slice(0, 4),
              },
              snapshotHistory: appendSnapshot(p.snapshotHistory, snap),
            }));
          }
          return { ...prev, [activeMarketId]: next };
        });

        setIsTrainingCustom(false);
        showToast(
          `Trained ${cleanConfig.name} on ${targetSeries.meta.shortName} (MAPE ${customResult.metrics.mape.toFixed(2)}%).`,
        );
      }, 40);
    },
    [activeMarketId, markets, persisted.rankingMetric, showToast],
  );

  const handleDeleteCustomModel = useCallback(
    (modelId: string) => {
      setPersisted((prev) => ({
        ...prev,
        customConfigsByMarket: {
          ...prev.customConfigsByMarket,
          [activeMarketId]: (prev.customConfigsByMarket[activeMarketId] ?? []).filter(
            (c) => c.id !== modelId,
          ),
        },
        pinnedOverlaysByMarket: {
          ...prev.pinnedOverlaysByMarket,
          [activeMarketId]: (prev.pinnedOverlaysByMarket[activeMarketId] ?? []).filter(
            (id) => id !== modelId,
          ),
        },
      }));
      setResultsByMarket((prev) => ({
        ...prev,
        [activeMarketId]: (prev[activeMarketId] ?? []).filter((r) => r.modelId !== modelId),
      }));
      showToast('Removed custom architecture from leaderboard.');
    },
    [activeMarketId, showToast],
  );

  const handleToggleOverlay = useCallback(
    (modelId: string) => {
      setPersisted((prev) => {
        const current = prev.pinnedOverlaysByMarket[activeMarketId] ?? [];
        const exists = current.includes(modelId);
        const next = exists
          ? current.filter((id) => id !== modelId)
          : [...current, modelId].slice(-4);
        return {
          ...prev,
          pinnedOverlaysByMarket: {
            ...prev.pinnedOverlaysByMarket,
            [activeMarketId]: next,
          },
        };
      });
    },
    [activeMarketId],
  );

  const handleChangeRankingMetric = useCallback((metric: RankingMetric) => {
    setPersisted((prev) => ({ ...prev, rankingMetric: metric }));
  }, []);

  const handleInspectModelInTerminal = useCallback(
    (modelId: string) => {
      setPersisted((prev) => {
        const current = prev.pinnedOverlaysByMarket[activeMarketId] ?? [];
        const next = current.includes(modelId) ? current : [...current, modelId].slice(-4);
        return {
          ...prev,
          pinnedOverlaysByMarket: {
            ...prev.pinnedOverlaysByMarket,
            [activeMarketId]: next,
          },
        };
      });
      setActiveView('forecast');
    },
    [activeMarketId],
  );

  const championsByMarket = useMemo(() => {
    const map: Partial<Record<MarketId, ModelRunResult | null>> = {};
    for (const id of MARKET_IDS) {
      map[id] = selectChampionModel(resultsByMarket[id] ?? [], persisted.rankingMetric);
    }
    return map;
  }, [persisted.rankingMetric, resultsByMarket]);

  const activeSeries = markets[activeMarketId];
  const activeResults = resultsByMarket[activeMarketId] ?? [];
  const sortedActiveResults = useMemo(
    () => sortModelResults(activeResults, persisted.rankingMetric),
    [activeResults, persisted.rankingMetric],
  );
  const activeChampion = sortedActiveResults[0] ?? null;
  const activePinnedOverlays = persisted.pinnedOverlaysByMarket[activeMarketId] ?? [];
  const activeCustomModels = useMemo(
    () => activeResults.filter((r) => r.isCustom),
    [activeResults],
  );
  const activeSnapshots = useMemo(
    () => persisted.snapshotHistory.filter((s) => s.marketId === activeMarketId),
    [activeMarketId, persisted.snapshotHistory],
  );

  const activeMeta = MARKET_META_MAP[activeMarketId];

  return (
    <div className="min-h-screen flex bg-[#0B101B] text-[#E5ECF6]">
      <Sidebar
        activeMarketId={activeMarketId}
        activeView={activeView}
        markets={markets}
        champions={championsByMarket}
        onSelectMarket={(id) => {
          setActiveMarketId(id);
          if (activeView === 'overview') setActiveView('forecast');
        }}
        onSelectView={setActiveView}
        mobileOpen={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
      />

      <div className="flex-1 min-w-0 flex flex-col">
        {/* Top Bar Contract: Breadcrumb on left, 2 clean actions on right */}
        <header className="sticky top-0 z-30 flex h-14 items-center justify-between gap-3 border-b border-slate-800/80 bg-[#0B101B]/95 px-4 lg:px-7 backdrop-blur-xs">
          <div className="flex items-center gap-2.5 min-w-0 flex-1 overflow-hidden">
            <button
              type="button"
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open navigation menu"
              className="lg:hidden inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-slate-800 text-slate-300 hover:bg-slate-800/60"
            >
              <Menu className="h-4 w-4" />
            </button>

            <nav
              aria-label="Breadcrumb"
              className="flex items-center gap-1.5 text-xs min-w-0 overflow-hidden"
            >
              <span className="text-slate-400 whitespace-nowrap shrink-0 hidden sm:inline">
                Meridian
              </span>
              <span className="text-slate-600 hidden sm:inline" aria-hidden="true">
                /
              </span>
              <span className="text-slate-200 sm:text-slate-300 truncate">
                {VIEW_TITLES[activeView]}
              </span>
              {activeView !== 'overview' && (
                <>
                  <span className="text-slate-600 hidden sm:inline" aria-hidden="true">
                    /
                  </span>
                  <span className="font-medium text-slate-100 truncate hidden sm:inline">
                    {activeMeta.shortName}
                  </span>
                </>
              )}
            </nav>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={handleRefreshFeed}
              disabled={isLoadingFeed}
              aria-label="Refresh market feed"
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-[#111827] px-2.5 sm:px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-600 hover:bg-slate-800/70 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoadingFeed ? 'animate-spin' : ''}`} />
              <span className="hidden sm:inline">Refresh Feed</span>
            </button>

            <button
              type="button"
              onClick={handleRunBenchmark}
              disabled={isBenchmarking || isLoadingFeed}
              className="inline-flex items-center gap-1.5 rounded-md bg-sky-500 px-3 sm:px-3.5 py-1.5 text-xs font-semibold text-slate-950 hover:bg-sky-400 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0"
            >
              <Play className="h-3.5 w-3.5 fill-current" />
              {isBenchmarking ? 'Scoring…' : 'Run Benchmark'}
            </button>
          </div>
        </header>

        {/* Non-intrusive floating status toast */}
        {statusMessage && (
          <div
            role="status"
            aria-live="polite"
            className="fixed bottom-5 right-5 z-40 max-w-sm rounded-md border border-sky-500/35 bg-[#0E1422]/95 px-4 py-2.5 text-xs text-sky-200 shadow-xl"
          >
            {statusMessage}
          </div>
        )}

        <main className="flex-1 w-full max-w-[1380px] mx-auto px-4 lg:px-7 py-6">
          {!activeChampion ? (
            <div className="space-y-4" aria-busy="true">
              <div className="h-8 w-64 rounded-md bg-slate-800/60" />
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
                {[0, 1, 2, 3].map((idx) => (
                  <div
                    key={idx}
                    className="h-24 rounded-lg border border-slate-800/80 bg-[#111827]"
                  />
                ))}
              </div>
              <div className="h-96 rounded-lg border border-slate-800/80 bg-[#111827]" />
            </div>
          ) : activeView === 'forecast' ? (
            <ForecastTerminalView
              series={activeSeries}
              sortedResults={sortedActiveResults}
              champion={activeChampion}
              pinnedOverlayIds={activePinnedOverlays}
              horizon={horizon}
              range={range}
              rankingMetric={persisted.rankingMetric}
              onChangeHorizon={setHorizon}
              onChangeRange={setRange}
              onToggleOverlay={handleToggleOverlay}
              onOpenLab={() => setActiveView('lab')}
              onOpenArena={() => setActiveView('arena')}
            />
          ) : activeView === 'arena' ? (
            <ModelArenaView
              series={activeSeries}
              sortedResults={sortedActiveResults}
              champion={activeChampion}
              pinnedOverlayIds={activePinnedOverlays}
              rankingMetric={persisted.rankingMetric}
              horizon={horizon}
              marketSnapshots={activeSnapshots}
              onChangeRankingMetric={handleChangeRankingMetric}
              onToggleOverlay={handleToggleOverlay}
              onDeleteCustomModel={handleDeleteCustomModel}
              onOpenLab={() => setActiveView('lab')}
            />
          ) : activeView === 'lab' ? (
            <NeuralNetLabView
              series={activeSeries}
              champion={activeChampion}
              customModels={activeCustomModels}
              isTraining={isTrainingCustom}
              onTrainCustomModel={handleTrainCustomModel}
              onDeleteCustomModel={handleDeleteCustomModel}
              onInspectInTerminal={handleInspectModelInTerminal}
            />
          ) : (
            <GlobalOverviewView
              markets={markets}
              champions={championsByMarket}
              allResultsByMarket={resultsByMarket}
              horizon={horizon}
              rankingMetric={persisted.rankingMetric}
              onSelectMarket={(id) => {
                setActiveMarketId(id);
                setActiveView('forecast');
              }}
            />
          )}
        </main>
      </div>
    </div>
  );
}

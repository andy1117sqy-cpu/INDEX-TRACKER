import type {
  BenchmarkSnapshot,
  MarketId,
  NeuralConfig,
  PersistedWorkspaceState,
  RankingMetric,
} from '../types/market';

const STORAGE_KEY = 'meridian_index_lab_state_v1';

const EMPTY_CONFIGS: Record<MarketId, NeuralConfig[]> = {
  nasdaq: [],
  sse: [],
  kospi: [],
  nikkei: [],
  hsi: [],
};

const EMPTY_OVERLAYS: Record<MarketId, string[]> = {
  nasdaq: [],
  sse: [],
  kospi: [],
  nikkei: [],
  hsi: [],
};

export function createDefaultPersistedState(): PersistedWorkspaceState {
  return {
    customConfigsByMarket: {
      nasdaq: [],
      sse: [],
      kospi: [],
      nikkei: [],
      hsi: [],
    },
    pinnedOverlaysByMarket: {
      nasdaq: [],
      sse: [],
      kospi: [],
      nikkei: [],
      hsi: [],
    },
    rankingMetric: 'rmse',
    snapshotHistory: [],
  };
}

export function loadPersistedState(): PersistedWorkspaceState {
  if (typeof window === 'undefined') return createDefaultPersistedState();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return createDefaultPersistedState();
    const parsed = JSON.parse(raw) as Partial<PersistedWorkspaceState>;

    const rankingMetric: RankingMetric =
      parsed.rankingMetric === 'mape' || parsed.rankingMetric === 'directionalAccuracy'
        ? parsed.rankingMetric
        : 'rmse';

    return {
      customConfigsByMarket: {
        ...EMPTY_CONFIGS,
        ...(parsed.customConfigsByMarket ?? {}),
      },
      pinnedOverlaysByMarket: {
        ...EMPTY_OVERLAYS,
        ...(parsed.pinnedOverlaysByMarket ?? {}),
      },
      rankingMetric,
      snapshotHistory: Array.isArray(parsed.snapshotHistory)
        ? parsed.snapshotHistory.slice(0, 60)
        : [],
    };
  } catch {
    return createDefaultPersistedState();
  }
}

export function savePersistedState(state: PersistedWorkspaceState): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Ignore storage quota errors gracefully
  }
}

export function appendSnapshot(
  history: readonly BenchmarkSnapshot[],
  snapshot: BenchmarkSnapshot,
): BenchmarkSnapshot[] {
  return [snapshot, ...history].slice(0, 60);
}

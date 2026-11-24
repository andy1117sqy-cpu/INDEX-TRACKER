import { BarChart3, Cpu, Globe2, LineChart, X } from 'lucide-react';
import { MARKET_CATALOG } from '../data/marketsCatalog';
import type {
  MarketId,
  MarketSeries,
  ModelRunResult,
  WorkspaceView,
} from '../types/market';
import { formatIndexPrice, formatSignedPercent } from '../utils/format';

interface SidebarProps {
  activeMarketId: MarketId;
  activeView: WorkspaceView;
  markets: Partial<Record<MarketId, MarketSeries>>;
  champions: Partial<Record<MarketId, ModelRunResult | null>>;
  onSelectMarket: (id: MarketId) => void;
  onSelectView: (view: WorkspaceView) => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

const NAV_ITEMS: ReadonlyArray<{
  id: WorkspaceView;
  label: string;
  icon: typeof LineChart;
}> = [
  { id: 'forecast', label: 'Forecast Terminal', icon: LineChart },
  { id: 'arena', label: 'Model Arena', icon: BarChart3 },
  { id: 'lab', label: 'Neural Net Lab', icon: Cpu },
  { id: 'overview', label: 'Global Overview', icon: Globe2 },
];

function MiniSparkline({ values, positive }: { values: readonly number[]; positive: boolean }) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const w = 54;
  const h = 20;
  const points = values
    .map((v, idx) => {
      const x = (idx / (values.length - 1)) * w;
      const y = h - ((v - min) / span) * (h - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true" className="shrink-0">
      <polyline
        fill="none"
        stroke={positive ? '#10B981' : '#EF4444'}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

export function Sidebar({
  activeMarketId,
  activeView,
  markets,
  champions,
  onSelectMarket,
  onSelectView,
  mobileOpen,
  onCloseMobile,
}: SidebarProps) {
  const sidebarBody = (
    <div className="flex h-full flex-col bg-[#0E1422] border-r border-slate-800/80">
      <div className="flex h-14 items-center justify-between px-5 border-b border-slate-800/80">
        <span className="font-display text-base font-semibold tracking-tight text-slate-100 whitespace-nowrap">
          Meridian Index Lab
        </span>
        <button
          type="button"
          onClick={onCloseMobile}
          aria-label="Close navigation menu"
          className="lg:hidden inline-flex h-9 w-9 items-center justify-center rounded-md text-slate-400 hover:bg-slate-800/70 hover:text-slate-100"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
        <nav aria-label="Workspace views" className="space-y-1">
          <p className="px-2 pb-1.5 text-xs font-medium text-slate-400">Workspace</p>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  onSelectView(item.id);
                  onCloseMobile();
                }}
                aria-current={isActive ? 'page' : undefined}
                className={`flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-sky-500/12 text-sky-300'
                    : 'text-slate-300 hover:bg-slate-800/60 hover:text-slate-100'
                }`}
              >
                <Icon className="h-4 w-4 shrink-0" />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div>
          <div className="flex items-center justify-between px-2 pb-2">
            <p className="text-xs font-medium text-slate-400">Tracked Indexes</p>
            <span className="font-mono text-xs tabular-nums text-slate-500">5 Markets</span>
          </div>

          <div className="space-y-1" role="listbox" aria-label="Regional equity indexes">
            {MARKET_CATALOG.map((meta) => {
              const series = markets[meta.id];
              const bars = series?.bars ?? [];
              const latest = bars[bars.length - 1];
              const prev = bars[bars.length - 2];
              const pctChange =
                latest && prev && prev.close > 0
                  ? ((latest.close - prev.close) / prev.close) * 100
                  : 0;
              const positive = pctChange >= 0;
              const spark = bars.slice(-28).map((b) => b.close);
              const champ = champions[meta.id];
              const isSelected = activeMarketId === meta.id;

              return (
                <button
                  key={meta.id}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => {
                    onSelectMarket(meta.id);
                    onCloseMobile();
                  }}
                  className={`w-full rounded-md px-2.5 py-2.5 text-left transition-colors border ${
                    isSelected
                      ? 'bg-[#151F33] border-sky-500/35'
                      : 'bg-transparent border-transparent hover:bg-slate-800/45'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs font-mono text-slate-400">{meta.countryCode}</span>
                        <span className="text-sm font-semibold text-slate-100 truncate">
                          {meta.shortName}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 truncate mt-0.5">{meta.region}</p>
                    </div>
                    <MiniSparkline values={spark} positive={positive} />
                  </div>

                  <div className="mt-2 flex items-baseline justify-between gap-2">
                    <span className="font-mono text-xs font-medium tabular-nums text-slate-200">
                      {latest ? formatIndexPrice(latest.close) : 'Loading…'}
                    </span>
                    <span
                      className={`font-mono text-xs font-medium tabular-nums ${
                        positive ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {latest ? formatSignedPercent(pctChange) : '—'}
                    </span>
                  </div>

                  {champ && (
                    <p className="mt-1 text-[11px] text-slate-400 truncate">
                      Best: <span className="text-slate-300">{champ.modelName}</span>
                    </p>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="border-t border-slate-800/80 px-4 py-3 text-xs text-slate-500">
        Out-of-sample 25-session holdout validation across all five regional benchmarks.
      </div>
    </div>
  );

  return (
    <>
      <aside className="hidden lg:block w-64 shrink-0 h-screen sticky top-0">{sidebarBody}</aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-black/65"
            aria-hidden="true"
            onClick={onCloseMobile}
          />
          <div className="relative z-10 w-72 max-w-[85vw] h-full">{sidebarBody}</div>
        </div>
      )}
    </>
  );
}

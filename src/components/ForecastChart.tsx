import { useEffect, useMemo, useRef, useState } from 'react';
import type {
  ChartRange,
  ForecastHorizon,
  ModelRunResult,
  PriceBar,
} from '../types/market';
import { formatIndexPrice, formatShortDate } from '../utils/format';

interface ForecastChartProps {
  bars: readonly PriceBar[];
  champion: ModelRunResult;
  challengers: readonly ModelRunResult[];
  horizon: ForecastHorizon;
  range: ChartRange;
  currency: string;
}

const CHALLENGER_COLORS = ['#F59E0B', '#A78BFA', '#34D399', '#FB7185', '#F97316'];

interface PlotPoint {
  date: string;
  actual?: number;
  champHoldout?: number;
  champForecast?: number;
  b80Low?: number;
  b80High?: number;
  b95Low?: number;
  b95High?: number;
  challengerValues: Record<string, number | undefined>;
  phase: 'history' | 'holdout' | 'forecast';
}

export function ForecastChart({
  bars,
  champion,
  challengers,
  horizon,
  range,
  currency,
}: ForecastChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(860);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      const nextW = Math.max(320, Math.floor(el.getBoundingClientRect().width));
      setWidth(nextW);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const height = 400;
  const pad = { top: 22, right: 74, bottom: 34, left: 16 };

  const { points, holdoutStartIdx, forecastStartIdx, yMin, yMax } = useMemo(() => {
    const historyCount = range === '3M' ? 66 : range === '6M' ? 132 : 252;
    const visibleBars = bars.slice(-Math.min(historyCount, bars.length));

    const holdoutMap = new Map<string, number>();
    champion.holdoutDates.forEach((d, idx) => {
      holdoutMap.set(d, champion.holdoutPredicted[idx]);
    });

    const challengerHoldoutMaps = challengers.map((ch) => {
      const m = new Map<string, number>();
      ch.holdoutDates.forEach((d, idx) => m.set(d, ch.holdoutPredicted[idx]));
      return { id: ch.modelId, map: m };
    });

    const pts: PlotPoint[] = [];
    let hStart = -1;

    visibleBars.forEach((bar) => {
      const champHold = holdoutMap.get(bar.date);
      if (champHold !== undefined && hStart === -1) {
        hStart = pts.length;
      }
      const chVals: Record<string, number | undefined> = {};
      challengerHoldoutMaps.forEach(({ id, map }) => {
        chVals[id] = map.get(bar.date);
      });

      pts.push({
        date: bar.date,
        actual: bar.close,
        champHoldout: champHold,
        challengerValues: chVals,
        phase: champHold !== undefined ? 'holdout' : 'history',
      });
    });

    const fStart = pts.length;
    const slicedForecastDates = champion.forecastDates.slice(0, horizon);

    slicedForecastDates.forEach((date, idx) => {
      const chVals: Record<string, number | undefined> = {};
      challengers.forEach((ch) => {
        chVals[ch.modelId] = ch.forecastValues[idx];
      });

      pts.push({
        date,
        champForecast: champion.forecastValues[idx],
        b80Low: champion.band80Low[idx],
        b80High: champion.band80High[idx],
        b95Low: champion.band95Low[idx],
        b95High: champion.band95High[idx],
        challengerValues: chVals,
        phase: 'forecast',
      });
    });

    let minVal = Number.POSITIVE_INFINITY;
    let maxVal = Number.NEGATIVE_INFINITY;

    const consider = (v?: number) => {
      if (v !== undefined && Number.isFinite(v)) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    };

    pts.forEach((p) => {
      consider(p.actual);
      consider(p.champHoldout);
      consider(p.champForecast);
      consider(p.b95Low);
      consider(p.b95High);
      Object.values(p.challengerValues).forEach(consider);
    });

    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
      minVal = 0;
      maxVal = 100;
    }
    const span = maxVal - minVal || maxVal * 0.05 || 1;
    const margin = span * 0.08;

    return {
      points: pts,
      holdoutStartIdx: hStart,
      forecastStartIdx: fStart,
      yMin: minVal - margin,
      yMax: maxVal + margin,
    };
  }, [bars, champion, challengers, horizon, range]);

  const innerW = Math.max(100, width - pad.left - pad.right);
  const innerH = height - pad.top - pad.bottom;

  const xAt = (idx: number) =>
    pad.left + (points.length <= 1 ? 0 : (idx / (points.length - 1)) * innerW);
  const yAt = (val: number) => pad.top + ((yMax - val) / (yMax - yMin || 1)) * innerH;

  const buildPath = (getter: (p: PlotPoint) => number | undefined, bridgeIdx?: number): string => {
    let d = '';
    let started = false;
    for (let i = 0; i < points.length; i++) {
      let v = getter(points[i]);
      if (bridgeIdx !== undefined && i === bridgeIdx && v === undefined) {
        v = points[i].actual;
      }
      if (v === undefined || !Number.isFinite(v)) {
        started = false;
        continue;
      }
      const cmd = started ? 'L' : 'M';
      d += `${cmd}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)} `;
      started = true;
    }
    return d.trim();
  };

  const buildBandPolygon = (
    lowGetter: (p: PlotPoint) => number | undefined,
    highGetter: (p: PlotPoint) => number | undefined,
  ): string => {
    if (forecastStartIdx <= 0 || forecastStartIdx >= points.length) return '';
    const anchorIdx = forecastStartIdx - 1;
    const anchorVal = points[anchorIdx]?.actual;
    if (anchorVal === undefined) return '';

    const upper: string[] = [`${xAt(anchorIdx).toFixed(1)},${yAt(anchorVal).toFixed(1)}`];
    const lower: string[] = [`${xAt(anchorIdx).toFixed(1)},${yAt(anchorVal).toFixed(1)}`];

    for (let i = forecastStartIdx; i < points.length; i++) {
      const lo = lowGetter(points[i]);
      const hi = highGetter(points[i]);
      if (lo === undefined || hi === undefined) continue;
      const x = xAt(i).toFixed(1);
      upper.push(`${x},${yAt(hi).toFixed(1)}`);
      lower.push(`${x},${yAt(lo).toFixed(1)}`);
    }

    return [...upper, ...lower.reverse()].join(' ');
  };

  const yTicks = useMemo(() => {
    const count = 5;
    const ticks: number[] = [];
    for (let i = 0; i <= count; i++) {
      ticks.push(yMin + ((yMax - yMin) * i) / count);
    }
    return ticks;
  }, [yMin, yMax]);

  const xTickIndices = useMemo(() => {
    if (points.length === 0) return [];
    const desired = width < 560 ? 4 : 7;
    const step = Math.max(1, Math.floor((points.length - 1) / (desired - 1)));
    const idxs: number[] = [];
    for (let i = 0; i < points.length; i += step) idxs.push(i);
    const lastIdx = points.length - 1;
    const prevLast = idxs[idxs.length - 1];
    if (prevLast !== lastIdx) {
      if (lastIdx - prevLast < step * 0.6) {
        idxs[idxs.length - 1] = lastIdx;
      } else {
        idxs.push(lastIdx);
      }
    }
    return idxs;
  }, [points.length, width]);

  const actualPath = buildPath((p) => p.actual);
  const holdoutPath = buildPath((p) => p.champHoldout);
  const forecastPath = buildPath((p) => p.champForecast, forecastStartIdx - 1);
  const band95Poly = buildBandPolygon((p) => p.b95Low, (p) => p.b95High);
  const band80Poly = buildBandPolygon((p) => p.b80Low, (p) => p.b80High);

  const hoveredPoint = hoverIdx !== null ? points[hoverIdx] : null;

  const handlePointerMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = e.clientX - rect.left - pad.left;
    const ratio = Math.max(0, Math.min(1, relX / innerW));
    const idx = Math.round(ratio * (points.length - 1));
    setHoverIdx(idx);
  };

  return (
    <div ref={containerRef} className="relative w-full select-none" data-testid="forecast-chart">
      {/* Legend strip */}
      <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 text-xs text-slate-300">
        <span className="inline-flex items-center gap-2">
          <span className="h-0.5 w-4 bg-slate-200 inline-block" />
          Actual Close ({currency})
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-0.5 w-4 border-t-2 border-dashed border-sky-400 inline-block" />
          Champion Validation Fit
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-0.5 w-4 bg-sky-400 inline-block" />
          Champion Forecast ({horizon}D)
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-2.5 w-4 rounded-xs bg-sky-500/20 border border-sky-400/30 inline-block" />
          80% / 95% Confidence Cone
        </span>
        {challengers.map((ch, idx) => (
          <span key={ch.modelId} className="inline-flex items-center gap-2">
            <span
              className="h-0.5 w-4 inline-block"
              style={{ backgroundColor: CHALLENGER_COLORS[idx % CHALLENGER_COLORS.length] }}
            />
            {ch.modelName}
          </span>
        ))}
      </div>

      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="block w-full overflow-visible"
        role="img"
        aria-label={`Price history and ${horizon}-day forecast chart for ${champion.modelName}`}
        onMouseMove={handlePointerMove}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {/* Subtle forecast region backdrop */}
        {forecastStartIdx > 0 && forecastStartIdx < points.length && (
          <rect
            x={xAt(forecastStartIdx - 1)}
            y={pad.top}
            width={Math.max(0, xAt(points.length - 1) - xAt(forecastStartIdx - 1))}
            height={innerH}
            fill="rgba(56, 189, 248, 0.04)"
          />
        )}

        {/* Horizontal gridlines & right-side price axis */}
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
                strokeWidth="1"
              />
              <text
                x={pad.left + innerW + 8}
                y={y + 4}
                fill="#94A3B8"
                fontSize="11"
                fontFamily="JetBrains Mono, monospace"
              >
                {formatIndexPrice(val)}
              </text>
            </g>
          );
        })}

        {/* Vertical phase split markers */}
        {holdoutStartIdx > 0 && (
          <g>
            <line
              x1={xAt(holdoutStartIdx)}
              x2={xAt(holdoutStartIdx)}
              y1={pad.top}
              y2={pad.top + innerH}
              stroke="rgba(148, 163, 184, 0.28)"
              strokeDasharray="3 3"
              strokeWidth="1"
            />
            <text
              x={xAt(holdoutStartIdx) + 5}
              y={pad.top + 12}
              fill="#94A3B8"
              fontSize="10"
              fontFamily="Public Sans, sans-serif"
            >
              Holdout Test
            </text>
          </g>
        )}

        {forecastStartIdx > 0 && forecastStartIdx < points.length && (
          <g>
            <line
              x1={xAt(forecastStartIdx - 1)}
              x2={xAt(forecastStartIdx - 1)}
              y1={pad.top}
              y2={pad.top + innerH}
              stroke="rgba(56, 189, 248, 0.5)"
              strokeDasharray="4 3"
              strokeWidth="1"
            />
            <text
              x={xAt(forecastStartIdx - 1) + 5}
              y={pad.top + 12}
              fill="#7DD3FC"
              fontSize="10"
              fontFamily="Public Sans, sans-serif"
            >
              Forecast ({horizon}D)
            </text>
          </g>
        )}

        {/* Confidence cones */}
        {band95Poly && <polygon points={band95Poly} fill="rgba(56, 189, 248, 0.09)" />}
        {band80Poly && <polygon points={band80Poly} fill="rgba(56, 189, 248, 0.16)" />}

        {/* Challenger overlay curves */}
        {challengers.map((ch, idx) => {
          const color = CHALLENGER_COLORS[idx % CHALLENGER_COLORS.length];
          const chPath = buildPath((p) => p.challengerValues[ch.modelId], forecastStartIdx - 1);
          return (
            <path
              key={ch.modelId}
              d={chPath}
              fill="none"
              stroke={color}
              strokeWidth="1.6"
              strokeDasharray="4 2.5"
              strokeLinejoin="round"
            />
          );
        })}

        {/* Historical actual price */}
        <path
          d={actualPath}
          fill="none"
          stroke="#E2E8F0"
          strokeWidth="1.85"
          strokeLinejoin="round"
        />

        {/* Champion holdout backtest fit */}
        <path
          d={holdoutPath}
          fill="none"
          stroke="#38BDF8"
          strokeWidth="1.75"
          strokeDasharray="5 3"
          strokeLinejoin="round"
        />

        {/* Champion future forecast trajectory */}
        <path
          d={forecastPath}
          fill="none"
          stroke="#38BDF8"
          strokeWidth="2.4"
          strokeLinejoin="round"
        />

        {/* Terminal forecast anchor dot */}
        {points.length > 0 && points[points.length - 1].champForecast !== undefined && (
          <circle
            cx={xAt(points.length - 1)}
            cy={yAt(points[points.length - 1].champForecast!)}
            r="3.5"
            fill="#38BDF8"
            stroke="#0B101B"
            strokeWidth="1.5"
          />
        )}

        {/* X-axis dates */}
        {xTickIndices.map((idx) => (
          <text
            key={idx}
            x={xAt(idx)}
            y={height - 10}
            textAnchor="middle"
            fill="#94A3B8"
            fontSize="11"
            fontFamily="JetBrains Mono, monospace"
          >
            {formatShortDate(points[idx].date)}
          </text>
        ))}

        {/* Interactive crosshair */}
        {hoverIdx !== null && hoveredPoint && (
          <g>
            <line
              x1={xAt(hoverIdx)}
              x2={xAt(hoverIdx)}
              y1={pad.top}
              y2={pad.top + innerH}
              stroke="rgba(226, 232, 240, 0.35)"
              strokeWidth="1"
            />
            {(hoveredPoint.actual ?? hoveredPoint.champForecast) !== undefined && (
              <circle
                cx={xAt(hoverIdx)}
                cy={yAt((hoveredPoint.actual ?? hoveredPoint.champForecast)!)}
                r="4"
                fill={hoveredPoint.phase === 'forecast' ? '#38BDF8' : '#F8FAFC'}
                stroke="#0B101B"
                strokeWidth="1.5"
              />
            )}
          </g>
        )}
      </svg>

      {/* Crosshair inspection card */}
      {hoverIdx !== null && hoveredPoint && (
        <div
          className="pointer-events-none absolute top-11 z-20 w-60 rounded-md border border-slate-700/80 bg-[#0E1422]/95 px-3 py-2.5 text-xs shadow-lg"
          style={{
            left:
              xAt(hoverIdx) > width * 0.6
                ? Math.max(12, xAt(hoverIdx) - 256)
                : Math.min(width - 252, xAt(hoverIdx) + 14),
          }}
        >
          <div className="flex items-center justify-between border-b border-slate-800 pb-1.5 mb-1.5">
            <span className="font-mono font-medium text-slate-200">{hoveredPoint.date}</span>
            <span className="text-slate-400 capitalize">{hoveredPoint.phase}</span>
          </div>

          <div className="space-y-1 font-mono tabular-nums">
            {hoveredPoint.actual !== undefined && (
              <div className="flex justify-between">
                <span className="text-slate-400">Actual</span>
                <span className="text-slate-100">{formatIndexPrice(hoveredPoint.actual)}</span>
              </div>
            )}
            {hoveredPoint.champHoldout !== undefined && (
              <div className="flex justify-between">
                <span className="text-sky-300">Champion Fit</span>
                <span className="text-sky-300">{formatIndexPrice(hoveredPoint.champHoldout)}</span>
              </div>
            )}
            {hoveredPoint.champForecast !== undefined && (
              <div className="flex justify-between">
                <span className="text-sky-300">Forecast</span>
                <span className="text-sky-200 font-semibold">
                  {formatIndexPrice(hoveredPoint.champForecast)}
                </span>
              </div>
            )}
            {hoveredPoint.b95Low !== undefined && hoveredPoint.b95High !== undefined && (
              <div className="flex justify-between text-[11px] text-slate-400">
                <span>95% Band</span>
                <span>
                  {formatIndexPrice(hoveredPoint.b95Low)} – {formatIndexPrice(hoveredPoint.b95High)}
                </span>
              </div>
            )}
            {challengers.map((ch, idx) => {
              const val = hoveredPoint.challengerValues[ch.modelId];
              if (val === undefined) return null;
              return (
                <div key={ch.modelId} className="flex justify-between text-[11px]">
                  <span
                    className="truncate pr-2"
                    style={{ color: CHALLENGER_COLORS[idx % CHALLENGER_COLORS.length] }}
                  >
                    {ch.modelName}
                  </span>
                  <span className="text-slate-200">{formatIndexPrice(val)}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

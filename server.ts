import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import {
  MARKET_IDS,
  MARKET_META_MAP,
  buildFallbackMarketSeries,
  isMarketId,
} from './src/data/marketsCatalog.ts';
import type { MarketId, MarketSeries, PriceBar } from './src/types/market.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 3000;

/**
 * Strict server-side allowlist mapping approved market IDs to Yahoo Finance symbols.
 * Prevents arbitrary upstream URL requests or symbol injection.
 */
const ALLOWED_SYMBOLS: Record<MarketId, string> = {
  nasdaq: '^IXIC',
  sse: '000001.SS',
  kospi: '^KS11',
  nikkei: '^N225',
  hsi: '^HSI',
};

interface CacheEntry {
  series: MarketSeries;
  fetchedAt: number;
}

const CACHE_TTL_MS = 15 * 60 * 1000;
const marketCache = new Map<MarketId, CacheEntry>();

interface YahooChartResponse {
  chart?: {
    result?: Array<{
      timestamp?: number[];
      indicators?: {
        quote?: Array<{
          open?: Array<number | null>;
          high?: Array<number | null>;
          low?: Array<number | null>;
          close?: Array<number | null>;
          volume?: Array<number | null>;
        }>;
      };
    }>;
  };
}

async function fetchYahooSeries(marketId: MarketId): Promise<MarketSeries | null> {
  const symbol = ALLOWED_SYMBOLS[marketId];
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=1y&interval=1d`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4500);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; MeridianIndexLab/1.0)',
        Accept: 'application/json',
      },
    });
    if (!response.ok) return null;

    const payload = (await response.json()) as YahooChartResponse;
    const result = payload.chart?.result?.[0];
    const timestamps = result?.timestamp;
    const quote = result?.indicators?.quote?.[0];
    if (!timestamps || !quote?.close || timestamps.length < 60) return null;

    const bars: PriceBar[] = [];
    const seenDates = new Set<string>();

    for (let i = 0; i < timestamps.length; i++) {
      const close = quote.close[i];
      if (typeof close !== 'number' || !Number.isFinite(close) || close <= 0) continue;

      const date = new Date(timestamps[i] * 1000).toISOString().slice(0, 10);
      if (seenDates.has(date)) continue;
      seenDates.add(date);

      const openRaw = quote.open?.[i];
      const highRaw = quote.high?.[i];
      const lowRaw = quote.low?.[i];
      const volRaw = quote.volume?.[i];

      const open =
        typeof openRaw === 'number' && Number.isFinite(openRaw) && openRaw > 0 ? openRaw : close;
      const high =
        typeof highRaw === 'number' && Number.isFinite(highRaw) && highRaw > 0
          ? highRaw
          : Math.max(open, close);
      const low =
        typeof lowRaw === 'number' && Number.isFinite(lowRaw) && lowRaw > 0
          ? lowRaw
          : Math.min(open, close);
      const volume =
        typeof volRaw === 'number' && Number.isFinite(volRaw) && volRaw >= 0
          ? Math.round(volRaw)
          : 0;

      bars.push({
        date,
        open: Number(open.toFixed(2)),
        high: Number(high.toFixed(2)),
        low: Number(low.toFixed(2)),
        close: Number(close.toFixed(2)),
        volume,
      });
    }

    if (bars.length < 60) return null;

    return {
      meta: MARKET_META_MAP[marketId],
      source: 'live',
      lastUpdated: new Date().toISOString(),
      bars,
    };
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function getMarketSeries(marketId: MarketId, forceRefresh = false): Promise<MarketSeries> {
  const now = Date.now();
  const cached = marketCache.get(marketId);
  if (!forceRefresh && cached && now - cached.fetchedAt < CACHE_TTL_MS) {
    return {
      ...cached.series,
      source: cached.series.source === 'live' ? 'cached' : 'fallback',
    };
  }

  const live = await fetchYahooSeries(marketId);
  if (live) {
    marketCache.set(marketId, { series: live, fetchedAt: now });
    return live;
  }

  const fallback = buildFallbackMarketSeries(marketId);
  marketCache.set(marketId, { series: fallback, fetchedAt: now });
  return fallback;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '256kb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', markets: MARKET_IDS });
  });

  app.get('/api/markets', async (req, res) => {
    const forceRefresh = req.query.refresh === '1';
    try {
      const entries = await Promise.all(
        MARKET_IDS.map(async (id) => [id, await getMarketSeries(id, forceRefresh)] as const),
      );
      const markets = Object.fromEntries(entries) as Record<MarketId, MarketSeries>;
      res.json({ markets, serverTime: new Date().toISOString() });
    } catch {
      res.status(500).json({ error: 'Unable to load market index data.' });
    }
  });

  app.get('/api/markets/:marketId', async (req, res) => {
    const rawId = String(req.params.marketId || '').toLowerCase();
    if (!isMarketId(rawId)) {
      res.status(400).json({ error: 'Unsupported market identifier.' });
      return;
    }
    const forceRefresh = req.query.refresh === '1';
    try {
      const series = await getMarketSeries(rawId, forceRefresh);
      res.json(series);
    } catch {
      res.status(500).json({ error: 'Unable to fetch market series.' });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Meridian Index Lab server listening on http://localhost:${PORT}`);
  });
}

startServer();

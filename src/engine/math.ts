import type { ModelMetrics } from '../types/market';

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}

export function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i];
  return sum / values.length;
}

export function std(values: readonly number[], avg?: number): number {
  if (values.length <= 1) return 1e-6;
  const m = avg ?? mean(values);
  let sumSq = 0;
  for (let i = 0; i < values.length; i++) {
    const d = values[i] - m;
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / values.length) || 1e-6;
}

export function computeMetrics(
  actual: readonly number[],
  predicted: readonly number[],
  priorClose?: number,
): ModelMetrics {
  const n = Math.min(actual.length, predicted.length);
  if (n === 0) {
    return { rmse: 0, mae: 0, mape: 0, r2: 0, directionalAccuracy: 0 };
  }

  let sqErr = 0;
  let absErr = 0;
  let apeSum = 0;
  const avgActual = mean(actual.slice(0, n));
  let ssTot = 0;

  let dirHits = 0;
  let dirTotal = 0;

  for (let i = 0; i < n; i++) {
    const a = actual[i];
    const p = predicted[i];
    const err = p - a;
    sqErr += err * err;
    absErr += Math.abs(err);
    apeSum += Math.abs(err) / Math.max(1e-6, Math.abs(a));
    const diff = a - avgActual;
    ssTot += diff * diff;

    const prevA = i === 0 ? priorClose : actual[i - 1];
    if (prevA !== undefined && Number.isFinite(prevA)) {
      const actualDir = a - prevA >= 0 ? 1 : -1;
      const predDir = p - prevA >= 0 ? 1 : -1;
      if (actualDir === predDir) dirHits++;
      dirTotal++;
    }
  }

  const rmse = Math.sqrt(sqErr / n);
  const mae = absErr / n;
  const mape = (apeSum / n) * 100;
  const r2 = ssTot > 1e-9 ? 1 - sqErr / ssTot : 0;
  const directionalAccuracy = dirTotal > 0 ? (dirHits / dirTotal) * 100 : 50;

  return {
    rmse: Number(rmse.toFixed(2)),
    mae: Number(mae.toFixed(2)),
    mape: Number(mape.toFixed(3)),
    r2: Number(clamp(r2, -5, 1).toFixed(4)),
    directionalAccuracy: Number(directionalAccuracy.toFixed(1)),
  };
}

/**
 * Solves (X^T X + lambda * I) w = X^T y via Gaussian elimination with partial pivoting.
 */
export function solveRidgeRegression(
  X: readonly (readonly number[])[],
  y: readonly number[],
  lambda: number,
  penalizeFirst = false,
): number[] {
  const n = X.length;
  if (n === 0) return [];
  const p = X[0].length;

  const A: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const b = new Array<number>(p).fill(0);

  for (let i = 0; i < n; i++) {
    const row = X[i];
    const yi = y[i];
    for (let j = 0; j < p; j++) {
      b[j] += row[j] * yi;
      const rj = row[j];
      for (let k = j; k < p; k++) {
        A[j][k] += rj * row[k];
      }
    }
  }

  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) {
      A[j][k] = A[k][j];
    }
    const reg = j === 0 && !penalizeFirst ? 1e-8 : Math.max(1e-8, lambda);
    A[j][j] += reg;
  }

  return solveLinearSystem(A, b);
}

export function solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const A = matrix.map((row) => row.slice());
  const b = rhs.slice();

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxAbs = Math.abs(A[col][col]);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(A[row][col]);
      if (val > maxAbs) {
        maxAbs = val;
        pivotRow = row;
      }
    }

    if (maxAbs < 1e-12) continue;

    if (pivotRow !== col) {
      [A[col], A[pivotRow]] = [A[pivotRow], A[col]];
      [b[col], b[pivotRow]] = [b[pivotRow], b[col]];
    }

    const pivot = A[col][col];
    for (let row = col + 1; row < n; row++) {
      const factor = A[row][col] / pivot;
      if (factor === 0) continue;
      for (let k = col; k < n; k++) {
        A[row][k] -= factor * A[col][k];
      }
      b[row] -= factor * b[col];
    }
  }

  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i + 1; j < n; j++) {
      sum -= A[i][j] * x[j];
    }
    const diag = A[i][i];
    x[i] = Math.abs(diag) > 1e-12 ? sum / diag : 0;
  }
  return x;
}

export function generateFutureTradingDates(lastDateStr: string, count: number): string[] {
  const result: string[] = [];
  const cursor = new Date(`${lastDateStr}T00:00:00Z`);
  if (Number.isNaN(cursor.getTime())) return result;

  while (result.length < count) {
    cursor.setUTCDate(cursor.getUTCDate() + 1);
    const dow = cursor.getUTCDay();
    if (dow !== 0 && dow !== 6) {
      result.push(cursor.toISOString().slice(0, 10));
    }
  }
  return result;
}

export function buildConfidenceBands(
  forecastValues: readonly number[],
  holdoutActual: readonly number[],
  holdoutPredicted: readonly number[],
): {
  band80Low: number[];
  band80High: number[];
  band95Low: number[];
  band95High: number[];
} {
  const logResiduals: number[] = [];
  const n = Math.min(holdoutActual.length, holdoutPredicted.length);
  for (let i = 0; i < n; i++) {
    const a = Math.max(1e-6, holdoutActual[i]);
    const p = Math.max(1e-6, holdoutPredicted[i]);
    logResiduals.push(Math.log(p / a));
  }

  const sigma = clamp(std(logResiduals), 0.0045, 0.035);
  const band80Low: number[] = [];
  const band80High: number[] = [];
  const band95Low: number[] = [];
  const band95High: number[] = [];

  for (let h = 1; h <= forecastValues.length; h++) {
    const center = forecastValues[h - 1];
    const horizonScale = Math.sqrt(h) * (1 - 0.004 * Math.min(h, 30));
    const spread80 = 1.2816 * sigma * horizonScale;
    const spread95 = 1.96 * sigma * horizonScale;

    band80Low.push(Number((center * Math.exp(-spread80)).toFixed(2)));
    band80High.push(Number((center * Math.exp(spread80)).toFixed(2)));
    band95Low.push(Number((center * Math.exp(-spread95)).toFixed(2)));
    band95High.push(Number((center * Math.exp(spread95)).toFixed(2)));
  }

  return { band80Low, band80High, band95Low, band95High };
}

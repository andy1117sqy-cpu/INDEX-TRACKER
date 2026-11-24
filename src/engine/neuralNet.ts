import type {
  ActivationName,
  EpochLossPoint,
  ModelRunResult,
  NeuralConfig,
  PriceBar,
} from '../types/market';
import {
  buildSupervisedDataset,
  computeLogReturns,
  extractFeatureVector,
  fitScaler,
  inverseTransformTarget,
  transformFeatures,
  transformTarget,
} from './features';
import {
  buildConfidenceBands,
  clamp,
  computeMetrics,
  generateFutureTradingDates,
  mulberry32,
} from './math';

export function sanitizeNeuralConfig(raw: NeuralConfig): NeuralConfig {
  const layers = (raw.hiddenLayers?.length ? raw.hiddenLayers : [16])
    .slice(0, 3)
    .map((u) => Math.round(clamp(Number(u) || 16, 4, 64)));

  const activation: ActivationName =
    raw.activation === 'relu' || raw.activation === 'silu' ? raw.activation : 'tanh';

  return {
    id: raw.id || `custom-mlp-${Date.now()}`,
    name: (raw.name || 'Custom MLP').trim().slice(0, 48) || 'Custom MLP',
    hiddenLayers: layers,
    activation,
    lookback: Math.round(clamp(Number(raw.lookback) || 12, 5, 30)),
    learningRate: Number(clamp(Number(raw.learningRate) || 0.008, 0.0005, 0.05).toFixed(4)),
    epochs: Math.round(clamp(Number(raw.epochs) || 85, 20, 200)),
    l2: Number(clamp(Number(raw.l2) || 0.002, 0, 0.05).toFixed(4)),
    residualSkip: Boolean(raw.residualSkip),
    seed: raw.seed ?? 42,
    isCustom: Boolean(raw.isCustom),
  };
}

export function formatArchitectureLabel(config: NeuralConfig): string {
  const stack = config.hiddenLayers.join('→');
  const skipTag = config.residualSkip ? ' + skip' : '';
  return `MLP [${stack}] ${config.activation}${skipTag} · L${config.lookback}`;
}

function activate(z: number, fn: ActivationName): number {
  if (fn === 'relu') return z > 0 ? z : 0.01 * z; // Leaky ReLU avoids dead neurons
  if (fn === 'silu') {
    const s = 1 / (1 + Math.exp(-clamp(z, -20, 20)));
    return z * s;
  }
  return Math.tanh(clamp(z, -15, 15));
}

function activateDeriv(z: number, a: number, fn: ActivationName): number {
  if (fn === 'relu') return z > 0 ? 1 : 0.01;
  if (fn === 'silu') {
    const s = 1 / (1 + Math.exp(-clamp(z, -20, 20)));
    return s * (1 + z * (1 - s));
  }
  return 1 - a * a;
}

interface LayerWeights {
  inDim: number;
  outDim: number;
  W: Float64Array; // row-major [outDim * inDim]
  b: Float64Array; // [outDim]
  mW: Float64Array;
  vW: Float64Array;
  mB: Float64Array;
  vB: Float64Array;
}

class FeedforwardMLP {
  private layers: LayerWeights[];
  private skipW: Float64Array | null;
  private mSkip: Float64Array | null;
  private vSkip: Float64Array | null;
  private activation: ActivationName;
  private residualSkip: boolean;
  private inputDim: number;
  private step = 0;

  constructor(
    inputDim: number,
    hiddenLayers: readonly number[],
    activation: ActivationName,
    residualSkip: boolean,
    seed: number,
  ) {
    this.inputDim = inputDim;
    this.activation = activation;
    this.residualSkip = residualSkip;
    const rng = mulberry32(seed);

    const dims = [inputDim, ...hiddenLayers, 1];
    this.layers = [];

    for (let l = 0; l < dims.length - 1; l++) {
      const inDim = dims[l];
      const outDim = dims[l + 1];
      const scale =
        activation === 'relu' && l < dims.length - 2
          ? Math.sqrt(2 / inDim)
          : Math.sqrt(1 / (inDim + outDim));

      const W = new Float64Array(outDim * inDim);
      for (let i = 0; i < W.length; i++) {
        // Approximate zero-mean Gaussian via Box-Muller
        const u1 = Math.max(1e-12, rng());
        const u2 = rng();
        const g = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        W[i] = g * scale;
      }

      this.layers.push({
        inDim,
        outDim,
        W,
        b: new Float64Array(outDim),
        mW: new Float64Array(outDim * inDim),
        vW: new Float64Array(outDim * inDim),
        mB: new Float64Array(outDim),
        vB: new Float64Array(outDim),
      });
    }

    if (residualSkip) {
      this.skipW = new Float64Array(inputDim);
      const skipScale = Math.sqrt(0.25 / inputDim);
      for (let i = 0; i < inputDim; i++) {
        this.skipW[i] = (rng() * 2 - 1) * skipScale;
      }
      this.mSkip = new Float64Array(inputDim);
      this.vSkip = new Float64Array(inputDim);
    } else {
      this.skipW = null;
      this.mSkip = null;
      this.vSkip = null;
    }
  }

  predict(x: readonly number[]): number {
    let curr = x;
    const lastLayerIdx = this.layers.length - 1;

    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const next = new Array<number>(layer.outDim);
      for (let o = 0; o < layer.outDim; o++) {
        let sum = layer.b[o];
        const rowOffset = o * layer.inDim;
        for (let i = 0; i < layer.inDim; i++) {
          sum += layer.W[rowOffset + i] * curr[i];
        }
        next[o] = l === lastLayerIdx ? sum : activate(sum, this.activation);
      }
      curr = next;
    }

    let out = curr[0];
    if (this.residualSkip && this.skipW) {
      for (let i = 0; i < this.inputDim; i++) {
        out += this.skipW[i] * x[i];
      }
    }
    return clamp(out, -6, 6);
  }

  evaluateMse(X: readonly (readonly number[])[], y: readonly number[]): number {
    if (X.length === 0) return 0;
    let sumSq = 0;
    for (let i = 0; i < X.length; i++) {
      const err = this.predict(X[i]) - y[i];
      sumSq += err * err;
    }
    return sumSq / X.length;
  }

  trainEpoch(
    X: readonly (readonly number[])[],
    y: readonly number[],
    learningRate: number,
    l2: number,
  ): number {
    const n = X.length;
    if (n === 0) return 0;

    const lastIdx = this.layers.length - 1;
    const gradW = this.layers.map((l) => new Float64Array(l.W.length));
    const gradB = this.layers.map((l) => new Float64Array(l.b.length));
    const gradSkip = this.residualSkip ? new Float64Array(this.inputDim) : null;

    let totalSqErr = 0;

    for (let sample = 0; sample < n; sample++) {
      const x0 = X[sample];
      const target = y[sample];

      // Forward pass storing pre-activations & activations
      const zs: number[][] = [];
      const acts: (readonly number[])[] = [x0];

      let curr: readonly number[] = x0;
      for (let l = 0; l < this.layers.length; l++) {
        const layer = this.layers[l];
        const z = new Array<number>(layer.outDim);
        const a = new Array<number>(layer.outDim);
        for (let o = 0; o < layer.outDim; o++) {
          let sum = layer.b[o];
          const rowOffset = o * layer.inDim;
          for (let i = 0; i < layer.inDim; i++) {
            sum += layer.W[rowOffset + i] * curr[i];
          }
          z[o] = sum;
          a[o] = l === lastIdx ? sum : activate(sum, this.activation);
        }
        zs.push(z);
        acts.push(a);
        curr = a;
      }

      let pred = curr[0];
      if (this.residualSkip && this.skipW) {
        for (let i = 0; i < this.inputDim; i++) {
          pred += this.skipW[i] * x0[i];
        }
      }

      const err = pred - target;
      totalSqErr += err * err;

      // dL/dpred for MSE = (2 / n) * err
      let delta = [clamp(2 * err, -6, 6)];

      if (gradSkip) {
        for (let i = 0; i < this.inputDim; i++) {
          gradSkip[i] += delta[0] * x0[i];
        }
      }

      // Backpropagation through layers
      for (let l = lastIdx; l >= 0; l--) {
        const layer = this.layers[l];
        const prevAct = acts[l];
        const gW = gradW[l];
        const gB = gradB[l];

        for (let o = 0; o < layer.outDim; o++) {
          const d = delta[o];
          gB[o] += d;
          const rowOffset = o * layer.inDim;
          for (let i = 0; i < layer.inDim; i++) {
            gW[rowOffset + i] += d * prevAct[i];
          }
        }

        if (l > 0) {
          const prevZ = zs[l - 1];
          const prevA = acts[l];
          const nextDelta = new Array<number>(layer.inDim).fill(0);
          for (let i = 0; i < layer.inDim; i++) {
            let back = 0;
            for (let o = 0; o < layer.outDim; o++) {
              back += layer.W[o * layer.inDim + i] * delta[o];
            }
            nextDelta[i] = clamp(back * activateDeriv(prevZ[i], prevA[i], this.activation), -5, 5);
          }
          delta = nextDelta;
        }
      }
    }

    // Adam parameter update
    this.step++;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const b1Corr = 1 - Math.pow(beta1, this.step);
    const b2Corr = 1 - Math.pow(beta2, this.step);
    const invN = 1 / n;

    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const gW = gradW[l];
      const gB = gradB[l];

      for (let idx = 0; idx < layer.W.length; idx++) {
        const g = clamp(gW[idx] * invN + l2 * layer.W[idx], -5, 5);
        layer.mW[idx] = beta1 * layer.mW[idx] + (1 - beta1) * g;
        layer.vW[idx] = beta2 * layer.vW[idx] + (1 - beta2) * g * g;
        const mHat = layer.mW[idx] / b1Corr;
        const vHat = layer.vW[idx] / b2Corr;
        layer.W[idx] -= (learningRate * mHat) / (Math.sqrt(vHat) + eps);
      }

      for (let o = 0; o < layer.b.length; o++) {
        const g = clamp(gB[o] * invN, -5, 5);
        layer.mB[o] = beta1 * layer.mB[o] + (1 - beta1) * g;
        layer.vB[o] = beta2 * layer.vB[o] + (1 - beta2) * g * g;
        const mHat = layer.mB[o] / b1Corr;
        const vHat = layer.vB[o] / b2Corr;
        layer.b[o] -= (learningRate * mHat) / (Math.sqrt(vHat) + eps);
      }
    }

    if (this.residualSkip && this.skipW && this.mSkip && this.vSkip && gradSkip) {
      for (let i = 0; i < this.inputDim; i++) {
        const g = clamp(gradSkip[i] * invN + l2 * 1.5 * this.skipW[i], -5, 5);
        this.mSkip[i] = beta1 * this.mSkip[i] + (1 - beta1) * g;
        this.vSkip[i] = beta2 * this.vSkip[i] + (1 - beta2) * g * g;
        const mHat = this.mSkip[i] / b1Corr;
        const vHat = this.vSkip[i] / b2Corr;
        this.skipW[i] -= (learningRate * mHat) / (Math.sqrt(vHat) + eps);
      }
    }

    return totalSqErr / n;
  }

  snapshotWeights(): { W: Float64Array[]; b: Float64Array[]; skipW: Float64Array | null } {
    return {
      W: this.layers.map((l) => new Float64Array(l.W)),
      b: this.layers.map((l) => new Float64Array(l.b)),
      skipW: this.skipW ? new Float64Array(this.skipW) : null,
    };
  }

  restoreWeights(snap: { W: Float64Array[]; b: Float64Array[]; skipW: Float64Array | null }): void {
    for (let l = 0; l < this.layers.length; l++) {
      this.layers[l].W.set(snap.W[l]);
      this.layers[l].b.set(snap.b[l]);
    }
    if (this.skipW && snap.skipW) {
      this.skipW.set(snap.skipW);
    }
  }
}

export function runNeuralModel(
  bars: readonly PriceBar[],
  rawConfig: NeuralConfig,
  holdoutSize = 25,
  forecastHorizon = 30,
): ModelRunResult {
  const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const config = sanitizeNeuralConfig(rawConfig);

  const closes = bars.map((b) => b.close);
  const dates = bars.map((b) => b.date);
  const dataset = buildSupervisedDataset(closes, config.lookback);

  const effectiveHoldout = Math.min(holdoutSize, Math.max(10, Math.floor(dataset.X.length * 0.18)));
  const splitIdx = dataset.X.length - effectiveHoldout;

  const trainXRaw = dataset.X.slice(0, splitIdx);
  const trainYRaw = dataset.y.slice(0, splitIdx);
  const valXRaw = dataset.X.slice(splitIdx);
  const valYRaw = dataset.y.slice(splitIdx);
  const valTargetIndices = dataset.targetIndex.slice(splitIdx);

  // Reserve last 15% of training window as internal early-stopping validation set (zero holdout leakage!)
  const innerValSize = Math.max(12, Math.floor(trainXRaw.length * 0.16));
  const innerSplit = Math.max(20, trainXRaw.length - innerValSize);

  const scaler = fitScaler(trainXRaw, trainYRaw);
  const allTrainX = trainXRaw.map((x) => transformFeatures(x, scaler));
  const allTrainY = trainYRaw.map((y) => transformTarget(y, scaler));

  const fitX = allTrainX.slice(0, innerSplit);
  const fitY = allTrainY.slice(0, innerSplit);
  const earlyValX = allTrainX.slice(innerSplit);
  const earlyValY = allTrainY.slice(innerSplit);

  const valX = valXRaw.map((x) => transformFeatures(x, scaler));

  const seedHash = config.id
    .split('')
    .reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) >>> 0, config.seed ?? 42);

  const mlp = new FeedforwardMLP(
    fitX[0].length,
    config.hiddenLayers,
    config.activation,
    config.residualSkip,
    seedHash,
  );

  // Effective weight decay keeps noisy financial return curves smooth and generalizable
  const effectiveL2 = Math.max(config.l2, 0.008);
  let bestValLoss = Number.POSITIVE_INFINITY;
  let bestWeights = mlp.snapshotWeights();

  const lossHistory: EpochLossPoint[] = [];
  for (let epoch = 1; epoch <= config.epochs; epoch++) {
    const trainLoss = mlp.trainEpoch(fitX, fitY, config.learningRate, effectiveL2);
    const earlyLoss = mlp.evaluateMse(earlyValX, earlyValY);
    if (earlyLoss < bestValLoss) {
      bestValLoss = earlyLoss;
      bestWeights = mlp.snapshotWeights();
    }
    lossHistory.push({
      epoch,
      trainLoss: Number(trainLoss.toFixed(4)),
      valLoss: Number(earlyLoss.toFixed(4)),
    });
  }

  // Restore best-generalizing internal checkpoint before scoring on untouched out-of-sample holdout
  mlp.restoreWeights(bestWeights);

  // Out-of-sample one-step holdout predictions in price space
  const holdoutDates: string[] = [];
  const holdoutActual: number[] = [];
  const holdoutPredicted: number[] = [];

  for (let i = 0; i < valX.length; i++) {
    const targetDay = valTargetIndices[i];
    const prevPrice = closes[targetDay - 1];
    const predNorm = mlp.predict(valX[i]);
    const predRet = clamp(inverseTransformTarget(predNorm, scaler), -0.045, 0.045);
    const predPrice = prevPrice * Math.exp(predRet);

    holdoutDates.push(dates[targetDay]);
    holdoutActual.push(closes[targetDay]);
    holdoutPredicted.push(Number(predPrice.toFixed(2)));
  }

  const priorClose = closes[valTargetIndices[0] - 1];
  const metrics = computeMetrics(holdoutActual, holdoutPredicted, priorClose);

  // Iterative multi-step future forecast using full historical return buffer
  const rollingReturns = computeLogReturns(closes);
  const forecastDates = generateFutureTradingDates(dates[dates.length - 1], forecastHorizon);
  const forecastValues: number[] = [];
  let runningPrice = closes[closes.length - 1];

  for (let h = 0; h < forecastHorizon; h++) {
    const rawFeat = extractFeatureVector(rollingReturns, rollingReturns.length, config.lookback);
    const normFeat = transformFeatures(rawFeat, scaler);
    const predNorm = mlp.predict(normFeat);
    const rawPredRet = inverseTransformTarget(predNorm, scaler);
    // Gentle horizon damping keeps long-horizon neural rollouts stable
    const damp = Math.pow(0.975, h);
    const driftMean = scaler.targetMean * 0.65;
    const stepRet = clamp(driftMean + (rawPredRet - driftMean) * damp, -0.028, 0.028);

    runningPrice *= Math.exp(stepRet);
    forecastValues.push(Number(runningPrice.toFixed(2)));
    rollingReturns.push(stepRet);
  }

  const bands = buildConfidenceBands(forecastValues, holdoutActual, holdoutPredicted);
  const endedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();

  return {
    modelId: config.id,
    modelName: config.name,
    family: 'neural',
    architectureLabel: formatArchitectureLabel(config),
    isCustom: Boolean(config.isCustom),
    neuralConfig: config,
    trainedAt: new Date().toISOString(),
    trainingMs: Math.max(1, Math.round(endedAt - startedAt)),
    metrics,
    holdoutDates,
    holdoutActual,
    holdoutPredicted,
    forecastDates,
    forecastValues,
    ...bands,
    lossHistory,
  };
}

// Online learning algorithms for asset selection, plus the backtest that runs them.
//
// Each round t the learner commits to a distribution p_t over K assets, then the
// period's gross returns x_t (price ratio) are revealed. Bandit learners observe
// only the return of the asset they picked; full-information learners see all.
//
// Learners that pick a single asset are scored by expected log-wealth,
//   L_T = Σ_t E[ log x_{I_t,t} ] = Σ_t p_t · log x_t,
// and their regret against the best single asset in hindsight is
//   R_T = max_i Σ_t log x_{i,t} − L_T.
// Portfolio learners (EG, equal weight) hold p_t as weights, earn log(p_t · x_t)
// and are measured against the same benchmark.

(function (root) {
  'use strict';

  // ---------- helpers ----------

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sample(p, rand) {
    let u = rand(), acc = 0;
    for (let i = 0; i < p.length; i++) { acc += p[i]; if (u < acc) return i; }
    return p.length - 1;
  }

  // p_i ∝ exp(eta * s_i), computed stably
  function softmax(s, eta) {
    let m = -Infinity;
    for (const v of s) if (v > m) m = v;
    const w = s.map(v => Math.exp(eta * (v - m)));
    const z = w.reduce((a, b) => a + b, 0);
    return w.map(v => v / z);
  }

  function argmax(v) {
    let best = 0;
    for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
    return best;
  }

  function oneHot(K, i) { const p = new Array(K).fill(0); p[i] = 1; return p; }

  // Bernoulli KL divergence
  function klBern(p, q) {
    const e = 1e-12;
    p = Math.min(Math.max(p, e), 1 - e);
    q = Math.min(Math.max(q, e), 1 - e);
    return p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
  }

  // ---------- learners ----------
  // Interface: probs(t) → distribution for round t (1-indexed);
  //            update(t, chosen, r, x) with r ∈ [0,1]^K scaled rewards, x gross returns.
  //            Bandit learners must only read r[chosen].

  // UCB1 (Auer, Cesa-Bianchi & Fischer, 2002)
  function UCB1(K) {
    const n = new Array(K).fill(0), s = new Array(K).fill(0);
    return {
      probs(t) {
        for (let i = 0; i < K; i++) if (n[i] === 0) return oneHot(K, i);
        const idx = n.map((ni, i) => s[i] / ni + Math.sqrt(2 * Math.log(t) / ni));
        return oneHot(K, argmax(idx));
      },
      update(t, i, r) { n[i] += 1; s[i] += r[i]; },
    };
  }

  // KL-UCB (Garivier & Cappé, 2011), Bernoulli KL, exploration ln t
  function KLUCB(K) {
    const n = new Array(K).fill(0), s = new Array(K).fill(0);
    function index(i, t) {
      const mu = s[i] / n[i], bound = Math.log(t) / n[i];
      let lo = mu, hi = 1;
      for (let k = 0; k < 30; k++) {
        const mid = (lo + hi) / 2;
        if (klBern(mu, mid) > bound) hi = mid; else lo = mid;
      }
      return lo;
    }
    return {
      probs(t) {
        for (let i = 0; i < K; i++) if (n[i] === 0) return oneHot(K, i);
        const idx = [];
        for (let i = 0; i < K; i++) idx.push(index(i, t));
        return oneHot(K, argmax(idx));
      },
      update(t, i, r) { n[i] += 1; s[i] += r[i]; },
    };
  }

  // EXP3 with anytime learning rate and the loss-based importance-weighted
  // estimator (Lattimore & Szepesvári, Bandit Algorithms, ch. 11)
  function EXP3(K) {
    const L = new Array(K).fill(0); // cumulative estimated losses
    let p = new Array(K).fill(1 / K);
    return {
      probs(t) {
        const eta = Math.sqrt(Math.log(K) / (t * K));
        p = softmax(L.map(v => -v), eta);
        return p;
      },
      update(t, i, r) { L[i] += (1 - r[i]) / p[i]; },
    };
  }

  // EXP3++ (Seldin & Slivkins, 2014): EXP3 plus gap-dependent exploration,
  // which gives O(√(KT)) adversarial regret and O(log² T) stochastic regret.
  function EXP3PP(K, c = 18) {
    const L = new Array(K).fill(0);
    let p = new Array(K).fill(1 / K);
    return {
      probs(t) {
        const beta = 0.5 * Math.sqrt(Math.log(K) / (t * K));
        const minL = Math.min(...L);
        const eps = L.map(Li => {
          const gap = t > 1 ? Math.min(1, (Li - minL) / (t - 1)) : 1;
          const xi = gap > 0 ? c * Math.log(t) ** 2 / (t * gap * gap) : Infinity;
          return Math.min(1 / (2 * K), beta, xi);
        });
        const rho = softmax(L.map(v => -v), beta);
        const sumEps = eps.reduce((a, b) => a + b, 0);
        p = rho.map((r, a) => (1 - sumEps) * r + eps[a]);
        return p;
      },
      update(t, i, r) { L[i] += (1 - r[i]) / p[i]; },
    };
  }

  // Hedge / exponential weights with full information, anytime rate √(8 ln K / t)
  function Hedge(K) {
    const S = new Array(K).fill(0);
    return {
      probs(t) { return softmax(S, Math.sqrt(8 * Math.log(K) / t)); },
      update(t, i, r) { for (let a = 0; a < K; a++) S[a] += r[a]; },
    };
  }

  // Exponentiated Gradient portfolio (Helmbold, Schapire, Singer & Warmuth, 1998).
  // Gradient of log-wealth: x_i / (w · x). Centring it at 1 does not change the update.
  function EG(K, eta) {
    let w = new Array(K).fill(1 / K);
    return {
      probs() { return w; },
      update(t, i, r, x) {
        const wx = w.reduce((a, wi, k) => a + wi * x[k], 0);
        const v = w.map((wi, k) => wi * Math.exp(eta * (x[k] / wx - 1)));
        const z = v.reduce((a, b) => a + b, 0);
        w = v.map(vi => vi / z);
      },
    };
  }

  function EqualWeight(K) {
    const w = new Array(K).fill(1 / K);
    return { probs() { return w; }, update() {} };
  }

  // ---------- catalogue ----------

  const ALGORITHMS = [
    { id: 'ucb1',  name: 'UCB1',   feedback: 'bandit', randomized: false, holds: 'pick',
      make: K => UCB1(K) },
    { id: 'klucb', name: 'KL-UCB', feedback: 'bandit', randomized: false, holds: 'pick',
      make: K => KLUCB(K) },
    { id: 'exp3',  name: 'EXP3',   feedback: 'bandit', randomized: true,  holds: 'pick',
      make: K => EXP3(K) },
    { id: 'exp3pp', name: 'EXP3++', feedback: 'bandit', randomized: true, holds: 'pick',
      make: K => EXP3PP(K) },
    { id: 'hedge', name: 'Hedge',  feedback: 'full',   randomized: false, holds: 'pick',
      make: K => Hedge(K) },
    { id: 'eg',    name: 'EG',     feedback: 'full',   randomized: false, holds: 'portfolio',
      make: K => EG(K, EG_ETA) },
  ];

  const EG_ETA = 0.5;

  const BENCHMARKS = [
    { id: 'equal', name: 'Equal weight', holds: 'portfolio', make: K => EqualWeight(K) },
  ];

  // Rebalancing frequencies. `scale` maps a log-return g to a reward in [0,1]
  // via r = clip(½ + g / (2·scale)), so ±scale is the full reward range.
  const FREQUENCIES = {
    daily:   { label: 'Daily',   scale: 0.05, runs: 40 },
    weekly:  { label: 'Weekly',  scale: 0.10, runs: 100 },
    monthly: { label: 'Monthly', scale: 0.20, runs: 200 },
  };

  // ---------- data preparation ----------

  function periodKey(dateStr, freq) {
    if (freq === 'daily') return dateStr;
    if (freq === 'monthly') return dateStr.slice(0, 7);
    // weekly: Monday of the week (UTC)
    const d = new Date(dateStr + 'T00:00:00Z');
    const dow = (d.getUTCDay() + 6) % 7;
    d.setUTCDate(d.getUTCDate() - dow);
    return d.toISOString().slice(0, 10);
  }

  // Indices of the last trading day in each period, from `start` onwards.
  function periodEnds(dates, freq, start) {
    const idx = [];
    for (let i = 0; i < dates.length; i++) {
      if (dates[i] < start) continue;
      const last = i === dates.length - 1 || periodKey(dates[i + 1], freq) !== periodKey(dates[i], freq);
      if (last) idx.push(i);
    }
    return idx;
  }

  // ---------- backtest ----------

  function backtest(data, opts) {
    const tickers = opts.tickers;
    const K = tickers.length;
    const freq = FREQUENCIES[opts.freq];
    const runs = opts.runs || freq.runs;
    const ends = periodEnds(data.dates, opts.freq, opts.start);
    const T = ends.length - 1;
    if (K < 2 || T < 2) return null;

    const prices = tickers.map(tk => data.prices[tk]);
    const X = [], G = [], R = [];
    for (let t = 1; t <= T; t++) {
      const a = ends[t - 1], b = ends[t];
      const x = prices.map(p => p[b] / p[a]);
      const g = x.map(Math.log);
      X.push(x); G.push(g);
      R.push(g.map(v => Math.min(1, Math.max(0, 0.5 + v / (2 * freq.scale)))));
    }

    const dates = ends.map(i => data.dates[i]);
    const assetLogW = tickers.map(() => [0]);
    const bestLogW = [0];
    for (let t = 0; t < T; t++) {
      let m = -Infinity;
      for (let i = 0; i < K; i++) {
        const v = assetLogW[i][t] + G[t][i];
        assetLogW[i].push(v);
        if (v > m) m = v;
      }
      bestLogW.push(m);
    }

    function run(spec) {
      const nRuns = spec.randomized ? runs : 1;
      const logW = new Float64Array(T + 1);
      const nextP = new Array(K).fill(0);
      for (let s = 0; s < nRuns; s++) {
        const rand = mulberry32(1234567 + 7919 * s);
        const alg = spec.make(K);
        let L = 0;
        for (let t = 1; t <= T; t++) {
          const p = alg.probs(t);
          const x = X[t - 1], g = G[t - 1];
          if (spec.holds === 'portfolio') {
            let wx = 0;
            for (let i = 0; i < K; i++) wx += p[i] * x[i];
            L += Math.log(wx);
          } else {
            for (let i = 0; i < K; i++) L += p[i] * g[i];
          }
          alg.update(t, sample(p, rand), R[t - 1], x);
          logW[t] += L / nRuns;
        }
        const pNext = alg.probs(T + 1);
        for (let i = 0; i < K; i++) nextP[i] += pNext[i] / nRuns;
      }
      const regret = Array.from(logW, (v, t) => bestLogW[t] - v);
      return { id: spec.id, name: spec.name, logWealth: Array.from(logW), regret, next: nextP };
    }

    return {
      T, K, tickers, dates,
      years: (Date.parse(dates[T]) - Date.parse(dates[0])) / (365.25 * 864e5),
      assets: tickers.map((tk, i) => ({ ticker: tk, logWealth: assetLogW[i] })),
      bestLogWealth: bestLogW,
      bestAsset: tickers[argmax(tickers.map((_, i) => assetLogW[i][T]))],
      algorithms: ALGORITHMS.map(run),
      benchmarks: BENCHMARKS.map(b => run(Object.assign({ randomized: false }, b))),
    };
  }

  const api = { ALGORITHMS, BENCHMARKS, FREQUENCIES, backtest, periodEnds,
                UCB1, KLUCB, EXP3, EXP3PP, Hedge, EG, mulberry32 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.OnlinePortfolio = api;
})(typeof window !== 'undefined' ? window : globalThis);

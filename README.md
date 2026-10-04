# Learning a portfolio online: bandits on real markets

A single-page interactive demo in which online learning algorithms decide, period by period, which assets to hold. They learn only from the returns they observe. On every visit the page loads the latest market data and replays the full history, then plots each learner's wealth and its regret against the best asset in hindsight.

## Algorithms

Bandit feedback (the learner only sees the return of the asset it holds):

- **UCB1**: Auer, Cesa-Bianchi & Fischer (2002).
- **KL-UCB**: Garivier & Cappé (2011), with a Bernoulli KL index.
- **EXP3**: Auer et al. (2002), with an anytime learning rate and the loss-based importance-weighted estimator.
- **EXP3++**: Seldin & Slivkins (2014), which adds gap-dependent exploration to get best-of-both-worlds guarantees.

Full information (the learner sees every asset's return):

- **Hedge**: exponential weights.
- **EG**: Exponentiated Gradient portfolio, Helmbold et al. (1998). It holds a diversified, rebalanced portfolio.

Benchmarks: the best single asset in hindsight (the regret benchmark), an equal-weight rebalanced portfolio, and buy-and-hold for each asset.

## Assets

SPY (S&P 500), GLD (gold), TLT and IEF (US Treasuries), AAPL, MSFT, NVDA, JPM and XOM, all as daily total-return prices from 2010 onwards. To change the universe, edit `ASSETS` in `scripts/fetch_prices.py` and re-run it.

## How it works

- `data/prices.json` holds daily adjusted closes. A GitHub Action (`.github/workflows/update-data.yml`) refreshes it after every US trading day by running `scripts/fetch_prices.py` (yfinance) and committing the result.
- `algorithms.js` contains the learners and the backtest. It runs in the browser, so changing the rebalancing frequency, the start year or the asset set recomputes everything instantly.
- `index.html` fetches the data, runs the backtest and draws the charts with Chart.js. The current settings are kept in the URL, so a link reproduces the same view.

Rewards for the bandits are log-returns mapped affinely to [0,1] (±5% daily, ±10% weekly, ±20% monthly span the range). Single-asset learners are scored by expected log-wealth, and EXP3/EXP3++ are averaged over many seeded runs.

## Running locally

```bash
python3 -m http.server
# open http://localhost:8000
```

The page has to be served over HTTP, not opened as a file, because it fetches `data/prices.json`.

## Link

You can try the demo [here](https://anderartola.github.io/online-portfolio-demo/).

## License

MIT.

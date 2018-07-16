# スクリプトの説明
## calculate\_predictions.py
事後確率を用いて曲線を予測する。

## calculate\_predictions\_with\_multinormal.py
事後確率を用いて曲線を予測する。事前確率を多次元ガウスにしたもの。

## params.py
各種パラメータを設定する。

## sample.py
MCMCを用いてベイス学習する。

## sample\_with\_multinormal.py
MCMCを用いてベイス学習する。事前確率を多次元ガウスにしたもの。

## utils.py
共有関数を提供する。

## visualize\_posteriors.py
事後確率を描画する。

## visualize\_posteriors\_with\_multinormal.py
事後確率を描画する。事前確率を多次元ガウスにしたもの。

## visualize\_predictions.py
予測した曲線を描画する。

## visualize\_std.py
予測した標準偏差を描画する。

# 実行手順
## 事前確率を1次元ガウス関数の配列で定義したとき
- sample.py: ベイズ学習する。
- visualize\_posteriors.py: 事後確率を描画する。
- calculate\_predictions.py: 事後確率を用いて予測値を計算する。
- visualize\_predictions.py: 予測曲線を描画する。
- visualize\_std.py: 標準偏差を描画する。

## 事前確率を多次元ガウス関数で定義したとき
- sample\_with\_multinormal.py: ベイズ学習する。
- visualize\_posteriors\_with\_multinormal.py: 事後確率を描画する。
- calculate\_predictions\_with\_multinormal.py: 事後確率を用いて予測値を計算する。
- visualize\_predictions.py: 予測曲線を描画する。
- visualize\_std.py: 標準偏差を描画する。

# メモ
## 事前確率を1次元ガウス関数の配列で定義したとき
M=8としたときのパラメータは以下の通り。
- DATASET_PATH = './dataset.txt'
- M = 8
- ALPHA = 0.1
- SIGMA = 0.015
- TAU = 1 / SIGMA\*\*2
- ITER = 40000000
- THIN = 2000
- BURN = ITER // 2
- PICKLE\_NAME = 'linear\_regression.pkl'
- XCOUNT = 50
- XMIN = 0
- XMAX = 4
- YMEANS\_PATH = 'ymeans.npy'
- YSTDS\_PATH = 'ystds.npy'
- IXS\_PATH = 'ixs.npy'
- RESULT\_PNG\_PATH = 'bayes.png'

計算時間は419m5.323s

## 事前確率を多次元ガウス関数の配列で定義したとき
M=8のとき

### trial-1
- DATASET_PATH = './dataset.txt'
- M = 8
- ALPHA = 0.1
- SIGMA = 0.015
- TAU = 1 / SIGMA\*\*2
- ITER = 40000000
- THIN = 2000
- BURN = ITER // 2
- PICKLE\_NAME = 'linear\_regression.pkl'
- XCOUNT = 50
- XMIN = 0
- XMAX = 4
- YMEANS\_PATH = 'ymeans.npy'
- YSTDS\_PATH = 'ystds.npy'
- IXS\_PATH = 'ixs.npy'
- RESULT\_PNG\_PATH = 'bayes.png'

計算時間は24m。

### trial-2
- DATASET\_PATH = './dataset.txt'
- M = 8
- ALPHA = 0.1
- SIGMA = 0.015
- TAU = 1 / SIGMA\*\*2
- ITER = 80000000
- THIN = 4000
- BURN = ITER // 2
- OUT\_DIR\_PATH = 'results'
- PICKLE\_PATH = os.path.join(OUT\_DIR\_PATH, 'linear\_regression.pkl')
- CSV\_PATH = os.path.join(OUT\_DIR\_PATH, 'linear\_regression.csv')
- XCOUNT = 50
- XMIN = 0
- XMAX = 4
- YMEANS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ymeans.npy')
- YSTDS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ystds.npy')
- IXS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ixs.npy')
- RESULT\_PNG\_PATH = os.path.join(OUT\_DIR\_PATH, 'bayes.png')
- YPREDICTIONS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ypredictions.npy')

計算時間は53m。

### trial-3
- DATASET\_PATH = './dataset.txt'
- M = 8
- ALPHA = 0.1
- SIGMA = 0.015
- TAU = 1 / SIGMA\*\*2
- ITER = 160000000
- THIN = 8000
- BURN = ITER // 2
- OUT\_DIR\_PATH = 'results'
- PICKLE\_PATH = os.path.join(OUT\_DIR\_PATH, 'linear\_regression.pkl')
- CSV\_PATH = os.path.join(OUT\_DIR\_PATH, 'linear\_regression.csv')
- XCOUNT = 50
- XMIN = 0
- XMAX = 4
- YMEANS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ymeans.npy')
- YSTDS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ystds.npy')
- IXS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ixs.npy')
- RESULT\_PNG\_PATH = os.path.join(OUT\_DIR\_PATH, 'bayes.png')
- YPREDICTIONS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ypredictions.npy')

計算時間は92m。

### trial-4
- DATASET\_PATH = './dataset.txt'
- M = 8
- ALPHA = 0.1
- SIGMA = 0.015
- TAU = 1 / SIGMA\*\*2
- ITER = 50000000
- THIN = 2500
- BURN = ITER // 2
- OUT\_DIR\_PATH = 'results'
- PICKLE\_PATH = os.path.join(OUT\_DIR\_PATH, 'linear\_regression\_trial-4.pkl')
- CSV\_PATH = os.path.join(OUT\_DIR\_PATH, 'linear\_regression\_trial-4.csv')
- XCOUNT = 50
- XMIN = 0
- XMAX = 4
- YMEANS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ymeans.npy')
- YSTDS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ystds.npy')
- IXS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ixs.npy')
- RESULT\_PNG\_PATH = os.path.join(OUT\_DIR\_PATH, 'bayes.png')
- YPREDICTIONS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ypredictions.npy')
- ANSWER\_PATH = os.path.join(OUT\_DIR\_PATH, 'answer\_curve.txt')

計算時間は30m。

### trial-5
- DATASET\_PATH = './dataset.txt'
- M = 8
- ALPHA = 0.1
- SIGMA = 0.015
- TAU = 1 / SIGMA]\*\*2
- ITER = 60000000
- THIN = 3000
- BURN = ITER // 2
- OUT\_DIR\_PATH = 'results'
- PICKLE\_PATH = os.path.join(OUT\_DIR\_PATH, 'linear\_regression\_trial-5.pkl')
- CSV\_PATH = os.path.join(OUT\_DIR\_PATH, 'linear\_regression\_trial-5.csv')
- XCOUNT = 50
- XMIN = 0
- XMAX = 4
- YMEANS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ymeans.npy')
- YSTDS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ystds.npy')
- IXS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ixs.npy')
- RESULT\_PNG\_PATH = os.path.join(OUT\_DIR\_PATH, 'bayes.png')
- YPREDICTIONS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ypredictions.npy')
- ANSWER\_PATH = os.path.join(OUT\_DIR\_PATH, 'answer\_curve.txt')

### trial-6
- DATASET\_PATH = './dataset.txt'
- M = 8
- ALPHA = 0.1
- SIGMA = 0.015
- TAU = 1 / SIGMA\*\*2
- ITER = 70000000
- THIN = 3500
- BURN = ITER // 2
- OUT\_DIR\_PATH = 'results'
- PICKLE\_PATH = os.path.join(OUT\_DIR\_PATH, 'linear\_regression\_trial-6.pkl')
- CSV\_PATH = os.path.join(OUT\_DIR\_PATH, 'linear\_regression\_trial-6.csv')
- XCOUNT = 50
- XMIN = 0
- XMAX = 4
- YMEANS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ymeans.npy')
- YSTDS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ystds.npy')
- IXS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ixs.npy')
- RESULT\_PNG\_PATH = os.path.join(OUT\_DIR\_PATH, 'bayes.png')
- YPREDICTIONS\_PATH = os.path.join(OUT\_DIR\_PATH, 'ypredictions.npy')
- ANSWER\_PATH = os.path.join(OUT\_DIR\_PATH, 'answer\_curve.txt')

44m0.329s。これが一番厳密解に近い。

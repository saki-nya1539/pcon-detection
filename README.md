# Pコン跡の画像認識（SDNET2018 / W）

256x256 のコンクリート壁面画像に Pコン跡が含まれるかを判定する（1: あり / 0: なし）。

## 動作環境（スコアを出した環境）
- Google Colab（GPU: NVIDIA T4）
- Python 3.12、PyTorch、timm、pandas、scikit-learn、Pillow、NumPy（`requirements.txt`）
- ローカル PC でも動作可。GPU が無い場合は自動で軽量設定（EfficientNet-B0 / 256px / 3fold×6epoch）に切り替わる（スコアは下がる）

## 必要なファイル（`pcon_detect.py` と同じフォルダ、Colab なら /content に置く）
- `pcon_detect.py`
- `train_label.csv`、`test.txt`（コンテスト配布物）
- SDNET2018 の zip（https://digitalcommons.usu.edu/all_datasets/48/ の Download。例: `DATA_Maguire_20180517_ALL.zip`）。展開済みフォルダでも可

## 実行方法
1. Colab で「ランタイム → ランタイムのタイプを変更 → T4 GPU」
2. 上記ファイルを /content にアップロード
3. `pcon_detect.py` の中身をセルに貼って実行（ローカルなら `python pcon_detect.py`）
4. 約40分で以下が出力される
   - `submission.csv` … 提出ファイル（しきい値 0.30、ヘッダなし、test.txt の順）
   - `submission_oof_threshold.csv` … 交差検証で F1 最大のしきい値による予測（参考）
   - `oof_predictions.csv` … 学習画像の交差検証予測（誤り分析用）
   - `test_prob.npy` … 評価用画像の予測確率

## 手法の概要
- モデル: ImageNet 事前学習済み EfficientNet-B3（timm `tf_efficientnet_b3.ns_jft_in1k`）を 288px で追加学習
- 検証: 撮影元の写真 ID 単位の StratifiedGroupKFold（3 分割）
- 不均衡対策: 1 バッチの 25% が陽性になるようオーバーサンプリング
- 拡張: 左右・上下反転、90 度回転、明るさ・コントラスト（切り抜きは使わない）
- 推論: 3 fold のアンサンブル + 4 種の TTA
- しきい値: 0.30

GPU 演算の非決定性により、再実行時のスコアは小数点以下でわずかに変動する場合がある。

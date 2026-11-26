# =====================================================================
# Pコン跡 検出（SDNET2018 / W のみ）  VS Code（ローカルPC）でも Google Colab でも動く版
#
#   このスクリプトと同じフォルダに置くもの:
#     - SDNET2018 の zip（約500MB）か、それを展開したフォルダ（中に W/CW, W/UW がある）
#     - train_label.csv, test.txt（コンテスト配布物）
#
#   事前準備（ローカルPCのターミナルで1回だけ）:
#     NVIDIA GPU あり: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#     GPU なし / Mac : pip install torch torchvision
#     共通           : pip install timm pandas scikit-learn pillow
#
#   実行: VS Code でこのファイルを開いて右上の ▶（Run Python File）
#   出力: 同じフォルダに submission.csv（ヘッダなし、test.txt の順）
#         oof_predictions.csv（学習画像の交差検証予測：誤り分析用）
# =====================================================================

import os, sys, glob, zipfile, random, time
try:
    import timm  # noqa
except ImportError:  # Colab 等で timm が無ければ入れる
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
import numpy as np, pandas as pd
import torch, torch.nn as nn
import timm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

# ---------- 設定 ----------
try:
    WORK = os.path.dirname(os.path.abspath(__file__))   # スクリプトを置いたフォルダ
except NameError:
    WORK = "/content" if os.path.exists("/content") else os.getcwd()   # Colab / Jupyter
SDNET_ZIP   = None            # 例 r"C:\Users\me\Downloads\SDNET2018.zip"。None なら WORK 内の zip を自動探索
MODEL_NAME  = "tf_efficientnet_b3.ns_jft_in1k"   # 速さ優先なら "efficientnet_b0.ra_in1k"
IMG_SIZE    = 288             # 256 のままでも可。少し拡大すると小さい穴が見やすい
N_FOLDS     = 3
EPOCHS      = 10
BATCH       = 48
LR          = 3e-4
POS_RATIO   = 0.25            # 1バッチ中の陽性の割合（オーバーサンプリング）
SEED        = 42
QUICK       = False           # True にすると 2fold×2epoch で動作確認だけ
FINAL_THRESHOLD = 0.30        # 最終提出に使った判定しきい値（評価用画像では確信度が低めに出るため OOF 最適値より下げた）

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"            # Apple シリコンの Mac
else:
    device = "cpu"
if device != "cuda":
    # GPU が無いと重いモデルは数時間かかるので軽量設定に自動で切り替え
    MODEL_NAME, IMG_SIZE, N_FOLDS, EPOCHS, BATCH = "efficientnet_b0.ra_in1k", 256, 3, 6, 32
if QUICK:
    N_FOLDS, EPOCHS = 2, 2
print("device:", device, "| model:", MODEL_NAME, IMG_SIZE, f"{N_FOLDS}fold x {EPOCHS}ep")

# ---------- 1. データ展開 ----------
def find_root():
    hits = glob.glob(f"{WORK}/**/W/CW", recursive=True)
    return os.path.dirname(os.path.dirname(hits[0])) if hits else None

root = find_root()
if root is None:
    zips = [p for p in glob.glob(os.path.join(WORK, "*.zip")) if os.path.getsize(p) > 100e6]
    if SDNET_ZIP is None and not zips:
        sys.exit(f"SDNET2018 が見つかりません。zip か展開済みフォルダを {WORK} に置いてください")
    zp = SDNET_ZIP or zips[0]
    print("unzip:", zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(f"{WORK}/sdnet")
    root = find_root()
    if root is None:   # zip の中にさらに zip がある場合
        for inner in glob.glob(f"{WORK}/sdnet/**/*.zip", recursive=True):
            with zipfile.ZipFile(inner) as z:
                z.extractall(os.path.dirname(inner))
        root = find_root()
print("SDNET root:", root)

train_df = pd.read_csv(f"{WORK}/train_label.csv", header=None, names=["path", "label"])
test_paths = [l.strip() for l in open(f"{WORK}/test.txt") if l.strip()]
# 撮影元の写真ID（"7069-3.jpg" → "7069"）でグループ化 → 同じ写真が学習と検証にまたがらない
train_df["group"] = train_df["path"].str.split("/").str[-1].str.split("-").str[0]
print(f"train {len(train_df)} (pos {train_df.label.sum()}), test {len(test_paths)}")

# ---------- 2. 画像をメモリに読み込み（Drive/ディスクI/Oを避けて高速化） ----------
def load(p):
    im = Image.open(os.path.join(root, p)).convert("RGB")
    if im.size != (IMG_SIZE, IMG_SIZE):
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)

def load_all(paths):
    with ThreadPoolExecutor(16) as ex:
        return np.stack(list(ex.map(load, paths)))

t0 = time.time()
X_train = load_all(train_df.path.tolist())
X_test  = load_all(test_paths)
y_train = train_df.label.values.astype(np.float32)
print(f"loaded in {time.time()-t0:.0f}s", X_train.shape, X_test.shape)

# ---------- 3. 学習まわり ----------
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255

def to_tensor(xb):  # uint8 NHWC -> normalized NCHW (GPU)
    x = torch.from_numpy(xb).to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    return (x - MEAN) / STD

def augment(x):
    # Pコン跡は円形で向きに意味がない → 反転・90度回転は安全。
    # 切り抜き系は端のPコン跡を消してラベルを壊すので使わない。
    if random.random() < 0.5: x = x.flip(3)
    if random.random() < 0.5: x = x.flip(2)
    k = random.randint(0, 3)
    if k: x = torch.rot90(x, k, (2, 3))
    b = x.shape[0]
    x = x * (1 + 0.2 * (torch.rand(b, 1, 1, 1, device=device) - 0.5)) \
          + 0.3 * (torch.rand(b, 1, 1, 1, device=device) - 0.5)          # 明るさ・コントラスト
    return x

def batches_balanced(idx_pos, idx_neg, n_steps):
    n_pos = max(1, int(BATCH * POS_RATIO))
    for _ in range(n_steps):
        b = np.concatenate([np.random.choice(idx_pos, n_pos),
                            np.random.choice(idx_neg, BATCH - n_pos, replace=False)])
        np.random.shuffle(b)
        yield b

@torch.no_grad()
def predict(model, X, tta=True):
    model.eval(); out = []
    for i in range(0, len(X), 128):
        x = to_tensor(X[i:i+128])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            views = [x, x.flip(3), x.flip(2), torch.rot90(x, 1, (2, 3))] if tta else [x]
            p = torch.stack([torch.sigmoid(model(v).float().squeeze(1)) for v in views]).mean(0)
        out.append(p.cpu().numpy())
    return np.concatenate(out)

def best_threshold(y, p):
    ths = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, (p >= t).astype(int)) for t in ths]
    i = int(np.argmax(f1s)); return ths[i], f1s[i]

# ---------- 4. 交差検証で学習 ----------
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train_df)); test_pred = np.zeros(len(test_paths))

for fold, (tr, va) in enumerate(sgkf.split(X_train, y_train, train_df.group)):
    seed_all(SEED + fold)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps_per_epoch = len(tr) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps_per_epoch, pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    pos, neg = tr[y_train[tr] == 1], tr[y_train[tr] == 0]
    best_f1, best_state, best_p = -1, None, None

    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = 0
        for b in batches_balanced(pos, neg, steps_per_epoch):
            x = augment(to_tensor(X_train[b])); y = torch.from_numpy(y_train[b]).to(device)
            y = y * 0.95 + 0.025                                      # 軽いラベルスムージング
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = lossf(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        p = predict(model, X_train[va], tta=False)
        th, f1 = best_threshold(y_train[va], p)
        print(f"fold{fold} ep{ep:2d} loss {tot/steps_per_epoch:.4f}  valF1 {f1:.4f} (th {th:.2f})  {time.time()-t0:.0f}s")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    oof[va] = predict(model, X_train[va])
    test_pred += predict(model, X_test) / N_FOLDS
    print(f"== fold{fold} best valF1 {best_f1:.4f}")
    del model, opt; torch.cuda.empty_cache()

# ---------- 5. しきい値を全OOFで決めて提出ファイル作成 ----------
th, f1 = best_threshold(y_train, oof)
print(f"\nOOF F1 = {f1:.4f}  at threshold {th:.2f}")
pd.DataFrame({"path": train_df.path, "label": y_train.astype(int), "prob": oof}) \
  .to_csv(f"{WORK}/oof_predictions.csv", index=False)

np.save(f"{WORK}/test_prob.npy", test_pred)
# 参考: OOF で F1 最大となるしきい値での予測
pd.DataFrame({"path": test_paths, "pred": (test_pred >= th).astype(int)}) \
  .to_csv(f"{WORK}/submission_oof_threshold.csv", header=False, index=False)
# 最終提出ファイル（しきい値 FINAL_THRESHOLD）
pred = (test_pred >= FINAL_THRESHOLD).astype(int)
pd.DataFrame({"path": test_paths, "pred": pred}).to_csv(f"{WORK}/submission.csv", header=False, index=False)
print(f"submission.csv (threshold {FINAL_THRESHOLD}): {len(pred)} rows, predicted positives {pred.sum()} ({pred.mean():.2%})")

# AI 間取り分析

ローカルのOllama/Qwen3-VLで間取り画像の空間構造をJSON化し、その結果と `knowledge.md` 全文を使って固定の100点採点表で評価するアプリです。

処理の流れは次のとおりです。

1. 部屋・扉・窓・隣接関係を画像から構造化する。
2. 初回構造を保持したまま、各接続だけを `accept / reject / uncertain` で再照合する。
3. 構造化結果だけを画像上の事実として採点JSONを生成する。
4. Pythonが不自然な接続、根拠ID、確認状態、項目別上限を検証する。
5. Pythonが確定点とMarkdownレポートを生成する。

確認不能な項目は満点にできません。玄関・廊下とバルコニーの直結、一般居室とトイレ・浴室の直結、浴室と廊下の直結、重複境界などは通行経路から機械的に除外されます。

## セットアップ

1. 仮想環境を作成し、`pip install -r requirements.txt` を実行します。
2. Ollamaで `ollama pull qwen3-vl:4b-instruct` を実行します。
3. `streamlit run apps/streamlit_app.py` で起動します。

標準設定は `.env` の `ANALYSIS_PROVIDER=ollama`、`OLLAMA_MODEL=qwen3-vl:4b-instruct`、`OLLAMA_NUM_CTX=16384` です。4Bモデルの重複誤認を避けるため `OLLAMA_MAX_IMAGES=1` で高解像度の全体図だけを渡します。Geminiを予備に使う場合だけ `ENABLE_GEMINI_FALLBACK=true` と有効な `GEMINI_API_KEY` を設定します。

8Bでも、初回構造と `knowledge.md` 全文を採点へ渡すため `OLLAMA_NUM_CTX=16384` を推奨します。

## テスト

```powershell
python -m unittest discover -s tests -v
```

単体テストは外部APIを呼び出さず、知識全文の組み込み、採点配分、呼び出し形式、異常系を検証します。

`floor_photo` 内の画像をローカルQwenで分析する場合は、Ollamaを起動した状態で次を実行します。

```powershell
python live_backtest.py
```

任意の画像フォルダも直接指定できます。たとえば、ローカルの評価セット
`floor_sample` を分析する場合は次のとおりです。元画像はGitに追加せず、
結果だけを `backtest_results/floor_sample` に保存します。

```powershell
python live_backtest.py --input-dir floor_sample --results-dir backtest_results/floor_sample
```

初回実行では `.analysis_cache` に検証済みの構造を保存します。同じ画像・モデル・抽出設定では、2回目以降は画像認識を省略して採点だけを実行します。`knowledge.md`だけを変更した場合も構造キャッシュを再利用します。

画像認識をやり直す場合:

```powershell
python live_backtest.py --refresh-cache
```

キャッシュを一切使わない場合:

```powershell
python live_backtest.py --no-cache
```

採点用JSONは、検証済み構造からbbox、扉座標、拒否済み接続、冗長な画像根拠を除いた圧縮形式です。元の完全な構造JSONは結果ファイルとキャッシュに保持されます。

正解データとの接続精度は次で測定できます。

```powershell
python tools/evaluate_ground_truth.py "backtest_results/シャルマンフジ住之江公園ドシール__qwen3-vl_8b-instruct-q4_K_M.structure.json" "ground_truth/シャルマンフジ住之江公園ドシール.json"
```

結果ファイル名にはモデル名が付くため、4Bと8Bの結果は上書きされません。

## ローカル評価セットの正解データ化

`floor_sample` のようなローカル画像セットは、まず分析して構造JSONを出力します。

```powershell
python live_backtest.py --input-dir floor_sample --results-dir backtest_results/floor_sample
```

その後、構造JSONを下書きにした正解データテンプレートを生成します。

```powershell
python tools/prepare_ground_truth.py --images-dir floor_sample --ground-truth-dir ground_truth/floor_sample --structure-dir backtest_results/floor_sample
```

各 `ground_truth/floor_sample/*.json` の `connections` を元画像と照合して修正し、確認を終えたファイルだけ `review_status` を `approved` に変更します。未確認の下書きは集計対象になりません。

先に構造JSONなしでテンプレートを作成していた場合は、分析後に次のように `--overwrite` を付けて実行すると、AIの採用接続を下書きに入れ直せます。人手で修正を始めた後は上書きしません。

```powershell
python tools/prepare_ground_truth.py --images-dir floor_sample --ground-truth-dir ground_truth/floor_sample --structure-dir backtest_results/floor_sample --overwrite
```

確認済みデータを一括集計するには次を実行します。複数モデルの結果が同居する場合は、`--model` に結果ファイル名のモデル部分を指定します。

```powershell
python tools/evaluate_benchmark.py --structure-dir backtest_results/floor_sample --ground-truth-dir ground_truth/floor_sample --model qwen3-vl_8b-instruct-q4_K_M --output backtest_results/floor_sample/benchmark.json
```

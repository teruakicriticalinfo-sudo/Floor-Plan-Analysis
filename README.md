# AI 間取り分析

ローカルのOllama/Qwen3-VLで間取り画像の空間構造をJSON化し、その結果と `knowledge.md` 全文を使って固定の100点採点表で評価するアプリです。

処理の流れは次のとおりです。

1. 図面領域と階を特定する。複数階では図面を階ごとに切り出して部屋の位置を再読取する。
2. 階ごとに扉・隣接関係を読み取り、専用工程で窓と設備を再確認する。
3. 初回構造を保持したまま、各接続だけを `accept / reject / uncertain` で再照合する。
4. 構造化結果と `knowledge.md` 全文から採点JSONを生成する。
5. Pythonが階の欠落、不自然な接続、根拠ID、項目別上限を検証してレポートを生成する。

確認不能な項目は満点にできません。玄関・廊下とバルコニーの直結、一般居室とトイレ・浴室の直結、浴室と廊下の直結、重複境界などは通行経路から機械的に除外されます。
複数階の所属が曖昧な場合は採点を停止します。別階の階段同士は `vertical_links` に未検証候補として記録し、通行可能な接続とはみなしません。家具寸法、収納容量、段差などが読み取れない場合は該当項目を中立点以下に抑えます。
接続候補の大半が機械検証で拒否された場合は総合点を出さず、レポートを「採点保留」にします。構造JSONと拒否理由は保存するため、人手確認や再読取に使用できます。
階別の拡大画像から部屋を読めない場合は初回の部屋一覧を使い、該当階を記録して採点を保留します。

## セットアップ

1. 仮想環境を作成し、`pip install -r requirements.txt` を実行します。
2. Ollamaで `ollama pull qwen3-vl:4b-instruct` を実行します。
3. `streamlit run apps/streamlit_app.py` で起動します。

標準設定は `.env` の `ANALYSIS_PROVIDER=ollama`、`OLLAMA_MODEL=qwen3-vl:4b-instruct`、`OLLAMA_NUM_CTX=16384` です。4Bモデルの重複誤認を避けるため `OLLAMA_MAX_IMAGES=1` で高解像度の全体図だけを渡します。Geminiを予備に使う場合だけ `ENABLE_GEMINI_FALLBACK=true` と有効な `GEMINI_API_KEY` を設定します。8Bモデルで初回読取が10分を超える環境では、`.env` に `OLLAMA_TIMEOUT=1800`（秒）を設定します。

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

特定の1件だけを再分析する場合は、`--image` を使います。

```powershell
python live_backtest.py --input-dir floor_sample --results-dir backtest_results/floor_sample --image sample1.webp --refresh-cache
```

初回実行では `.analysis_cache` に検証済みの構造を保存します。同じ画像・モデル・抽出設定では、2回目以降は画像認識を省略して採点だけを実行します。`knowledge.md`だけを変更した場合も構造キャッシュを再利用します。
認識の途中で失敗した場合も、図面領域・階別の部屋・接続・窓設備の完了済み工程を `.analysis_cache/*.stages.json` に保存します。通常の再実行では完了済み工程を再利用します。`--refresh-cache` は古い途中保存を無視して作り直し、`--no-cache` は読み書きしません。
今回の階別読取への変更でキャッシュの版が更新されたため、変更後の初回実行は `CACHE MISS` になります。窓と設備の専用工程が増えたため、以前より実行時間は長くなる可能性があります。

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

複数階で同じ部屋名がある場合は、`floor_qualified: true` として接続の両端を `1F:廊下`、`2F:LDK` のように階付きで記録できます。画像だけでは判定できない接続は `excluded_connections` に記録すると採点から除外されます。`sample1.json` は目視した暫定下書きで、利用者が接続を確認するまでは `draft` のままです。単体の暫定比較には `python tools/evaluate_ground_truth.py <structure.json> <ground_truth.json>` を使用します。

先に構造JSONなしでテンプレートを作成していた場合は、分析後に次のように `--overwrite` を付けて実行すると、AIの採用接続を下書きに入れ直せます。人手で修正を始めた後は上書きしません。

```powershell
python tools/prepare_ground_truth.py --images-dir floor_sample --ground-truth-dir ground_truth/floor_sample --structure-dir backtest_results/floor_sample --overwrite
```

確認済みデータを一括集計するには次を実行します。複数モデルの結果が同居する場合は、`--model` に結果ファイル名のモデル部分を指定します。

```powershell
python tools/evaluate_benchmark.py --structure-dir backtest_results/floor_sample --ground-truth-dir ground_truth/floor_sample --model qwen3-vl_8b-instruct-q4_K_M --output backtest_results/floor_sample/benchmark.json
```

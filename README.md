# AI 間取り分析

ローカルのOllama/Qwen3-VLで間取り画像の空間構造をJSON化し、その結果と `knowledge.md` 全文を使って固定の100点採点表で評価するアプリです。

分析は2段階です。

1. 部屋・扉・窓・隣接関係を画像から構造化する。
2. 構造化結果だけを画像上の事実として採点JSONを生成する。
3. Pythonが不自然な接続、根拠ID、確認状態、項目別上限を検証する。
4. Pythonが確定点とMarkdownレポートを生成する。

確認不能な項目は満点にできません。玄関・廊下とバルコニーの直結、一般居室とトイレ・浴室の直結、重複境界などは通行経路から機械的に除外されます。

## セットアップ

1. 仮想環境を作成し、`pip install -r requirements.txt` を実行します。
2. Ollamaで `ollama pull qwen3-vl:4b-instruct` を実行します。
3. `streamlit run apps/streamlit_app.py` で起動します。

標準設定は `.env` の `ANALYSIS_PROVIDER=ollama`、`OLLAMA_MODEL=qwen3-vl:4b-instruct`、`OLLAMA_NUM_CTX=16384` です。4Bモデルの重複誤認を避けるため `OLLAMA_MAX_IMAGES=1` で高解像度の全体図だけを渡します。Geminiを予備に使う場合だけ `ENABLE_GEMINI_FALLBACK=true` と有効な `GEMINI_API_KEY` を設定します。

## テスト

```powershell
python -m unittest discover -s tests -v
```

単体テストは外部APIを呼び出さず、知識全文の組み込み、採点配分、呼び出し形式、異常系を検証します。

`floor_photo` 内の画像をローカルQwenで分析する場合は、Ollamaを起動した状態で次を実行します。

```powershell
python live_backtest.py
```

正解データとの接続精度は次で測定できます。

```powershell
python tools/evaluate_ground_truth.py "backtest_results/シャルマンフジ住之江公園ドシール.structure.json" "ground_truth/シャルマンフジ住之江公園ドシール.json"
```

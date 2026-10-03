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

階別の部屋再読取が失敗した画像、または高画質なのに窓・設備の結果が不足する画像は、次のように部分修復できます。過去に修復を試みた階も再試行し、途中保存に残る図面領域や正常な階の読取を再利用します。部屋一覧が更新された階の扉・接続は再読取し、窓・設備は不足した種類だけ個別に確認します。最後の接続再照合は画像を使って再実行します。修復後も主要な部屋が欠けていれば採点を保留します。通常の再実行では完成済み構造キャッシュを使い、画像認識は省略します。

```powershell
python live_backtest.py --input-dir floor_sample --results-dir backtest_results/floor_sample --image sample1.webp --repair-fallbacks
```

洗面所の取り違えや窓の見落としを調べるときは、現在の構造キャッシュを使って該当箇所だけ拡大再読取できます。洗面所は浴室との周辺領域、窓は居室の外壁に近い細帯を調べます。読取位置を元画像へ戻し、部屋外周との距離・確信度を記録した確認ページとJSONを `targeted_vision_results` に保存します。承認済みの手動補正範囲は金色の破線で表示します。候補は誤認防止のため採点・正解データには自動反映しません。画像や構造キャッシュが変わらなければ小領域の結果も再利用します。

```powershell
python targeted_vision_review.py --image sample1.webp
```

`--refresh` で小領域を再読取、`--max-window-rooms 3` で窓を調べる部屋数を制限できます。`--structure` で別の構造JSONを指定できます。現在のキャッシュがない画像は、先に `live_backtest.py` を実行してください。sample1の実測では、Qwen 8Bは小領域化後も洗面所を「確認不能」、窓を候補なしと返しました。この結果を精度改善済みと扱わず、確認ページで未検出箇所を追跡します。

sample1の窓位置を人手で確かめるための別ページは `manual_corrections/sample1_window_review.html` です。利用者提供の赤枠画像から24個の矩形を元画像へ位置合わせし、赤の窓矩形20個と紫の外部扉矩形4個として保存しました。矩形の個数は物理的な窓の枚数と同義ではありません。以前のW1〜W5・M1〜M15点指定は履歴として残し、位置の基準には赤枠を用います。窓と外部扉を混同せず、Qwenの検出結果とも分けて保存します。`manual_corrections/sample1_windows_draft.json` を修正したら次のコマンドでページを再生成できます。全窓の網羅性と同じ開口の片数はまだ未確認のため、採点や正式なベンチマークへは反映しません。

```powershell
python tools/render_window_review.py
```

確認された赤枠に対する暫定的な検出再現率だけを測る場合は、現在の画像認識キャッシュを`--structure`に指定して`tools/evaluate_window_annotations.py`を実行します。外部扉は窓の分母から外し、未確認の網羅性を理由にprecisionと正式な全窓精度は出しません。sample1では20個の窓矩形に対し、座標検証を通ったモデル窓は0件でした。

赤枠を推論に使わず、階ごとの図面領域を重なり合う小パッチに分けてQwen 8Bで窓・外部扉を再読取する実験は次で実行できます。各パッチの応答は `targeted_vision_results/*.wall_patches.json` に逐次保存され、同条件の再実行ではキャッシュを使います。赤枠との照合は全パッチの推論後に行い、JSONレポートと重ね合わせHTMLを出力します。この実験結果は通常の採点には反映しません。

```powershell
python window_patch_backtest.py
python window_patch_backtest.py --size-px 96 --stride-px 64
```

sample1の実測では標準の30パッチ・細分化した70パッチのどちらも赤枠20個との一致は0個でした。両設定でQwenが出した窓候補2件はキッチン設備付近の誤認とみられ、細長い窓記号ではないため後段で除外しました。外部扉4箇所を窓とする採用候補は0件ですが、外部扉そのものの検出も0件であり、扉識別の成功は確認できていません。小パッチ化による精度改善は確認できていないため、窓の判定を自動採点へ組み込まないでください。元画像の解像度向上か、別の画像認識手段での比較が次の検証課題です。

同じ640×461ピクセルのsample1と30パッチ、赤枠20個で、別モデルも比較しました。高解像度の同一図面は作業フォルダ内にはなく、単純な拡大画像を高解像度の原画として扱っていません。数値は赤枠への暫定的な位置一致で、正式な窓検出精度ではありません。

| 画像認識モデル | 生の窓候補 | 採用した窓候補 | 赤枠との一致 | 採用した外部扉候補 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL 8B | 2 | 0 | 0/20 | 0 |
| Qwen3-VL 4B | 0 | 0 | 0/20 | 0 |
| Gemma 3 4B | 45 | 0 | 0/20 | 0 |

Gemmaの生窓候補45件のうち37件は切出し全体に近い大きなbboxで、残る8件も細長い窓記号の形ではありませんでした。生候補の中心が赤枠に偶然近い1件もありますが、位置が不正確なため一致に数えていません。外部扉の生候補2件も切出し全体を囲むため除外しました。3モデルとも改善は確認できず、現在の低解像度画像に対する小パッチ再読取は採点へ反映しません。結果の個別JSONと重ね合わせHTMLは `targeted_vision_results` に保存しています。

### 画像処理による窓候補と人手確認

黒い直線の壁が明るく途切れる箇所を、Pillowだけで「開口候補」として抽出できます。窓か扉かは自動確定しません。`sample1` では候補42件が既知の窓赤枠20箇所すべてに届きましたが、候補には扉・室内開口なども混ざります。sample1を見ながら検出条件を調整したため20/20は独立評価ではなく、他の図面での性能は未確認です。検出時には赤枠データを渡さず、抽出後にのみ位置照合します。

```powershell
python window_cv_review.py
```

生成される `targeted_vision_results/sample1.wall-gap-v1.review.html` を開き、青枠の候補を選んで「窓」「室内扉」「外部扉」「窓・扉ではない」「判別不能」に分類してください。赤枠は必要なときだけ表示できます。分類はブラウザ内に一時保存されますが、必ず「確認結果JSONを保存」で書き出してください。保存したJSONはページから再読込できます。`*.candidates.json` は自動候補、`*.coverage.json` は既知の赤枠との暫定照合であり、利用者の分類結果とは別です。確認結果は採点へ自動反映しません。

初版の確認ページには「室内扉」がなく、利用者は扉全般を「外部扉」として選びました。修正版では室内扉・外部扉を分け、旧JSONを読み込むと旧「外部扉」だけを「扉（内外未分類）」に戻します。窓や「窓・扉ではない」はそのまま残ります。手元の旧JSONから分類を引き継いだ再確認ページを生成するには、次を実行します。元のダウンロードJSONは変更しません。分類が未完了の間は下書きで、採点へ反映しません。

```powershell
python window_cv_review.py --review-file "C:\Users\teru\Downloads\sample1_window_cv_review.json"
```

`targeted_vision_results/sample1.wall-gap-v1.recheck.html` で未分類の扉だけ確認し、再度「確認結果JSONを保存」を押してください。初回の42候補は窓21件、扉全般11件、非開口10件でした。既存の窓赤枠20件はすべて窓と判定され、赤枠外の窓候補はC038です。赤枠の外部扉は候補を一対一対応させると3/4件で、R17は未検出です。これらはsample1の開発用結果であり、一般的な認識精度ではありません。

再確認済みの分類は `manual_corrections/sample1_window_cv_review.json` に保存しています。42候補の内訳は窓21件、室内扉7件、外部扉4件、非開口10件で、未分類はありません。画像ハッシュ・候補ID・位置の照合を通した人手ラベルですが、採点やモデルの推論結果には自動反映しません。既知の赤枠20窓との一致は検出器の開発に使った同じ画像での確認値であり、独立した精度評価ではありません。

別画像のsample2では、利用者が赤枠10件をすべて窓と確認しました。元画像へ位置合わせした下書きは `manual_corrections/sample2_openings_draft.json`、重ね合わせ画像は `targeted_vision_results/sample2.user_redboxes.registration.png` です。現行検出器の候補は1件で、指定された窓枠への位置一致は0/10件でした。赤枠の全窓網羅性と位置合わせの最終承認は未確認のため、正式な全窓recallやprecisionは出しません。この描画スタイルに現行検出器を適用して採点しないでください。

赤枠がまだない別画像でも候補と確認ページを作れます。この場合は位置一致を未測定と表示します。

```powershell
python window_cv_review.py --image floor_sample/sample2.webp
```

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

複数階で同じ部屋名がある場合は、`floor_qualified: true` として接続の両端を `1F:廊下`、`2F:LDK` のように階付きで記録できます。画像だけでは判定できない接続は `excluded_connections` に記録すると採点から除外されます。図面に入口がないことを確認した居室は `confirmed_no_entrance` に記録すると、その部屋へ架空の接続を作った結果を検出できます。`sample1.json` の2階主寝室は図面上に入口が描かれていないと利用者が確認済みですが、他の接続はまだ暫定下書きのため `draft` のままです。単体の暫定比較には `python tools/evaluate_ground_truth.py <structure.json> <ground_truth.json>` を使用します。

先に構造JSONなしでテンプレートを作成していた場合は、分析後に次のように `--overwrite` を付けて実行すると、AIの採用接続を下書きに入れ直せます。人手で修正を始めた後は上書きしません。

```powershell
python tools/prepare_ground_truth.py --images-dir floor_sample --ground-truth-dir ground_truth/floor_sample --structure-dir backtest_results/floor_sample --overwrite
```

確認済みデータを一括集計するには次を実行します。複数モデルの結果が同居する場合は、`--model` に結果ファイル名のモデル部分を指定します。

```powershell
python tools/evaluate_benchmark.py --structure-dir backtest_results/floor_sample --ground-truth-dir ground_truth/floor_sample --model qwen3-vl_8b-instruct-q4_K_M --output backtest_results/floor_sample/benchmark.json
```

集計には画像総数、構造JSONがある件数、正解データの確認状態も含まれます。承認済み正解データが0件のときはprecision/recall/F1を`null`とし、未測定の精度を0%と誤表示しません。

## 位置・接続の人手補正

小さな画像ではローカル8Bモデルの座標が不安定な場合があります。`manual_corrections/sample1.json` には、利用者が確認した4つの直接接続、5つの部屋範囲、4つの開口位置を記録しています。この補正は承認済みですが、人手補正後の精度はモデル本来の精度を示すベンチマークには含めません。

`manual_corrections/sample1_review.html` をブラウザで開くと、元画像に補正候補の範囲と開口位置を重ねて確認できます。

```powershell
python tools/apply_reviewed_corrections.py --structure backtest_results/floor_sample_hall_repair/sample1__qwen3-vl_8b-instruct-q4_K_M.structure.json --corrections manual_corrections/sample1.json --output backtest_results/manual_reviewed/sample1.structure.json
```

通常のバックテストに補正を明示的に組み込む場合は `--corrections-dir manual_corrections` を付けます。承認済み補正は `__manual_approved` の別名ファイルに出力します。下書きは自動的にスキップされ、`--preview-draft-corrections` を追加した場合だけ採点保留の `__manual_draft` に出力します。構造キャッシュには補正前の画像認識結果を残すので、人手補正とモデル単独の結果を区別できます。

```powershell
python live_backtest.py --input-dir floor_sample --results-dir backtest_results/manual_reviewed --image sample1.webp --corrections-dir manual_corrections
```

元画像と照合してbbox・開口座標の両方を確認できた場合にのみ、修正JSONの `bbox_review_status`、`opening_position_review_status`、`review_status` を `approved` に変更します。旧形式の `geometry_review_status` も読み取れます。未承認データは `--preview-draft` なしでは適用できず、試算結果を採点へ渡しても採点保留になります。

`sample1` の承認済み補正を既存の構造に適用すると、暫定正解データとの直接接続は8/8件一致します。ただし正解データ自体が `draft` であり、人手補正後の照合値です。新しい個別再確認では設備候補を検出できましたが、窓は座標が部屋外周と合わないため採用せず、1階の階別部屋一覧も洗面所を落とすため不完全と判定しました。採点は引き続き保留です。接続補正が成功しただけで、総合点を確定しないでください。

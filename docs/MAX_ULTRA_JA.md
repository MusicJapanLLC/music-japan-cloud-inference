# Music Japan MAX / ULTRA — 実際の公開LLMによる高思考モード

OpenAI公式公開ウェイト **gpt-oss-20b** と **gpt-oss-120b** を、GitHubの無料Linux CPUジョブ上で実推論する。誰かのOpenAI/Anthropic APIキーや有料推論サービスを呼び出さない。

## 使い方（追加料金ゼロ）

1. [Music Japan MAX / ULTRA ワークフロー](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/workflows/max-ultra-reasoning.yml) を開く。
2. **Run workflow** を選ぶ。ブランチは **feature/mj-frontier-research-factory-20261018**。
3. `execute=true`、`mode=max` または `mode=ultra`、`public_question` に一般公開して構わない文章だけ入力する。
4. GitHub Actionsのジョブ **20B → 120B → SHA検証** が自動で順番に動く。
5. 結果は最後の **Prove** ジョブの **Summary**、証拠ファイル `mj-max-ultra-independent-verification`、または実行ログから確認できる。

## 実装の差

| パラメータ | MAX | ULTRA |
|---|---:|---:|
| 20B初期推論＋反証・再推論 | 2回 | 3回 |
| 20B出力予算（1回） | 1,280トークン | 2,048トークン |
| OpenAI公式公開120Bの最終推論 | 1回 | 2回 |
| 120B出力予算（1回） | 384トークン | 512トークン |
| 実際のネイティブ思考設定 | high | high |
| 実モデルAPIクレジット使用 | 0 | 0 |
| 公開GitHub CPU VM | 使用 | 使用 |

**MAX/ULTRAはMusic Japan独自の推論ワークフロー名であり、非公開GPT-6やClaudeの権利・ウェイト・プラン設定ではない。** 構成モデルと推論パラメータはコード内で明示する。

## 全ジョブの証拠の扱い

- 初期段階で実際のgpt-oss20Bウェイトの固定SHA・20B実行回数・推論回答を記録。
- 最終段階でSHA一致の公開gpt-oss120Bを実際に読み込み、20Bの回答を参考資料として含めて推論する。
- 120Bは受け取った20B証拠の SHA-256、元質問のSHA、モデルID、ウェイトSHA、推論量を照合。証拠が一致しない場合は生成を開始しない。
- 最終証拠検証で20Bと120Bの出力、回数、使用モデルの正当性、回答テキスト、独立計算結果と追加コストゼロを検証。
- 数学や日付は安全な独立ツールで質問から再計算できる範囲で確認し、ツール補正と生のモデル出力を分けて記録。

## 公開GitHub環境の範囲

ここに書いた質問・回答は公開リポジトリの実行ログ・成果物として閲覧され得る。**企業秘密、顧客データ、LINE、Gmail、契約情報、個人情報を入力しないこと。**

GitHub ActionsのCPUは通常4CPU/約15.6GiB RAMで、SSDはエフェメラル。無料の公開リポジトリ用標準ランナーを利用する。一回の120Bウェイトは約63.4GBで、起動・推論に数分〜数十分かかる。

**最上位品質と認定するには、GPT-6や他社商用モデルに対する同一タスク・統計的に有意な実測が必要。** 3段階モデルが長く考えれば常に賢くなるわけではない。

## 公開実機証拠

- [gpt-oss120B 実モデルロードと437の回答](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39095219646)
- [gpt-oss120B highで任意の数理質問に回答](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39096510916)
- [MAX実モデル連携](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39098211563)
- [ULTRA実モデル連携](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39098436310)

## ソース

- `reasoning_profiles.py`: モード別の実計算量と証拠検証。
- `cloud_owner_chat.py`: 20Bの実モデル推論。
- `official120b_chat.py`: 20B参照を伴う元公開120Bの高思考推論。
- `verify_max_ultra_chain.py`: モデル証拠と改ざんチェック。
- `render_official120b.py`: GitHub UIに証拠付き回答を表示。
- `.github/workflows/max-ultra-reasoning.yml`: 上記一括実行。

永久稼働サーバーではなく、**GitHubのワークフローを起動するたびに実モデルを読み込んで回答するクラウド実行基盤**。無断の権限昇格・有償GPUの自動生成・非公開モデルの偽装を含まない。

# Music Japan Ultra — クラウド実行・ワンコマンド操作

**用途：** Music Japanが自分で GitHub上の無償Linux VM に公式公開の gpt-oss-20b と gpt-oss-120b の実モデルを起動し、複数回の高度推論、モデル間受け渡し、検算、改ざん検証を行って最終回答を受け取る。

OpenAI/Anthropic の推論APIに料金を払わず、GitHubの無償標準公開Runnerを使う。これは実際のオープンウェイトの推論であり、非公開の GPT-6/Claude モデル自体を示すものではない。

## 起動（追加のソフトウェア契約なし）

既存のGitHub認証が済んだPCで、GitHub CLI `gh` と Python 3.10以降を使う。

```powershell
gh repo clone MusicJapanLLC/music-japan-cloud-inference MusicJapanUltra -- --branch feature/mj-ultra-onecommand-20261018 --depth 1
cd MusicJapanUltra
python mj_ultra.py start --mode ultra --public-question "Calculate 314 times 29. Answer only the integer." --public
```

これで GitHub Actions の実行URLと run_id が表示される。処理はPCではなくGitHubのLinuxマシン上で進む。作業が終わったら：

```powershell
python mj_ultra.py status 39098211563
python mj_ultra.py collect 39098211563 --mode max --public-question "Calculate 314 times 29. Answer only the integer."
```

上記 `39098211563` は**既存の実測成功したMAXのサンプル**。ULTRAを起動した場合は新しく表示されたrun_idに置き換え、`--mode ultra`を指定する。

一気通貫で待ち合わせまで実行する場合：

```powershell
python mj_ultra.py run --mode ultra --public-question "What are the failure modes of a distributed message queue? Provide a concrete mitigation plan." --public --wait-seconds 7200
```

## 技術構成

- MAXはOpenAI公式公開gpt-oss20B 2回推論→OpenAI公式公開gpt-oss120B 1回の計3回
- ULTRAはgpt-oss20B 3回→gpt-oss120B 2回の計5回
- 元の実行ワークフロー：`.github/workflows/max-ultra-reasoning.yml`
- `mj_ultra.py` は既存の `gh` 認証に従って実行指示を送り、返されたrun_idのみ使用（推測で別の実行に接続しない）
- 終了時は GitHub Actionsの3つのJSON artifact を一時領域に取得。公式ウェイトモデルID・SHA256・質問ハッシュ・回答・実推論回数・クレジット0を再照合
- 実行結果は元のGitHub Actions Summaryにも残る
- 使えるアクションは start / run / status / collect。説明文ではなく実行可能な Python CLI

## 実証済みの受入テスト

- 実モデル連携の成功実行： https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39098211563
- このクライアントのCloud Linux API実データ読み戻しと暗号的証拠照合： https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39102037514
- 前者は gpt-oss20B 2回、gpt-oss120B 1回の実推論、最終回答「9106」、合計933.36秒を記録
- 後者はGitHub経由で元の3つの成果物を再取得し、実際のモデル間検証証拠から同じ「9106」を返却。46個のCLI・プロバイダ証拠検証試験も成功

## 入力情報の扱い

**GitHub公開リポジトリのワークフローを利用するので入力する質問・生成結果も公開され得る。** 企業情報、請求書、顧客名・LINE・Gmailの会話、内部DBや秘密鍵は入力しない。CLIの `--public` はこれを了承したときのみ指定する。

モデルの秘密鍵は一切要求しない。既存 `gh` のログインセッションから端末上で実行する。ユーザーに新しい有料契約やプロバイダの推論クレジット消費を要求しない。

## 注意すべき完成判定

- 「既存の本物の公開120Bを使うMAX」については実推論と証拠の読み戻しに成功
- ULTRAの本番向けエンドツーエンドテストは、モード `ultra` のrunで全ジョブが成功し5回の生成が裏付けられていることを確認する
- GitHub Actionsは一時実行環境。常時オンラインの推論API／独立GPUクラウドとは別の要件
- GPT-6や他社非公開モデルと同等の性能は、未見の難問/コード/多言語の第三者比較試験で判断する

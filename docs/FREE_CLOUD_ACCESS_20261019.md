# 無料クラウドLLMの利用手順

作成日：2026-10-19 JST

**追加支払いをせず、既存の公開モデルをPCに依存せず呼び出すための手順です**

利用条件は無料のみです。有料API、チャージ、カード登録による有料枠、有料GPUへの切替はこの手順に含めません。匿名APIで試す2モデルは明示的に選択し、失敗しても別モデルへ自動で切り替えません。

## 利用できる経路と確認範囲

| 経路 | 用途 | 認証・条件 | 確認範囲 |
| --- | --- | --- | --- |
| `qwen-flash-next` | 公開情報を使った短い生成・合成課題 | 既存の匿名公開API、会社の鍵を送らない | 固定したcatalog・応答モデルIDを検査するCLIを実装 |
| `deepseek-flash-0731` | 同上、別モデルを明示選択 | 既存の匿名公開API、会社の鍵を送らない | 同じCLIで独立した1回の要求として検査 |
| Chat-Deepの公開Web画面 | 人がブラウザから短い質問をする | 画面の無料回数上限に従う | 今回のブラウザ確認で合成課題に1回回答、後述のモデル区別あり |
| ModelScopeのDeepSeek-V4-Pro-0813画面 | 1650B対象の無料枠を本人アカウントで確認する候補 | ログイン・メール確認等、利用可能な魔粒が必要 | 無料表示と2魔粒/回を確認、本人残高と実推論は未確認 |

CLIの2モデルは、対象の`DeepSeek-V4-Pro-0813`とは別モデルです。出力を1650Bモデルの結果として扱いません。

## 1. 最初は接続せず、実行予定を見る

リポジトリのルートから実行します。

```bash
python3 free_cloud_inference.py --model qwen-flash-next
python3 free_cloud_inference.py --model deepseek-flash-0731
```

既定値は`PLAN`です。ネットワーク接続、プロンプトファイルの読取、レシートの作成を行いません。`--verify`を付けても、`--execute`がなければ生成しません。

## 2. 新しい合成課題で1回だけ検証する

Qwen側を選ぶ場合：

```bash
python3 free_cloud_inference.py \
  --execute --public --verify \
  --model qwen-flash-next \
  --receipt free-qwen-check-01.json
```

DeepSeek Flash側を選ぶ場合：

```bash
python3 free_cloud_inference.py \
  --execute --public --verify \
  --model deepseek-flash-0731 \
  --receipt free-deepseek-check-01.json
```

`--verify`は毎回、新しい16桁のhex nonce、3桁同士の掛け算、単語の反転を作ります。モデルにはnonceの完全な転記と計算結果をJSONで返すよう求めます。保存済み回答の再生や、過去の正答を生成結果に使う処理はありません。

`--receipt`には、存在しない新しいファイル名を指定してください。親ディレクトリは事前に用意します。レシートは0600の権限で排他作成し、同名ファイルを上書きしません。

## 3. 自分の公開プロンプトを使う

UTF-8のテキストファイルを用意し、`--verify`の代わりに`--prompt-file`を指定します。

例えば、次の内容を`public-question.txt`に保存します。

```text
Write a short Python function that returns a reversed copy of a string.
Include one example. Use only the information in this request.
```

```bash
python3 free_cloud_inference.py \
  --execute --public \
  --model qwen-flash-next \
  --prompt-file public-question.txt \
  --receipt free-public-answer-01.json
```

`--public`は、この入力を外部の公開モデルへ送ってよいという明示です。会社の非公開DB、顧客情報、LINE・メール本文、認証情報は入力しません。入力した公開プロンプトと実際の回答はレシートに残ります。

任意の質問への回答は`GENERATED`になります。これは、指定モデルの申告IDと正常な生成終了を確認した状態です。回答の事実関係やコードの正しさを自動で証明した状態ではないため、用途に応じた確認を行います。

## 4. 上限と失敗時の動作

| 項目 | 制限・動作 |
| --- | --- |
| モデル選択 | `qwen-flash-next`または`deepseek-flash-0731`のみ |
| ネットワーク要求 | catalog確認のGET後、生成POSTを最大1回 |
| 全体の期限 | 150秒。ネットワーク子プロセスを親が監督 |
| 入力 | UTF-8で最大8,192 bytes |
| 出力 | 最大384 tokens。`--max-tokens`で下げることは可能 |
| 成功する終了条件 | 完全なSSE、`finish_reason=stop`、非空の最終回答、固定モデルIDの一致 |
| usage | 返された場合、非負の整数とprompt＋completion＝totalを検査 |
| 失敗 | 自動再試行、自動fallback、有料経路への変更なし |
| レシート | 成功・採点不合格・接続エラー等を保存。部分回答が取得できた場合も、失敗を成功に変えず保持 |

catalogが一致しなければ生成POSTへ進みません。429、認証要求、接続切断、モデルID変更、`length`終了を成功に読み替えません。子プロセスの強制終了などで要求がどこまで進んだか分からない場合、レシートの要求回数等は`null`になり得ます。

クライアントにはAPI鍵の読取、プロキシ経由の接続、リダイレクト追従、応答キャッシュがありません。公開運営者自身のホスティング費用や将来の提供継続は不明です。「匿名で呼べる」という確認を、無制限の利用枠や継続稼働の保証として扱いません。

## 5. GitHub上でPCを経由せず実行する

対象リポジトリは[MusicJapanLLC/music-japan-cloud-inference](https://github.com/MusicJapanLLC/music-japan-cloud-inference)です。

今回の実装・workflowは`fix/1650b-cloud-gb300-20261019`ブランチにあります。

- [CLI](https://github.com/MusicJapanLLC/music-japan-cloud-inference/blob/fix/1650b-cloud-gb300-20261019/free_cloud_inference.py)
- [Free public model verificationの定義](https://github.com/MusicJapanLLC/music-japan-cloud-inference/blob/fix/1650b-cloud-gb300-20261019/.github/workflows/free-cloud-verification.yml)
- [Actionsの実行一覧](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions)

workflowは公開リポジトリの標準`ubuntu-24.04` runnerを使います。モデルへのHTTP要求と検証処理はGitHub側で動くため、Windows PCやRemote Desktopの起動に依存しません。モデルの重みをGitHub runnerへ載せる実装ではなく、既存の公開運営者のAPIへ接続する実装です。

GitHub公式の料金説明では、公開リポジトリの標準GitHub-hosted runnerは無料です。このworkflowはlarger runnerや有料GPUを選びません。これは時間を限定した開発検証jobであり、常設の推論サーバーではありません。

起動条件は次のとおりです。

- 通常のPR・pushではoffline検査と`PLAN`確認を行う
- 手動実行の`run_free_check`は既定`false`。明示的に`true`へ変更した場合だけ実推論へ進む
- 手動実行のモデル選択は`qwen-flash-next`、`deepseek-flash-0731`、または`all`
- `all`は2モデルをそれぞれ独立して1回ずつ検証する指定。片方の失敗をもう片方で成功に置き換える処理ではない
- 指定ブランチの対象ファイルpushに`[run-free-model-check]`マーカーを明示した場合も実行対象になる
- 同じrunの再実行では実推論jobを起動しない

workflowの定義がブランチにあることと、GitHubで手動起動可能な状態になっていることは別です。画面/APIからの起動可否はGitHub側のworkflow登録状態も確認してください。

**今回追加したworkflowの実推論は、この案内の作成時点では結果待ちです。実行結果はworkflowログの`MJ_FREE_RECEIPT`と`grade`で確認してください。**

合成課題の合格では、少なくとも以下を照合します。

```text
status == GENERATED_AND_GRADED
grade.passed == true
selected_model == 実行時に選んだモデル
inference_requests_attempted == 1
finish_reason == stop
```

レシートの`raw_answer`、`expected`、`completion_id`、`reported_model`、`usage`、経過時間、GitHubのcommit/run情報を併せて確認します。jobの緑表示、HTTP 200、catalog取得だけでは新しい実回答の証拠になりません。

## 6. モデルIDの対応と証明できる範囲

### Qwen Flash Next

```text
選択名: qwen-flash-next
公開host: apathy-exe-qwen3-8-flash-next.hf.space
出典モデル名: Qwen/Qwen3.8-Flash-Next
固定した上流申告ID:
/root/.cache/huggingface/hub/models--AtomicChat--Qwen3.8-Flash-Next-GGUF/snapshots/1c1f33abc8e835db3215390e8e14c6cc726f5843/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf
```

### DeepSeek Flash 0731

```text
選択名: deepseek-flash-0731
公開host: apathy-exe-deepseek-v4-flash-0731.hf.space
出典モデル名: deepseek-ai/DeepSeek-V4-Flash-0731
固定した上流申告ID:
/app/model_unified/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf
```

固定IDの一致で確認するのは、公開運営者のcatalogと応答が同じモデルを申告したことです。上流の重み全バイト、量子化の正確性、実モデルの完全な同一性をこちらで検証したものではありません。`weight_bytes_attested=false`を維持します。

## 7. ブラウザだけで使う無料候補

### Chat-Deep：手動の会話用

[公開チャット画面](https://chat-deep.ai/deepseek-chat/)で、今回のブラウザ操作による1回の回答を確認しました。

```text
課題のnonce: 04f5576330ce62b2
掛け算: 924 × 725
実回答で確認した積: 669900
```

その時点の画面は通常50回/日、Pro10回/日の上限を表示し、使用後は`1/50`、`Pro 1/10`になりました。[FAQ](https://seek-chat.com/faq/)の旧記述には100回/日という差があるため、利用時の画面表示を確認します。

FAQはFast/ProをDeepSeek-V4.1-Flashの思考モード切替として説明しています。画面の「Pro」という名称だけで`DeepSeek-V4-Pro-0813`を使ったとは判断できません。今回の回答成功は、1650B対象の実推論成功としては数えません。

FAQはAPIを提供せず、自動化を許可しない旨を記載しています。この候補は人が公開画面を使う用途で案内し、API化・自動ループ・回数上限の回避には使いません。

### ModelScope：exact 0813の無料枠を確認する候補

[DeepSeek-V4-Pro-0813のモデルページ](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro-0813)には、対象のexactモデル名、無料表示、`2魔粒/次`を確認しました。

[ポイント獲得画面](https://www.modelscope.cn/magicube/usage?tab=earn)には、ログインによる200魔粒の表示と、メール確認等の条件があります。ポイントに関する[公式告知](https://www.modelscope.cn/posts/435591)も参照できます。

今回確認できたのはUI上の条件です。本人アカウントのログイン完了、メール確認、実際に使える残高、当該モデルへ使える権利は確認していません。**この経路の実推論回数は0です。** 無料表示から、無条件・無制限に利用できるとは推定しません。

## 根拠・関連ファイル

- [GitHub Actionsの料金説明](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
- [CLIのソース](../free_cloud_inference.py)
- [CLIのoffline検証](../test_free_cloud_inference.py)
- [今回のworkflow](../.github/workflows/free-cloud-verification.yml)
- [以前のQwen匿名canaryの保存レシート](https://github.com/MusicJapanLLC/music-japan-cloud-inference/blob/92868904094bbd3d26c0e8346ab894445f51888f/evidence/public-cloud-live-canary-39138454573.json)：過去の別課題・別コードの記録であり、今回の実推論結果とは分ける
- [Chat-Deep公開画面](https://chat-deep.ai/deepseek-chat/)／[FAQ](https://seek-chat.com/faq/)
- [ModelScope exact 0813](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro-0813)／[ポイント画面](https://www.modelscope.cn/magicube/usage?tab=earn)／[公式告知](https://www.modelscope.cn/posts/435591)

この案内は確認済みの実装と観測範囲を記載しています。提供側の料金、無料枠、モデル表示が変わった場合は、その時点の一次情報と実行レシートを優先します。

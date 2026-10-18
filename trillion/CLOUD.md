# 1650B 公式クラウド接続

`trillion_cloud.py` は DeepSeek の公式 `https://api.deepseek.com/chat/completions` に最大1回だけ接続します。ローカルPCのGPUや893GBの重み取得は不要です。既定動作はネットワーク接続も認証読取も行わない計画表示です。

```bash
python trillion/trillion_cloud.py
python -m unittest discover -s trillion -p 'test_trillion_cloud.py' -q
```

実行には、正規の自社・本人用 `DEEPSEEK_API_KEY`、利用可能な残高、公開可能な入力、明示的な課金許可と利用上限が必要です。キーは環境変数か `--api-key-file` から読み、コマンド引数としてキー文字列を渡しません。次の例は **有料になり得る操作** です。この実装作業では実行していません。

```bash
python trillion/trillion_cloud.py --execute --public --allow-charge \
  --prompt-file /path/to/public-prompt.txt \
  --receipt /path/to/new-receipt.json \
  --max-input-bytes 1024 --max-output-tokens 128 \
  --timeout-seconds 120 --max-cost-usd 0.01
```

`--public` は入力者による公開可の申告です。秘密情報の自動判定ではありません。入力はユーザーメッセージ1件のテキストに限定し、バイト上限はJSON化したメッセージ全体で判定します。非thinking、非streamingで出力上限を指定し、POSIXのメインスレッド上でタイムアウトを設けます。再試行、別モデルへの切替、クライアントでの応答再利用はありません。提供元の自動プロンプトキャッシュはあり得ます。

## モデルと料金の確認

実行直前に認証なしで[公式料金表](https://api-docs.deepseek.com/quick_start/pricing/)を1回読み、HTML表の列と行を対応させます。`deepseek-v4-pro` が `DeepSeek-V4-Pro-0813` に対応し、接続先が公式APIであり、両時間帯の料金が確認済み上限以下である場合だけ推論を送信します。モデル変更、表構造の変更、料金上昇時には停止します。確認済み料金上限は入力cache miss 1.32 USD/100万token、出力3.96 USD/100万tokenです。実行時のページSHA256、抜粋行、読取時刻を保存します。

`--max-cost-usd` は、入力のUTF-8バイト数にテンプレート分1024tokenを加えた保守的な見積もりと、最大出力、確認済みピーク料金による**実行許可の閾値**です。正確なtokenizer計測でも、提供元が強制する請求上限でもありません。実応答のusageでも再判定しますが、送信後の中断・拒否・タイムアウトを無課金とは扱いません。厳密な請求上限は提供元アカウント側の制御が別途必要です。

## 成功を何で判定するか

receiptにはリクエストSHA256、公開リクエスト、開始・終了時刻、経過秒数、公式の対応表証拠、実際の応答ID・model・content・usage・finish_reasonを別々に残します。正常終了 `stop`、期待モデル、空でない応答、整合するusageを満たす場合に限り `inference_verified: true` とします。打切り応答は内容を残して不合格とし、通信エラーの本文や認証キーは記録しません。既存receiptは上書きできません。

これは提供元が報告する0813版での実行証拠です。`model.lock.json` のHF revision `72e1d3230f6c080a530b0a1d46f8eb4602340597` と、提供元がロードした全重みのSHA一致を証明するものではありません。常に `weight_revision_attested: false` とします。対応表の取得だけを実推論成功と数えません。

## GitHub上で実行する場合

既存の `Verify pinned 1.6T runtime and optimizer` workflowを拡張しています。通常のpush／pull requestでは実装テストと公開料金表の読取だけを行い、推論用の秘密情報を渡しません。`official-model-metadata` jobはGitHubホストから公式ページへ直接TLSで1回接続し、モデル対応表のreceiptを保存します。

実推論の `official-cloud-canary` jobは `workflow_dispatch` かつ `allow_one_paid_call=true` の初回attemptに限り、実装テストと公開モデル対応表の検証が両方成功した後に動きます。この入力は既定falseです。手動のRe-runでも推論jobは再実行しません。必要な本人の `DEEPSEEK_API_KEY` はGitHubの既存リポジトリsecretとして安全に登録しておく必要があります。認証の登録や有料実行を、この実装作業で実施したという意味ではありません。

明示実行した場合も、生成は最大1回、入力のJSON化後上限1024 bytes、出力128 tokens、全体待機120秒、事前の料金見積許可閾値0.01 USDに固定します。ランダムなnonceと掛け算の公開課題を生成し、返答をJSONとして採点します。生成されたコードは実行しません。GitHubのrun ID／commit／要求／応答／採点をartifactへ保存し、推論が失敗した場合も取得できたreceiptを残します。

workflowの成功、モデル対応表の一致、算術課題の合格は、それぞれ別の確認です。実推論jobがskipされたrunを、1650Bの生成成功とは扱いません。GitHubの標準ホストから公式APIへ直接送信し、所有PCのlistenerやトンネルへ委譲する処理はありません。

仕様の一次資料: [公式Chat Completion API](https://api-docs.deepseek.com/api/create-chat-completion/)、[公式モデルカード](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)。

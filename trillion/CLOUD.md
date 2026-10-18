# 1650B 公開メタデータの確認 — 無料限定

現在の方針は **無料限定** です。有料DeepSeek APIの実行提案とGitHubの課金許可入力を撤回しました。`trillion_cloud.py --execute` は、以前の課金許可引数が付いていても、入力ファイル・認証情報の読取やHTTP接続より前に `PAID_INFERENCE_DISABLED_BY_FREE_ONLY_POLICY` で停止します。

## 現在できること

```bash
python trillion/trillion_cloud.py
python -m unittest discover -s trillion -p 'test_trillion_cloud.py' -q
```

既定動作はネットワーク接続、認証読取、推論を行わない計画表示です。`execution_policy: free_only`、`paid_inference_enabled: false`、`max_inference_requests: 0` を表示します。公開表の解析と従来の応答検証関数は、隔離した実装テストのために残しています。これらのテストは実API呼出ではありません。

GitHubの `Verify pinned 1.6T runtime and optimizer` workflowは、オフラインの実装検証と、認証なしでの[公式モデル対応表](https://api-docs.deepseek.com/quick_start/pricing/)の読取だけを行います。`official-model-metadata` jobはGitHubホストのUbuntuから直接TLSで1回GETし、`deepseek-v4-pro` と `DeepSeek-V4-Pro-0813` の対応、表のSHA256、時刻、run ID、commitをログに残します。推論リクエストは0回です。`allow_one_paid_call` 入力、推論job、推論用secretの参照はありません。

このworkflowは追加のartifact uploadやActions cacheを使いません。[GitHubの公式課金説明](https://docs.github.com/en/billing/concepts/product-billing/github-actions)では、公開リポジトリの標準GitHub-hosted runnerは無料で、ログとjob summaryはartifact storage使用量に算入されません。公開リポジトリ・標準runnerという条件を維持します。

## 1650Bについて確認できていないこと

対象は `deepseek-ai/DeepSeek-V4-Pro-0813`、固定HF revisionは `72e1d3230f6c080a530b0a1d46f8eb4602340597` です。対象重みは66ファイル、892,744,322,880 bytesです。この作業では重みファイルの取得も、対象モデルの実推論も行っていません。

公開対応表の一致やworkflowの成功は、モデルが新しい返答を生成した証拠にはなりません。対応表receiptは `inference_performed: false`、`weight_revision_attested: false` を維持します。提供元のサービス名と、固定revisionの全重みSHA一致も別の確認です。

無料の小さいモデルを代替として実行できた場合は、その実際のモデル名・revision・重みSHA・新規返答・実行場所を個別に記録します。その結果を1650Bの実行成功や同等の性能として表示しません。今回の新規実行は、今回取得した要求と応答が揃ってから判定します。

一次資料: [公式モデルカード](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)、[公式モデル対応表](https://api-docs.deepseek.com/quick_start/pricing/)、[GitHub Actions課金条件](https://docs.github.com/en/billing/concepts/product-billing/github-actions)。

# 現行公開重みを使う、時間を限定したLLM開発評価

基準日: 2026-10-18

この評価は、公開モデルの実ファイルを取得・全量検証し、固定したCPUランタイムで公開合成課題を一回生成する開発用の再現手順です。第三者の推論APIを呼ぶ公開Qwenサービスとは別の経路で、モデルファイルと起動条件を管理します。学習・追加学習を実施した証拠や、常設の推論サービスではありません。

成功したQwen 3.8 27B評価は[公開PR #3](https://github.com/MusicJapanLLC/music-japan-cloud-inference/pull/3)、[run 39097846403](https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39097846403)、job 117587910044に対応します。記録は [evidence/qwen38-ownweights-39097846403.json](evidence/qwen38-ownweights-39097846403.json) にあります。

## 確認できた結果

| 項目 | Qwenの観測結果 |
|---|---|
| プロファイル | qwen38-27b-iq3s |
| 入力 | 公開可能な合成課題。計算、反転後の小文字化、依存工程の終了時刻、提案と決定の区別 |
| 重みの検証 | 12,040,883,104バイトすべてを読み、固定したサイズ・SHA-256に一致 |
| 実行資源 | 論理CPU 4、RAM 15.61 GiB、空きディスク85.47 GiB |
| ロード待ち | 7.174秒 |
| 生成リクエスト | 534.051秒、一回のみ |
| 評価処理全体 | 576.83秒。ランタイムのビルド時間を含まない |
| 出力 | stop終了、最終テキストあり、切り詰めの報告なし |
| 使用量の上流ローカル報告 | prompt 218、completion 380、total 598トークン |
| 内容の採点 | 外側のJSON Markdownフェンスを解釈すると4項目すべて期待値に一致 |
| 元の出力形式 | 厳密JSONは不合格。strict_json=false、strict_contract_passed=false |
| 設定 | context 2,048、出力上限1,024、thinking有効、要求effort low |
| 費用経路の記録 | model_api_credits_used=0、paid_compute_allocated=false |

[同じrunのNemotron job 117587910037のレシート](evidence/nemotron35-ownweights-39097846403.json)は、全59,764,210,568バイト（約59.8 GB）の重み検証と177.867秒でのロードまで成功しました。その後、thinking有効の一回の生成要求は900.198秒でタイムアウトしました。ロード成功と最終回答の生成成功を分け、この実行を正答や生成成功に数えません。この文書の最終回答が確認できた成功結果はQwen分だけです。

## 固定した重みとランタイム

| 対象 | 固定値 |
|---|---|
| 重みリポジトリ | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| リポジトリrevision | 4ca720788d1e01f1bff70c033e0d0028fd02e502 |
| ファイル名 | Qwen3.8-27B-UD-IQ3_S.gguf |
| 正確なファイル長 | 12,040,883,104 bytes |
| ファイルSHA-256 | d847e2c1e4aa276e4b7b8e9ad7628050e61e165d49ab995407bc36677a6f3864 |
| llama.cppソースcommit | 848bbda9041decd87904f7eb5a2990863ecce03b |
| レシートの自己ハッシュ | b47a92716a44bba27c2eafc32457da8fbf51979abe6a77b9f6478eb10ebaca4b |

設定の原本は [current_frontier_eval.py](current_frontier_eval.py) のPROFILESとLLAMA_COMMITです。[ワークフロー](.github/workflows/current-frontier-eval.yml)はランタイムの完全なGit commitを取得し、checkout後にも一致を確認してビルドします。評価ドライバーは実行前にバージョン出力のcommit接頭辞も確認します。ランタイム実行ファイルの全バイトをSHA-256で固定したという主張ではありません。

重みは一時ファイルへ取得し、全ファイルのサイズとSHA-256が一致した後に最終ファイル名へ移します。既存キャッシュも毎回全量検証し、不一致なら生成へ進みません。今回の証拠は公開API運営者の自己申告だけではなく、評価環境上で実際に読み取ったモデルファイルの検証を含みます。

## 再現手順

### 1. ソースと無生成確認

公開PR #3または上記runが実行したソースを確認し、そのrevisionをcheckoutします。再現時はGitHubのrun画面のhead SHAも記録し、後から変更されたブランチ先端と混同しません。

    python3 -m unittest -v test_current_frontier_eval.py
    python3 current_frontier_eval.py --profile qwen38-27b-iq3s

二つ目は既定のdry runです。重みをダウンロードせず、モデルを起動せず、生成しません。

### 2. 固定ランタイムのビルド

十分な一時ディスクを持つLinux環境で実行します。以下のソースcommitとCPU向けフラグはワークフローと同じです。

    git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
    git -C llama.cpp checkout --detach 848bbda9041decd87904f7eb5a2990863ecce03b
    git -C llama.cpp rev-parse HEAD
    cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
    cmake --build llama.cpp/build --target llama-server --config Release -j 4

rev-parseの結果が完全な固定commitと一致することを確認します。ワークフローでは浅いfetchを使いますが、生成対象のcommitとビルドフラグは上と同じです。

### 3. 明示的な一回の評価

    python3 current_frontier_eval.py --execute \
      --profile qwen38-27b-iq3s \
      --binary ./llama.cpp/build/bin/llama-server \
      --weights-dir ./current-model \
      --output ./current-eval/receipt.json

このコマンドは公開重みをダウンロードし、資源・整合性・ランタイムrevisionを確認してから、一回だけ固定合成課題を生成します。モデルAPI鍵の入力や、有料推論プロバイダーへの接続はありません。評価を繰り返す場合は新しい明示的な実行として記録し、失敗を消して最良結果だけを残さないでください。

GitHub上では、公開リポジトリでのworkflow_dispatchからrun_free_frontier_eval=trueを明示する経路があります。従来のrun_free_frontier_eval経路はQwenとthinking有効Nemotronの二つのプロファイルを別jobで実行するため、Qwenだけの一回を意図する場合に、そのまま一モデル用の操作だと解釈しないでください。後述の新しい非thinking追試は別の専用jobです。特定研究ブランチへのpushと明示マーカーで評価を許可する条件もワークフローに記載されています。通常のPR・pushではオフラインテストとdry runを行います。

## 資源と時間の制限

評価開始前にRAM 15 GiB以上と、全重みファイルの合計サイズに8 GiBを加えた空きディスクを要求します。今回のQwenの必要空き容量は19.21 GiBでした。条件を満たさなければRESOURCE_GATE_FAILEDで停止し、別の有料マシンやモデルへ切り替えません。

モデルはCPU llama-serverを4スレッド、並列要求1、GPU offloadなしで起動します。mmap、fit off、no-repackを明示し、従来二プロファイルのcontext 2,048、batch 64、microbatch 32です。後述の非thinking追試だけはcontext 1,024です。小さなRAM容量で実際に動いたことは、重み全体がRAMに常駐したことや、高速に実行できることを意味しません。

ドライバーの評価予算は2,100秒で、重み取得・起動・生成が残り時間を共有します。一ファイルのダウンロードは最大1,100秒で再試行なし、readiness待ちは最大420秒、生成HTTP待ちは最大900秒で、いずれも残り予算に制限されます。GitHub jobはランタイムのビルドも含め40分の上限を持ちます。予算超過や起動失敗を成功へ変換しません。

ローカル待受先は127.0.0.1の一時ポートだけです。起動時に作る一時トークンで保護し、Web UIと画像用mmprojを有効にしません。子プロセスへ渡す環境は必要なパス・言語設定等とその一時トークンに限定し、GitHubトークンや会社の推論鍵を渡しません。モデルプロセスは処理のfinallyで終了し、job終了後に常設サーバーとして残しません。

GitHubの標準公開runnerは、このリポジトリの時間を限定したLLM開発・評価jobに使っています。一般向けの常設無料計算サーバー、無料の無期限推論API、GPU割当済み環境として案内しません。実行条件、利用可能な資源、適用される利用枠はその実行時点で確認します。ワークフローは公開リポジトリであることを実行条件にし、追加の有料計算資源を要求しません。

## 測定・品質・証拠の読み方

レシートのfinal_contentは元のモデル出力です。今回は外側にMarkdownのjsonフェンスが付いたため、「JSONオブジェクトだけ」という指示には違反しました。採点器は元の厳密JSON採点と、完全な外側フェンス一組だけを解釈した内容採点を別々に記録します。後者の4項目正答を、元の出力形式まで含む完全合格と表示しません。

status=completedとプロセス終了コード0は、有効な最終回答とstop終了を確認した実行成功を表します。採点4項目の完全成功を必須条件にした状態値ではないため、利用側はgradingも確認します。length終了はoutput_truncatedとして記録し、完了回答には数えません。今回のレシートにはstopが返されています。

load_secondsは子プロセス起動後のreadiness待ち時間、generation_secondsは生成HTTP要求の経過時間です。576.83秒の全体時間は、評価ドライバー開始から終了処理までで、別stepのコンパイルは含みません。raw_response内のpredicted_ms等はランタイム自身の内訳報告で、HTTP全体の時間と同じ測定ではありません。

レシートには元の応答、usage、要求したthinking設定、prompt本文とハッシュ、全重みのピン、環境の資源量を保持します。要求effort lowの記録を、内部計算量が独立に証明されたという意味にはせず、reasoning_effort_attested=falseを維持します。vision_tested=falseで、画像理解を実証した結果でもありません。

receipt_sha256は、そのフィールド自体を入れる前のレコードを、キー順・区切り・UTF-8を固定してハッシュした値です。ファイル全体のSHA-256とは区別します。jobログのMJ_EVAL_RECEIPT行にもレコードを残します。ワークフローは重み・モデルキャッシュ・成果物アーカイブをアップロードしません。

この一回の実行は、現在の公開重みを自社管理の一時ランタイムで実行できたことと、限定した4項目の内容一致の証拠です。最先端の他モデルとの同等性、業務全般の完成度、長文・画像・ツール機能の品質、常時稼働の可用性を示す結果ではありません。自社APIの公開モデル経路と、この重み管理経路を別々に測り、以後の固定した業務評価セットへ接続します。

## 明示的なNemotron非thinking追試（実測待ち）

元のNemotronはロードできてもthinking有効の生成でタイムアウトしたため、別プロファイルnemotron35-120b-iq3xxs-instructを追加しました。現在は実行可能な設定とオフラインテストを用意した段階で、生成成功や高速化を確認した結果ではありません。元の失敗を上書きせず、新しい一回の比較として記録します。自動のfallbackや自動再試行ではありません。

| 設定 | 追試の指定 |
|---|---|
| 重み | 元Nemotronと同じunsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF |
| revision | 75a5829055f69a0359085ae6bbd0a81b60bc05c6 |
| サイズ・SHA-256 | 元プロファイルの3分割ファイルと完全に同じピン。PROFILESのfilesを参照 |
| ランタイム | 同じllama.cpp commit 848bbda9041decd87904f7eb5a2990863ecce03b |
| テンプレート | enable_thinking=falseのみ。medium_effortもreasoning_effortも送らない |
| context / 出力上限 | 1,024 / 256トークン |
| 課題 | 同じ固定公開合成4項目。プロンプト内容を変更しない |
| 要求回数 | 一回。タイムアウトやlength終了は失敗として保持 |

無生成確認と明示実行は次のとおりです。既存のQwen、thinking有効Nemotronの設定は変更していません。

    python3 current_frontier_eval.py --profile nemotron35-120b-iq3xxs-instruct
    python3 current_frontier_eval.py --execute \
      --profile nemotron35-120b-iq3xxs-instruct \
      --binary ./llama.cpp/build/bin/llama-server \
      --weights-dir ./current-model \
      --output ./current-eval/nemo-instruct-receipt.json

専用workflow job evaluate_nemo_instructは公開リポジトリの標準ubuntu-24.04で、research/current-frontier-ownweights-20261018ブランチだけを対象にします。実行条件は、push時の明示マーカー[run-free-nemo-instruct-eval]、またはそのブランチへの手動実行でrun_free_nemo_instruct_eval=trueを指定することです。手動boolは既定falseです。レビュー前のコミットには実行マーカーを付けません。

この追試マーカーまたは手動boolが選択された場合、従来のevaluate二モデルmatrixを明示的に無効にします。両方の手動boolをtrueにした場合も追試専用jobだけが対象になり、Qwenやthinking有効Nemotronを再実行しません。専用jobにはmatrixがなく、指定するプロファイルは非thinkingの一つだけです。

資源ゲート、全重みの再検証、40分job上限、2,100秒の共有評価予算、900秒以下の生成待ち、一時loopbackトークン、環境変数制限、子プロセス終了は従来と同じです。画像は試さず、重み・キャッシュ・成果物アーカイブのアップロードも行いません。dry runと実行レシートに要求したテンプレート、context、出力上限を記録します。測定済みレシートを得るまで、この追試の速度・正答・完了を断定しません。

<img src='./data/info/readme/001.png' width='1500'>  

# Rainforest-Connection-Species-Audio-Detection

[Rainforest-Connection-Species-Audio-Detection](https://www.kaggle.com/c/rfcx-species-audio-detection/overview) コンペのリポジトリ

デバッグ実行: `ipython3 --pdb exp.py`  
docker build: `sh build_docker.sh`  
docker run: `sh run_docker.sh -p 8713 -g 0`  
	- gpu使わない場合は `-g -1` とする


## Links

- [googledrive](https://drive.google.com/drive/u/1/folders/1oTq6R1t5OZCwaVR9u9ChoA_SRZi6r7Ol)
- [issue board](https://github.com/fkubota/kaggle-Rainforest-Connection-Species-Audio-Detection/projects/1)
- [yukiさんのgithub](https://github.com/yuki-a4/rfcx-species-audio-detection/)

## Paper
hoge

## Task
**Description(DeepL)**

朝の鳥のさえずりやカエルの夕べの鳴き声を楽しまない人はいませんか？動物は甘い歌や自然の雰囲気だけではありません。熱帯雨林の種の存在は、気候変動や生息地の減少の影響を示す良い指標となります。これらの種は目で見るよりも耳で聞く方が簡単なので、地球規模で機能する音響技術を使用することが重要です。機械学習技術によって提供されるようなリアルタイムの情報は、人間が環境に与える影響を早期に発見することを可能にします。その結果、より効果的な保全管理の意思決定が可能になる可能性があります。

種の多様性や豊富さを評価する従来の方法は、コストがかかり、空間的にも時間的にも制限がある。また、ディープラーニングによる自動音響識別は成功しているが、モデルは種ごとに多数の訓練サンプルを必要とする。このため、保全活動の中心となる希少種への適用には限界があります。そこで、限られたトレーニングデータでノイズの多い音風景の中で高精度な種の検出を自動化する方法が解決策となります。

レインフォレスト・コネクション（RFCx）は、遠隔地の生態系を保護し、研究するための世界初のスケーラブルなリアルタイムモニタリングシステムを開発しました。ドローンや人工衛星のような視覚的な追跡システムとは異なり、RFCxは音響センサーに依存しており、年間を通して選ばれた場所で生態系のサウンドスケープを監視しています。RFCxの技術は、地域のパートナーが適応的管理の原則に基づいて野生生物の回復と回復の進捗状況を測定することを可能にする包括的な生物多様性モニタリングプログラムをサポートするために進歩してきました。また、RFCxモニタリングプラットフォームは、解析のための畳み込みニューラルネットワーク（CNN）モデルを作成する機能も備えています。

このコンテストでは、熱帯のサウンドスケープ録音から鳥やカエルの種を自動検出します。限られた音響的に複雑なトレーニングデータでモデルを作成します。鳥やカエルの音だけではなく、虫の音が1～2匹聞こえてくることが予想されますが、これはモデルがフィルタリングして除去する必要があります。

成功すれば、急速に拡大している科学分野、つまり自動化された環境音響モニタリングシステムの開発に貢献することができます。その結果、リアルタイムの情報が得られれば、人間の環境への影響を早期に発見できるようになり、環境保全をより迅速かつ効果的に行うことができるようになります。


**Data**

このコンテストでは、多数の種の音を含むオーディオファイルが与えられます。あなたの課題は、各テストオーディオファイルについて、与えられた種のそれぞれがオーディオクリップの中で聴こえる確率を予測することです。トレーニングファイルには、種の識別と種が聴こえた時間の両方が含まれていますが、時間の定位はテストの予測には含まれていません。

トレーニングデータには、トレーニングを支援するために偽陽性ラベルの発生も含まれていることに注意してください。

ファイル
- train_tp.csv - 真の正の種ラベルのトレーニングデータ．
- train_fp.csv - 擬陽性種ラベルのトレーニングデータ．
- sample_submission.csv - 正しいフォーマットのサンプル提出ファイル。
- train/ - トレーニング用の音声ファイル
- test/ - テスト用の音声ファイル; タスクは各音声ファイルに含まれる種を予測することです.
- tfrecords/{train,test}を使用しています。- 競技データは TFRecord 形式で、Recording_id, audio_wav (16bit PCM 形式でエンコードされている) および label_info (列車のみ) を含み、以下の列 (Recording_id を除く) を区切り文字列として提供しています。

列
- recording_id - 録音用の一意の識別子
- species_id - 種の一意の識別子
- songtype_id - ソングタイプの一意の識別子。
- t_min - アノテーションされた信号の開始秒数．
- f_min - アノテーションされた信号の低周波数．
- t_max - アノテーションされた信号の終了秒数．
- f_max- 注釈された信号の上限周波数
- is_tp- [tfrecords のみ] ラベルが train_tp (1) または train_fp (0) ファイルのものかどうかの指標。

**shapeについて**
- n_class = 24
- recording_id:
	- train = 1132
	- test = 1992(public: 21%)
		- public 21% = 418
		- private 7% = 1574


## Log
### 20201210
- チームマージ

### 20200108
- Kaggle日記始動

### 20200109
- Dockerで環境構築

### 20200110
- Dockerで環境構築。たぶん完了。

- `train_fp.csv` について書かれている[discussion](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197866)
	- yukiさんに教えてもらった！

- うらたつさんに教えてもらったLwLRAPについての[notebook](https://www.kaggle.com/osciiart/understanding-lwlrap-sorry-it-s-in-japanese)
	- 今回の評価指標について書かれている。

- 簡単なEDAしてる[notebook](https://www.kaggle.com/mrutyunjaybiswal/rainforest-tfrecords-with-audio-eda-metrics)

### 20200111
- pandasguiでいろいろ見てた
	- train_tp.csv

|column|Type	|Count	|N Unique	|Mean	|StdDev	|Min	|Max|
|---|---	|---	|--- 	|---	|---	|---	|---|
|f_max	|float64	|7781	|30	|6074.830	|3386.040	|843.75	|13687.5|
|f_min	|float64	|7781	|24	|2827.996	|2515.604	|93.75	|10687.5|
|recording_id	|string	|7781	|3958	|	|	|	||
|songtype_id	|int64	|7781	|2	|1.346	|0.959	|1.0	|4.0|
|species_id	|int64	|7781	|24	|12.138	|7.068	|0.0	|23.0|
|t_max	|float64	|7781	|6089	|31.267	|17.496	|0.768	|59.994|
|t_min	|float64	|7781	|6090	|28.627	|17.461	|0.0107	|59.301|


### 20200112
- baseline作成をしていた
	- datasetの作成途中

- nb001
	- EDA
	- 1 recording_id に複数のラベルを持つリスト
		- 151コ
		- unique: 67
		- 内訳:
			- 1:    1065
			- 2:     55
			- 3:       8
			- 4:       3
			- 5:       1
		- `'03b96f209', '053aeb7bd', '11c2c02e5', '160add406', '16553d5cd',
       '178b835e3', '1aa00dc63', '21e2f2977', '287bf77ec', '2bcddf9a5',
       '2d09eb065', '2dc763e67', '2eb098e76', '33d0f2685', '34340b225',
       '349095631', '3c621e663', '400b7210c', '41829d963', '43d34d63c',
       '48fb5143f', '534db172e', '551385b05', '55b2b19d1', '561ed4362',
       '59a9eb657', '5b1e3b55b', '5bfe1dec6', '5db2e86fe', '5f8eecc9e',
       '5f9b4785b', '60b260508', '69aacafc4', '6bf2953a8', '6d93f853d',
       '71cf9646b', '728459067', '77299bde7', '774912d66', '7a9d46229',
       '9251fdbdd', '942ca05c0', '9a76cab9c', 'a2441a74b', 'a993402e2',
       'b056e5bc2', 'b55d2f7b4', 'b62b5a988', 'b7485fa88', 'bc9dd660e',
       'bd62d4fa2', 'bf964d1fa', 'c12e0a62b', 'c91cae4aa', 'cb5ddad47',
       'ccee900dd', 'd2cb96229', 'd58429096', 'd59d099b3', 'd80fab44f',
       'dd38bef4b', 'e42215aa0', 'e6de52902', 'ed2f84e75', 'ee3dc0bc6',
       'f3f82b897', 'f97ababc1'`

	- 1 recording_id に複数のspecies_idを持つリスト
		- 64コ
		- unique: 27
		- 内訳:
			- 1:    1105
			- 2:      25
			- 3:      2
		- `'178b835e3', '1aa00dc63', '2bcddf9a5', '2d09eb065', '2eb098e76',
       '34340b225', '349095631', '400b7210c', '43d34d63c', '551385b05',
       '561ed4362', '5db2e86fe', '5f8eecc9e', '5f9b4785b', '60b260508',
       '6d93f853d', '71cf9646b', '77299bde7', '7a9d46229', 'b55d2f7b4',
       'b7485fa88', 'bf964d1fa', 'c12e0a62b', 'c91cae4aa', 'd58429096',
       'e42215aa0', 'ed2f84e75'`

### 20200113
- exp001
	- dataset部分の作成

### 20200114
- exp001
	- model部分の作成

### 20200115	
- exp001
	- train_fold部分の作成

### 20200116
- exp001
	- train_fold部分の作成
	- resnet18の実装
	- debugモードの追加

### 20200117
- exp001
	- train_cv の実装

### 20200118
- exp001
	- なぜか学習がすすまない
		- [x] Adamはだめだとyukiさんが言ってた
			- SGDでもだめだった
		- [x] label周りは?
			- 問題なさそう
		- [x] BCEのせい？
			- BCEWithLogitsLossでも安定しない
		- [x] Spectrogramにちゃんとラベル部分は入ってる？
			- 入ってた
		- [x] LRを0.001から0.01にしてみた。
			- かわらず
		- [x] get_index_foldでrecording_ids_shf が使われてなかったミスを治した
			- 結局収束せず
		- [x] trainner内のdelをコメントアウトしてみた
		- [x] periodを10秒にしてみた
		- [x] resnet50で試してみた
		- [x] modelの場所確認
			- こいつが犯人だったぁぁぁぁああああ！

### 20200119
- exp001
	- get_index_foldにn_classesをチェックするassertを入れた


### 20200129
- 久々に復帰...
- exp001
	- ベースライン
	- lwlrapは未実装
	- result
		- time: 2h36m
		- oof_accuracy: 0.717105


### 20200130
- run_docker.shにg=-1の場合分けを追加
- exp002
	- accのグラフ追加
	- lwlrap追加
	- result
		- time: 2h35m
		- oof_accuracy: 0.719572
		- oof_lwlrap: 0.817672


### 20200131
- exp003
	- base: exp002
	- pretrained=Falseにしてみた
	- result
		- pretrainedってめっちゃ効くんだな
		- time: 2h36m
		- oof_accuracy: 0.5181
		- oof_lwlrap: 0.7196

- exp004
	- base: exp002(lwlrap=0.817672)
	- sweepできるように改良した
	- splitのseedを5714, 5715で回す(exp002はseed:5713)
	- split依存結構あるなぁ。
	- run001
		- split.seed: 5714
		- result
			- time: 2h31m
			- oof_accuracy: 0.6842
			- oof_lwlrap: 0.7948
	- run002
		- split.seed: 5715
		- result
			- time: 2h31m
			- oof_accuracy: 0.7113
			- oof_lwlrap: 0.8131


- exp005
	- base: exp002(lwlrap=0.817672)
		- period: 5
		- shift_duration: 4
	- periodとshift_durationをsweepしてみる
	- run001
		- period: 3
		- shift_duration: 2
		- result
			- time: 2h30m程度
			- oof_accuracy: 0.7245
			- oof_lwlrap: 0.8245
	- run002
		- period: 7
		- shift_duration: 4
		- result
			- time: 2h30m程度
			- oof_accuracy: 0.727
			- oof_lwlrap: 0.8235
	- run003
		- period: 20
		- shift_duration: 7
		- result
			- time: 2h30m程度
			- oof_accuracy: 0.6793
			- oof_lwlrap: 0.7946


### 20200201
- 今日からカンム！！
- exp006
	- base: exp002(lwlrap=0.817672)
		- melspec.fmin: 0
	- melspec_params.fminのsweepをしてみる
	- run001
		- fmin: 5
		- result
			- time: 2h30m程度
			- oof_lwlrap: 0.81
	- run002
		- fmin: 22
		- result
			- time: 2h30m程度
			- oof_lwlrap: 0.8196
	- run003
		- fmin: 50
		- result
			- time: 2h30m程度
			- oof_lwlrap: 0.8241
	- run004
		- fmin: 90
		- result
			- time: 2h30m程度
			- oof_lwlrap: 0.826

### 20200203
- exp007
	- base: exp002(lwlrap=0.817672)
		- melspec.fmax: sr/2 = 24000
	- melspec_params.fmax のsweepをしてみる
	- f_max最大値は13687.5
	- run001
		- fmax: 14000
		- result
			- time: 2h30m程度
			- oof_lwlrap: 0.8276
	- run002
		- fmax: 18000
		- result
			- time: 2h30m程度
			- oof_lwlrap: 0.8236
	- run003
		- fmax: 23000
		- result
			- time: 2h30m程度
			- oof_lwlrap: 0.8192

- exp008
	- base: exp002(lwlrap=0.817672)
	- exp006, exp007で、melspectrogramのパラメータを変えたほうがいいとわかった
	- 与えられたデータのf_min, f_maxのそれぞれの最小値、最大値はf_min=93.75, f_max=13687.5なので以下のようにする
		- fmin = 90
		- fmax = 14000
	- run001
		- fmin: 90
		- fmax: 14000
		- result
			- oof_lwlrap: 0.8236

### 20200204
- exp009
	- base: exp008(lwlrap=0.8236)
	- Resnet18: resnet18にGlobalMaxPoolingに変更してみた
	- run001
		- result
			- 大失敗！！
			- oof_lwlrap: 0.3944

	
- exp010
	- base: exp008(lwlrap=0.8236)
	- Resnet18_2: resnet18に GAPとGMPの和を追加
	- gap_ratio: 1はexp008と同じ結果になるはず
	- 混ぜる比率をハイパラにした
	- result
		- run001
			- gap_ratio: 1
			- oof_lwlrap: 0.8207
		- run002
			- gap_ratio: 0.75
			- oof_lwlrap: 0.8298
		- run003
			- gap_ratio: 0.5
			- oof_lwlrap: 0.8149
		- run004
			- gap_ratio: 0.3
			- oof_lwlrap: 0.7972
		- run005
			- gap_ratio: 0.1
			- oof_lwlrap: 0.5763

### 20200205
- exp011
	- base: exp008(lwlrap=0.8236)
	- Resnet18_2: resnet18に GAPとGMPの和を追加
	- gap_ratio: 1はexp008と同じ結果になるはず
	- exp010で0.75付近がよかったのでその周辺を探す
	- result
		- gap_ratioは 0.9が一番よさそう
		- run001
			- gap_ratio: 0.9
			- oof_lwlrap: 0.8377 <---- 過去最高スコア
		- run002
			- gap_ratio: 0.8
			- oof_lwlrap: 0.8288
		- run003
			- gap_ratio: 0.7
			- oof_lwlrap: 0.8259

- exp012
	- base: exp008(lwlrap=0.8236)
	- yukiさんとのディスカッションで、GAPとGMPのconcatはどうだろうかという話になった。やってみる。
	- model_name: Resnet18_3
	- result
		- run001
			- gap_ratio: 0.9
			- oof_lwlrap: 0.8396  <---- 過去最高スコア
		- run002
			- gap_ratio: 0.8
			- oof_lwlrap: 0.8321
		- run003
			- gap_ratio: 0.7
			- oof_lwlrap: 0.8354


### 20200206
- exp013
	- base: exp008(lwlrap=0.8236)
	- 背景の情報をどれだけ使っているかを見極めるために、labelに関係なくクロップする
	- result
		- run001
			- 完全にランダムでもaccuracyが0.5049あるのか...
			- oof_lwlrap: 0.658
			- oof_accuracy: 0.5049

### 20200213
- exp014
	- base: exp013(lwlrap=0.658)
	- consusion_matrixを実装
	- result
		- confusion_matrixを見る感じ、相変わらずclass3が苦手っぽい。
		- ↑のことから、背景で学習してるとは判断できない。もしさらに調査するならラベルが混入してないか確認しながらやるべき。
		- run001

### 20200215
- exp015
	- base: exp013(lwlrap=0.658)
	- mono_to_colorを改造してhpssを入れる
	- imageの3枚をoriginal, original_p, original_h みたいな構成を取る。
	- result
		- run001

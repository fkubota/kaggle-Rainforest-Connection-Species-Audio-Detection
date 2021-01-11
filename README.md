<img src='./data/info/readme/001.png' width='1500'>  

# Rainforest-Connection-Species-Audio-Detection

[Rainforest-Connection-Species-Audio-Detection](https://www.kaggle.com/c/rfcx-species-audio-detection/overview) コンペのリポジトリ

## Links

- [googledrive](https://drive.google.com/drive/u/1/folders/1oTq6R1t5OZCwaVR9u9ChoA_SRZi6r7Ol)
- [issue board](https://github.com/fkubota/kaggle-Rainforest-Connection-Species-Audio-Detection/projects/1)

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



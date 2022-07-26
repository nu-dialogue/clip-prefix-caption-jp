# clip-prefix-caption-jp
本リポジトリは画像キャプション生成手法[ClipCap](https://arxiv.org/abs/2111.09734)の日本語版実装です．ClipCapは，大規模汎用画像分類モデル[CLIP](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf)と大規模汎用言語モデル[GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)をベースにした手法です．本リポジトリでは，日本語版GPT-2と日本語版MS COCOデータセットを用いて実装してあります．自作データセットで学習・推論する際のチュートリアルも公開しています．

- Official implementation of ClipCap: [GitHub](https://github.com/rmokady/CLIP_prefix_caption)
- Japanese GPT-2 model : [![](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-rinna%2Fjapanese--gpt2--medium-brightgreen)](https://huggingface.co/rinna/japanese-gpt2-medium)
- Japanese MS COCO: [Paper](https://aclanthology.org/P16-1168/), [GitHub](https://github.com/yahoojapan/YJCaptions)

## Inference Examples on MS COCO

<table>
  <tr>
    <td><img src="example_images/COCO_val2014_000000499388.jpg" ></td>
    <td><img src="example_images/COCO_val2014_000000232842.jpg" ></td>
    <td><img src="example_images/COCO_val2014_000000250345.jpg" ></td>
  </tr>
  <tr>
    <td>街路樹のある道路を車が走っています。</td>
     <td>広場でたくさんの人が凧揚げをしています。</td>
     <td>紙の箱の中にピザが入っています。</td>
  </tr>
 </table>
 
 <table>
  <tr>
    <td><img src="example_images/COCO_val2014_000000380510.jpg" ></td>
    <td><img src="example_images/COCO_val2014_000000148403.jpg" ></td>
    <td><img src="example_images/COCO_val2014_000000271429.jpg" ></td>
  </tr>
  <tr>
    <td>時計の針は10時20分を指しています。</td>
     <td>青空の下に時計台が建っています。</td>
     <td>野球のバッターがボールを打とうとしています。</td>
  </tr>
 </table>

## Requirements
Python >= 3.7
```bash
pip instal -r requirements.txt
```

## チュートリアル
- COCOデータセットやSFCOCOデータセットで学習したモデルでの推論: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nu-dialogue/clip-prefix-caption-jp/blob/master/notebooks/sfc2022_clipcap.ipynb)
  - SFCの授業で用いたものです．
- 自作データセットの用意から学習・推論まで: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nu-dialogue/clip-prefix-caption-jp/blob/master/notebooks/original_clipcap.ipynb)

## 用意するデータの形式
`data/`直下に，任意のデータセット名のディレクトリを作成し，キャプション一覧データ（`captions.csv`）と画像格納ディレクトリ（`images/`）を置く．

### `data/`ディレクトリ下のデータ例
データセット名が`original`の場合
```
data/
  └original/ # データセット名
    ├images/ # 画像データを含んだフォルダ
    │  ├001.jpeg # 画像ファイル名は何でもよい（連番である必要はない）
    │  ├002.jpeg
    │  └...
    │
    └captions.csv # 画像ファイル名とそのキャプション文のペアリスト
```

### captions.csvの中身
- **画像ファイル名**と**キャプション文**が対になったcsvファイル
  - 画像ファイル名は絶対パスや相対パスではなく，ファイル名そのもの（拡張子付き）とする．
- 例
  ```csv
  001.jpeg,スケボーに興じる一人の男性がいます。
  002.jpeg,ゲレンデでスキーをしている人がいます。
  ...
  ```
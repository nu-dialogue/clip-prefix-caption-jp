# clip-prefix-caption-jp
CLIP Prefix captioningの日本語版実装
- CLIP prefix captioning
    - Paper: [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/abs/2111.09734)
    - Official implementation: [github](https://github.com/rmokady/CLIP_prefix_caption)
- Japanese GPT-2 model
    - Hugging Face: [japanese-gpt2-medium](https://huggingface.co/rinna/japanese-gpt2-medium)
- MS COCO Dataset
    - Paper: [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
    - Download: [MS COCO](https://cocodataset.org/#download)
- Japanese version of the MS COCO
    - Paper: [Cross-Lingual Image Caption Generation](https://aclanthology.org/P16-1168/)
    - Data: [github](https://github.com/yahoojapan/YJCaptions)

## ライブラリ
- 基本
    ```bash
    pip instal -r requirements.txt
    ```
- JupyterNotebookを使用する場合
    ```bash
    pip install ipykernel
    pip install ipywidgets widgetsnbextension
    ```

## チュートリアル
- 学習済みモデルを用いた推論のみ: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ohashi56225/clip-prefix-caption-jp/blob/master/notebooks/sfc_inference.ipynb)
- キャプションデータの用意から学習，推論まで: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ohashi56225/clip-prefix-caption-jp/blob/master/notebooks/sfc_train_test.ipynb)
### 用意するキャプションの形式
- キャプション一覧データ
    - 推奨データパス：`data/<データセット名>/captions.json`
    - 内容：キャプション文（caption），キャプションID（id），画像名（image_name）からなる辞書のリスト
    - 内容例
        ```json
        [
            {
                "caption": "暗くされた部屋の中で、テーブルに置かれたノートパソコンのモニター画面が壁の大きなスクリーンに映し出されています。", 
                "id": 1, 
                "image_name": "COCO_val2014_000000131075.json"
            }, 
            {
                "caption": "プロジェクターにパソコンの画像が映し出されています。", 
                "id": 2, 
                "image_name": "COCO_val2014_000000131075.json"
            }
        ]
        ```
- 画像データ
    - 推奨データパス：`data/<データセット名>/images`
    - 内容：キャプションの対象画像
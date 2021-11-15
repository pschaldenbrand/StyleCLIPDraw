# StyleCLIPDraw
#### Peter Schaldenbrand, Zhixuan Liu, Jean Oh September 2021

To be featured in the 2021 NeurIPS Workshop on Machine Learning and Design. 
[ArXiv pre-print](https://arxiv.org/abs/2111.03133).

StyleCLIPDraw adds a style loss to the [CLIPDraw (Frans et al. 2021)](https://arxiv.org/pdf/2106.14843.pdf) [(code)](https://github.com/kvfrans/clipdraw) text-to-drawing synthesis model to allow artistic control of the synthesized drawings in addition to control of the content via text.  Whereas performing decoupled style transfer on a generated image only affects the texture, our proposed coupled approach is able to capture  a  style in both texture and shape, suggesting that the style of the drawing is coupled with the drawing process itself.

Checkout our [code on Colab](https://colab.research.google.com/github/pschaldenbrand/StyleCLIPDraw/blob/master/Style_ClipDraw.ipynb)

<p align="left">
    <img src="images/styleclipdraw_main_example.png" height="300" title="">
</p>

## Method


<!-- <p align="left">
    <img src="images/method.PNG" height="400" title="">
</p> -->

<p align="left">
    <img src="images/method_animated.gif" height="450" title="">
</p>

Unlike most other image generation models, CLIPDraw produces drawings consisting of a series of Bezier curves defined by a list of coordinates, a color, and an opacity.  The drawing begins as randomized Bezier curves on a canvas and is optimized to fit the given style and text. The StyleCLIPDraw model architecture is shown above.  The brush strokes are rendered into a raster image via  differentiable model.  There are two losses for StyleCLIPDraw that correspond to each input.  The text input and the augmented raster drawing are fed the the CLIP model and the difference in embeddings are compared using cosine distance to compute a loss that encourages the drawing to fit the text input.  The image is augmented to avoid finding shallow solutions to optimizing through the CLIP model.  The raster image and the style image are fed through early layers of the VGG-16  model and the difference in extracted features form the loss that encourages the drawings to fit the style of the style image.



## Results


<p align="left">
    <img src="images/results.png" height="400" title="">
</p>


## StyleCLIPDraw vs. CLIPDraw then Style Transfer


<p align="left">
    <img src="images/style_transfer_results.png" height="600" title="">
</p>

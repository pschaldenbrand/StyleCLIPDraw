# StyleCLIPDraw

StyleCLIPDraw adds a style loss to the [CLIPDraw (Frans et al. 2021)](https://arxiv.org/pdf/2106.14843.pdf) [(code)](https://github.com/kvfrans/clipdraw) text-to-drawing synthesis model to allow artistic control of the synthesized drawings in addition to control of the content via text.  Whereas performing decoupled style transfer on a generated image only affects the texture, our proposed coupled approach is able to capture  a  style in both texture and shape, suggesting that the style of the drawing is coupled with the drawing process itself.

<p align="left">
    <img src="images/styleclipdraw_main_example.png" height="400" title="">
</p>

## Method


<p align="left">
    <img src="images/method.PNG" height="400" title="">
</p>



## Results


<p align="left">
    <img src="images/results.png" height="400" title="">
</p>


## StyleCLIPDraw vs. CLIPDraw then Style Transfer


<p align="left">
    <img src="images/style_transfer_results.png" height="600" title="">
</p>

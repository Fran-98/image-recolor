# Image Recolor 🖌️
Final project of the course "Diplomatura en Ciencia de Datos - UNNE"

## Architecture
The proposed neural network is using a UNET architecture, whichs main use is on the field of image segmentation. For this project is being re-imagined as an image transformation NN.
![UNET architecture](assets\unet.png)
Some changes were made to the UNET so it can perform the task better, such as adding attention gates into the decoder blocks and adding a 

![Proposed architecture](assets\unet_propuesta.png)

## Datasets
- https://www.kaggle.com/datasets/balraj98/monet2photo
- https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite
- https://www.kaggle.com/datasets/marcinrutecki/old-photos

## Related works
- [Colorful Image Colorization](https://arxiv.org/abs/1603.08511)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

## Extras
As the final and intended use of the neural network is for recoloring vintage photos a proccesing library was needed to transform modern photos into black and white vintage photos. As there was no open source solution for this, an alternative was created and is available on [pypi](https://pypi.org/project/vintager/) and [github](https://github.com/Fran-98/vintager) as "vintager".


## TODO
- [ ] Limpiar readme y completarlo
- [ ]

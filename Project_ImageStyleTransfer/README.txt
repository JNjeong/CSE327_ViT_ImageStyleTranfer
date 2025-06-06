[Project Overview]
- ViT+SNN-based Style Transfer and various Hybrid architecture experiments
- Includes comparison with CNN(VGG)-based style transfer, analysis of ViT feature limitations, and experimental improvements with Hybrid structures

[Main Usage]
1. Training:
   python train.py
   - Result images are saved in the output/hybrid_vitstyle folder, with filenames containing epoch, content, style, style_weight, and other key information

[Folder Structure]
- data/content: Content images
- data/style: Style images
- output/hybrid_vitstyle: ViT+SNN/Hybrid result images
- output/cnn: CNN(VGG) style transfer result images
- checkpoints: Trained model parameters
- models, utils: Main network and utility code

[Key Hyperparameters]
- style_weight: Style effect strength (increases by epoch, up to 1.0)
- perceptual_weight: Perceptual (VGG feature) loss weight
- content_weight: Content loss weight

[Notes]
- Content images are used in order (1, 2, 3, ...), style images are used once each, then assigned randomly
- Adding perceptual loss (VGG feature loss) was experimentally found to be the most effective for both detail and style effect
- You can adjust style_weight, perceptual_weight, etc. to control the balance between style strength and content detail

[How to run]
- For CNN: python cnn_style_transfer.py 
- For ViT: python train.py
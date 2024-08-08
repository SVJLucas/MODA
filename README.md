# MODA
MODA (Multimodal Object Description Assistant) is an AI model that describes fashion items by combining visual and textual data. Powered by FashionCLIP and OPT, MODA provides descriptions of various fashion objects. 

<p align="center">
  <img width="700" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/c5fff5bf-13a8-4999-8c11-27736503581e">
</p> 

<p align="center">
  <img width="700" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/4010d574-1a82-4594-82b9-5071e0459daa">
</p> 

## Training

The MODA model is designed to generate detailed descriptions for fashion items by integrating visual and textual data. This hybrid approach leverages the capabilities of both the Fashion CLIP model and the OPT-125M language model. The Fashion CLIP model processes the input image to generate an image embedding, which is then projected to a suitable size through a non-linear projection layer. Simultaneously, the OPT tokenizer converts the text description into token embeddings. These image and text embeddings are combined into a sequence of embeddings, which are then fed into the OPT-125M model. This pre-trained transformer model processes the combined embeddings to generate or refine the descriptive text, enhancing the detail and accuracy of fashion item descriptions.
<p align="center">
  <img width="1200" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/9f846bb8-13c5-413b-8de6-7e05677879ae">
</p> 

 The training process employs the AdamW optimizer with a learning rate of 1e-3, and the learning rate is adjusted using a StepLR scheduler. The CrossEntropyLoss function is used to compute the loss, and training is conducted on a CUDA-enabled GPU when available. The model is trained for 20 of epochs, taking 1h44min to train on a Google Colab A100, with gradient accumulation and periodic model saving to ensure stability and performance.
 
<p align="center">
  <img width="600" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/85813aac-d8a0-4c7e-9ef9-fb81f5f62b3d">
</p> 




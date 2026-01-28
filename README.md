#Introduction
Proto-FiNet is an open-source semantic segmentation toolbox based on PyTorch, pytorch lightning and timm, which mainly focuses on developing advanced Vision Transformers for remote sensing image segmentation. The following is a brief introduction to the method.

Semantic segmentation of remote sensing imagery is a pivotal task in computer vision, bridging advanced computational techniques with critical real-world applications such as land use mapping, environmental monitoring, and urban planning.  Despite remarkable progress driven by deep learning—particularly convolutional neural networks (CNNs) and Transformers—existing methods still struggle with complex backgrounds, multi-scale targets, and limited annotated data, leading to suboptimal performance for small objects and fine-grained categories.  To address these challenges, we propose ProtoFiNet, a novel semantic segmentation model that integrates prototype learning with foundation model knowledge to enhance feature representation and segmentation accuracy.  Unlike conventional approaches relying on randomly initialized prototypes, ProtoFiNet enriches prototypes with semantic information from the pre-trained ScaleMAE model, fine-tuned via Low-Rank Adaptation (LoRA) for efficient adaptation to remote sensing scenarios.  The model incorporates a Contextual Prototype Refinement Branch (CPRB) for dynamic prototype updating, coupled with two novel loss functions—Prototype Disentanglement Optimization (PDO) and Prototype Compactness Contrastive (PCC) Loss—to enhance intra-class compactness and inter-class separability.  Here we show that ProtoFiNet achieves state-of-the-art performance across three benchmark datasets: ISPRS Vaihingen (mIoU = 84.68%), ISPRS Potsdam (mIoU = 87.64%), and LoveDA (mIoU = 52.81%, outperforming existing methods by 1.56–3.1% in mIoU.  These results highlight the model’s effectiveness in segmenting small targets and complex backgrounds, offering a robust technical pathway for applying large pre-trained models to remote sensing image analysis.  Beyond specific applications, this work contributes to the broader field of computer vision by demonstrating how semantic prototype guidance can mitigate limitations in low-data and multi-scale segmentation tasks, with implications for diverse image analysis domains requiring precise pixel-level classification.

##wwww


{


}

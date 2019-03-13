# paper-list

## Embedding and Metric Learning Approaches:
1. Siamese neural networks for one-shot image recognition 
    * G. Koch, R. Zemel, and R. Salakhutdinov. In ICML Workshop, 2015.
    * Input: 2 (paired input) Output: 1 (same or different)

2. Learning to Compare: Relation Network for Few-Shot Learning
    * F. Sung, Y. Yang, L. Zhang, T. Xiang, P. H.S. Torr and T.M. Hospedales. In CVPR, 2018.
    * A RN is able to classify images of new classes by computing relation scores between query images and few examples of each new class without further updating the network.

3. Prototypical Networks for Few-shot Learning
    * J. Snell, K. Swersky, and R. S. Zemel. In NIPS, 2017.
    * Learn a metric space in which classification can be performed by computing distances to prototype representations of each class.

4. Meta-Learning for Semi-Supervised Few-Shot Classification
    * Improvement of Prototypical Network (consider unlabeled examples).

## Gradient Base Optimization (Easy to Fine-Tune):
1. Optimization as a model for few-shot learning
    * S. Ravi and H. Larochelle. In ICLR, 2017.
    * Use LSTM to learn approximate parameter updates specifically for the scenario, and also learn a general initialization of the learner.

2. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (MAML)
    * C. Finn, P. Abbeel, and S. Levine. In ICML, 2017.
    * The parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on this task.

3. Meta-sgd: Learning to learn quickly for few shot learning
    * Z. Li, F. Zhou, F. Chen and H. Li. arXiv preprint arXiv:1707.09835, 2017.

4. Recasting Gradient-Based Meta-Learning as Hierarchical Bayes
    * E. Grant, C. Finn, S. Levine, T. Darrell, T. Griffiths. In ICLR, 2018.
    
## Memory Based:
1. Meta-Learning with Memory-Augmented Neural Networks (MANN)
    * A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra, and T. Lillicrap. In ICML, 2016.

2. Meta Networks
    * T. Munkhdalai and H. Yu. arXiv preprint arXiv:1703.00837, 2017.
    * Continuous learning or incrementally learning new concepts on the fly.MetaNet consists of a base learner, a meta learner and is equipped with an external memory(MANN).
    * Use gradient for meta information.

## Mixture:
1. Learning to Learn with Conditional Class Dependencies
    * X. Jiang, M. Havaei, F. Varno, G. Chartrand, N. Chapados, S. Matwin. Submitted to ICLR,2019.
    * Make use of structured information (metric space information) from the label space at Conditional Batch Normalization to help representation learning. 
    * Use MAML for meta train.

2. Meta-Learning with Latent Embedding Optimization
    * A. A. Rusu, D. Rao, J. Sygnowski, O. Vinyals, R. Pascanu, S. Osindero, R. Hadsell. Submitted to ICLR, 2019.
    * Learn a data-dependent latent generative representation of model parameters, and performing gradient-based meta-learning in this low-dimensional latent space.
    * Use encoder and decoder.

## Incremental Learning:
1. Incremental Learning of Object Detectors without Catastrophic Forgetting [[pdf]](https://arxiv.org/pdf/1708.06977.pdf)
    - K. Shmelkov, C. Schmid, K. Alahari. ICCV, 2017.

2. iCaRL: Incremental Classifier and Representation Learning. [[pdf]](https://arxiv.org/pdf/1611.07725.pdf)
    - S.A. Rebuffi, A. Kolesnikov, G. Sperl, C. H. Lampert. CVPR, 2017.

3. Dynamic Few-Shot Visual Learning without Forgetting [[pdf]](https://arxiv.org/pdf/1804.09458.pdf)
    - S. Gidaris, N. Komodakis. CVPR, 2018.
    
4. Gradient Episodic Memory for Continual Learning. [[pdf]](https://arxiv.org/pdf/1706.08840.pdf)
    - D. Lopez-Paz and M. Ranzato. NIPS, 2017.

5. Continual lifelong learning with neural networks : A review. [[pdf]](https://www.sciencedirect.com/science/article/pii/S0893608019300231)
    - G. I. Parisi, R. Kemker, J. L. Part , C. Kanan, S. Wermter. Neural Network, 2019.
6. End-to-End Incremental Learning. [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf)
     - Francisco M. Castro, Manuel J. Mar´ın-Jimenez, Nicolas Guil, Cordelia Schmid3, and Karteek Alahari3

## Spectral Clustering
1. Spectral Graph Reduction for Efficient Image and Streaming Video Segmentation [[pdf]](https://fgalasso.bitbucket.io/Files/Equivalence/Spectral_equiv_cvpr14.pdf)
   - Fabio Galasso, Margret Keuper, Thomas Brox, Bernt Schiele. IEEE, 2014.
2. Deep Spectral Clustering Learning [[pdf]](https://www.cs.toronto.edu/~urtasun/publications/law_etal_icml17.pdf)
   - Marc T. Law 1 Raque, Urtasun, Richard S. Zemel. ICML, 2017.
3. SpectralNet: Spectral Clustering using Deep Neural Networks [[pdf]](https://openreview.net/pdf?id=HJ_aoCyRZ)
   - Uri Shaham, Kelly Stanton, Henry Li, Ronen Basri, Boaz Nadler, Yuval Kluger. ICRL, 2018.
   
## Information Distillation   
1. Learning without Forgetting [[pdf]](https://arxiv.org/pdf/1606.09282.pdf)
   - Zhizhong Li, Derek Hoiem, Member, IEEE
2. Distilling the Knowledge in a Neural Network [[pdf]](https://arxiv.org/pdf/1503.02531.pdf)
   - Hinton, G., Vinyals, O., Dean, J. In: NIPS workshop, 2014.

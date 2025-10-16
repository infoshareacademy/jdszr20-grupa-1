# ML-based fake news classification models and online fake news gamification app
The final group project in the infoShare Academy Data Science boot camp, conceived and developed by **ADAM**:
 - **A**licja Szpunar-Szałek (<a href="https://github.com/AlicjaSzpunarSzalek">AlicjaSzpunarSzalek</a>)
 - **D**ominika Kokoryk (<a href="https://github.com/dominija">dominija</a>)
 - **A**drian Komuda (<a href="https://github.com/Naioku">Naioku</a>)
 - **M**ichał Alenowicz (<a href="https://github.com/michal-alenowicz">michal-alenowicz</a>)

## Overview
We combined several real-world fake news datasets available online to train and validate various machine learning models for binary classifiaction of news of up to 50 words (labels: fake = 0 or fake = 1). Embeddings obtained from Hugging Face sentence-transformers were used. The obtained metrics were compared with those obtained using more "traditional" TF-IDF vectorization. Ultimately, Support Vector Classifier with polynomial (degree = 2) kernel was chosen and used with embeddings from 'all-mpnet-base-v2', achieving the accuracy of 0.84 on a balanced test set. Additionally, a bonus synthetic validation dataset was obtained from ChatGPT and curated for quality. In this case the best generalization was shown by the XGBoost classifier (accuracy of 0.81 on unseen synthetic data). **Both models are showcased in <a href="https://huggingface.co/spaces/Naioku/adam">our online app</a>, which raises awareness of the fake news problem by gamification. Can you spot the single fake piece of news among the 5 randomly drawn from our test datasets? Our models will be guessing as well. Can you beat them?**

- PDF presentation <a href="presentations/Presentation_short_PL_ADAM.pdf">in Polish</a> (English version coming soon)
- <a href="https://huggingface.co/spaces/Naioku/adam">Online app</a> (three modes: play vs. SVC on test data / play vs. XGB on ChatGPT data / input your own text and test the SVC model )

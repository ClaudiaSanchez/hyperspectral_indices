# Genetic Programming for the generation of Hyperspectral Indices in Crops

Nowadays, the use of hyperspectral images in the agroindustry is growing because it represents a non-invasive methodology for identifying the internal properties of food products. Hyperspectral indices are common in satellite and aerial images for identifying vegetation, water, etc. This respository proposes a methodology for generating hyperspectral indices on a minor scale to identify dry matter in crops based on Genetic Programming. We use the correlation coefficients Pearson and Kendal as fitness functions. We use the datasets of four crops: apple, broccoli, leek, and mushroom. 

The description of the files is as follows:
- clau_genetic_programming.py contains the main code of Genetic Programming
- clau_gp_experiments.ipynb has the code for running the experiments on the four datasets: apple, broccoli, leek, and mushroom.
- clau_GPtree_fromExpersion.ipynb validates the results of the indices generated in the experiments

The datasets are the ones published in:
[Malounas, I., Vierbergen, W., Kutluk, S., Zude-Sasse, M., Yang, K., Zhao, M., ... & Fountas, S. (2024). SpectroFood dataset: A comprehensive fruit and vegetable hyperspectral meta-dataset for dry matter estimation. Data in Brief, 52, 110040.](https://www.sciencedirect.com/science/article/pii/S2352340924000143)


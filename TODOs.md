### CV: 
- [x] Separate each episode in 2 parts
- [x] Implement an inner CV loop:
  - [x] Train model for each hyperparam combi
  - [x] Compute metrics
  - [x] For each hyperparam combi, average metrics
  - [x] Find best hyperparam combi based on average metrics
  - [x] Train full model with chosen hyperparam options on the full inner CV data
- [x] Implement outer CV loop:
  - [x] Compute metrics on test fold for each model type
  - [x] Compare models based on metrics computed on test fold


### SimMod 1
Detecting Kermit and Gentlemen
#### Video features:
- [ ] Color histogram:
  - [x] Create color histogram
  - [x] Check color histograms and dominant color for Kermit and Gentlemen
  - [x] Use the colors as feature for classifier (present in the frame or not)
  - [x] Kermit Hue 25-50
- [ ] Use pictures with colors changed to primary colors by clustering as input for blob detection
- [ ] Try to improve feature for lines for gentlemen balcony (do as feature lines present in a specific part of the image and intersection)
- [x] Foreground detection + optical flow
#### Audio features:
- [x] RMS
- [x] ZCR
- [x] FT + 1st peak
#### Run experiments:
- [ ] Random Forest
- [ ] KNN
- [ ] Decision Tree


### SimMod 2
Detecting Swedish chef and Miss Piggy
#### Video features:
- [ ] i vector
- [ ] DCT
- [ ] texture with DCT
#### Audio features:
- [x] MFCC 
- [ ] ???
- [ ] ???

#### Run experiments:
- [ ] SVM
- [ ] Markov
- [ ] Gaussian

### Hand-In & Evaluation
#### SimMod 1
- [ ] ROC curves
- [ ] precision-recall curves
#### SimMod 2
- [ ] ROC curves
- [ ] precision-recall curves
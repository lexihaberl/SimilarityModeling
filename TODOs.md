### CV: 
- [ ] Separate each episode in 2 parts
- [ ] Implement an inner CV loop:
  - [ ] Train model for each hyperparam combi
  - [ ] Compute metrics
  - [ ] For each hyperparam combi, average metrics
  - [ ] Find best hyperparam combi based on average metrics
  - [ ] Train full model with chosen hyperparam options on the full inner CV data
- [ ] Implement outer CV loop:
  - [ ] Compute metrics on test fold for each model type
  - [ ] Compare models based on metrics computed on test fold


### SimMod 1
Detecting Kermit and Gentlemen
#### Video features:
- [ ] Color histogram:
  - [ ] Create color histogram
  - [ ] Check color histograms and dominant color for Kermit and Gentlemen
  - [ ] Use the colors as feature for classifier (present in the frame or not)
- [ ] Use pictures with colors changed to primary colors by clustering as input for blob detection
- [ ] Try to improve feature for lines for gentlemen balcony (do as feature lines present in a specific part of the image and intersection)
- [ ] Foreground detection + optical flow
#### Audio features:
- [x] RMS
- [x] ZCR
- [ ] FT + 1st peak
#### Run experiments:
- [ ] Random Forest
- [ ] KNN
- [ ] Vector Space


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
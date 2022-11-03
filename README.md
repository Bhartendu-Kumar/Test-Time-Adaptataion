# Test-Time-Adaptatation
We propose a novel method for TEST TIME MODEL ADAPTATION and demonstrated for OPTICAL FLOW

## For adapting to the test dataset: TENT paper proposes minimizing Entropy of Prediction as loss function

## But it is Highly Non-Trivial to find ENTROPY of OPTICAL FLOW Models as these are real number output and there are no classes.

## Thus we propose, novel loss: L(.) = variance + C (change in mean)

## The chracteristics of this loss functions are: (i) differentiable , (ii) closely approximates entropy , (iii) proved to be working 

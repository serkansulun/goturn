This is the implementation of the object tracker, GOTURN: Generic Object Tracking Using Regression Networks, written in Julia language using Knet: Koc University Deep Learning Framework.

Paper: https://arxiv.org/pdf/1604.01802.pdf

Needed datasets: For training, VOT2014. For testing ALOV300+.

Arrangement of the dataset folders should be as the following:

|-goturn.jl

|-datasets

|---trn

|-----annotations

|-----images

|---tst

|-------(video folders)

|---val

|-----annotations

|-----images

Check inside main function for arguments. Default arguments provide best tested results.

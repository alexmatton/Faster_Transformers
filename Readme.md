### Faster Transformers

Faster Transformers is the project Amaury Sabran (@https://github.com/amaurySabran) and I decided to work on for Stanford CS224N 2018/19 class.

The purpose of the project was to apply the Transformer Architecture to summarization. We benchmarked several architectures close to the Transformer and derived new models that make it both faster and more efficient for this specific task.
We also analyzed the speed of each component of the Transformer to determine where the overall architecture can be improved for real-life applications.

Our final report can be found here: https://github.com/Nutemm/Faster_Transformers/blob/master/Project_final_report%20CS224N.pdf

## Requirements

* Install torchvision (https://pytorch.org/)
* Install our fairseq fork from sources: https://github.com/amaurySabran/fairseq
* Install tensorflow (for dataloading)
* Install py-rouge (pip install py-rouge)

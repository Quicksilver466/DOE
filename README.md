# Disentanglement of Experts

This project is potentially a novel attempt on task based MoEs(Mixture of Experts) and its potential extensions. This project aims to solve the issues faced due to dearth of task specific data. For ex. Suppose we are creating an assistant for legal domain, where the assistant would answer any legal questions asked to it. Training this assistant might require specific legal data of lawyers talking to GPT or other lawyers in a conversational style. But obtaining this data might not be an easy endeavour since curating such data is expensive and trying to find an open source variation might not yeild any result as this is quite a niche domain. This is where DOE comes in. With DOE we would have one expert who would be trained on general and authentic legal texts, codes, statutes and hearings(the raw next token completion task, not the instruction tuning SFT/DPO) and another expert who would be well versed in handling the generic coversational flow of data. When these experts would work together we should have with us an assistant who can answer legal questions coherently. This paradigm can be extended anywhere as long as we are able to decompose a specific task into some rational number of generic tasks.

This idea has taken shape at the courtesy of motivations from a diverse set of techniques and past works. The basic technical idea and goal of this project in its rudimentary form is to have a Transformer based model that hosts plethora of experts pertaining to their field of specializations.

These experts code / architecture wise are just like the experts we have in Mixture of Experts i.e MLP layer replaced by specialized MLP layers(experts / sparse MoE layers). The difference is in the routing mechanism where instead of routing individual tokens to different experts, we would be routing complete sequence to different experts. Another difference would again be in the routing mechanism and the number of experts. In normal MoE layers the number of experts are fixed and the routing network is made to only cater those experts. But with this project I hope to make the model more flexible with respect to number of experts and type of experts. Each expert would be associated with a set of gating weights which would decide whether or not the expert should process the input or not. Essentialy each expert would choose the type of input it wants to process.

![DOE-gate-fixed](https://github.com/Quicksilver466/DOE/assets/40929815/ea719338-811f-4ad6-914f-59bf71a9d393)


The next goal is to further modify the experts so that any and all modalities can be tackled. This will open tons of multimodal applications for the Model.

The last and most important goal is to establish a way to induce cohesion between the Experts. This goal is highlighted in the example mentioned at the beginning.

This format can be extended everywhere and even in other modalities. This will eliminate the need for task specific data and make the task more approachable making it a significant advantage for the user. Apart from that I expect the network to maintain the performance advantages achieved by the traditional MoEs and even amplify it as we would know the complete batching after processing the input only once.

There are other architectual details which I plan to elaborate once the experimentation is done.

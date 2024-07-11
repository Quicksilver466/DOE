# Disentanglement of Experts

This project is a novel attempt on task based MoEs(Mixture of Experts) and its potential extensions. This idea has taken shape at courtesy of motivations from a diverse set of techniques and past works. The basic idea and goal of this project in its rudimentary form is to have a Transformer based model that hosts plethora of experts pertaining to their field of specializations.

These experts code / architecture wise are just like the experts we have in Mixture of Experts i.e MLP layer replaced by specialized MLP layers(experts / sparse MoE layers). The difference is in the routing mechanism where instead of routing individual tokens to different experts, we would be routing complete sequence to different experts. Another difference would again be in the routing mechanism and the number of experts. In normal MoE layers the number of experts are fixed and the routing network is made to only cater those experts. But with this project I hope to make the model more flexible with respect to number of experts and type of experts. Each expert would be associated with a set of gating weights which would decide whether or not the expert should process the input or not. Essentialy each expert would choose the type of input it wants to process.

![DOE-gate-fixed](https://github.com/Quicksilver466/DOE/assets/40929815/ea719338-811f-4ad6-914f-59bf71a9d393)


The next goal is to further modify the experts so that any and all modalities can be tackled. This will open tons of multimodal applications for the Model.

The last goal is to establish a way to induce cohesion between the Experts. So for example if we are training an LLM to be a Legal Assistant, then we might need conversation data between a Lawyer and a LLM assitant. But this is a very niche domain, and we might not always have the data for the very specific task we have at hand. So the goal would be that we would use authentic and expansive Legal documents and texts to pretrain(the raw next token completion task, not the instruction tuning SFT/DPO) the legal expert, while have the conversational expert be trained(SFT and DPO training to follow chat format and follow instructions) on generic conversations dataset like Alapaca or SlimOrca. Thus one expert would have all the understanding about the Legal text obtained through easily available general information and another expert would be trained on following instructions and chat format. After working together they should use their combined knowledge to answer specific Legal queries.

This format can be extended everywhere and even in other modalities. This will eliminate the need for task specific data and make the task more approachable making it a significant advantage for the user. Apart from that I expect the network to maintain the performance advantages achieved by traditional MoEs and even amplify it as we would know the complete batching after processing the input only once.

There are other architectual details which I plan to elaborate once the experimentation is done.

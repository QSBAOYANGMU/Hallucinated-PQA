# Hallucinated-PQA. This paper has been accepted by EXPERT SYSTEMS WITH APPLICATIONS (ESWA)
Hallucinated-PQA:No Reference Point Cloud Quality Assessment via Injecting Pseudo-Reference Features##
Point clouds (PCs) have been increasingly applied in business and life, but a variety of PC transmission and generation systems inevitably produce various types of distortions. Therefore, it is of great significance to design an objective point cloud quality assessment (PCQA) particularly in a no-reference manner to evaluate the PC systems in the actual situation. However, due to the lack of original PCs, the existing no-reference (NR) PCQA metrics cannot perceive the feature changes caused by distortions, resulting in inaccurate quality prediction. In addition, dependent on the viewing habits of the subjects in the subjective evaluation experiment, a strong contextual correlation among intra- and inter-view local regions at different scales naturally exists, but the existing projection-based no-reference PCQA only using concatenation or average pooling operations cannot reflect this relationship of multi-view features fusion. Considering the above challenges, we propose a novel hallucination-guided NR PCQA framework, namely Hallucinated-PQA. Specifically, we introduce a distortion restoration network to correct multiple projected images in preprocessing to provide pseudo-reference information for NR PCQA. In the feature extraction of distorted PCs, we de-signed a hallucination injection block (HIB) by utilizing feature differences to assist the feature description of distorted PCs, and a multi-view and multi-scale context fusion (MMCF) module to construct the contextual correlation among intra- and inter-view local regions at different scales. Experimental results show that our Hallucinated-PQA can achieve comparable or better performance than state-of-the-art (SOTA) metrics on four open PCQA databases.  The source code will be released at https://github.com/QSBAOYANGMU/Hallucinated-PQA.
The code is coming soon!
![image](https://github.com/QSBAOYANGMU/Hallucinated-PQA/assets/91246967/266e4042-2fd9-4f1a-be55-10ea44acff81)
![image](https://github.com/QSBAOYANGMU/Hallucinated-PQA/assets/91246967/751c5554-5f83-4ca0-aee3-e4f64c367972)
![image](https://github.com/QSBAOYANGMU/Hallucinated-PQA/assets/91246967/89f3309b-aa53-40f3-af4e-0e3465e4f969)


We also uploaded the k-fold predictions of Hallucinated-PQA in the  WPC(740) , SJTU-PCQA(378),IRPC(54),and M-PCCD(232) databases.


![Prediction](https://user-images.githubusercontent.com/91246967/230887584-b6b37656-0e46-4b91-a05a-0a940add6808.png)

WPC_1:1:banana
2:cauliflower
3:mushroom
4:pineapple

WPC_2:5:bag
6:biscuits
7:cake
8:flowerpot

WPC_3:9:glasses_case
10:honeydew_melon
11:house
12:pumpkin

WPC_4:13:litchi
14:pen_container
15:ping-pong_bat
16:puer_tea

WPC_5:17:ship
18:statue
19:stone
20:tool_box






For the k-fold split of WPC(740) and SJTU-PCQA(378) databases, please refer to VQA-PC:https://github.com/zzc-1998/VQA_PC

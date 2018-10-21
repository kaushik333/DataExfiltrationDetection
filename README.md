# DataExfiltrationDetection | PyTorch

This is a Machine Learning framework to detect Data Exfiltration; specifically DNS exfiltration. 
We will be using the [Data Exfiltration Toolkit](https://github.com/sensepost/DET) framework to generate synthetic data to test out our algorithm. Specifically, we will be using it to generate a mixture of "Normal" traffic and "Malicious" traffic. This ML framework is aimed at doing a packet wise classification of Normal/Malicious. The overall pipeline of the algorithm is illustrated below: [pic](https://github.com/kaushik333/DataExfiltrationDetection/blob/master/pipeline.PNG)

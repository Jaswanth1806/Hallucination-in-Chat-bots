# Hallucination in Chat-bots

**Faithful Benchmark for Information-Seeking Dialogue - Fact Hallucinations Detection and Prevention**

This repository contains code for the final project of the course "CSE 635: Natural Langauge Processing and Information Retrieval" and detailed instructions to run the code.


**Abstract:** <br>
Chatbots can provide a convenient and efficient means of communication, but they are not immune to errors. Hallucination is one of the most significant problems associated with chatbots. When a chatbot provides factually incorrect responses, it can be detrimental to the user and the chatbot's performance. Hallucinations can result in users receiving misinformation or even harm. To address this issue, we propose a model to detect and prevent hallucinations in Information-Seeking Dialogue-based NLP systems. In this project, we explore various techniques to identify and reduce hallucinations in NLG models. Our approach involves combining rule-based techniques and machine learning algorithms to detect potential hallucinations in the model's output. Once identified, we prevent the model from generating such responses in the future by modifying the training data, reweighting model parameters, or introducing a feedback loop with human experts. Our project aims to enhance user trust, improve the overall quality of chatbots, and reduce the potential harm caused by misinformation.




**Instructions to run the code:**

There are two folders present in this repository - milestone-2 and milestone-3. The milestone-2 folder contains our baseline model code which is RoBERTa-large model. The milestone-3 folder contains our final model code which is a modified version of the RoBERTa-large model.

1) For task-1: "python task1.py" - it will train and test the model. And after training, it will save the generated model as task1_model.pth. If you want to just test the model, you can comment out trainer.fit and torch.save lines from the code.
2) For task-2: BEGIN and VRM Classification: "python task2.py" - it will train and test the model. After training, it will save the generated model as task21_model.pth and task22_model.pth respectively. If you want to just test the model, you can comment out trainer.fit and torch.save lines from the code.
3) For task-3: "python task3.py" - you need to run task-1 before you run task-3 so that the task1_model.pth is generated which will then be used to classify whether the generated responses are hallucinated or not.

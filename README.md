# pytorch_transformer_translator
**Created a transformer from scratch, from it's original paper (Attention Is All You Need) using pytorch have a look at model_new.py for details).**

The language data is from huggingface (use pip install datasets to get it). I've only trained my model on english to italian, but other langauges combinations are possible if you change
the configuration in trainandgetdata.py  

The model itself is in the model_new.py file.   

I then used the trainandgetdata.py file for training. Training can be automatically resumed from the latest weights file (produced after each epoch) if for whatever reason it crashes.
The number of epochs can be modified in the trainandgetdata.py file.  

The inference.py file can be used for basic inferencing (i.e. to test it) after training. It automatically finds the weights file of the latest epoch. Unfortunately, I had to ommit
the weight files as they are each around 800MB (the code will automatically find the latest weights file in the folder and use it.)  

I used use startflaskserver.py to create a web sever. After running this python script, go to 127.0.0.1:5000 to try it out.  

Demo:
![demooftransformertranslator](https://github.com/user-attachments/assets/c58146c5-975e-431b-b6d3-6809156ee6cf)

As you can see, it works in essence. The translation is actually closer to "Here's a sentence" - but remember, I only trained the model for 30 epochs (around 9 or 10 hours on my RTX
4060 GPU.)

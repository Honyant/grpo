# janky GRPO v1

So I tried implementing GRPO from a blog post I found ["A vision researcher's guide to some RL stuff: PPO & GRPO"](https://yugeten.github.io/posts/2025/01/ppogrpo/) at like 3am. ggs
Basically I tried making LLMs better at the task of fermi estimation using GRPO. I found a dataset of fermi estimation questions and answers on huggingface [https://huggingface.co/datasets/voidful/fermi](https://huggingface.co/datasets/voidful/fermi)

## running this:

You need a gpu with a solid amount of memory. The more memory, the bigger the group size (though I will implement a batch accmulation for group advantage later).
```
pip install -r requirements.txt
python grpo.py
```


## how it works:
Go read the blog post but its basic stuff - rewards for right answers, implemnted group advantage and the grpo loss (absolutely ridiculously simple (can you even call it rl if it throws the bellman equation out the window!)) from the blog, and implmented some kl penalties against the base model.

maybe ml/rl theory is a scam lmfao

## everything that went wrong:

- Tiny models (1B params) are useless for this (who could've guessed), their priors are only capable of guessing terribly and the rewards are wayyy too sparse to learn anything. at least for this dataset.
- but RAHHHHH i got reward:
![picture 0](https://i.imgur.com/Ngcoilc.png)  

- The sad gpu im running this on is dying (JENSEN GIVE ME a 8x GB200 NVLINK CLUSTER FOR MY BIRTHDAY)
- Deepseek is actually capping about their expenditure, this needs serious compute to run
- Might work better with bigger models but good luck running those on a tight budget. ill try getting a little more compute and trying it on that.

## summary:

I did get GRPO kinda working! It got some reward (unfortunately rarely). And in theory this base algorithm doesnt need much more modification to actually work (unless im missing something), and all thats needed is scale (bitter lesson baby). I shall sleep in peace now.

## future work:

- implement batch accmulation for group advantage
- try it on bigger models, with more data, and more compute, or with longer sequence lengths
- more 𝓅𝓇𝑜𝒻𝑒𝓈𝓈𝒾𝑜𝓃𝒶𝓁 readme and make a blog post
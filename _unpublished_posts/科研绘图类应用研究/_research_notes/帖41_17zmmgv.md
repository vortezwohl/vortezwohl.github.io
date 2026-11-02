# 帖41 Tikz via AI
- URL: https://www.reddit.com/r/LaTeX/comments/17zmmgv/
- r/LaTeX | 2023-11-20 | score:15 | ratio:0.9 | 评论:13(实抓12, moreLeft:0) | 作者:Silberauge | flair:
- 正文: Hello TeX-Community, With the currentbirse of AI I wonder, whether there is a decent AI to produce Tex-code for tikz-figures. Are charGPT and their likes capable? If so, which is the current best? Greetings Silberauge

[2023-11-20|swissbuechi|13] I've used chatGPT to create a simple diagram with overlapping circles. It took me about 3–4 tries to get it exactly how I've imagined it to look. The AI was definitely capable of understanding my initial request and corrections.
[2023-11-20|thesolitaire|4] chatGPT can write tikz, but I can't vouch for the quality. I think (as with most things) GPT-4 is somewhat better at it.
[2023-11-20|diaracing|3] As long as the chatbot doesn't compile the code and interpret the output, we will be struggling with it.
  [2023-11-21|MMazinCC|1] Tbh it is very good. I first used latex for my thesis and if not for gpt I would not be able to create the amazing figures that I have in my thesis rn. But to be fair I used pgfplots which is much easier as it doesn't need to really interpret the output to check if things are placed coreectly, which I found to be an issue it faces with tikz figures.
[2023-11-20|Mahkda|3] I asked chatGPT to draw me a house in tikz, it looked like a 3 years old drawing, but it worked (and used some functionality that I didn't know about tikz) but I had to ask him to fix his mistakes
[2023-11-21|MMazinCC|2] Gpt plus with a plugin for code writing is brilliant for this. However its skills in pgfplots are much higher compared to creating tikz figures using tikz directly. Usually it is unaware of where to place things so things are almost always out of place. However, what I do is I ask it to always leave the x and y positions empty and I would fill them up myself.
[2024-06-05|adityaguru149|2] https://github.com/potamides/DeTikZify is new and seems nice
[2025-10-07|basilyusuf1709|1] You should try: https://tools.useoctree.com/tools/tikz-generator - it’s a free tool to do what you are asking
[2023-11-20|ca326|1] I've tried it with Chat GPT and I was a bit disappointed. It could give me the code to create simple diagrams, which compiled very well. But when using these simple diagrams and adding elements that made them more complicated, I couldn't make it understand what I wanted it to do. Not even after several correction rounds. The code compiled, but the output didn't make much sense to me.
[2023-11-27|Raldditor|1] Has anyone been able to upload an image of a figure/sketch and get GPT-4 to produce TikZ code that recreates it?
  [2024-03-08|[deleted]|1] working on it rn
    [2024-03-08|[deleted]|1] its an iterative process that has to use NLP as a go-between - the main issue is ensuring a clean interpretation.
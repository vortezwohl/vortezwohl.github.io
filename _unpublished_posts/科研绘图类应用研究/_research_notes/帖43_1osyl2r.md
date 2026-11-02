# 帖43 Hello with origin pro
- URL: https://www.reddit.com/r/PhD/comments/1osyl2r/
- r/PhD | 2025-11-09 | score:19 | ratio:0.81 | 评论:26(实抓26, moreLeft:0) | 作者:wohren_osso | flair:Seeking advice-academic
- 正文: Hi! So, after using peak fitting on a graph, I can't move the peaks. Is like for them to have an offset so they're not overlapping each other, but it seems to be like they're looked in place. Even if I go to the stack menu in okie details and check the individual offset option, I can't move anything. Does any one know what I have to do to get to move them? P.S. sorry for the bad photos

[2025-11-09|AutoModerator|1] It looks like your post is about needing advice. Please make sure to include your *field* and *location* in order for people to give you accurate advice. *I am a bot, and this action was performed automatically. Please [contact the moderators of this subreddit](/message/compose/?to=/r/PhD) if you have any questions or concerns.*
[2025-11-09|Physix_R_Cool|35] Bro, you have fitted the model to the data. Ain't no moving that.
  [2025-11-10|wohren_osso|7] What I want is to have them stacked so the curves are visible. There are four curves in that photo but, since they're overlapped, they're barely visible.
    [2025-11-10|IDontWantToGrowUpYet|11] I would make your raw signal a dashed line, and play with the transparency and colours of the fitted curves. Don’t move these if you fitted them elsewhere.
    [2025-11-10|Physix_R_Cool|4] Set the alpha of the lines to like 0.5
    [2025-11-13|Ok-Ease5589|1] It seems there is the data(black), two gaussian fits(or lorrentzian or some other distribution) for the two peaks (green) and the sum of the two fits (red) plotted here. I would not consider the data a curve it is a collection of discrete points that you are fitting to. Plotting data as anything other than individual points with error bars I feel is intentionally misleading to the reader, but many people may disagree. There should be fitting parameters associated with the two gaussian fit curves that you can use to reconstruct them in another plotting software or even in origin itself. You can also move each of the fit curves to their own layers and stack them however you want and change the…[截断]
[2025-11-10|Turtle-from-hell|7] Set measured data as symbols, and approximation as dashed Everything will be visible Or if you wanna just lift them up, you can viabdifferent layer added forbthe approximation, but then you need the additional axis as well, to keep data real, so that makes no sense hahaha
[2025-11-10|[deleted]|3] Wait, do you guys use special software to plot?? Am I too stupid to just use Matplotlib and Plotly, Lol
  [2025-11-10|Physix_R_Cool|5] In some fields the students don't learn programming
  [2025-11-10|Marz1panCake|2] Depending on the data type, OriginPro can significantly accelerate your workflow. It can also make it very slow. It depends on data type, plot type, experience, and whether analysis is required. I recommend my students to learn both.
    [2025-11-11|QuasiNomial|2] Hard disagree, python is faster in literally every way. Origin is only good if you can’t program basic plotting.
      [2025-11-13|Ok-Ease5589|2] Origin is good for making plots that you can later export to Illustrator to make it look nice for publication.
[2025-11-10|andrewsb8|2] Just changed some of the lines to dotted, dashed, etc to make them more visible. Add a legend so its apparent that curves are there that overlap.
[2025-11-10|ElephantShell22|2] Is that Raman data? 👀
[2025-11-10|vivazapataa|2] Isn't it from Le Petit Prince?
[2025-11-09|cman674|1] Go to plot setup and make sure the series you want to stack are grouped.
  [2025-11-10|wohren_osso|0] Yup, already checked.
    [2025-11-10|cman674|1] did you set an amount for the offset then for each individual curve?
      [2025-11-10|wohren_osso|1] Don't even get the option.
[2025-11-10|thecheshirejack|1] Unironically use the spectroscopy widget from Orange Datamining. It's open source and genuinely has a much better UI plus a bunch of auxiliary analysis tools
[2025-11-11|Science_zaddy|1] If you add a legend indicating the labeling for the multiple lines then the audience will understand that the lines are overlapping perfectly. Why do you need to stack them?
[2025-11-09|Hackeringerinho|-8] Arbitrarily moving the peaks to "better fit" the data is intellectual dishonesty. Change your model.
  [2025-11-10|wohren_osso|12] 🤦‍♂️I wanna offset them so they're more visible, not change their values.
    [2025-11-10|Hackeringerinho|1] So you're going to arbitrarily move them to make them more visible yes? Play with the line type and thickness.
  [2025-11-10|buckeyebooty|3] Hey folks, we got ourselves a reviewer 2 here!
  [2025-11-13|Ok-Ease5589|2] Fuck the down votes you're right.
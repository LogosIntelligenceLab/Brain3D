Using these files requires that you have already agreed to the Natural Scenes Dataset's Terms and Conditions: https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions

Webdatasets only contain behavioral information, .tar filename numbering correspondings to the scanning session of the subject.

(Always use "new_test" instead of "test" in the wds folders, "test" refers to using the old NSD data from before they released the full set of scanning sessions.) 

behavior numpy files correspond to the following order, in relation to the variables listed here: https://cvnlab.slite.page/p/fRv4lz5V2F/Untitled

"-1" values were used in place of NaN.

behavior = {
- "cocoidx": int(behav.iloc[jj]['73KID'])-1, #0
- "subject": subject,                        #1
- "session": int(behav.iloc[jj]['SESSION']), #2
- "run": int(behav.iloc[jj]['RUN']),         #3
- "trial": int(behav.iloc[jj]['TRIAL']),     #4
- "global_trial": int(i * (tar + 1)),        #5
- "time": int(behav.iloc[jj]['TIME']),       #6
- "isold": int(behav.iloc[jj]['ISOLD']),     #7
- "iscorrect": iscorrect,                    #8
- "rt": rt, # 0 = no RT                      #9
- "changemind": changemind,                  #10
- "isoldcurrent": isoldcurrent,              #11
- "iscorrectcurrent": iscorrectcurrent,      #12
- "total1": total1,                          #13
- "total2": total2,                          #14
- "button": button,                          #15
- "shared1000": is_shared1000,               #16
}
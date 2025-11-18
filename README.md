# Belief-Networks-Hidden-Markov-Models
Fall 2025 CS 362/562

1. Give an example of a word which was correctly spelled by the user, but which was incorrectly “corrected” by the algorithm. Why did this happen? 
  Example: "you". 
  Reason: see below

2. Give an example of a word which was incorrectly spelled by the user, but which was still incorrectly “corrected” by the algorithm. Why did this happen?
   Example:"tooday".
   Reason: see below

3. Give an example of a word which was incorrectly spelled by the user, and was correctly corrected by the algorithm. Why was this one correctly corrected, while the previous two   were not?
   Example:"heallo"
   Reason: see below

Answer to 1 - 3: 
    In order to handle a situation which happend in my earlier version of the program, where a wrong typed word would win out by the emission probability despite a significantly lower transition probability (for example, in the case "hallo", although the transition probability from "h" to "e" might be higher than "h" to "a", but the candidate "hallo" can still win out because the emission probability from "a" to "a" is 1, which is higher than the emission probability from "e" to "a". Depending on the situation, emission probability may overpower an unfavourable transition probability), I imitate the real-world spelling checker to introduce prior probabilities in the word level. Although this move greatly increases the accuracy for what is within the training dataset, it basically limits its effect there. For anything not fed in the training data, it will try to find the most similar match in the provided correct words, which in most cases would lead to wrong suggestions. "you" and "tooday" failed, because their corresponding correct words were not in the training data. "heallo" is correctly corrected, because "hello" was in the training data.

4. How might the overall algorithm’s performance differ in the “real world” if that training dataset is taken from real typos collected from the internet, versus synthetic typos (programmatically generated)?
   A viterbi decoder trained on real human typos will perform significantly better. While programmatically generated typos are more likely to be randomly distributed, human users do not make mistakes equally likely. They are more prone to some typos than others, which might reflect specific underlying psychological activities, where as the programmatically generated typos lack this layer.
   
  

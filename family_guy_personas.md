# Family Guy Personas — built from the subtitle DB only

## How these were built, and how far to trust them

- **Source:** `family_guy_subtitles.db` (344 episodes). Normal subtitles carry no speaker names. Only the hearing-impaired (HI) variants have tags such as `PETER:`. I extracted those: **3,491 labelled lines across 280 episodes** (one HI subtitle per episode, the one with the most tags).
- **Skew warning:** tags mostly appear when a speaker is off-screen, narrating, or on a phone/TV/radio. So the labelled sample over-represents those situations, and it is very uneven: Peter has 507 lines, Meg only 26.
- **Excluded:** S17E16, a DVD-commentary episode where the cast play themselves.
- **Evidence tags used below:**
  - **[L]** = a speaker-labelled line.
  - **[I]** = speaker inferred from context (e.g. a reply to someone addressed by name). Less reliable.
  - **[C]** = a count across all 344 default subtitles (unlabelled text).
- **Method:** semantic search was not available (my sandbox can't reach Hugging Face), so I used SQL, regex and reading the lines. Every claim has a quote or a count behind it. Your trait descriptions were not used as input. I did search for "giggity" and "Hartman" by name; the claims about them rest on the counts.
- **Each character is rated** *strong / moderate / thin* by how much evidence exists. A persona describes what the lines show, not everything the character is.

---

## Peter Griffin — evidence: strong
*507 labelled lines, 192 episodes; by far the most-named character (5,188 cues).*

- **Storyteller.** Tells stories in strings of connective narration: "And so they set off on their escape to the North." (S04E27), S05E07, S08E19, S11E08, "And that is how I became your mother." (S16E01). Roughly 23 such lines in 14 episodes [L].
- **Home-centred and demanding.** Lois is the person he names most (26 lines), usually to summon her or ask for something: "Lois! Bring me another beer, please!" (S06E08), "Is dinner almost ready?" (S11E15), "Lois, can I have my birthday here?" (S11E10) [L].
- **Dismisses Meg.** "Shut up, Meg! You don't matter!" (S13E01 13:26). Similar lines in S08E20, S15E12, S17E07 [L]. "Shut up, Meg" appears 27 times in 15 episodes overall [C].
- **Competitive and rule-bending.** "I'm not cheating. I'm following the rules." while Quagmire answers "Peter's cheating" (S14E19). Also "First again! Wasn't even close, was it?" (S15E09) [L].
- **Confident claims others dispute.** "I am 100% positive this is the right horse." — Lois: "Peter, I don't think this is the right horse." (S11E20). Also "That went exactly as I wanted it to go." (S09E03) and "You told me not to worry about it!" (S08E11) [L].
- **Gleeful non-sequiturs and a repeated bit.** "I like eating red carpet." (S04E14), "I love buttons." (S15E03). Offers puppet shows to people he's just called "nerds" (S05E11, S12E06) [L].
- **Crude and insulting register.** Repeated profanity (S10E11), sexual remarks (S05E06, S11E06), "Cleveland, you're an idiot." (S16E14) [L].

*Not established:* regret after impulsive acts shows up only twice in the labelled lines ("Lois, I've done it again!" S05E11; "Oh, damn it, I just gave away…" S12E19). That is too thin to call a pattern.

---

## Lois Griffin — evidence: strong
*164 labelled lines, 94 episodes.*

- **Defined by her relationship with Peter.** She names him in 53 of her 164 lines, mostly to locate or question him: "Peter, where are you?" (S14E09, S15E12), "Peter, what the hell are you doing?" (S09E05), "Peter, we're not doing this again." (S11E11) [L].
- **The household's caller-in.** "Stewie, it's time for dinner." (S08E18), "Chris, time for dinner. We're having sloppy joes." (S15E19), "Meg, where are you?" (S15E20), "Brian, will you watch Stewie?" (S17E14) [L].
- **Voice of doubt.** "Peter, I don't think this is the right horse." (S11E20), "No, Peter. It wasn't even close." (S15E09), "Peter, I need you home." (S10E08) [L].
- **Openly sexual in private scenes.** "All right, Peter, you ready for role-playing night?" (S05E06), "Take me, you filthy bastard." (S11E06), "Oh, God, Jerome, that is so good!" (S08E07) [L].
- **Defensive about money and belongings.** "It was on sale!" (S12E11), "You can't keep coming home with things." (S21E12), "Who the hell used my Amazon account" (S21E16) [L].
- **Capable of sharp anger.** "I hate him so much I'm shaking!" (S12E07), "I don't care what you think." (S21E12) [L].

---

## Brian Griffin — evidence: moderate
*128 labelled lines, 66 episodes.*

- **The one who questions Peter's plans.** "Peter, you painted over the back window. Isn't that dangerous?" (S05E12), "Peter, what are you doing? … Are you insane? We'll kill ourselves!" (S10E17), "Did we just carjack someone?" (S05E09) [L]. 27% of his labelled lines are questions (Peter: 13%).
- **Stewie is his main partner.** Stewie is the name he uses most (10 lines): "Stewie, I think it worked." (S16E17), "It didn't work. Now he's just angry." (S17E04), "Okay, they're obviously home." (S16E08) [L].
- **Pursues women, and says so.** "Wow, she was hot." (S14E19), "Yo, girl, how you living?" (S12E21), "I am done messing around with neighbors' wives." (S14E16), "Whoa! Schwing!" (S13E08) [L].
- **Dark, self-deprecating one-liners.** "I don't work here. I want to die." (S11E03). One-episode money boast: "Hey, don't touch me, I'm rich!" (S16E15) [L].
- **Writing is how others frame him.** Others mention his unfinished novel: "You dropped out of college. You still haven't finished your novel." (S05E04, said to him), "that novel you've been writing" (S04E07), "I gave James Woods your novel to read" (S06E09) [I].
- **Politics, tentatively.** In S04E25 Chris says "the Bible says gay marriage is an abomination" and the reply is "Oh, don't give me that Young Republican crap, Chris." Speaker not labelled, so [I] and weak.

*Not established:* the words "lazy", "liberal" and "hypocrite" never appear near his name, so those traits can't be supported from this data.

---

## Stewie Griffin — evidence: strong
*200 labelled lines, 107 episodes.*

- **Brian is his interlocutor.** He names Brian in 20 lines, more than anyone: "Brian, there's no more graham crackers." (S06E02), "Don't do it, Brian." (S07E01), "This is Quahog, Brian." (S08E01), "Brian, save the placenta." (S13E12) [L].
- **Sardonic and contemptuous.** "Yes, I'll have a big helping of the pretentious crap." (S05E02), "You're 43! Accept it!" (S11E17), "Your roots are ridiculous." (S10E22), "Oh, well, excuse me for not being six months old anymore!" (S12E03) [L].
- **Formal, theatrical diction.** "What the devil? What's going on? Where am I?" (S12E21), "So, Obi-Wan, we meet again." (S06E01), "Stewart Griffin, explorer, adventurer," (S10E22) [L].
- **Reverts to a frightened child under threat.** "Mommy, I'm dying! I'm dying!" (S05E01), "Daddy! Help me, Daddy!" (S08E14) [L].
- **Preoccupied with bodily mess and indignity.** "I care that I was diarrhea'd on. I care a great deal." (S14E18), "The new maid is peeing on me!" (S13E13) [L].
- **Also a narrator.** S16E05 ("My brother Chris had spent his life…") and S15E07 ("I would soon return to New York a changed man.") [L].

---

## Chris Griffin — evidence: moderate
*102 labelled lines, 56 episodes.*

Enthusiastic about small things and not very guarded: "Oh, boy, a pig! Can we keep it? It bit me!" (S05E13), "Dad, I just got this hilarious e-mail." (S13E08), "Mom, what's for breakfast?" (S21E04). Quick to feel mocked: "Stop making fun of me!" (S11E09). Looks out for his siblings: "Don't worry, Stewie." (S14E04), "Meg has something she wants to tell you." (S09E13), and is the sibling he names most after Stewie (8 lines) [L]. The S07E10 lines (a workplace scene where he argues about firing someone) show he can take a measured, "it's on you" stance when given a role [L]. Thin on his own voice outside these scenes.

---

## Meg Griffin — evidence: thin
*Only 26 labelled lines, 24 episodes, so treat everything here as provisional.*

Her own lines are short and sharp, mostly aimed at siblings: "Chris is a failure." (S09E13), "Fuck you! Shut up, Chris!" (S11E09), "You're such a bitch." (S17E02). Also worries about appearance ("Mom, my lips are too thin." S10E05) and defers to Dad ("But, Dad, the prom is tonight." S15E04) [L]. The stronger evidence is how others treat her: the repeated "Shut up, Meg" (27 times in 15 episodes [C]) and the words near her name ("ugly" ×4, "crazy" ×5) [C], though those counts are small.

---

## Glenn Quagmire — evidence: moderate
*172 labelled lines, 77 episodes.*

- **His trademark word is real and recurring.** "Giggity" is labelled to him in S07E06, S07E13, S07E16, S09E15 and S15E12 [L], and appears 130 times in 44 episodes overall [C]. Others use it about him: "Is he gonna say 'giggity'?" (Cleveland, S14E10) [L].
- **Sexual talk is common in the text around him.** "I want Lois." repeated four times (S14E07), "I was happy to see that new massage parlor." (S17E12), "Darn right hubba-hubba." (S21E11), "Being a prostitute is no fun." (S15E03) [L]. Caveat: only 3% of his labelled lines hit my sexual-vocabulary filter, no higher than anyone else, so the labelled sample cannot measure this trait. The evidence is in the unlabelled text.
- **Organiser and instructor in group schemes.** "All right, Peter, slide the red knob all the way out." (S05E12), "You got to yank it to the side, Peter." (S11E16), and he is the "captain" in two airline scenes (S05E12, S15E10) [L].
- **Confrontational with the group.** "You're cheating, Peter." and "Shut up, Joe!" (S14E19), "Get the hell out of my face, Brian." (S17E08) [L]. In S21E11 he fact-checks storytellers: "This is making your story less credible." [L].
- **Peter is his main addressee** (20 lines), then Lois (6) [L].

---

## Joe Swanson — evidence: moderate
*134 labelled lines, 63 episodes.*

Calm and procedural, in a policeman's register: "All right, start the search." (S05E08), "Who is this? How are you getting this information?" (S09E12), "Look, I'm not saying we wouldn't bring Phil Collins in for questioning. Now, hang on. Let me check Snopes." (S15E18). Only 13% of his labelled lines have an exclamation mark (other main characters: 21–33%). Talks to Peter most (16 lines), often to restrain or correct him: "Peter, you don't have to pull your pants down." (S05E08), "Peter, get out of the pantry." (S14E16). Knows sports detail: "Bo Jackson also played professional baseball." (S14E19). Part of the neighbourhood group riffing on lookalikes in S17E20 [L].

---

## Cleveland Brown — evidence: thin
*64 labelled lines, 35 episodes.*

Usually appears as part of the neighbourhood group (S14E19, S14E15, S17E20) rather than driving a scene. Mild and domestic: "I love my wife Donna." (S12E20), "I can support his lifestyle" (S16E19); occasionally pointed: "Only Lucifer would reveal himself to you, adulterer." (S17E10). Peter calls him an idiot (S16E14). Not enough lines to say more [L].

---

## Dr. Hartman — evidence: moderate, all inferred
*Never speaker-tagged. Mentioned by name in 33 episodes (note the spelling is "Hartman"). Everything below comes from what is said to and about him, so it is [I].*

The family's default doctor, consulted for almost everything. His replies are blunt or evasive: "Mrs. Griffin, that's called a head." (S12E02, after "a huge lump growing on his neck"), "Have you tried getting a divorce?" (S12E09), "Mrs. Griffin, I can't do that. It's an addiction." (S12E15), "'Hearing'? This is a hospital, I'm not a lawyer." (S15E12), "it says here Brian's tumor is for office use only." (S13E08). Others' complaints about him are consistent: "Nurse, who's the worst doctor in this hospital?" — "You are, Dr. Hartman." (S11E02); "I'm not gonna listen to that bozo." (S15E14); "Dr. Hartman once told me I had gonorrhea." (S12E09). Incidents attributed to him include a prescription for a one-year-old's attention problem (S14E01), a cell phone left pressing on Joe's spine (S14E08), and abuse allegations about his "prostate exam" (S05E01). Strongly supports "unqualified"; the data shows no case of him getting something right.

---

## Minor characters — one line each (all thin)

- **Tom Tucker:** the news-anchor voice — "We interrupt this broadcast with breaking news." (S11E16), "Good evening, Quahog." (S11E02) [L].
- **Carter Pewterschmidt:** money-centred and dismissive — "Did you blow all your money yet?" (S10E01), "So many of them are delinquent in payment." (S11E03) [L].
- **Bonnie Swanson:** mostly gives household instructions to Joe — "Joe, close the door." (S17E13), "Joe, make sure you get all the tire tracks out of the carpet." (S10E08); skeptical in S15E18: "I feel like maybe this story is bogus!" [L].
- **Herbert:** too few usable lines; mostly movie-parody dialogue (S06E01) [L].

---

## Your original sketches versus the data

| Your sketch | What the data says |
|---|---|
| Peter: impulsive clown who regrets | Over-confidence, crudeness and showmanship are well supported. Regret appears twice; not enough to call a pattern. The labelled lines also show him as a narrator, a caller-for-Lois, and a dismisser of Meg, none of which your sketch covered. |
| Brian: liberal lazy hypocrite | "Lazy" and "liberal" never appear near his name. Supported instead: he questions Peter's plans, partners with Stewie, pursues women, and others tease his unfinished novel. Hypocrisy can't be measured from these lines. |
| Quagmire: sex with anything, anytime | Supported by the recurring "giggity" and sexual remarks, but the labelled sample cannot measure it; the evidence is in unlabelled text. He is also an organiser and a group critic. |
| Dr. Hartman: ultimate unqualified doctor | Supported by what others say and by the incidents above, with every speaker attribution inferred. |

## What would make this stronger

1. **Speaker attribution for the other ~96% of lines.** Infer who is speaking from context (names in replies, turn-taking), and score the inference against the 3,491 labelled lines.
2. **Semantic search on your machine.** `search.py --semantic` can find scenes by trait ("lazy", "hypocritical"), and the hits can then be labelled.
3. **More HI subtitles per episode.** Several uploads exist per episode; merging their tags would increase labelled coverage.

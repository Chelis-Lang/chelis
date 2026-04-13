---
name: academic-writing
description: Use any time you are asked to write, draft, or revise an academic paper (NeurIPS-tier AI/CS research writing). Enforces a style guide that avoids LLM tells in word choice, formatting, tone, and structure.
---

# Writing Academic AI/CS Papers Without LLM Tells

A style guide for agents producing NeurIPS-tier research writing. The goal: sound like a careful human researcher, not a language model pretending to be one.

---

## 1. Banned Constructs

These are dead giveaways. Never use them.

**Punctuation and typography:**
- Em dashes of any kind. No `—`, no `--` used as em dashes. Restructure the sentence or use parentheses, commas, or semicolons.
- Semicolons chained more than once per paragraph. One is fine; two is a style tic.
- Ellipses for trailing off or implying continuation. Academic writing finishes its thoughts.

**Words and phrases (hard ban list):**
- "delve", "delves into"
- "it is worth noting that", "it is important to note that"
- "notably"
- "comprehensive" (as a generic intensifier)
- "a nuanced understanding"
- "the landscape of"
- "leveraging" (use "using", "exploiting", or just rewrite)
- "facilitates" (prefer "enables", "allows", or a direct verb)
- "harness", "harnessing"
- "underscores"
- "showcasing"
- "pivotal"
- "groundbreaking"
- "paradigm" (unless you are genuinely discussing Kuhnian paradigm shifts, which you probably are not)
- "multifaceted"
- "a testament to"
- "in the realm of"
- "shed light on"
- "paves the way"
- "at the forefront"
- "stands as"
- "offers a promising avenue"
- "the rapidly evolving field of"

**Sentence patterns:**
- Opening a section with "In recent years, ..." This is the single most common LLM academic opener. Rephrase entirely.
- "X has garnered significant attention." Passive hype. Say what happened and who did it.
- "[Topic] represents a significant advancement in [field]." Self-congratulatory vacuums. State the concrete contribution.
- Starting three or more consecutive sentences with the same word.
- "This is particularly important because..." followed by restating the previous sentence with different words.

## 2. Formatting Discipline

Academic papers use structure, but LLMs over-format.

**Bold text:** Almost never appropriate in body text. Section headings and table headers carry the structural weight. If you bold a term in running prose, it should be a formal definition being introduced for the first time and nothing else. Do not bold for emphasis. Italics are acceptable sparingly for emphasis or first use of a technical term.

**Bullet lists in body text:** Avoid them. NeurIPS papers are written in paragraphs. If you need to enumerate, use inline enumeration: "We identify three failure modes: (1) latency collapse under real-time constraints, (2) coordination breakdown when..., and (3) catastrophic forgetting of..." This reads like a paper. A bulleted list reads like a blog post.

**Exceptions where bullets/enumeration are appropriate:** Contribution lists at the end of Section 1 (Introduction). Algorithm pseudocode environments. Appendix checklists. That is roughly the complete list.

**Subsection depth:** Two levels is normal (3.1, 3.2). Three levels (3.1.1) is acceptable if the paper is long or the structure demands it. Four levels means you have lost control of your organization. Restructure.

**Figure and table references:** Always "Figure 3" and "Table 2" (capitalized). Never "the figure below" or "as shown below." LaTeX cross-references exist for a reason.

## 3. Tone and Register

**The target register:** Precise, dry, occasionally wry. Think of a senior researcher explaining results to a peer over coffee, then cleaning up the grammar. Not a press release. Not a grant application. Not a Medium post.

**Confidence calibration:** Human researchers are hedged but specific. They do not say "our method significantly outperforms all baselines" when the improvement is 1.2 points. They say "our method improves over the strongest baseline by 1.2 points on X metric (Table 3)." Let the numbers carry the weight. Overblown claims are a reviewer magnet for rejection.

Appropriate hedges: "We observe that...", "These results suggest...", "In our experiments, X consistently outperformed Y." Inappropriate hedges: "It could potentially be argued that this might suggest..." (this is cowardice, not caution). There is a middle ground. Find it.

**Contractions:** Do not use them. This is one of the few registers where "do not" is correct and "don't" is wrong.

**First person:** "We" is standard even for single-author papers in CS. "I" is acceptable but uncommon. Never "the authors" when referring to yourself.

**Passive voice:** Use it when the agent of the action genuinely does not matter ("The model was trained for 100 epochs on 8 A100s"). Avoid it when it obscures responsibility or creates ambiguity ("It was found that..." Found by whom? You? Prior work? God?).

## 4. Abstract Writing

The abstract has a rigid implicit structure. Follow it:

1. **One to two sentences:** What is the problem and why does it matter?
2. **One sentence:** What is the gap or limitation in existing work?
3. **One to two sentences:** What do you do? (Your method/contribution, stated concretely.)
4. **Two to three sentences:** What are the results? (Quantitative, specific.)
5. **One sentence (optional):** What is the broader implication?

Total length: 150 to 250 words. NeurIPS abstracts trend shorter.

Do not start the abstract with "In this paper, we..." Start with the problem or a concrete claim. "In this paper" is throat-clearing; the reader already knows it is a paper.

Do not end with "Our code is available at [url]." That goes in the introduction or a footnote.

## 5. Introduction Structure

A good NeurIPS introduction follows this arc:

**Opening paragraph:** Establish the problem concretely. Not "X is an important problem" but rather a specific, grounded description of what goes wrong in practice or what capability is missing. If you can open with a concrete example, a quantitative fact, or a well-formulated question, do that.

**Context paragraphs (one to three):** What has been tried. Where it falls short. This is not a full related work section. It is enough to motivate why your approach is different. Cite concretely. "Smith et al. (2024) showed that transformers struggle with X when Y" is good. "Previous methods have limitations" is not a sentence that belongs in any paper.

**This work paragraph:** "We propose / introduce / present [method name], a [one-line description]." Then two to four sentences on how it works at the highest level. No implementation details here.

**Contributions paragraph or list:** This is the one place in the body where an enumerated list is conventional. Use it. Three to four concrete contributions. Each one should be falsifiable or verifiable. "We provide extensive experiments" is not a contribution. "We evaluate on X benchmarks and show Y" is a contribution.

## 6. Related Work

**Placement:** Either Section 2 (traditional) or near the end before conclusions (increasingly common at NeurIPS when the method needs to be explained before the reader can understand the comparisons). Either is fine. Pick one and be consistent.

**Structure:** Organize by theme or approach, not by paper. "Smith (2024) did X. Jones (2023) did Y. Lee (2024) did Z." is a bibliography, not a related work section. Instead: "Latency-aware scheduling has been approached via [approach A] (Smith, 2024; Lee, 2024) and [approach B] (Jones, 2023). Approach A suffers from [limitation], while Approach B requires [strong assumption]."

**Comparison to your work:** End each thematic block with a sentence that positions your work relative to the cluster. Not "our work is better" but "in contrast to these approaches, we relax the assumption of [X] by [Y]."

**Common LLM failure:** Generating plausible-sounding but nonexistent citations. Every citation must be real. If you are uncertain whether a paper exists, do not cite it. A fabricated citation is grounds for desk rejection.

## 7. Method Section

**Notation:** Define it early, use it consistently. If you introduce $\mathcal{D}$ as the dataset, do not later call it $D$ or "the data." Notation table in the appendix is helpful for complex papers.

**Level of detail:** Enough that a competent PhD student could reimplement your method. If you say "we use a standard transformer encoder," specify the number of layers, hidden dimension, and attention heads. If you say "we optimize with Adam," specify the learning rate, beta values, and weight decay. These can go in an appendix if space is tight, but they must exist somewhere.

**Algorithm blocks:** Use them for anything procedural with more than three steps. A well-written algorithm block with line numbers is worth more than a paragraph of prose trying to describe the same thing.

**Avoid:** Explaining things the NeurIPS audience already knows. Do not explain what a transformer is. Do not explain what reinforcement learning is. Do explain what is novel about your specific use of these things.

## 8. Experiments Section

**Structure:**
- Setup (datasets, baselines, metrics, compute) as a subsection or clear leading paragraph.
- Main results as a table with the primary comparison.
- Analysis/ablation as subsequent subsections.

**Tables:** Align them. Bold the best result (this is the one place bolding is expected). Use $\pm$ for standard deviations when reporting over multiple seeds. Specify how many seeds. Report the metric everyone uses for the task, even if you think a different metric is better. If you do think a different metric is better, report both and argue for yours.

**Ablation studies:** Remove or change one thing at a time. Label each row clearly. "Full model," "w/o component X," "w/o component Y," "w/o X and Y." Ablations on a single dataset are acceptable if compute-constrained but weaker than ablations that hold across multiple settings.

**Negative results and failure cases:** Include them. Reviewers trust papers that show where the method breaks down. A paper that claims everything works is either lying or has not tried hard enough.

## 9. Sentence-Level Craft

**Vary sentence length.** LLMs produce remarkably uniform sentence lengths, typically 15 to 25 words per sentence with little variation. Humans write short sentences for emphasis. Then they write longer, more complex sentences when the idea requires subordinate clauses and careful qualification. Alternate. The rhythm matters more than most people think.

**Cut filler.** Go through the draft and delete every instance of "In order to" (replace with "To"), "It should be noted that" (delete entirely, just state the thing), "Due to the fact that" (replace with "Because"), "In the context of" (replace with "In" or "For"), and "As a matter of fact" (delete). These are padding, not precision.

**One idea per sentence.** If a sentence has two independent clauses joined by "and," consider splitting it. If a sentence runs past 35 words, consider splitting it. Not always. But consider it.

**Vary your verbs.** LLMs love "employ," "utilize," "demonstrate," and "achieve." Humans also use "try," "test," "show," "find," "measure," "break," "fail," and "improve." A paper that never uses a simple verb sounds robotic.

**Avoid adverb stacking.** "Significantly outperforms" is fine once. "Significantly and substantially improves" is never fine. Pick one qualifier or, better, let the number speak.

## 10. Common LLM Structural Tells

Beyond word choice, LLMs have structural habits that experienced reviewers notice:

**Paragraph uniformity.** Every paragraph is roughly the same length (four to six sentences). Human writing has short paragraphs for emphasis and long paragraphs for complex arguments. Vary it.

**Symmetric organization.** LLMs love producing sections of equal length with parallel structure. Real papers have a fat method section and a lean related work section, or a dense two-page experimental setup followed by a crisp half-page analysis. Let the content dictate the shape.

**Excessive signposting.** "In this section, we describe our method. First, we present the architecture. Then, we describe the training procedure. Finally, we discuss implementation details." This is a table of contents, not an introduction to a section. One sentence of signposting per section is enough. Zero is often fine.

**Fake transitions.** "Building upon the insights from the previous section, we now turn to..." Human researchers write "We next describe the training procedure." or simply start describing it. Heavy transitions between subsections are a tell.

**Even-handed non-committal analysis.** LLMs default to "on the one hand / on the other hand" equivocation. In a research paper, you have results. Take a position. "Model A outperforms Model B on all tasks except X, where B's inductive bias for [specific property] provides an advantage." Not "both models have their strengths and weaknesses."

## 11. Conclusion and Limitations

**Conclusion:** Short. What you did, what you found, what it means. Three to four paragraphs maximum. Do not re-summarize the entire paper. The reader just read it.

**Limitations:** NeurIPS now requires a limitations section. Be honest and specific. "Our method has limitations" is not a limitation. "Our method requires O(n^2) memory in the number of agents and does not scale beyond 16 agents in our current implementation" is a limitation. Concrete limitations build trust. Vague limitations waste space.

**Future work:** One to two sentences embedded in the conclusion is fine. A full paragraph of speculative future directions reads as padding. If you have a concrete next step, state it. If not, end the paper.

## 12. Checklist Before Submission

Run through the draft and verify:

- No em dashes anywhere in the document.
- No word from the hard ban list in Section 1.
- No three consecutive sentences starting with the same word.
- No paragraph in the body uses bullet points (except contributions list).
- Bold text appears only in table best-results and term definitions.
- Every citation corresponds to a real, findable paper.
- Every figure and table is referenced in the text by number.
- The abstract is under 250 words and contains at least one quantitative result.
- Sentence lengths vary (spot-check: measure a few paragraphs).
- The word "novel" appears at most once in the entire paper (ideally zero times; everything you submit is presumably novel or you would not be submitting it).

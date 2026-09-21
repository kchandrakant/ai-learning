# Module 6: Staying Current

AI moves fast. Techniques from two years ago may already be obsolete. Staying current isn't about reading everything — it's about efficiently filtering signal from noise and knowing where to look when you need something specific.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Find relevant papers and resources efficiently
- Filter high-signal information sources
- Build a sustainable system for staying updated
- Know where to look for specific topics
- Avoid information overload

---

## 📚 Part 1: The Information Landscape

### The Speed of Change

| Year | Major Development |
|------|-------------------|
| 2017 | Transformers ("Attention Is All You Need") |
| 2018 | BERT, GPT |
| 2019 | GPT-2 |
| 2020 | GPT-3, Vision Transformers |
| 2021 | CLIP, Codex, LoRA |
| 2022 | ChatGPT, Stable Diffusion, InstructGPT |
| 2023 | GPT-4, LLaMA, Claude 2, Mixtral |
| 2024 | Claude 3, Gemini, open-weight models explode |
| 2025 | Agents, reasoning models, MCP/A2A protocols |
| 2026 | ... |

**The half-life of ML knowledge is ~2 years.** Techniques you learn today may need updating. The fundamentals (math, core algorithms) last longer, but applications and best practices evolve rapidly.

### Information Sources by Type

| Source | Update Speed | Signal/Noise | Depth |
|--------|-------------|--------------|-------|
| arXiv preprints | Hours | Low | High |
| Twitter/X | Hours | Very low | Low |
| Blog posts | Days-weeks | Medium | Medium |
| Newsletters | Weekly | High | Medium |
| Conference papers | 6-12 months | High | High |
| Courses/books | 1-2 years | High | Very high |

**Trade-off:** Speed vs. quality. The fastest sources have the most noise.

---

## 📚 Part 2: Primary Sources

### arXiv — The Firehose

[arxiv.org](https://arxiv.org) hosts preprints (papers before peer review). Most ML research appears here first.

**Key categories:**
- `cs.LG` — Machine Learning
- `cs.CL` — Computation and Language (NLP)
- `cs.CV` — Computer Vision
- `cs.AI` — Artificial Intelligence
- `stat.ML` — Statistics: Machine Learning

**How to use:**
```
# Daily listings
https://arxiv.org/list/cs.LG/recent

# Search
https://arxiv.org/search/?query=transformer&searchtype=all
```

**Warning:** 50-100+ new papers daily in cs.LG. You cannot read them all. Use filters (below).

### Semantic Scholar — Smart Search

[semanticscholar.org](https://www.semanticscholar.org/)

Better than Google Scholar for ML:
- AI-powered recommendations
- "Highly influential citations" highlights
- Research feeds based on your interests
- TLDR summaries for papers

**Pro tip:** Create an account and build a "library" of papers. It learns your interests.

### Papers With Code — Papers + Implementations

[paperswithcode.com](https://paperswithcode.com/)

Essential features:
- Papers linked to GitHub implementations
- State-of-the-art leaderboards by task
- Datasets with benchmarks
- Methods and components explained

**Use for:**
- "What's the current best model for [task]?"
- "Is there code for this paper?"
- Discovering papers by browsing tasks

### Connected Papers — Visualize Relationships

[connectedpapers.com](https://www.connectedpapers.com/)

Enter a paper, get a visual graph of related papers:
- **Prior work:** What this paper builds on
- **Derivative work:** What built on this paper
- **Similar work:** Papers in the same area

**Use for:**
- Understanding a new research area
- Finding papers you missed
- Literature reviews

---

## 📚 Part 3: Curated Sources

These sources filter the firehose for you.

### Newsletters

| Newsletter | Focus | Frequency |
|------------|-------|-----------|
| [The Batch](https://www.deeplearning.ai/the-batch/) | General AI news | Weekly |
| [ImportAI](https://jack-clark.net/) | AI policy + research | Weekly |
| [Davis Summarizes Papers](https://dblalock.substack.com/) | Paper summaries | Weekly |
| [The Gradient](https://thegradient.pub/) | In-depth articles | Bi-weekly |
| [Ahead of AI](https://magazine.sebastianraschka.com/) | LLMs, research | Weekly |

**Recommendation:** Subscribe to 2-3 newsletters. More becomes noise.

### Blogs — High-Quality Explanations

| Blog | Known For |
|------|-----------|
| [Lil'Log](https://lilianweng.github.io/) | Comprehensive survey posts |
| [Jay Alammar](https://jalammar.github.io/) | Visual explanations (Illustrated Transformer) |
| [Andrej Karpathy](https://karpathy.ai/) | Deep, accessible explanations |
| [Chip Huyen](https://huyenchip.com/blog/) | MLOps, practical ML |
| [Sebastian Raschka](https://sebastianraschka.com/blog/) | LLMs, research insights |
| [Eugene Yan](https://eugeneyan.com/) | Production ML, RecSys |

### YouTube — Video Explanations

| Channel | Style |
|---------|-------|
| [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) | Paper deep-dives |
| [Two Minute Papers](https://www.youtube.com/@TwoMinutePapers) | Quick summaries |
| [3Blue1Brown](https://www.youtube.com/@3blue1brown) | Visual math/ML explanations |
| [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) | Neural networks from scratch |
| [StatQuest](https://www.youtube.com/@statquest) | Statistics and ML basics |

---

## 📚 Part 4: Social Sources

### Twitter/X

High noise, but can be high signal if you curate carefully.

**Key accounts to follow:**

| Account | Focus |
|---------|-------|
| @kaborepharma | Breaking ML research |
| @_akhaliq | Paper announcements |
| @ylecun | AI research (Meta) |
| @iaborepharma | Paper summaries |
| @AndrewYNg | AI education |
| @sama | OpenAI, AI industry |
| @ClementDelangue | Hugging Face, open source |

**Lists > Individual follows:** Create or subscribe to ML-focused lists to reduce noise.

**Warning:** Twitter rewards hot takes over accuracy. Verify claims before believing them.

### Reddit

| Subreddit | Focus |
|-----------|-------|
| r/MachineLearning | Research discussion |
| r/LocalLLaMA | Open-source LLMs |
| r/learnmachinelearning | Learning resources |
| r/artificial | General AI |

**Quality:** r/MachineLearning has good paper discussions. Others vary.

### Discord/Slack

Many communities have active Discords:
- Hugging Face Discord
- Eleuther AI Discord
- LangChain Discord
- Various paper reading groups

**Pro:** Real-time discussion, quick answers
**Con:** Time-consuming, easy to get distracted

---

## 📚 Part 5: Conferences

Major venues where peer-reviewed research is published:

### Top ML Conferences

| Conference | Focus | When |
|------------|-------|------|
| NeurIPS | General ML | December |
| ICML | General ML | July |
| ICLR | Representation learning | May |
| CVPR | Computer vision | June |
| ACL | NLP | July |
| EMNLP | NLP | November |

### Following Without Attending

- **Conference proceedings:** Released online (free)
- **Best paper awards:** High-signal filter
- **Recorded talks:** Often on YouTube or SlidesLive
- **Twitter threads:** People summarize key papers

### Conference Calendar

- Papers submitted ~6 months before conference
- Accepted papers announced ~2-3 months before
- Papers often on arXiv before acceptance

---

## 📚 Part 6: Building Your System

### The Funnel Approach

```
        arXiv firehose (100+ papers/day)
                  ↓
        Newsletters filter (10-20/week)
                  ↓
        Your reading list (3-5/week)
                  ↓
        Deep reading (1-2/week)
                  ↓
        Notes/implementation (few/month)
```

### Weekly Routine (Example)

**Monday (30 min):**
- Skim newsletter digests
- Add interesting papers to reading list

**Wednesday (1 hour):**
- Pass 1-2 on 2-3 papers from your list
- One Pass 3 on an important paper

**Friday (30 min):**
- Check Papers With Code for new SOTAs in your area
- Browse Twitter for discussions

**Ongoing:**
- Bookmark interesting threads/posts
- Note concepts to learn later

### Tools for Organization

| Tool | Use |
|------|-----|
| Zotero | Paper library management |
| Notion | Notes and summaries |
| Readwise | Highlight management |
| Pocket/Instapaper | Save articles for later |
| Feedly | RSS feeds |

### What to Ignore

You don't need to track everything. Focus areas depend on your goals.

**If you're learning foundations:**
- Focus on established, cited papers
- Ignore most arXiv papers
- Prioritize courses and tutorials

**If you're doing research:**
- Track your specific area closely
- Skim adjacent areas
- Follow key researchers

**If you're building products:**
- Focus on practical techniques
- Prioritize implementations over theory
- Watch for new libraries/tools

---

## 📚 Part 7: Avoiding Information Overload

### Signs of Overload

- Hundreds of unread tabs/bookmarks
- FOMO about every new paper
- Reading widely but not deeply
- Knowing about techniques but not understanding them

### Strategies

**1. Set limits:**
- Max 30 min/day on "staying current"
- Max 2-3 newsletters
- Unsubscribe aggressively

**2. Batch processing:**
- Designate specific times for paper reading
- Don't context-switch throughout the day

**3. Depth over breadth:**
- Better to deeply understand 5 papers than skim 50
- Implementation teaches more than reading

**4. Just-in-time learning:**
- Learn things when you need them
- Not everything requires immediate attention

**5. Accept missing things:**
- You WILL miss papers. That's okay.
- Important ideas resurface

---

## 📚 Part 8: Topic-Specific Resources

### LLMs and NLP

- Papers: ACL, EMNLP, arXiv cs.CL
- Blogs: Lil'Log, Jay Alammar
- Code: Hugging Face Transformers
- Discussions: r/LocalLLaMA

### Computer Vision

- Papers: CVPR, ICCV, ECCV, arXiv cs.CV
- Code: timm library, Hugging Face
- Benchmarks: Papers With Code vision tasks

### MLOps / Production ML

- Blogs: Chip Huyen, Eugene Yan
- Courses: MLOps Zoomcamp, Made With ML
- Tools: MLflow, Weights & Biases, DVC

### Agents

- Papers: arXiv, LangChain blog
- Code: LangChain, LlamaIndex, AutoGen
- Discussions: LangChain Discord

---

## 🏋️ Exercises

### Exercise 1: Build Your Feed
1. Create a Semantic Scholar account
2. Save 5 papers relevant to your interests
3. Check the "Recommended" feed after a few days

### Exercise 2: Newsletter Audit
1. Subscribe to 2 newsletters from the list
2. Read them for 2 weeks
3. Keep only those providing consistent value

### Exercise 3: Find Current SOTA
Using Papers With Code, find:
1. Current best model for ImageNet classification
2. Current best model for machine translation (WMT)
3. A paper with open-source code for text summarization

### Exercise 4: Design Your System
Write down your personal "staying current" system:
1. Which sources will you follow? (max 5)
2. When will you read? (specific times)
3. How will you organize/save papers?
4. What will you explicitly ignore?

---

## ✅ Solutions

<details>
<summary>Click to reveal solutions</summary>

### Exercise 3: Find Current SOTA (answers will change over time)

Check Papers With Code directly for current answers:

1. **ImageNet classification:**
   - Go to paperswithcode.com/sota/image-classification-on-imagenet
   - As of late 2024: Models like CoCa, PaLI achieve ~91%+ top-1 accuracy

2. **Machine translation (WMT):**
   - Go to paperswithcode.com/sota/machine-translation-on-wmt2014-english-german
   - Recent models achieve 35+ BLEU

3. **Text summarization with code:**
   - Go to paperswithcode.com/task/text-summarization
   - Look for papers with GitHub links (indicated by code icon)
   - Examples: PEGASUS, BART, LED all have implementations

### Exercise 4: Design Your System (example answer)

```markdown
## My Staying Current System

### Sources (5 max)
1. The Batch newsletter (weekly)
2. Ahead of AI newsletter (weekly)
3. Papers With Code (weekly check)
4. Yannic Kilcher YouTube (select videos)
5. r/MachineLearning (occasional browse)

### Schedule
- Monday 7-7:30 AM: Read newsletters
- Wednesday lunch: One paper (Pass 1-2)
- Saturday morning: Deep reading or implementation

### Organization
- Zotero for paper library
- Notion for reading notes
- Template for each paper summary

### Explicitly Ignoring
- Daily arXiv firehose
- Twitter drama
- Every new model release announcement
- Papers outside my focus area (vision, RL)
```

</details>

---

## 🔗 What's Next?

You now have tools to keep learning after this course ends. Complete the **Self-Assessment** to verify you're ready for Course 01: ML Foundations.

---

## 📖 Quick Reference: Key Resources

### Must-Have Bookmarks
```
https://arxiv.org/list/cs.LG/recent
https://www.semanticscholar.org/
https://paperswithcode.com/
https://www.connectedpapers.com/
```

### Newsletters to Consider
```
The Batch: deeplearning.ai/the-batch
ImportAI: jack-clark.net
Ahead of AI: magazine.sebastianraschka.com
```

### Video Channels
```
Yannic Kilcher: youtube.com/@YannicKilcher
Andrej Karpathy: youtube.com/@AndrejKarpathy
3Blue1Brown: youtube.com/@3blue1brown
```

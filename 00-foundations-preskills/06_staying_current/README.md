# Module 6: Staying Current

AI moves fast. Techniques from two years ago may already be obsolete. Staying current isn't about reading everything — it's about efficiently filtering signal from noise and knowing where to look when you need something specific.

---

## 🎯 Why This Matters

The field moves faster than any individual can track. But you don't need to track everything — you need a **system** that surfaces what matters for your work.

```
┌─────────────────────────────────────────────────────────────────────┐
│                 The Information Overload Problem                    │
│                                                                     │
│   Daily arXiv submissions (cs.LG): 50-100+ papers                   │
│   Twitter ML posts: Thousands                                       │
│   New models announced: Weekly                                      │
│   Blog posts: Hundreds                                              │
│                                                                     │
│   You cannot read everything. You shouldn't try.                    │
│                                                                     │
│   The goal is NOT: "Know everything happening in AI"                │
│   The goal IS: "Have a system to find what I need, when I need it"  │
│                                                                     │
│   This module teaches you to build that system.                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The Half-Life of ML Knowledge

| Knowledge Type | Half-Life | Examples |
|----------------|-----------|----------|
| Math fundamentals | Decades | Linear algebra, calculus, probability |
| Core algorithms | 5-10 years | Backprop, SGD, attention mechanism |
| Architectures | 2-5 years | Transformers, CNNs, specific models |
| Best practices | 1-2 years | Training recipes, hyperparameters |
| SOTA models | Months | GPT-X, latest checkpoints |
| Library APIs | Months | Specific function calls |

**Implication:** Focus your deep learning on fundamentals. Stay loosely aware of trends. Deep dive only when you need something specific.

---

## 📚 Part 1: The Information Landscape

### Source Types Compared

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Information Source Spectrum                         │
│                                                                     │
│   Speed          ◀─────────────────────────────────────▶           │
│   (fast)                                           (slow)           │
│                                                                     │
│   Twitter  Blogs  Newsletters  Papers  Conferences  Books           │
│     │        │        │          │          │         │             │
│   Hours    Days    Weekly    Months    6-12mo     1-2yr             │
│                                                                     │
│   Noise         ◀─────────────────────────────────────▶            │
│   (high)                                           (low)            │
│                                                                     │
│   Twitter  Blogs  arXiv    Newsletters  Conferences  Books          │
│                                                                     │
│   Depth          ◀─────────────────────────────────────▶           │
│   (shallow)                                        (deep)           │
│                                                                     │
│   Twitter  News  Blogs  Newsletters  Papers  Books                  │
│                                                                     │
│   Strategy: Use fast sources for awareness, slow sources for depth  │
└─────────────────────────────────────────────────────────────────────┘
```

### Speed vs Quality Tradeoff

| Source | Update Speed | Signal/Noise | Depth | Best For |
|--------|--------------|--------------|-------|----------|
| Twitter/X | Hours | Very Low | Shallow | Breaking news, trends |
| arXiv | Hours | Low-Medium | High | Latest research |
| Blogs | Days-Weeks | Medium | Medium | Explanations, tutorials |
| Newsletters | Weekly | High | Medium | Curated summaries |
| Conference papers | 6-12 months | High | High | Vetted research |
| Books/Courses | 1-2 years | Very High | Very High | Foundations |

---

## 📚 Part 2: Primary Sources

### arXiv — The Research Firehose

[arxiv.org](https://arxiv.org) hosts preprints before peer review. Most ML research appears here first.

**Key Categories:**

| Category | Focus | Daily Volume |
|----------|-------|--------------|
| cs.LG | Machine Learning | 50-100+ |
| cs.CL | Computation and Language (NLP) | 30-50 |
| cs.CV | Computer Vision | 40-70 |
| cs.AI | Artificial Intelligence | 20-40 |
| stat.ML | Statistical ML | 10-20 |

**How to use arXiv:**
```
Daily listings:    https://arxiv.org/list/cs.LG/recent
Search:            https://arxiv.org/search/
Alerts:            https://arxiv.org/help/subscribe (email alerts)
```

> **Warning:** You cannot read 50+ papers per day. Use arXiv for targeted searches, not browsing. Let curated sources filter for you.

### Semantic Scholar — Smart Search

[semanticscholar.org](https://www.semanticscholar.org/)

Better than Google Scholar for ML because:
- AI-powered recommendations
- "Highly influential citations" highlights
- Research feeds based on your interests
- TLDR summaries for papers

**Pro tips:**
- Create an account and build a library
- Use "Research Feeds" for personalized recommendations
- Check "Highly Influential Citations" to find seminal papers

### Papers With Code — Research + Implementation

[paperswithcode.com](https://paperswithcode.com/)

Essential features:
- Papers linked to GitHub implementations
- State-of-the-art leaderboards by task
- Methods and components explained
- Datasets with benchmarks

**Use for:**
- "What's the current best model for [task]?"
- "Is there code for this paper?"
- Discovering papers by browsing tasks

### Connected Papers — Visualize the Literature

[connectedpapers.com](https://www.connectedpapers.com/)

Enter a paper, get a visual graph showing:
- Prior work (what this paper builds on)
- Derivative work (what built on this paper)
- Similar work (papers in the same area)

**Use for:**
- Understanding a new research area
- Literature reviews
- Finding papers you missed

---

## 📚 Part 3: Curated Sources (High Signal)

These sources filter the firehose for you. **This is where you should spend most of your "staying current" time.**

### Newsletters — Weekly Digests

| Newsletter | Focus | Frequency | Why Subscribe |
|------------|-------|-----------|---------------|
| [The Batch](https://www.deeplearning.ai/the-batch/) | General AI | Weekly | Andrew Ng's perspective, accessible |
| [ImportAI](https://jack-clark.net/) | AI policy + research | Weekly | Policy implications, big picture |
| [Ahead of AI](https://magazine.sebastianraschka.com/) | LLMs, research | Weekly | Technical depth, practitioner focus |
| [Davis Summarizes Papers](https://dblalock.substack.com/) | Paper summaries | Weekly | Quick paper digests |
| [The Gradient](https://thegradient.pub/) | In-depth articles | Bi-weekly | Long-form analysis |
| [TLDR AI](https://tldr.tech/ai) | Daily news | Daily | Quick headlines |

**Recommendation:** Subscribe to 2-3 newsletters max. More becomes noise.

### Blogs — Deep Explanations

| Blog | Author | Known For | URL |
|------|--------|-----------|-----|
| Lil'Log | Lilian Weng | Comprehensive surveys | lilianweng.github.io |
| Jay Alammar | Jay Alammar | Visual explanations | jalammar.github.io |
| Karpathy's Blog | Andrej Karpathy | Deep, accessible | karpathy.ai |
| Chip Huyen | Chip Huyen | MLOps, practical ML | huyenchip.com/blog |
| Sebastian Raschka | Sebastian Raschka | LLMs, tutorials | sebastianraschka.com/blog |
| Eugene Yan | Eugene Yan | Production ML, RecSys | eugeneyan.com |
| colah's blog | Chris Olah | Neural network intuitions | colah.github.io |

### YouTube — Video Explanations

| Channel | Style | Best For |
|---------|-------|----------|
| [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) | Paper deep-dives | Understanding new papers |
| [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) | From-scratch tutorials | Building intuition |
| [3Blue1Brown](https://www.youtube.com/@3blue1brown) | Visual math | Understanding concepts |
| [Two Minute Papers](https://www.youtube.com/@TwoMinutePapers) | Quick summaries | Awareness |
| [StatQuest](https://www.youtube.com/@statquest) | Statistics basics | Foundation building |
| [Mutual Information](https://www.youtube.com/@EntropicEvidence) | Technical deep-dives | Research understanding |

---

## 📚 Part 4: Social Sources

### Twitter/X — High Noise, Potentially High Signal

Twitter is noisy, but can be high-signal if curated carefully.

**Key Accounts (Research):**

| Account | Focus |
|---------|-------|
| @_akhaliq | Paper announcements (prolific) |
| @ylecun | AI research (Meta, opinions) |
| @kaborepharma | Paper highlights |
| @AndrewYNg | AI education |
| @hardmaru | Research + creative ML |
| @jeffdean | Google AI |

**Key Accounts (Industry):**

| Account | Focus |
|---------|-------|
| @sama | OpenAI |
| @ClementDelangue | Hugging Face, open source |
| @EMostaque | Stability AI |
| @sataborepharma | Model releases |

**Twitter Strategy:**
1. Create a dedicated ML list (don't pollute your main feed)
2. Follow accounts, but view via list
3. Check 2-3 times per week, not daily
4. Mute aggressively

> **Warning:** Twitter rewards hot takes over accuracy. Verify claims before believing them.

### Reddit

| Subreddit | Focus | Quality | Frequency |
|-----------|-------|---------|-----------|
| r/MachineLearning | Research discussion | High | Daily |
| r/LocalLLaMA | Open-source LLMs | Medium-High | Very active |
| r/learnmachinelearning | Learning resources | Medium | Daily |
| r/artificial | General AI | Low-Medium | Daily |

**r/MachineLearning** has good paper discussions. The "What are you reading?" threads are useful.

### Discord/Slack Communities

| Community | Focus | Good For |
|-----------|-------|----------|
| Hugging Face Discord | Transformers, NLP | Quick questions, announcements |
| Eleuther AI Discord | Open research | Research discussions |
| MLOps Community Slack | Production ML | Practical advice |
| Weights & Biases Discord | ML experiments | Tool-specific help |

---

## 📚 Part 5: Conferences

### Top ML Venues

| Conference | Focus | When | Acceptance Rate |
|------------|-------|------|-----------------|
| NeurIPS | General ML | December | ~25% |
| ICML | General ML | July | ~25% |
| ICLR | Representation learning | May | ~30% |
| CVPR | Computer vision | June | ~25% |
| ACL | NLP | July | ~25% |
| EMNLP | NLP | November | ~25% |

### Following Without Attending

You don't need to attend conferences to benefit:

- **Proceedings:** Released online for free
- **Best paper awards:** High-signal filter
- **Recorded talks:** YouTube, SlidesLive
- **Twitter threads:** People summarize key papers
- **OpenReview:** ICLR reviews are public

### Conference Timeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Conference Paper Lifecycle                          │
│                                                                     │
│   Submission deadline                                               │
│        │                                                            │
│        ▼                                                            │
│   Reviews (2-3 months)                                              │
│        │                                                            │
│        ▼                                                            │
│   Decisions announced ← Papers often on arXiv here                  │
│        │                                                            │
│        ▼                                                            │
│   Camera-ready (1-2 months)                                         │
│        │                                                            │
│        ▼                                                            │
│   Conference ← Talks, networking                                    │
│        │                                                            │
│        ▼                                                            │
│   Proceedings released ← Full access                                │
│                                                                     │
│   Total: ~6-9 months from submission to conference                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Part 6: Building Your System

### The Funnel Approach

```
┌─────────────────────────────────────────────────────────────────────┐
│                 The Information Funnel                              │
│                                                                     │
│                    ┌───────────────────┐                            │
│                    │   arXiv firehose  │  100+ papers/day           │
│                    │     (awareness)    │                            │
│                    └─────────┬─────────┘                            │
│                              │                                      │
│                              ▼                                      │
│                    ┌───────────────────┐                            │
│                    │   Newsletters     │  10-20 papers/week         │
│                    │    (filtered)     │                            │
│                    └─────────┬─────────┘                            │
│                              │                                      │
│                              ▼                                      │
│                    ┌───────────────────┐                            │
│                    │  Your reading     │  3-5 papers/week           │
│                    │      list         │                            │
│                    └─────────┬─────────┘                            │
│                              │                                      │
│                              ▼                                      │
│                    ┌───────────────────┐                            │
│                    │   Deep reading    │  1-2 papers/week           │
│                    │   (Pass 2+3)      │                            │
│                    └─────────┬─────────┘                            │
│                              │                                      │
│                              ▼                                      │
│                    ┌───────────────────┐                            │
│                    │Notes/implement    │  Few per month             │
│                    └───────────────────┘                            │
│                                                                     │
│   Each level filters for the next. Don't skip levels.              │
└─────────────────────────────────────────────────────────────────────┘
```

### Weekly Routine Example

| Day | Activity | Time | Focus |
|-----|----------|------|-------|
| **Monday** | Skim newsletters | 30 min | What happened last week |
| **Wednesday** | Deep read 1 paper | 1 hour | One paper, Pass 2 |
| **Friday** | Check Papers With Code | 30 min | SOTA in your area |
| **Ongoing** | Bookmark interesting things | 5 min | Build reading list |

**Total: ~2-3 hours/week** — sustainable and effective.

### Tools for Organization

| Tool | Use For | Free? |
|------|---------|-------|
| **Zotero** | Paper library, citations | Yes |
| **Semantic Scholar** | Discovery, recommendations | Yes |
| **Notion** | Notes, reading log | Yes (basic) |
| **Readwise** | Highlight management | No |
| **Pocket/Instapaper** | Save articles | Yes (basic) |
| **Feedly** | RSS feeds | Yes (basic) |

### A Minimal System

```markdown
## My Staying Current System

### Weekly inputs (2 sources max)
1. The Batch newsletter
2. Ahead of AI newsletter

### Monthly check
- Papers With Code: SOTA in my focus area

### On-demand
- Semantic Scholar: When I need to find papers
- Connected Papers: When exploring a new area

### Organization
- Zotero: Paper library
- Notion: Reading notes (using template from Module 5)

### What I explicitly ignore
- Daily arXiv browsing
- Twitter drama
- Every new model announcement
- Areas outside my focus
```

---

## 📚 Part 7: Avoiding Information Overload

### Signs of Overload

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Warning Signs                                       │
│                                                                     │
│   ✗ Hundreds of unread tabs/bookmarks                               │
│   ✗ FOMO about every new paper                                      │
│   ✗ Reading widely but not deeply                                   │
│   ✗ Knowing about techniques but not understanding them             │
│   ✗ Spending more time reading about ML than doing ML               │
│   ✗ Guilt about papers you "should" have read                       │
│                                                                     │
│   If these sound familiar, you're consuming too much.               │
└─────────────────────────────────────────────────────────────────────┘
```

### Strategies for Sanity

**1. Set Time Limits**
- Max 30 min/day on "staying current"
- Batch process (don't check continuously)
- Schedule specific times

**2. Depth Over Breadth**
- Better to deeply understand 5 papers than skim 50
- Implementation teaches more than reading
- One paper per week, fully understood, beats 10 skimmed

**3. Just-in-Time Learning**
- Learn things when you need them
- Not everything requires immediate attention
- Trust that you can find information when needed

**4. Aggressive Filtering**
- Unsubscribe from low-value sources
- Mute topics/people that don't serve you
- Delete bookmarks older than 1 month (if you haven't read it, you won't)

**5. Accept Missing Things**
- You WILL miss papers. That's okay.
- Important ideas resurface
- No one reads everything

### What to Ignore by Stage

| Your Stage | Focus On | Ignore |
|------------|----------|--------|
| **Learning foundations** | Courses, books, classic papers | arXiv, Twitter, latest models |
| **Building projects** | Practical tutorials, documentation | Theoretical papers, benchmarks |
| **Doing research** | Papers in your area, methods | Industry news, product launches |
| **Industry work** | MLOps, production techniques | Pure research, academic metrics |

---

## 📚 Part 8: Topic-Specific Resources

### LLMs and NLP

| Resource Type | Recommendations |
|---------------|-----------------|
| Papers | ACL Anthology, EMNLP, arXiv cs.CL |
| Blogs | Lil'Log, Jay Alammar, Sebastian Raschka |
| Code | Hugging Face Transformers, LangChain |
| Community | r/LocalLLaMA, Hugging Face Discord |
| Newsletters | Ahead of AI, The Batch |

### Computer Vision

| Resource Type | Recommendations |
|---------------|-----------------|
| Papers | CVPR, ICCV, ECCV, arXiv cs.CV |
| Code | timm library, Hugging Face Vision |
| Benchmarks | Papers With Code (vision tasks) |
| Community | r/computervision |

### MLOps / Production ML

| Resource Type | Recommendations |
|---------------|-----------------|
| Blogs | Chip Huyen, Eugene Yan |
| Courses | MLOps Zoomcamp, Made With ML |
| Tools | MLflow, Weights & Biases, DVC |
| Books | "Designing ML Systems" by Chip Huyen |
| Community | MLOps Community Slack |

### Reinforcement Learning

| Resource Type | Recommendations |
|---------------|-----------------|
| Papers | NeurIPS, ICML (RL track) |
| Courses | DeepMind x UCL lectures |
| Code | Stable Baselines 3, CleanRL |
| Books | Sutton & Barto (free online) |

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

### Exercise 3: Find Current SOTA

Check Papers With Code directly for current answers (they change frequently):

1. **ImageNet classification:**
   - Go to paperswithcode.com/sota/image-classification-on-imagenet
   - As of 2024: Models achieving 90%+ top-1 accuracy

2. **Machine translation (WMT):**
   - Go to paperswithcode.com/sota/machine-translation-on-wmt2014-english-german
   - Recent models achieve 30+ BLEU

3. **Text summarization with code:**
   - Go to paperswithcode.com/task/text-summarization
   - Look for papers with GitHub links (code icon)
   - Examples: PEGASUS, BART, LED

### Exercise 4: Example System

```markdown
## My Staying Current System

### Sources (5 max)
1. The Batch newsletter (weekly)
2. Ahead of AI newsletter (weekly)  
3. Papers With Code (weekly SOTA check)
4. Yannic Kilcher YouTube (select videos)
5. r/MachineLearning (occasional browse)

### Schedule
- Monday 7-7:30 AM: Read newsletters
- Wednesday lunch: One paper (Pass 2)
- Saturday morning: Deep reading or implementation

### Organization
- Zotero for paper library
- Notion for reading notes
- Template from Module 5 for each paper

### Explicitly Ignoring
- Daily arXiv browsing
- Twitter ML drama
- Every new model announcement
- Papers outside my focus (RL, robotics)
- Benchmarks I don't care about
```

</details>

---

## 🎯 Key Takeaways

1. **You can't read everything.** Build a system that filters for you. Newsletters > raw arXiv browsing.

2. **The funnel approach works.** Let curated sources filter the firehose → your reading list → deep reads → implementation.

3. **Depth beats breadth.** One paper fully understood is worth more than 10 papers skimmed. Implement to truly learn.

4. **Foundations have long half-lives.** Invest heavily in fundamentals (math, core algorithms). They remain relevant for years.

5. **Just-in-time learning is valid.** You don't need to know everything now. Trust that you can find what you need when you need it.

6. **Set explicit boundaries.** Decide what you'll ignore. Unsubscribe aggressively. Time-box your "staying current" activities.

7. **Systems beat willpower.** A simple, consistent system (2 newsletters + weekly paper) beats sporadic intense reading.

---

## 🔗 What's Next?

You now have tools to keep learning after this course ends. Complete the **Self-Assessment** in the `self_assessment/` folder to verify you're ready for Course 01: ML Foundations.

---

## 📖 Quick Reference: Key Resources

### Bookmark These

```
https://arxiv.org/list/cs.LG/recent     — Latest ML papers
https://www.semanticscholar.org/        — Smart paper search
https://paperswithcode.com/             — SOTA + code
https://www.connectedpapers.com/        — Paper relationships
```

### Subscribe to These (Pick 2)

```
The Batch:    deeplearning.ai/the-batch
Ahead of AI:  magazine.sebastianraschka.com
TLDR AI:      tldr.tech/ai
ImportAI:     jack-clark.net
```

### Watch These

```
Yannic Kilcher:   youtube.com/@YannicKilcher
Andrej Karpathy:  youtube.com/@AndrejKarpathy
3Blue1Brown:      youtube.com/@3blue1brown
```

### Follow These Blogs

```
Lil'Log:       lilianweng.github.io
Jay Alammar:   jalammar.github.io
colah's blog:  colah.github.io
```

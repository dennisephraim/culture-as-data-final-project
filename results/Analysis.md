# Linguistic Analysis Report: Fed vs. Media (2021–2025)

This report details the findings of 22 analytical tasks comparing the communication styles of the Federal Reserve (FED) and financial media (Bloomberg and FT).

---

## Group I: Semantic Façades & Focus

### Task 1: Normalization Baseline
- **What it measures**: The total word count per month for each corpus. This establishes the denominator for all density-based metrics.
- **What it shows**: The corpus is well-balanced with ~1.2M total words across the period, providing a statistically significant baseline for time-series comparison.

### Task 2: Institutional Control Cluster
- **What it measures**: Frequency of terms like *mandate*, *framework*, *toolkit*, *transparency*, and *methodology*. These represent the "Stability Façade."
- **What it shows**: The Fed uses these terms at a much higher frequency (3-4x) than the Media. There is a notable spike in "framework" and "methodology" usage during 2023, coinciding with post-inflationary policy reviews.

### Task 3: Volatility Regime Cluster
- **What it measures**: Frequency of "Crisis Language" terms such as *surge*, *shock*, *unprecedented*, *crisis*, *spike*, and *panic*.
- **What it shows**: Media coverage is highly sensitive to volatility, showing massive spikes in early 2022 and mid-2023. The Fed's usage of these terms is muted and lags behind the Media by 2-3 months.

### Task 4: The "And" Multiplier
- **What it measures**: Regex search for 3+ abstract nouns joined by "and" (e.g., "resilience and stability and growth"). This identifies "Competence Stacking."
- **What it shows**: The Fed is a heavy user of "Competence Stacking," particularly in opening statements. This linguistic device creates an aura of comprehensive control by linking multiple desirable but abstract outcomes.

### Task 5: Narrative Life-Cycle
- **What it measures**: Peak-to-trough frequency of specific terms like "transitory" or "soft landing" to track "Narrative Mortality."
- **What it shows**: "Transitory" shows a classic boom-bust cycle, peaking in June 2021 and almost entirely disappearing by January 2022. "Soft landing" emerges as a dominant narrative in 2023 and 2024.

### Task 6: Collocate Sentiment
- **What it measures**: Top adjectives within a ±5 word window around "Inflation."
- **What it shows**: The Fed's collocates are institutional and technocratic (*core*, *consistent*, *pce*), while Media collocates are more visceral and global (*global*, *persistent*, *bad*, *eurozone*).

### Task 7: Quarterly TF-IDF
- **What it measures**: Top 20 "Defining Tokens" with the highest TF-IDF scores per quarter for each corpus.
- **What it shows**: The Fed's defining tokens track specific policy shifts (*cbdc* in 2021, *crypto* in 2022, *ai* in 2023). Media tokens are more event-driven and reactive.

---

## Group II: Structural Abstraction (Bankspeak)

### Task 8: Nominalization Ratio
- **What it measures**: The frequency of words ending in *-tion*, *-sion*, *-ment*, *-ity*, and *-ance*. High ratios indicate high abstraction and "Bankspeak."
- **What it shows**: The Fed consistently exhibits a higher nominalization ratio (peaking near 0.035) compared to Media (around 0.020). This confirms that Fed communications are structurally more abstract and formalized than financial journalism.

### Task 9: Noun-on-Noun Density
- **What it measures**: Sequences of 2+ nouns (e.g., "Expectations anchoring"). Measures information compression.
- **What it shows**: Counter-intuitively, Media coverage shows a higher noun-on-noun density than the Fed. This likely reflects the "headlinese" style of Bloomberg and FT, where multiple descriptors are stacked for brevity, whereas the Fed uses more complete (if abstract) grammatical structures.

### Task 10: Verb-to-Noun Ratio
- **What it measures**: Ratio of total nouns to total verbs. High ratios indicate a focus on "static things" rather than "events."
- **What it shows**: Both corpora are noun-heavy (ratios > 2.0), but the Fed maintains a slightly higher and more stable ratio, suggesting a consistent focus on institutional entities and conceptual frameworks over dynamic actions.

### Task 11: Subjectivity Index
- **What it measures**: Ratio of first-person pronouns (*I*, *we*) vs. impersonal entities (*The Committee*, *The Board*). Measures "Human Removal."
- **What it shows**: The Media has a significantly higher subjectivity index, often 2-3x that of the Fed. The Fed's language is heavily filtered through impersonal institutional identifiers, effectively "removing the human" from the policy decision-making process.

### Task 12: Passive Voice Density
- **What it measures**: Frequency of passive constructions using dependency parsing. Measures agency erasure.
- **What it shows**: The Fed utilizes passive voice constructions at a higher rate than the Media. This agency erasure (e.g., "rates were raised" instead of "we raised rates") is a hallmark of institutional accountability distancing.

### Task 13: Sentence Complexity
- **What it measures**: Average sentence length and syllables-per-word (Flesch-Kincaid components).
- **What it shows**: Fed sentences average 30-35 words, significantly longer than the Media's 25-30 word average. Combined with higher syllables-per-word, the Fed's communication remains consistently at a higher "grade level" of complexity.

---

## Group III: Temporality & Accountability

### Task 14: Gerund/Verb Ratio
- **What it measures**: Ratio of *-ing* verbs (fostering) to active past/present tense verbs. Identifies "Blurred Temporality."
- **What it shows**: The Fed uses gerunds at a significantly higher ratio than the Media. By favoring "ongoing" actions (*promoting*, *maintaining*) over completed actions, the Fed creates a sense of perpetual process that avoids definitive start/end points of accountability.

### Task 15: Specificity Gap
- **What it measures**: Frequency of specific time adverbs (*now*, *today*, *June*) vs. vague adverbs (*periodically*, *eventually*).
- **What it shows**: Media coverage is 40% more likely to use specific temporal anchors. The Fed relies on vague temporal adverbs, which allows for maximum policy flexibility and "forward guidance" that is never pinned to a specific calendar date.

### Task 16: Process Framing
- **What it measures**: Frequency of "Assessment" verbs (*monitoring*, *evaluating*, *assessing*, *reviewing*). Tracks "Institutional Stalling."
- **What it shows**: Process framing is a dominant feature of Fed rhetoric, staying high and stable regardless of the economic cycle. This suggests that the Fed frames its primary value as "the act of watching" rather than "the act of deciding."

### Task 17: Hedging & Modals
- **What it measures**: Frequency of modal verbs (*may*, *might*, *could*, *should*) as a percentage of total verbs. Measures uncertainty.
- **What it shows**: Fed communications contain a significantly higher density of modal hedging. This linguistic caution is designed to prevent "market surprises" but also serves to dilute the perceived agency of the Committee.

---

## Group IV: Relational Metrics

### Task 18: CPI/Nominalization Correlation
- **What it measures**: Correlation between the Fed's "Nominalization Ratio" and monthly CPI/Inflation data.
- **What it shows**: The correlation is near zero (-0.07). This is a critical finding: it suggests that "Bankspeak" (linguistic abstraction) is an **inherent structural feature** of the Fed rather than a reactive defense mechanism to high inflation. The Fed remains abstract even when inflation is low.

### Task 19: Narrative Lag
- **What it measures**: Time delay between "Volatility" terms appearing in Media vs. when they appear in Fed communications.
- **What it shows**: The cross-correlation peaks at a 2-month lag, confirming that the Media's "Crisis Language" leads the Fed's recognition of volatility. The Media acts as an early warning system (or amplifier), while the Fed maintains linguistic "poker face" for approximately 60 days.

### Task 20: Self-Referentiality
- **What it measures**: How often the Fed refers to its own "framework" or "methodology."
- **What it shows**: Self-referentiality in the Fed corpus spiked in 2023. As inflation became persistent, the Fed linguistically retreated into its own "Framework" to justify its delayed response, effectively using methodology as a shield.

### Task 21: Dissent Scarcity
- **What it measures**: Frequency of disagreement markers (*dissent*, *alternative*, *disagree*, *however*).
- **What it shows**: Markers of dissent are almost non-existent in Fed communications compared to the Media. This highlights the "consensus-at-all-costs" culture of the FOMC, where linguistic unity is prioritized over the public airing of policy alternatives.

### Task 22: Readability Delta
- **What it measures**: Time-series plot of Flesch-Kincaid Grade Level for FED vs. Media.
- **What it shows**: The "Readability Gap" is persistent. The Fed consistently communicates at a post-graduate level (Grade 16+), while Media coverage averages Grade 12-13. This gap represents a "Democratic Deficit" in the accessibility of monetary policy.

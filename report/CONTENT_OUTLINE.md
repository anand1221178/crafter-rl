# IEEE Report Content Outline
**8-Page Structure: Intro (1p) + PPO (3p) + DQN (4p)**

---

## PAGE 1: Introduction & Background

### Section 1.1: Introduction (0.4 pages)
**What to write:**
- Start with hook: "RL has achieved remarkable successes (AlphaGo, Atari) but challenges remain in sparse-reward environments"
- Introduce PPO vs DQN comparison
- State Crafter as testbed

**Key points:**
- Deep RL overview (2 sentences)
- Sparse reward challenge (2 sentences)
- Our contribution: systematic comparison (2 sentences)

### Section 1.2: The Crafter Benchmark (0.4 pages)
**What to write:**
- Description: "2D procedurally-generated survival game"
- Key challenges:
  - Sparse rewards (~1% of steps)
  - Long-horizon tasks (5-10 step sequences)
  - 64×64×3 visual observations
  - 22 achievements across gathering, crafting, combat
- Crafter Score = geometric mean of achievement rates

**Include:**
- Reference Fig X showing Crafter screenshot/overview
- List achievement categories: wood → table → pickaxe → coal → iron

### Section 1.3: Research Objectives (0.1 pages)
**What to write:**
- Implement PPO and DQN from scratch
- Systematic improvement through iterations
- Compare exploration strategies
- Document failures for scientific rigor

### Section 1.4: Contributions (0.1 pages)
**Bullet list:**
- PPO: 8.61% score (+69.5% improvement)
- DQN: [partner fills]
- 11+ experiments documented
- Open source code

---

## PAGES 2-4: PPO Implementation (Anand's Work)

### Section 3.1: Algorithm Overview (0.3 pages)
**What to write:**
- PPO clipped objective equation
- Explain probability ratio r_t(θ)
- Mention GAE for advantage estimation
- Why PPO: "Balances sample efficiency with stability"

**Include:**
- Key equation (clipped surrogate objective)
- Brief: on-policy learning, multiple epochs per rollout

### Section 3.2: Network Architecture (0.3 pages)
**What to write:**
- Shared CNN encoder: 3 conv layers (64×64×3 → 8×8×64)
- Actor head: FC layers → 17-dim action distribution
- Critic head: FC layers → scalar value
- Hidden dim: 512 (baseline) vs 1024 (improved)

**Include:**
- Reference architecture diagram if space
- Parameter counts: 2.1M → 3.8M

### Section 3.3: Evaluation 1 - Weak Baseline (0.4 pages)
**What to write:**
- Goal: Deliberately suboptimal to show clear improvement
- Hyperparameters:
  - lr = 1e-3 (too high)
  - entropy = 1e-4 (too low)
- Results: 5.08% score
- Achievements: Wood (90%), Sapling (85%), Table (75%)
- Failures: No coal, no tools, episode length 165

**Key insight:** "Agent learned basic skills but failed at tool crafting"

### Section 3.4: Improvement 1 - Hyperparameter Tuning (0.4 pages)
**What to write:**
- Changes: lr = 5e-4, entropy = 0.001
- Rationale: Standard configs from CleanRL, Stable-Baselines3
- Results: 7.10% (+39.8%)
- Improvements: Wood pickaxe 51%, sword 48%, zombie 48%

**Key insight:** "Literature-based hyperparameters unlocked baseline performance"

### Section 3.5: Improvement 2 - ICM Curiosity (0.6 pages)
**What to write:**
- Motivation: Sparse rewards need dense exploration signal
- ICM components:
  1. Feature encoder (CNN → 512D)
  2. Inverse model (predicts action from transition)
  3. Forward model (predicts next state, error = reward)
- Combined reward: (1-β)·extrinsic + β·intrinsic, β=0.2
- Results: 8.27% (+16.5%)
- Discovered: Coal (1%), Skeleton (3%), Zombie (54%)
- Intrinsic/extrinsic ratio: 1.6× → 0.4× (exploration→exploitation)

**Key insight:** "Curiosity-driven exploration discovered rare achievements through novelty-seeking"

**Include:**
- Equation for combined reward
- Reference ICM architecture diagram if space

### Section 3.6: Improvement 3 - Large Network + Low β (0.5 pages)
**What to write:**
- Two complementary changes:
  1. hidden_dim = 1024 (80% more params)
  2. β = 0.15 (25% less curiosity)
- Rationale: Capacity for complex behaviors + more exploitation focus
- Results: 8.61% (+4.1%)
- Total: 5.08% → 8.61% = +69.5%

**Key insight:** "Training metrics (5.14 reward) looked WORSE but evaluation (8.61%) was BETTER. Training rewards ≠ final performance!"

### Section 3.7: Failed Attempts (0.4 pages)
**What to write:**
- List 7 failed strategies with scores and lessons:

1. **RND Curiosity (6.11%, 3 attempts)**
   - Intrinsic reward too high (42.28 vs ICM's 3.08)
   - Curiosity trap: agent chased novelty, ignored goals
   - Lesson: Random features don't capture task-relevant novelty

2. **High Curiosity β=0.3 (6.38%)**
   - Another curiosity trap
   - Lesson: Balance exploration/exploitation carefully

3. **High Entropy 0.005 (4.86%), 0.002 (6.08%)**
   - Policy too stochastic, never converged
   - Lesson: Excessive randomness prevents learning

4. **Conservative LR 1e-4 (8.11%)**
   - Too slow for 1M steps
   - Lesson: Match LR to training budget

5. **Extended Training 1.5M (7.52%)**
   - More training ≠ better
   - PPO policy degraded after 1M
   - Lesson: Know when to stop

6. **Dual-Clip (5.79%)**
   - Asymmetric updates + curiosity = instability
   - Lesson: Test modifications carefully

**Include:**
- Reference Fig 4 (failed attempts bar chart)
- Emphasize: "11 total experiments shows rigorous methodology"

### Section 3.8: Implementation Details (0.1 pages)
**What to write:**
- Hardware: M4 Pro MacBook Pro (MPS GPU)
- Training: ~2.5 hours/run, ~125 FPS
- Total: 11 runs, ~30 hours
- Evaluation: 100 episodes per checkpoint
- Code: [GitHub link]

---

## PAGES 5-8: DQN Implementation (Partner's Work)

### Section 4.1: Algorithm Overview (0.5 pages)
**Partner fills:**
- Q-learning formulation
- Bellman equation
- Target networks
- Experience replay
- ε-greedy exploration

### Section 4.2: Network Architecture (0.3 pages)
**Partner fills:**
- CNN structure
- Dueling architecture (if used)
- Parameter count

### Section 4.3: Baseline (0.5 pages)
**Partner fills:**
- Initial DQN configuration
- Results
- Strengths/weaknesses observed

### Section 4.4-4.6: Improvements 1, 2, 3 (1.5 pages total)
**Partner fills:**
- Each improvement: 0.5 pages
- Strategy, rationale, results
- Show progression like PPO section

### Section 4.7: Failed Attempts (0.5 pages)
**Partner fills:**
- Document failures for rigor
- Match PPO's format

### Section 4.8: Implementation Details (0.2 pages)
**Partner fills:**
- Hardware, training time, code

### Section 4.9: DQN-Specific Analysis (0.5 pages)
**Partner fills:**
- Sample efficiency
- Replay buffer insights
- Exploration decay strategy
- Any unique findings

---

## PAGE 7-8: Comparison, Discussion, Conclusion

### Section 5: Comparative Analysis (0.8 pages)

#### 5.1: Performance Comparison (0.3 pages)
**What to write:**
- Table comparing final scores, training time, achievements
- PPO: 8.61%, DQN: [X%]
- Which won overall?

#### 5.2: Exploration Strategy Analysis (0.3 pages)
**What to write:**
- **PPO + ICM**: Dense curiosity rewards, discovered rare achievements
- **DQN + ε-greedy**: [Partner analyzes]
- Compare: Which exploration strategy better suited for sparse rewards?

#### 5.3: Sample Efficiency (0.2 pages)
**What to write:**
- Steps to reach 5%, 6%, 7% thresholds
- DQN typically more sample-efficient (off-policy)
- But did it help on Crafter?

### Section 6: Discussion (0.6 pages)

#### 6.1: Key Findings (0.3 pages)
**What to write:**
1. Policy gradient vs value-based on sparse rewards
2. Exploration matters: ICM >> ε-greedy for discovery
3. Network capacity helps generalization
4. More training can hurt (PPO @ 1.5M)
5. Training metrics ≠ evaluation performance

#### 6.2: Limitations (0.15 pages)
**What to write:**
- Single environment (Crafter)
- Limited compute (~30 hours)
- Manual hyperparameter search
- No hierarchical RL tested

#### 6.3: Future Work (0.15 pages)
**What to write:**
- Hybrid PPO+DQN approaches
- Hierarchical RL for compositional tasks
- Meta-learning for faster adaptation
- Multi-agent cooperation

### Section 7: Conclusion (0.3 pages)
**What to write:**
- Summary: Achieved 8.61% PPO, [X%] DQN through systematic experimentation
- Key takeaway: Curiosity-driven exploration critical for sparse rewards
- Documented 11+ experiments (rigorous methodology)
- Code publicly available
- [Winner algorithm] proved most effective for Crafter

---

## FIGURES TO INCLUDE

**Required figures (5 total):**
1. **Fig 1**: Score progression (all attempts) - `fig1_score_progression.pdf`
2. **Fig 2**: Achievement comparison - `fig2_achievement_comparison.pdf`
3. **Fig 3**: Summary metrics - `fig3_summary_metrics.pdf`
4. **Fig 4**: Failed attempts analysis - `fig4_failed_attempts.pdf`
5. **Fig 5**: Improvement breakdown - `fig5_improvement_breakdown.pdf`

**Optional (if space):**
- Crafter environment screenshot
- Network architecture diagram
- ICM module diagram
- Training curves over time

---

## WRITING TIPS

### For Introduction (Page 1):
- **Start strong**: "Deep reinforcement learning has achieved remarkable success..."
- **Be concise**: 1 sentence per key point
- **Use citations**: [1], [2] after claims
- **Set up comparison**: "We compare PPO (policy gradient) vs DQN (value-based)"

### For Methods (Pages 2-6):
- **Technical but clear**: Explain equations in words
- **Results first**: "Achieved X% (+Y% improvement)"
- **Then explain why**: "This worked because..."
- **Show progression**: Each improvement builds on previous

### For Results (Page 7):
- **Lead with numbers**: Table showing final scores
- **Compare directly**: "PPO achieved 8.61% vs DQN's X%"
- **Explain differences**: Why did one win?

### For Discussion (Page 8):
- **Big picture**: What did we learn about RL on sparse rewards?
- **Honest limitations**: Single environment, limited compute
- **Actionable insights**: "Practitioners should use curiosity for sparse rewards"

---

## LATEX COMPILATION

```bash
cd report/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use Overleaf for easier compilation.

---

## PAGE BUDGET ALLOCATION

- **Page 1**: Introduction + Crafter description
- **Pages 2-4**: Your PPO work (3 pages)
  - 0.3p: Algorithm
  - 0.3p: Architecture
  - 0.4p: Eval 1
  - 0.4p: Imp 1
  - 0.6p: Imp 2
  - 0.5p: Imp 3
  - 0.4p: Failed attempts
  - 0.1p: Details
- **Pages 5-8**: Partner's DQN work (4 pages)
- **Last 0.5p Page 7-8**: Comparison + Discussion

---

## NEXT STEPS

1. ✅ LaTeX template created (`report/main.tex`)
2. ✅ All figures generated (`results/paper_figures/`)
3. ⏳ Fill in PPO sections (you)
4. ⏳ Fill in DQN sections (partner)
5. ⏳ Write comparison/discussion together
6. ⏳ Compile and check page count
7. ⏳ Proofread and polish

**You have everything you need to write your 3 pages now!**

Key files:
- Template: `report/main.tex`
- Figures: `results/paper_figures/*.pdf`
- Content guide: `CLAUDE.md` (all your results)
- This outline: For structure

Good luck! 🚀

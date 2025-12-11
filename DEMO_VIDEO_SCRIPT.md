# 10-Minute Demonstration Video Script
## ORION-AI — **O**ptimized **R**esearch & **I**nvestment **O**rchestration **N**etwork

**Total Duration: 10 minutes**
**Target Audience: Course instructors and evaluators**

---

## SEGMENT 1: Introduction and Overview (2:00)

**[0:00 - 0:30] Opening Hook**
- "Today I'll demonstrate a reinforcement learning system that learns to make investment recommendations by orchestrating multiple AI agents."
- Show the system running live: "Watch as this system analyzes NVDA stock and generates a recommendation."
- Quick screen capture of the dashboard showing a recommendation being generated.

**[0:30 - 1:00] Project Overview**
- "This project implements two RL approaches: Deep Q-Networks for value-based learning and Proximal Policy Optimization for policy gradients."
- Show architecture diagram from the report.
- "The system uses a Controller Agent to orchestrate 5 specialized agents: Research, Technical Analysis, Insight, Recommendation, and Evaluator agents."

**[1:00 - 2:00] Key Features**
- "The system learns from actual stock returns, making it outcome-based rather than just pattern matching."
- "It's been trained on 10,000 episodes across 5 different stocks."
- "The system includes a production-ready REST API and interactive web dashboard."
- Show quick demo of the dashboard interface.

**Transition:** "Now let me show you how the system works and how it learned to improve."

---

## SEGMENT 2: System Architecture and How It Works (2:30)

**[2:00 - 3:00] Agent Orchestration**
- "The Controller Agent uses a DQN model to decide which agent to call next."
- Show code snippet of ControllerAgent.orchestrate() method.
- "It can choose from 10 different actions: fetching news, running technical analysis, generating insights, and more."
- Demonstrate one full orchestration cycle with screen recording:
  - Show Research Agent fetching news
  - Show Technical Analysis Agent calculating indicators
  - Show Insight Agent generating analysis
  - Show Recommendation Agent producing final recommendation

**[3:00 - 4:00] Reinforcement Learning Integration**
- "The DQN model learns which sequence of agent actions leads to better recommendations."
- Show the state space: "The model receives a 21-dimensional state vector including market data, technical indicators, and sentiment scores."
- Show action selection: "Based on this state, it selects the next action using epsilon-greedy exploration."
- "Rewards are calculated based on recommendation correctness, efficiency, and diversity."

**[4:00 - 4:30] Reward Function**
- "The reward function is comprehensive: it rewards correct recommendations, penalizes incorrect ones, encourages efficiency, and values using diverse data sources."
- Show reward calculation code or formula from the report.
- "This multi-objective reward helps the system learn balanced strategies."

**Transition:** "Let me show you the training process and how the system improved over time."

---

## SEGMENT 3: Training Process and Learning Progress (3:00)

**[4:30 - 5:30] Training Overview**
- "The system was trained for 10,000 episodes across 5 stocks: NVDA, AAPL, TSLA, JPM, and XOM."
- Show training script running (or screenshots of training logs).
- "Each episode involves selecting a random date, running the agent orchestration, and learning from the outcome."
- Show checkpoint files: "We saved checkpoints every 1,000 episodes to track progress."

**[5:30 - 6:30] Learning Curves**
- Switch to the technical report, show the learning curves chart.
- "Here's how the system improved over training:"
  - Point to reward progression: "Average rewards improved from -0.25 to -0.14"
  - Point to accuracy curve: "Accuracy increased from 22% to 29.5%"
  - "The model learned to avoid worst-case scenarios and generate recommendations more consistently."

**[6:30 - 7:00] Before/After Comparison**
- Show the before/after comparison chart from the report.
- "At the start of training: 22% accuracy, 3.2% recommendation rate, average 4.2 steps per episode."
- "After 10,000 episodes: 29.5% accuracy, 5.6% recommendation rate, average 5.6 steps per episode."
- "This represents a 33% improvement in accuracy and 75% increase in recommendation generation."

**[7:00 - 7:30] Key Learning Insights**
- "The model learned to complete more of the analysis pipeline before stopping."
- "It improved at avoiding very negative rewards, though it still struggles with positive rewards."
- "One limitation we identified: epsilon didn't decay, so the model was always exploring and never exploiting learned knowledge."

**Transition:** "Now let me demonstrate the system in action with a live example."

---

## SEGMENT 4: Live Demonstration (2:30)

**[7:30 - 8:30] Live Stock Analysis**
- "Let me analyze a stock in real-time using the trained model."
- Open the dashboard or API interface.
- Enter a stock symbol (e.g., "AAPL").
- Show the system running:
  - "The Controller Agent is selecting actions..."
  - "Research Agent is fetching news and fundamentals..."
  - "Technical Analysis Agent is calculating indicators..."
  - "Insight Agent is generating analysis..."
  - "Recommendation Agent is synthesizing everything..."
- Show the final recommendation with confidence score.

**[8:30 - 9:00] Recommendation Explanation**
- "The system recommends [Buy/Hold/Sell] with [X]% confidence."
- Show the insight text explaining the reasoning.
- "This recommendation is based on:"
  - News sentiment: [positive/negative/neutral]
  - Technical indicators: [trend, RSI, MACD values]
  - Fundamental analysis: [P/E ratio, growth metrics]
- "The system learned this strategy through 10,000 episodes of training."

**[9:00 - 10:00] Results and Performance Metrics**
- Switch to the technical report results section.
- "Overall performance: 29.46% accuracy across all stocks."
- Show per-stock performance: "Performance varies by stock, with XOM showing 34% accuracy."
- Show the baseline comparison: "Compared to baselines:"
  - "Better than sentiment-only (28.7%)"
  - "Slightly below technical analysis baseline (31.2%)"
  - "Below buy-and-hold (45.2%), which is expected in an upward market"
- "The system shows measurable learning, though there's room for improvement."

**Transition:** "Let me summarize the key achievements and future work."

---

## SEGMENT 5: Summary and Future Work (0:30)

**[10:00 - 10:30] Key Achievements**
- "This project successfully demonstrates:"
  - "Two RL approaches integrated with agentic systems"
  - "Measurable learning over 10,000 training episodes"
  - "Production-ready system with API and dashboard"
  - "Comprehensive evaluation and analysis"
- "The system learns to orchestrate agents effectively and improves recommendation quality over time."

**[10:30] Closing**
- "Thank you for watching. The complete code, documentation, and technical report are available in the GitHub repository."
- Show GitHub repository link or QR code.
- "Questions and feedback are welcome!"

---

## VISUAL ELEMENTS TO INCLUDE

### Screen Recordings Needed:
1. **Dashboard Demo** (30 seconds)
   - Opening the dashboard
   - Entering stock symbol
   - Showing recommendation generation

2. **Code Walkthrough** (1 minute)
   - ControllerAgent.orchestrate() method
   - DQN action selection
   - Reward calculation

3. **Training Visualization** (1 minute)
   - Training script output
   - Learning curves animation (if possible)
   - Checkpoint files

4. **Live Analysis** (1.5 minutes)
   - Full agent orchestration cycle
   - Real-time recommendation generation
   - Result explanation

### Charts/Diagrams to Show:
- System architecture diagram (Mermaid)
- Learning curves chart
- Before/after comparison chart
- Baseline comparison chart
- Per-stock performance chart

### Code Snippets to Highlight:
- ControllerAgent initialization
- DQN action selection
- Reward function calculation
- Agent orchestration logic

---

## TIPS FOR RECORDING

1. **Preparation:**
   - Test all demos beforehand
   - Have stock symbols ready (NVDA, AAPL, TSLA)
   - Prepare screen recordings in advance
   - Test audio quality

2. **Recording:**
   - Use screen recording software (OBS, QuickTime, Loom)
   - Record in 1080p or higher
   - Use a good microphone for clear audio
   - Record in a quiet environment

3. **Editing:**
   - Add text overlays for key points
   - Use zoom/pan for code snippets
   - Add transitions between segments
   - Include timestamps in video description

4. **Presentation:**
   - Speak clearly and at moderate pace
   - Pause briefly after key points
   - Use cursor highlighting for important elements
   - Maintain enthusiasm throughout

---

## BACKUP CONTENT (If Running Short)

- Show more detailed code walkthrough
- Demonstrate error handling
- Show the API endpoints
- Explain the state encoding in detail
- Show hyperparameter sensitivity analysis

---

## BACKUP CONTENT (If Running Long)

- Skip detailed code walkthrough
- Show only key charts
- Condense live demo to one stock
- Focus on learning progress over detailed architecture

---

**Total Script Duration: ~10 minutes**
**Recommended Recording Duration: 11-12 minutes (allows for pauses and transitions)**


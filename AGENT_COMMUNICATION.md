# Agent Communication Architecture

## Overview

Agents in this system **do NOT directly communicate with each other**. Instead, they share context through a **shared state store** (the `StockResearchEnv` environment) with **explicit context passing**. This is a hybrid design:

1. **Storage**: Shared state in `StockResearchEnv` (persists data throughout episode)
2. **Communication**: Context is explicitly extracted and passed as function parameters

This design pattern combines **"Shared State"** (Blackboard Architecture) with **"Explicit Context Passing"** for better clarity, testability, and decoupling.

## Communication Pattern: Shared State (Blackboard)

```
┌─────────────────────────────────────────────────────────────┐
│              StockResearchEnv (Shared State)                │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │ news_data│fundamenta│sentiment │ta_basic  │ insights │   │   
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
└─────────────────────────────────────────────────────────────┘
         ▲              ▲              ▲              ▲
         │              │              │              │
         │              │              │              │
    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
    │Research │   │   TA    │   │ Insight │   │   Rec   │
    │  Agent  │   │  Agent  │   │  Agent  │   │  Agent  │
    └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

## How It Works

### 1. **Shared State Store (StockResearchEnv)**

The `StockResearchEnv` acts as a **shared memory/context store** that all agents can read from and write to:

```python
class StockResearchEnv:
    def __init__(self):
        # Shared state variables - accessible by all agents
        self.news_data = None              # Written by Research Agent
        self.fundamentals_data = None      # Written by Research Agent
        self.sentiment_data = None          # Written by Research Agent
        self.ta_basic_data = None          # Written by TA Agent
        self.ta_advanced_data = None       # Written by TA Agent
        self.insights = []                  # Written by Insight Agent
        self.recommendation = None          # Written by Recommendation Agent
        self.confidence = 0.0               # Written by Recommendation Agent
```

### 2. **Agent Communication Flow**

#### Step 1: Research Agent Writes Data
```python
# In StockResearchEnv.step() when action='FETCH_NEWS'
self.news_data = self.research_agent.fetch_news(self.stock_symbol)
# ✅ Research Agent writes to shared state
```

#### Step 2: TA Agent Reads Previous Data, Writes New Data
```python
# In StockResearchEnv.step() when action='RUN_TA_BASIC'
self.ta_basic_data = self._run_ta_basic()
# ✅ TA Agent can read self.news_data (if needed)
# ✅ TA Agent writes to shared state
```

#### Step 3: Insight Agent Receives Context Explicitly
```python
# In StockResearchEnv._generate_insight()
# Context is EXTRACTED from shared state and PASSED explicitly
insight_agent.generate_insight(
    news_data=self.news_data,              # ✅ Explicitly passed from shared state
    fundamentals_data=self.fundamentals_data,  # ✅ Explicitly passed
    sentiment_data=self.sentiment_data,    # ✅ Explicitly passed
    ta_basic_data=self.ta_basic_data,      # ✅ Explicitly passed
    ta_advanced_data=self.ta_advanced_data # ✅ Explicitly passed
)
# ✅ Insight Agent writes to self.insights (stored back in shared state)
```

#### Step 4: Recommendation Agent Receives Context Explicitly
```python
# In StockResearchEnv._generate_recommendation()
# Context is EXTRACTED from shared state and PASSED explicitly
recommendation_agent.generate_recommendation(
    insights=self.insights,                # ✅ Explicitly passed from shared state
    news_data=self.news_data,              # ✅ Explicitly passed
    fundamentals_data=self.fundamentals_data,  # ✅ Explicitly passed
    ta_basic_data=self.ta_basic_data,     # ✅ Explicitly passed
    ta_advanced_data=self.ta_advanced_data # ✅ Explicitly passed
)
# ✅ Recommendation Agent writes to self.recommendation (stored back in shared state)
```

### 3. **Controller Agent Orchestration**

The `ControllerAgent` orchestrates the sequence but **does not pass data between agents**. It:
- Calls agents in sequence via `env.step(action)`
- Each agent reads/writes to the shared `env` state
- Agents never directly call each other

```python
# Controller Agent orchestration sequence
for action_name, agent_name in agent_sequence:
    # Execute action through environment
    next_state, reward, done, info = env.step(action_name)
    # ✅ Each step() call updates shared state
    # ✅ Next agent in sequence reads from updated shared state
```

## Context Availability

### ✅ **Agents DO Have Context**

Each agent has access to:
1. **All previously collected data** (via `env` shared state)
2. **Stock symbol** (passed during initialization)
3. **Current date/index** (stored in env)
4. **Historical price data** (stored in env)

### Example: Insight Agent Context

```python
def generate_insight(
    self,
    news_data: Optional[Dict[str, Any]] = None,        # ✅ Context from Research Agent
    fundamentals_data: Optional[Dict[str, Any]] = None, # ✅ Context from Research Agent
    sentiment_data: Optional[Dict[str, Any]] = None,     # ✅ Context from Research Agent
    ta_basic_data: Optional[Dict[str, Any]] = None,     # ✅ Context from TA Agent
    ta_advanced_data: Optional[Dict[str, Any]] = None,  # ✅ Context from TA Agent
    existing_insights: Optional[List[str]] = None,      # ✅ Previous insights
    stock_symbol: Optional[str] = None                  # ✅ Stock context
) -> List[str]:
    # Agent has FULL context of all previous work
```

### Example: Recommendation Agent Context

```python
def generate_recommendation(
    self,
    insights: List[str],                                 # ✅ Context from Insight Agent
    news_data: Optional[Dict[str, Any]] = None,         # ✅ Context from Research Agent
    fundamentals_data: Optional[Dict[str, Any]] = None, # ✅ Context from Research Agent
    ta_basic_data: Optional[Dict[str, Any]] = None,     # ✅ Context from TA Agent
    ta_advanced_data: Optional[Dict[str, Any]] = None    # ✅ Context from TA Agent
) -> Tuple[str, float, Dict[str, Any]]:
    # Agent has FULL context of all previous work
```

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    Controller Agent                          │
│              (Orchestrates sequence only)                    │
└──────────────────────────────────────────────────────────────┘
                            │
                            │ calls env.step(action)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              StockResearchEnv (Shared State)                 │
│                                                              │
│  Step 1: FETCH_NEWS                                          │
│    ├─ Research Agent writes: self.news_data                  │
│                                                              │
│  Step 2: FETCH_FUNDAMENTALS                                  │
│    ├─ Research Agent writes: self.fundamentals_data          │
│    └─ Can read: self.news_data (if needed)                   │
│                                                              │
│  Step 3: FETCH_SENTIMENT                                     │
│    ├─ Research Agent writes: self.sentiment_data             │
│    └─ Can read: self.news_data, self.fundamentals_data       │
│                                                              │
│  Step 4: RUN_TA_BASIC                                        │
│    ├─ TA Agent writes: self.ta_basic_data                    │
│    └─ Can read: all previous data                            │
│                                                              │
│  Step 5: RUN_TA_ADVANCED                                     │
│    ├─ TA Agent writes: self.ta_advanced_data                 │
│    └─ Can read: all previous data                            │
│                                                              │
│  Step 6: GENERATE_INSIGHT                                    │
│    ├─ Insight Agent reads: ALL previous data                 │
│    └─ Insight Agent writes: self.insights                    │
│                                                              │
│  Step 7: GENERATE_RECOMMENDATION                             │
│    ├─ Rec Agent reads: ALL previous data + insights          │
│    └─ Rec Agent writes: self.recommendation, confidence      │
│                                                              │
│  Step 8: EVALUATE_PERFORMANCE                                │
│    ├─ Evaluator Agent reads: ALL data + recommendation       │
│    └─ Evaluator Agent computes: reward, validation           │
└──────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### ✅ **Why Hybrid Approach (Shared State + Explicit Passing)?**

**Shared State (Storage):**
1. **RL Compatibility**: Environment naturally stores state for RL training
2. **State Persistence**: Data persists throughout entire episode
3. **Centralized Tracking**: All context in one place for debugging
4. **Reward Calculation**: Environment needs access to all data for rewards

**Explicit Context Passing (Communication):**
1. **Type Safety**: Clear function signatures show what data is needed
2. **Testability**: Agents can be tested in isolation with mock data
3. **Decoupling**: Agents don't need to know about environment structure
4. **Explicit Dependencies**: You can see exactly what each agent needs
5. **Reusability**: Agents can be used outside the environment

### ✅ **Context Persistence**

- Context persists throughout the entire episode
- Each agent can access **all previous work**
- No data is lost between agent calls
- Environment acts as a **persistent memory**

### ✅ **No Direct Agent-to-Agent Calls**

Agents **never** directly call each other:
- ❌ `insight_agent.generate_insight(research_agent.get_news())`  ← NOT USED
- ✅ `insight_agent.generate_insight(news_data=env.news_data)`  ← USED (explicit passing)

**Key Point**: We extract from shared state (`env.news_data`) and pass explicitly as parameters, rather than agents reading directly from the environment.

## Example: Full Context Chain

```python
# Episode starts
env = StockResearchEnv(stock_symbol='NVDA')
state = env.reset()

# Step 1: Research Agent fetches news
env.step('FETCH_NEWS')
# ✅ env.news_data = {'headlines': [...], 'sentiment_score': 0.65, ...}

# Step 2: Research Agent fetches fundamentals
env.step('FETCH_FUNDAMENTALS')
# ✅ env.fundamentals_data = {'pe_ratio': 45.2, 'revenue_growth': 0.15, ...}
# ✅ env.news_data still available

# Step 3: TA Agent runs basic analysis
env.step('RUN_TA_BASIC')
# ✅ env.ta_basic_data = {'rsi': 68.5, 'ma20': 150.2, ...}
# ✅ env.news_data, env.fundamentals_data still available

# Step 4: Insight Agent generates insights
env.step('GENERATE_INSIGHT')
# ✅ Insight Agent receives:
#    - news_data (from step 1)
#    - fundamentals_data (from step 2)
#    - ta_basic_data (from step 3)
# ✅ env.insights = ['Strong bullish sentiment...', 'RSI indicates...']

# Step 5: Recommendation Agent generates recommendation
env.step('GENERATE_RECOMMENDATION')
# ✅ Recommendation Agent receives:
#    - insights (from step 4)
#    - news_data (from step 1)
#    - fundamentals_data (from step 2)
#    - ta_basic_data (from step 3)
# ✅ env.recommendation = 'Buy', env.confidence = 0.82
```

## Summary

- **Communication Pattern**: Shared State (Blackboard Architecture)
- **Context Store**: `StockResearchEnv` instance
- **Agent Access**: All agents read/write to the same `env` object
- **Context Availability**: ✅ Full context - each agent sees all previous work
- **Orchestration**: Controller Agent sequences the calls, but doesn't pass data
- **No Direct Calls**: Agents never directly communicate with each other

This design ensures that:
1. ✅ Agents have full context of previous work
2. ✅ Data flows naturally through the shared state
3. ✅ System is modular and maintainable
4. ✅ Compatible with RL training (state is in the environment)


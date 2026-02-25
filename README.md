# AISCM Final Project

Multi-agent reinforcement learning simulation for a newsvendor supply chain problem with price and quantity optimization.

## Project Structure

### Core Files

- **`params.py`** - Configuration parameters for the simulation (rounds, demand distribution, action spaces, benchmarks)
- **`model.py`** - Main simulation model implementing the newsvendor environment using Mesa framework
- **`agents.py`** - Multi-agent implementations (Greedy, UCB, Thompson Sampling) for price and quantity decisions
- **`singleagents.py`** - Single-agent model where one agent controls both price and quantity decisions

### Notebooks

- **`execute.ipynb`** - Basic simulation runs and visualization of multi-agent scenarios
- **`experiment.ipynb`** - Comprehensive experiments comparing different agent configurations
- **`execute_llm_agent.ipynb`** - LLM-based agent experiments using GPT-4 for decision making
- **`supplier.ipynb`** - Supplier agent analysis with competitive vs. team player modes and price ceiling sensitivity

### Configuration

- **`pyproject.toml`** - Project dependencies and metadata
- **`.env`** - Environment variables (e.g., OpenAI API key for LLM agent)
- **`.gitignore`** - Git ignore rules

## Installation

```bash
pip install -e .
```

## Usage

Open any of the Jupyter notebooks to run experiments. Start with `execute.ipynb` for basic multi-agent simulations.

## Key Features

- Multi-agent reinforcement learning (Greedy, UCB, Thompson Sampling)
- Price and quantity optimization in newsvendor setting
- Supplier agent with competitive and cooperative modes
- LLM-based decision making
- Information sharing experiments
- Price ceiling sensitivity analysis

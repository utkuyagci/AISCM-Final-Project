# AISCM Final Project

Multi-agent reinforcement learning simulation for a newsvendor supply chain problem with price and quantity optimization.

Project Github Link: https://github.com/utkuyagci/AISCM-Final-Project

## Project Structure

### Core Files

- **`params.py`** - Configuration parameters for the simulation (rounds, demand distribution, action spaces, benchmarks)
- **`model.py`** - Main simulation model implementing the newsvendor environment using Mesa framework
- **`agents.py`** - Multi-agent implementations (Greedy, UCB, Thompson Sampling) for price and quantity decisions
- **`singleagents.py`** - Single-agent model where one agent controls both price and quantity decisions

### Notebooks

- **`execute.ipynb`** - Basic simulation runs and visualization of multi-agent scenarios
- **`experiment.ipynb`** - Comprehensive experiments comparing different agent configurations
- **`execute_llm_agent.ipynb`** - LLM-based agent experiments using Deepseek R1 for decision making
- **`supplier.ipynb`** - Supplier agent analysis with competitive vs. team player modes and price ceiling sensitivity

## Getting Started

### Prerequisites

- Python 3.14 or higher
- pip or poetry package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/utkuyagci/AISCM-Final-Project.git
   cd AISCM-Final-Project
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Set up environment variables (optional, for LLM agent)**
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `api_key=your_key_here` (For the LLM agent)

## Usage

Open any of the Jupyter notebooks to run experiments. 

## Key Features

- Multi-agent reinforcement learning (Greedy, UCB, Thompson Sampling)
- Price and quantity optimization in newsvendor setting
- Supplier agent with competitive and cooperative modes
- LLM-based decision making
- Information sharing experiments
- Price ceiling sensitivity analysis

## Contributors

- **Utku Yagci** - [utkuuguryagci07@gmail.com](mailto:utkuuguryagci07@gmail.com)
- **Mine Gokdere** - [gokdere.mine3@gmail.com](mailto:gokdere.mine3@gmail.com)
- **Patrick Poremba** - [patrick.poremba@tum.de](mailto:patrick.poremba@tum.de)
# Football Match Prediction System

A machine learning-based system for predicting football match outcomes using team form, head-to-head statistics, and goal difference metrics.

## Features

- **Real-time Data Integration**: Fetches up-to-date match statistics from football-data.org
- **Comprehensive Analysis**: Considers team form, head-to-head records, and goal difference
- **Advanced Modeling**: Uses a neural network for accurate match outcome predictions
- **User-friendly Interface**: Simple command-line interface for easy interaction
- **Historical Tracking**: Saves prediction history for future reference

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Football-data.org API key (free tier available)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd football-match-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API key:
   ```
   FOOTBALL_DATA_API_KEY=your_api_key_here
   ```

## Usage

1. Run the prediction system:
   ```bash
   python betting.py
   ```

2. Follow the on-screen prompts to enter the match details:
   - Home team name
   - Away team name
   - Competition (default: Premier League)
   - Season (default: current season)

3. The system will display the prediction along with:
   - Win/draw/loss probabilities
   - Expected goals (xG) for both teams
   - Team form metrics
   - Goal difference statistics

## Project Structure

- `betting.py`: Main prediction script and user interface
- `config.py`: Configuration and utility functions
- `fetch_match_stats.py`: Module for fetching and analyzing match statistics
- `fetch_head_to_head.py`: Module for retrieving head-to-head match data
- `goal_difference.py`: Module for calculating goal difference metrics
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not included in version control)
- `prediction_history.csv`: Saved prediction history (created after first run)

## Data Sources

- [football-data.org](https://www.football-data.org/): Match statistics and results
- [API Documentation](https://www.football-data.org/documentation/quickstart)

## Model Architecture

The prediction model uses a neural network with the following architecture:

1. Input Layer: 10 features (team form, goals scored/conceded, H2H stats, etc.)
2. Hidden Layer 1: 64 neurons with ReLU activation and dropout
3. Batch Normalization
4. Hidden Layer 2: 32 neurons with ReLU activation and dropout
5. Batch Normalization
6. Output Layer: 3 neurons with softmax activation (Home Win/Draw/Away Win)

The model is trained using categorical cross-entropy loss with the Adam optimizer and includes learning rate scheduling and early stopping for better performance.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to football-data.org for providing the football data API
- Inspired by various sports analytics projects and research papers
- Built with Python and TensorFlow

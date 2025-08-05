# Contributing to LongBall Analytics

Thank you for your interest in contributing to LongBall Analytics! This document provides guidelines for contributing to the project.

## Development Roadmap

We have organized our development priorities into comprehensive GitHub issues. Please check our [issue templates](.github/ISSUE_TEMPLATE/) for detailed improvement plans in the following areas:

### Model Performance Issues
- **Improve Pitching Projection Accuracy** - Address known issues with pitcher projections
- **Handle Limited MLB Experience Players Better** - Integrate minor league data for better rookie projections

### Feature Enhancements  
- **Add Confidence Intervals to Projections** - Implement uncertainty quantification
- **Implement Model Validation Framework** - Create backtesting system for model evaluation

### Technical Improvements
- **Migrate from Jupyter Notebooks to Production Pipeline** - Refactor to proper ML pipeline
- **Integrate Better Data Sources** - Research and implement improved data sources

### UI/UX Improvements
- **Modernize UI Design** - Overhaul interface with professional design principles
- **Enhance Trade Simulator Interface** - Add clickable player links and better navigation
- **Add Dynamic Player Statistics Visualization** - Implement interactive charts and visualizations

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Local Development Setup

#### Backend Setup
```bash
cd web-app/backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend Setup
```bash
cd web-app/frontend
npm install
npm run dev
```

#### Model Development
```bash
# Install model dependencies
pip install -r requirements.txt

# Navigate to models directory
cd models

# Open Jupyter notebooks for exploration
jupyter lab
```

## How to Contribute

### 1. Choose an Issue
- Look at our [comprehensive issue templates](.github/ISSUE_TEMPLATE/)
- Comment on issues you'd like to work on
- Start with issues labeled `good first issue` for easier entry points

### 2. Fork and Clone
```bash
git clone https://github.com/your-username/LSTMLB.git
cd LSTMLB
git checkout -b feature/your-feature-name
```

### 3. Development Workflow
- Make your changes following the issue's acceptance criteria
- Test your changes thoroughly
- Follow existing code style and conventions
- Update documentation as needed

### 4. Submit a Pull Request
- Push your changes to your fork
- Create a pull request with a clear description
- Reference the related issue in your PR description
- Ensure all tests pass (if applicable)

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Include docstrings for functions and classes
- Keep notebook outputs clean (clear before committing)

### TypeScript/React Code
- Follow existing ESLint configuration
- Use TypeScript interfaces for props and data structures
- Maintain component modularity
- Follow React best practices

### Documentation
- Update README.md if adding new features
- Document new APIs and data formats
- Include comments for complex algorithms
- Update issue templates if adding new categories

## Project Structure

```
LSTMLB/
├── .github/
│   └── ISSUE_TEMPLATE/        # Comprehensive issue templates
├── models/                    # ML models and notebooks
│   ├── *.ipynb               # Jupyter notebooks (current)
│   ├── MiLB/                 # Minor league related models
│   └── *.pkl                 # Trained model artifacts
├── web-app/
│   ├── frontend/             # React TypeScript frontend
│   └── backend/              # FastAPI Python backend
├── data/                     # Data storage
└── requirements.txt          # Python dependencies
```

## Issue Labels

We use the following labels to organize issues:

### Type Labels
- `bug` - Something isn't working
- `enhancement` - New feature or improvement
- `feature` - New functionality
- `technical-debt` - Code refactoring and improvements

### Priority Labels
- `high-priority` - Critical issues affecting core functionality
- `medium-priority` - Important improvements
- `low-priority` - Nice to have features

### Component Labels
- `model-performance` - ML model accuracy and training
- `ui/ux` - User interface and experience
- `data-sources` - Data integration and processing
- `infrastructure` - Development and deployment infrastructure
- `validation` - Testing and model validation
- `visualization` - Charts and data visualization

### Experience Labels
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed

## Model Development Guidelines

### Current Model Architecture
- LSTM-based projections for 15-year forecasts
- Separate models for batters, pitchers, fielding, and baserunning
- Current limitations acknowledged in README

### Improvement Areas
1. **Pitching Models**: Known accuracy issues need investigation
2. **Data Integration**: Minor league data for limited-experience players
3. **Validation**: Backtesting framework for model evaluation
4. **Pipeline**: Migration from notebooks to production code

### Data Considerations
- Playing time normalization (150 games hitters, 135 catchers, 32 starts, 65 innings relievers)
- WAR-to-dollar calculations with non-linear adjustments
- Sequence length of 5 years for LSTM training

## UI/UX Development Guidelines

### Current Tech Stack
- **Frontend**: React with TypeScript, Tailwind CSS, Vite
- **Backend**: FastAPI with Python
- **Deployment**: Netlify (frontend), Render (backend)

### Design Principles
- Professional appearance suitable for sports analytics
- Mobile-first responsive design
- Clear data visualization and statistics presentation
- Intuitive navigation and user workflows

## Testing Guidelines

### Model Testing
- Validate model performance on historical data
- Compare against existing projection systems where possible
- Test edge cases (rookies, injured players, etc.)

### Frontend Testing
- Test responsive design on multiple devices
- Validate user workflows and navigation
- Ensure accessibility compliance

### Backend Testing
- API endpoint testing
- Database integration testing
- Performance testing for model inference

## Community Guidelines

### Communication
- Be respectful and constructive in discussions
- Ask questions if you're unsure about implementation approaches
- Share your expertise and help others learn

### Collaboration
- Coordinate with others working on related issues
- Share progress updates in issue comments
- Provide feedback on pull requests

## Getting Help

- **Questions**: Use GitHub Discussions for general questions
- **Issues**: Create specific issues for bugs or feature requests
- **Documentation**: Check README.md and issue templates for guidance
- **Live Demo**: Try the application at https://longball-analytics.netlify.app

## Recognition

Contributors will be recognized in the project README and release notes. Significant contributions may result in collaborator access to the repository.

Thank you for helping make LongBall Analytics better! 🚀
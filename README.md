# Stochastic Climate Disaster Cost Estimator

A comprehensive probabilistic model for estimating climate disaster costs using advanced Monte Carlo simulations with economic regime switching and dynamic disaster frequency modeling.

## Overview

This React-based application implements a sophisticated stochastic model that combines climate risk assessment with economic cycle modeling to provide robust estimates of climate disaster costs. The model incorporates complex interactions between economic downturns and disaster impacts, showing how recessions and financial crises can amplify climate risks.

## Key Features

### Advanced Stochastic Modeling
- **Monte Carlo Simulation**: Run 500-5,000 simulations for comprehensive uncertainty quantification
- **Economic Regime Switching**: 5-state Markov chain model (Expansion, Slowdown, Recession, Depression, Financial Crisis)
- **Dynamic Disaster Frequencies**: Gamma-distributed frequencies with climate oscillations and non-stationarity
- **Probabilistic Real Estate**: Regime-dependent property valuations with economic interactions
- **Cross-Correlation Effects**: Models disaster clustering during economic stress periods

### Economic-Climate Interactions
- **Regime-Dependent Impacts**: Different disaster damage multipliers based on economic conditions
- **Preparedness Feedback**: Reduced disaster preparedness funding during economic downturns
- **Recovery Constraints**: Credit market conditions affecting post-disaster recovery
- **Infrastructure Investment**: Counter-cyclical government spending patterns
- **Flight-to-Safety Effects**: Property market dynamics during crises

### Statistical Analysis
- **Risk Metrics**: Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) calculations
- **Confidence Intervals**: User-configurable confidence levels (90%, 95%, 99%)
- **Tail Risk Analysis**: Comprehensive extreme event modeling
- **Regime Statistics**: Economic indicator distributions and transition probabilities

## Technical Implementation

### Random Number Generation
- Seeded Linear Congruential Generator for reproducible results
- Box-Muller transform for normal distributions
- Marsaglia-Tsang method for gamma distributions
- Ensures consistent results across runs with same seed

### Model Components

#### Economic Regime Model
```javascript
// 5 economic regimes with distinct characteristics
const regimes = {
  expansion: { gdp_growth: 2.8%, unemployment: 4.5%, property_multiplier: 1.15 },
  slowdown: { gdp_growth: 0.8%, unemployment: 6.5%, property_multiplier: 0.85 },
  recession: { gdp_growth: -2.5%, unemployment: 9.5%, property_multiplier: 0.65 },
  depression: { gdp_growth: -8.0%, unemployment: 18%, property_multiplier: 0.4 },
  financial_crisis: { gdp_growth: -4.5%, unemployment: 12%, property_multiplier: 0.55 }
};
```

#### Disaster Frequency Model
- Base frequencies calibrated to historical data
- Stochastic trends with random walk components
- Cyclic patterns (AMO, ENSO, monsoons)
- Climate scenario multipliers (conservative, moderate, aggressive)

#### Real Estate Valuation Model
- Property value evolution with economic sensitivity
- Interest rate effects and credit market conditions
- Sector-specific impacts (residential, commercial, industrial, infrastructure)
- Market bubble cycles and corrections

## Regional Models

### Gulf of Mexico
- **Disasters**: Hurricanes, Flooding, Storm Surge, Heat Waves
- **Base Property Value**: $6.75B total (residential: $2.8B, commercial: $1.9B, industrial: $0.85B, infrastructure: $1.2B)
- **Characteristics**: Hurricane-dominated risk profile with significant storm surge vulnerability

### Southeast Asia
- **Disasters**: Typhoons, Flooding, Sea Level Rise, Landslides
- **Base Property Value**: $10.8B total (residential: $4.2B, commercial: $3.1B, industrial: $1.4B, infrastructure: $2.1B)
- **Characteristics**: Monsoon-driven flooding with high coastal vulnerability

## Usage

### Basic Configuration
1. **Region**: Select between Gulf of Mexico or Southeast Asia
2. **Climate Scenario**: Choose conservative, moderate, or aggressive projections
3. **Simulations**: Set number of Monte Carlo runs (500-5,000)
4. **Confidence Level**: Configure statistical confidence intervals
5. **Random Seed**: Set seed for reproducible results

### Interpreting Results

#### Summary Statistics
- **Mean 30-Year Cost**: Expected total cost over projection period
- **Mean Annual Cost**: Average yearly expected cost
- **Peak Year**: Year with highest expected annual cost
- **VaR (95%)**: Value-at-Risk showing potential losses at 95% confidence
- **Economic Metrics**: GDP growth, unemployment, inflation averages

#### Visualizations
- **Annual Expected Costs**: Time series with confidence bands
- **Risk Metrics**: VaR and CVaR evolution
- **Cumulative Costs**: Total cost accumulation with uncertainty
- **Disaster Breakdown**: Cost attribution by disaster type
- **Economic Impact**: Regime effects on property values

## Model Validation and Limitations

### Strengths
- Comprehensive uncertainty quantification through Monte Carlo simulation
- Realistic economic-climate interaction modeling
- Regime-switching captures business cycle effects
- Cross-correlation modeling prevents underestimation of clustered events
- Empirically calibrated parameters based on historical data

### Limitations
- Model parameters based on historical relationships that may change
- Climate change could alter fundamental disaster-economy interactions
- Adaptation responses not fully captured in current implementation
- Regional models may not capture all local risk factors
- Economic regime transitions simplified compared to real-world complexity

### Important Disclaimers
This model is for research and planning purposes. Results should be interpreted within the context of the assumptions and limitations outlined above. The model incorporates significant uncertainty and should not be used as the sole basis for financial or policy decisions without additional analysis and expert consultation.

## Technical Requirements

### Dependencies
- React 18+ with hooks support
- Recharts for data visualization
- Lucide React for icons
- Tailwind CSS for styling
- Modern browser with JavaScript ES6+ support

### Performance Notes
- 5,000 simulations may take 10-30 seconds depending on hardware
- Memory usage scales linearly with number of simulations
- Recommend starting with 1,000 simulations for initial analysis
- Browser storage APIs not supported (uses in-memory state only)

## Model Architecture

The application follows a modular architecture with separate functions for:
- Economic regime generation and transitions
- Dynamic disaster frequency modeling
- Real estate valuation with economic interactions
- Urban growth patterns affected by economic cycles
- Statistical analysis and risk metric calculations

Each component includes comprehensive documentation and parameter explanations to facilitate understanding and potential modifications.

## Future Enhancements

Potential areas for model improvement include:
- Integration of additional climate variables (temperature, precipitation)
- Enhanced adaptation modeling with investment feedback loops
- Spatial risk modeling for sub-regional analysis
- Integration with external economic forecasting models
- Machine learning-based parameter calibration
- Real-time data integration for dynamic model updating

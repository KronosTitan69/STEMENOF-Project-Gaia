import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, AreaChart, Area, ResponsiveContainer, ScatterChart, Scatter, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, MapPin, AlertTriangle, DollarSign, Calendar, BarChart3, Shuffle, Target } from 'lucide-react';

/**
 * Advanced Stochastic Climate Disaster Cost Estimator
 * 
 * This component implements a comprehensive probabilistic model for estimating climate disaster costs
 * using Monte Carlo simulations. The model incorporates:
 * - Economic regime switching (Markov chains)
 * - Dynamic disaster frequency modeling with climate oscillations
 * - Stochastic real estate valuations with economic interactions
 * - Urban growth patterns affected by economic cycles
 * - Cross-correlation effects and tail risk analysis
 * 
 * Key Features:
 * - Multiple economic regimes (expansion, slowdown, recession, depression, financial crisis)
 * - Gamma-distributed disaster frequencies with non-stationarity
 * - Regime-dependent volatility and correlation clustering
 * - Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) calculations
 * - Comprehensive uncertainty quantification
 */
const ClimateDisasterCostEstimator = () => {
  // ===== STATE MANAGEMENT =====
  // User-configurable parameters for the simulation
  const [selectedRegion, setSelectedRegion] = useState('gulf_mexico');
  const [selectedScenario, setSelectedScenario] = useState('moderate');
  const [showDetails, setShowDetails] = useState(false);
  const [numSimulations, setNumSimulations] = useState(1000);
  const [confidenceInterval, setConfidenceInterval] = useState(95);
  const [randomSeed, setRandomSeed] = useState(12345);

  // ===== RANDOM NUMBER GENERATION =====
  /**
   * Seeded random number generator for reproducible Monte Carlo simulations
   * Uses a linear congruential generator (LCG) algorithm
   * @param {number} seed - Initial seed value for reproducibility
   * @returns {Function} Random number generator function [0, 1)
   */
  const seededRandom = (seed) => {
    let state = seed;
    return () => {
      // LCG formula: (a * seed + c) mod m
      state = (state * 9301 + 49297) % 233280;
      return state / 233280;
    };
  };

  /**
   * Box-Muller transform for generating normally distributed random variables
   * Converts uniform random variables to standard normal distribution
   * @param {number} mean - Mean of the normal distribution
   * @param {number} stdDev - Standard deviation of the normal distribution
   * @param {Function} rng - Random number generator function
   * @returns {number} Normally distributed random value
   */
  const normalRandom = (mean, stdDev, rng) => {
    const u1 = rng();
    const u2 = rng();
    // Box-Muller transformation
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0 * stdDev + mean;
  };

  /**
   * Gamma distribution random number generator
   * Used for modeling disaster frequencies (non-negative, right-skewed distributions)
   * Implements Marsaglia and Tsang's method for shape >= 1
   * @param {number} shape - Shape parameter (k) of gamma distribution
   * @param {number} scale - Scale parameter (θ) of gamma distribution
   * @param {Function} rng - Random number generator function
   * @returns {number} Gamma-distributed random value
   */
  const gammaRandom = (shape, scale, rng) => {
    // Handle case where shape < 1 using transformation
    if (shape < 1) {
      const u = rng();
      return gammaRandom(1 + shape, scale, rng) * Math.pow(u, 1 / shape);
    }
    
    // Marsaglia and Tsang's method for shape >= 1
    const d = shape - 1/3;
    const c = 1 / Math.sqrt(9 * d);
    
    let z, u, v;
    do {
      do {
        z = normalRandom(0, 1, rng);
        v = 1 + c * z;
      } while (v <= 0);
      
      v = v * v * v;
      u = rng();
    } while (u >= 1 - 0.0331 * z * z * z * z && Math.log(u) >= 0.5 * z * z + d * (1 - v + Math.log(v)));
    
    return d * v * scale;
  };

  // ===== DISASTER FREQUENCY MODELING =====
  /**
   * Dynamic disaster frequency model incorporating climate patterns and stochastic trends
   * 
   * Features:
   * - Base disaster frequencies calibrated to historical data
   * - Stochastic trends with random walk components
   * - Cyclic patterns representing climate oscillations (AMO, ENSO, etc.)
   * - Regime-dependent volatility adjustments
   * - Gamma-distributed frequencies to ensure non-negativity
   * 
   * @param {string} region - Geographic region (gulf_mexico, southeast_asia)
   * @param {number} year - Projection year (2024-2054)
   * @param {Function} rng - Seeded random number generator
   * @returns {Object} Disaster frequencies by type for the given year
   */
  const getDynamicDisasterModel = (region, year, rng) => {
    const yearIndex = year - 2024;
    
    // Regional disaster models with calibrated parameters
    const baseModels = {
      gulf_mexico: {
        hurricanes: { 
          baseMean: 6.2,              // Historical average annual frequency
          trendMean: 0.08,            // Expected annual increase due to climate change
          cyclicAmplitude: 1.5,       // Amplitude of natural cycles
          cyclicPeriod: 11,           // Atlantic Multidecadal Oscillation period
          volatility: 1.2,            // Year-to-year variability
          shapeParam: 4.2             // Gamma distribution shape (affects skewness)
        },
        flooding: { 
          baseMean: 12.4, 
          trendMean: 0.15,            // Higher trend due to urbanization + climate change
          cyclicAmplitude: 2.1, 
          cyclicPeriod: 7,            // Regional precipitation cycles
          volatility: 2.1,
          shapeParam: 6.8 
        },
        storm_surge: { 
          baseMean: 3.8, 
          trendMean: 0.12,            // Sea level rise contribution
          cyclicAmplitude: 0.8, 
          cyclicPeriod: 9,            // Coastal upwelling cycles
          volatility: 0.8,
          shapeParam: 3.1 
        },
        heat_waves: { 
          baseMean: 8.5, 
          trendMean: 0.18,            // Strong warming trend
          cyclicAmplitude: 1.5, 
          cyclicPeriod: 5,            // Urban heat island cycles
          volatility: 1.5,
          shapeParam: 5.2 
        }
      },
      southeast_asia: {
        typhoons: { 
          baseMean: 8.7, 
          trendMean: 0.10, 
          cyclicAmplitude: 1.8, 
          cyclicPeriod: 6,            // ENSO (El Niño-Southern Oscillation) cycle
          volatility: 1.4,
          shapeParam: 4.8 
        },
        flooding: { 
          baseMean: 15.2,             // Higher base due to monsoon patterns
          trendMean: 0.20, 
          cyclicAmplitude: 3.2, 
          cyclicPeriod: 4,            // Monsoon variability
          volatility: 2.8,
          shapeParam: 7.2 
        },
        sea_level_rise: { 
          baseMean: 2.1, 
          trendMean: 0.25,            // High vulnerability to SLR
          cyclicAmplitude: 0.3, 
          cyclicPeriod: 12,           // Pacific Decadal Oscillation
          volatility: 0.3,
          shapeParam: 2.8 
        },
        landslides: { 
          baseMean: 4.3, 
          trendMean: 0.05,            // Slower trend, mainly precipitation-driven
          cyclicAmplitude: 1.1, 
          cyclicPeriod: 8, 
          volatility: 0.9,
          shapeParam: 3.5 
        }
      }
    };

    const models = baseModels[region];
    const dynamicModel = {};

    // Generate stochastic frequencies for each disaster type
    Object.entries(models).forEach(([disasterType, params]) => {
      // Climate oscillation component (sinusoidal cycles)
      const cyclicComponent = params.cyclicAmplitude * Math.sin(2 * Math.PI * yearIndex / params.cyclicPeriod);
      
      // Stochastic trend with random walk innovation
      const trendVariation = normalRandom(0, 0.02, rng);
      const dynamicTrend = params.trendMean + trendVariation;
      
      // Compose base frequency with all components
      const meanFrequency = params.baseMean + 
                           (dynamicTrend * yearIndex) + 
                           cyclicComponent;
      
      // Generate from Gamma distribution to ensure non-negative frequencies
      const scale = params.volatility / Math.sqrt(params.shapeParam);
      const shape = params.shapeParam;
      
      // Ensure minimum frequency of 0 (no negative disasters)
      dynamicModel[disasterType] = Math.max(0, gammaRandom(shape, scale * meanFrequency / shape, rng));
    });

    return dynamicModel;
  };

  // ===== URBAN GROWTH MODELING =====
  /**
   * Dynamic urban growth model with economic regime interactions
   * 
   * Models how economic conditions affect:
   * - Population growth patterns
   * - Infrastructure development rates
   * - Coastal development (increasing exposure)
   * - Vulnerability accumulation over time
   * - Government policy responses during different economic regimes
   * 
   * @param {string} region - Geographic region
   * @param {number} year - Projection year
   * @param {Function} rng - Random number generator
   * @param {Object} economicState - Current economic regime and indicators
   * @returns {Object} Growth rates for different urban development aspects
   */
  const getDynamicUrbanGrowth = (region, year, rng, economicState) => {
    const yearIndex = year - 2024;
    
    // Regional urban growth models with economic sensitivity parameters
    const baseModels = {
      gulf_mexico: {
        population_growth: { 
          mean: 0.012,                // Base annual population growth rate
          volatility: 0.003,          // Volatility around the mean
          cyclePeriod: 20             // Demographic transition cycles
        },
        infrastructure_value_growth: { 
          mean: 0.035,                // Infrastructure investment rate
          volatility: 0.008, 
          cyclePeriod: 15             // Political/budget cycles
        },
        coastal_development_rate: { 
          mean: 0.028,                // Rate of coastal zone development
          volatility: 0.006, 
          cyclePeriod: 12             // Real estate cycles
        },
        vulnerability_increase: { 
          mean: 0.015,                // Rate of vulnerability accumulation
          volatility: 0.004, 
          cyclePeriod: 25             // Long-term planning cycles
        },
        // Economic sensitivity coefficients
        economic_sensitivity: { 
          gdp: 0.8,                   // GDP growth coefficient
          unemployment: -0.5          // Unemployment impact coefficient (negative)
        }
      },
      southeast_asia: {
        population_growth: { 
          mean: 0.018,                // Higher growth in developing regions
          volatility: 0.004, 
          cyclePeriod: 18 
        },
        infrastructure_value_growth: { 
          mean: 0.045,                // Rapid infrastructure development
          volatility: 0.012, 
          cyclePeriod: 10             // Shorter political cycles
        },
        coastal_development_rate: { 
          mean: 0.038,                // Intensive coastal development
          volatility: 0.009, 
          cyclePeriod: 8 
        },
        vulnerability_increase: { 
          mean: 0.022,                // Higher vulnerability accumulation
          volatility: 0.006, 
          cyclePeriod: 22 
        },
        economic_sensitivity: { 
          gdp: 1.0,                   // Higher economic sensitivity
          unemployment: -0.7 
        }
      }
    };

    const model = baseModels[region];
    const dynamicGrowth = {};

    // Calculate growth rates for each urban development aspect
    Object.entries(model).forEach(([growthType, params]) => {
      if (growthType === 'economic_sensitivity') return;
      
      // Economic cycle component (business cycle effects)
      const cycleComponent = 0.1 * params.mean * Math.sin(2 * Math.PI * yearIndex / params.cyclePeriod);
      
      // Economic regime impacts on growth
      const gdpImpact = economicState.gdpGrowth * model.economic_sensitivity.gdp;
      const unemploymentImpact = (economicState.unemployment - 0.05) * model.economic_sensitivity.unemployment;
      
      // Regime-specific effects
      const regimeImpact = economicState.regime === 'expansion' ? 0.002 :
                          economicState.regime === 'recession' ? -0.008 :
                          economicState.regime === 'depression' ? -0.015 :
                          economicState.regime === 'financial_crisis' ? -0.012 : -0.003;
      
      // Counter-cyclical government spending (infrastructure stimulus during downturns)
      const govSpendingEffect = economicState.regime === 'recession' && growthType === 'infrastructure_value_growth' ? 0.01 :
                               economicState.regime === 'depression' && growthType === 'infrastructure_value_growth' ? 0.015 : 0;
      
      // Random economic shocks (higher probability during downturns)
      const shockProbability = economicState.regime === 'expansion' ? 0.03 : 0.08;
      const shockMagnitude = rng() < shockProbability ? normalRandom(0, 0.01, rng) : 0;
      
      // Regime-dependent volatility (higher during crises)
      const crisisMultiplier = ['recession', 'depression', 'financial_crisis'].includes(economicState.regime) ? 1.5 : 1.0;
      const innovation = normalRandom(0, params.volatility * crisisMultiplier, rng);
      
      // Combine all effects with floor constraint
      dynamicGrowth[growthType] = Math.max(-0.08, 
        params.mean + cycleComponent + gdpImpact + unemploymentImpact + 
        regimeImpact + govSpendingEffect + shockMagnitude + innovation);
    });

    return dynamicGrowth;
  };

  // Track economic regime states (not React state to avoid unnecessary re-renders)
  const economicStates = {};

  // ===== ECONOMIC REGIME MODELING =====
  /**
   * Economic regime state generator using Markov chain transitions
   * 
   * Implements a hidden Markov model with 5 economic regimes:
   * - Expansion: Strong growth, low unemployment, rising property values
   * - Slowdown: Moderate growth, rising unemployment, cooling markets  
   * - Recession: Negative growth, high unemployment, declining property values
   * - Depression: Severe contraction, very high unemployment, major market declines
   * - Financial Crisis: Credit crunch, property market collapse, high volatility
   * 
   * Each regime has different:
   * - Duration distributions
   * - Economic indicator distributions (GDP, unemployment, inflation)
   * - Property value multipliers
   * - Disaster preparedness levels
   * - Transition probabilities to other regimes
   * 
   * @param {number} year - Current year
   * @param {Function} rng - Random number generator
   * @param {Object|null} previousState - Previous year's economic state
   * @returns {Object} Current economic regime with all indicators
   */
  const getEconomicRegime = (year, rng, previousState = null) => {
    const yearIndex = year - 2024;
    
    // Economic regime parameter definitions based on historical business cycle data
    const regimes = {
      expansion: { 
        duration_mean: 8.5,         // Average duration in years
        duration_std: 3.2, 
        gdp_growth: { mean: 0.028, std: 0.015 },        // Annual GDP growth rate
        unemployment: { mean: 0.045, std: 0.008 },       // Unemployment rate
        inflation: { mean: 0.025, std: 0.012 },          // Inflation rate
        property_multiplier: 1.15,                       // Property value multiplier
        disaster_preparedness: 1.1                       // Preparedness funding multiplier
      },
      slowdown: { 
        duration_mean: 2.8, 
        duration_std: 1.1,
        gdp_growth: { mean: 0.008, std: 0.020 },
        unemployment: { mean: 0.065, std: 0.012 },
        inflation: { mean: 0.018, std: 0.008 },
        property_multiplier: 0.85,
        disaster_preparedness: 0.95
      },
      recession: { 
        duration_mean: 1.8, 
        duration_std: 0.8,
        gdp_growth: { mean: -0.025, std: 0.035 },       // Negative growth
        unemployment: { mean: 0.095, std: 0.018 },
        inflation: { mean: 0.012, std: 0.015 },
        property_multiplier: 0.65,                      // Significant property decline
        disaster_preparedness: 0.8                      // Reduced preparedness spending
      },
      depression: { 
        duration_mean: 4.2,         // Longer duration for severe downturns
        duration_std: 2.1,
        gdp_growth: { mean: -0.08, std: 0.045 },        // Severe economic contraction
        unemployment: { mean: 0.18, std: 0.025 },       // Very high unemployment
        inflation: { mean: -0.005, std: 0.020 },        // Deflationary pressures
        property_multiplier: 0.4,                       // Major property market collapse
        disaster_preparedness: 0.6                      // Severely constrained preparedness
      },
      financial_crisis: {
        duration_mean: 2.5, 
        duration_std: 1.0,
        gdp_growth: { mean: -0.045, std: 0.025 },
        unemployment: { mean: 0.12, std: 0.020 },
        inflation: { mean: 0.008, std: 0.018 },
        property_multiplier: 0.55,                      // Credit-driven property decline
        disaster_preparedness: 0.75
      }
    };

    // Markov chain transition probability matrix
    // Rows = current state, columns = next state
    const transitionMatrix = {
      expansion: { 
        expansion: 0.85,            // High persistence during good times
        slowdown: 0.10, 
        recession: 0.04, 
        depression: 0.005, 
        financial_crisis: 0.005 
      },
      slowdown: { 
        expansion: 0.35,            // Moderate recovery probability
        slowdown: 0.40,             // Persistence in intermediate state
        recession: 0.20, 
        depression: 0.03, 
        financial_crisis: 0.02 
      },
      recession: { 
        expansion: 0.15, 
        slowdown: 0.25, 
        recession: 0.45,            // High persistence during downturns
        depression: 0.10,           // Risk of deeper contraction
        financial_crisis: 0.05 
      },
      depression: { 
        expansion: 0.05,            // Low probability of rapid recovery
        slowdown: 0.15, 
        recession: 0.25, 
        depression: 0.50,           // Very high persistence
        financial_crisis: 0.05 
      },
      financial_crisis: { 
        expansion: 0.10, 
        slowdown: 0.20, 
        recession: 0.35, 
        depression: 0.15,           // High risk of deeper crisis
        financial_crisis: 0.20      // Moderate persistence
      }
    };

    // Initialize or carry forward regime state
    let currentRegime = previousState?.regime || 'expansion';
    let yearsInRegime = previousState?.yearsInRegime || 0;
    
    // Evaluate regime transition based on Markov chain
    const random = rng();
    const transitions = transitionMatrix[currentRegime];
    let cumProb = 0;
    
    // Sample next regime from transition probabilities
    for (const [regime, prob] of Object.entries(transitions)) {
      cumProb += prob;
      if (random <= cumProb && yearsInRegime >= 1) {  // Minimum 1 year in each regime
        if (regime !== currentRegime) {
          currentRegime = regime;
          yearsInRegime = 0;
        }
        break;
      }
    }
    
    yearsInRegime++;
    const regimeData = regimes[currentRegime];
    
    // Generate economic indicators from regime-specific distributions
    const gdpGrowth = normalRandom(regimeData.gdp_growth.mean, regimeData.gdp_growth.std, rng);
    const unemployment = Math.max(0.02, normalRandom(regimeData.unemployment.mean, regimeData.unemployment.std, rng));
    const inflation = normalRandom(regimeData.inflation.mean, regimeData.inflation.std, rng);
    
    // Model external shocks (black swan events)
    // Higher probability during non-expansion regimes
    const shockProbability = currentRegime === 'expansion' ? 0.01 : 0.03;
    const externalShock = rng() < shockProbability ? normalRandom(-0.05, 0.02, rng) : 0;
    
    return {
      regime: currentRegime,
      yearsInRegime,
      gdpGrowth: gdpGrowth + externalShock,           // Include shock effects
      unemployment,
      inflation,
      propertyMultiplier: regimeData.property_multiplier,
      disasterPreparedness: regimeData.disaster_preparedness,
      isShockYear: Math.abs(externalShock) > 0.01     // Flag for significant shocks
    };
  };

  // ===== REAL ESTATE VALUATION MODEL =====
  /**
   * Dynamic real estate pricing model with comprehensive economic interactions
   * 
   * Models property values across different sectors with:
   * - Base appreciation rates by property type
   * - Economic regime impacts on valuations
   * - Market bubble cycles and corrections
   * - Interest rate sensitivity
   * - Flight-to-safety effects during crises
   * - Credit market condition impacts
   * - Regime-dependent volatility
   * 
   * @param {string} region - Geographic region
   * @param {number} year - Current year
   * @param {Function} rng - Random number generator
   * @param {Object} economicState - Current economic regime and conditions
   * @returns {Object} Property values and appreciation rates by sector
   */
  const getDynamicRealEstate = (region, year, rng, economicState) => {
    const yearIndex = year - 2024;
    
    // Regional real estate models with sector breakdowns (values in $/sq ft equivalent)
    const baseModels = {
      gulf_mexico: {
        residential_value: 2800,            // Single-family and multi-family
        commercial_value: 1900,             // Office, retail, hospitality
        industrial_value: 850,              // Manufacturing, logistics, energy
        infrastructure_value: 1200,         // Transportation, utilities, public
        base_appreciation: { 
          mean: 0.025,                      // Historical appreciation rate
          volatility: 0.015,                // Market volatility
          bubblePeriod: 18                  // Property cycle length
        },
        // Economic sensitivity coefficients for different indicators
        economic_sensitivity: { 
          gdp: 1.8,                         // GDP sensitivity multiplier
          unemployment: -1.2,               // Unemployment impact (negative)
          inflation: -0.8                   // Inflation impact (negative, real values)
        }
      },
      southeast_asia: {
        residential_value: 4200,            // Higher values in dense urban areas
        commercial_value: 3100, 
        industrial_value: 1400, 
        infrastructure_value: 2100, 
        base_appreciation: { 
          mean: 0.035,                      // Higher growth in developing markets
          volatility: 0.022,                // Higher volatility
          bubblePeriod: 12                  // Shorter cycles
        },
        economic_sensitivity: { 
          gdp: 2.1,                         // Higher economic sensitivity
          unemployment: -1.5, 
          inflation: -1.0 
        }
      }
    };

    const model = baseModels[region];
    
    // Property market bubble/cycle component
    const bubblePhase = (yearIndex % model.base_appreciation.bubblePeriod) / model.base_appreciation.bubblePeriod;
    const bubbleComponent = 0.01 * Math.sin(2 * Math.PI * bubblePhase);
    
    // Economic regime direct impact on property markets
    const regimeImpact = (economicState.propertyMultiplier - 1) * 0.3;
    
    // Economic indicator impacts (scaled to reasonable magnitudes)
    const gdpImpact = economicState.gdpGrowth * model.economic_sensitivity.gdp * 0.1;
    const unemploymentImpact = (economicState.unemployment - 0.05) * model.economic_sensitivity.unemployment * 0.1;
    const inflationImpact = (economicState.inflation - 0.02) * model.economic_sensitivity.inflation * 0.1;
    
    // Credit market tightness by regime (affects borrowing capacity)
    const creditTightness = economicState.regime === 'expansion' ? 0 : 
                           economicState.regime === 'recession' ? 0.02 :
                           economicState.regime === 'depression' ? 0.05 :
                           economicState.regime === 'financial_crisis' ? 0.08 : 0.01;
    
    // Regime-dependent market volatility multiplier
    const regimeVolatility = economicState.regime === 'expansion' ? 1.0 :
                            economicState.regime === 'slowdown' ? 1.3 :
                            economicState.regime === 'recession' ? 1.8 :
                            economicState.regime === 'depression' ? 2.5 :
                            economicState.regime === 'financial_crisis' ? 2.2 : 1.0;
    
    // Random market shock with regime-dependent volatility
    const marketShock = normalRandom(0, model.base_appreciation.volatility * regimeVolatility, rng);
    
    // Interest rate effects (regime-dependent base rates)
    const baseInterestRate = economicState.regime === 'expansion' ? 0.04 :
                            economicState.regime === 'recession' ? 0.015 :
                            economicState.regime === 'depression' ? 0.005 : 0.025;
    const interestRateEffect = -2.5 * (baseInterestRate - 0.025);  // Negative relationship
    
    // Flight-to-safety effects during crises (affects different property types differently)
    const flightToSafety = ['recession', 'depression', 'financial_crisis'].includes(economicState.regime) ? 
                          { 
                            residential: 0.95,      // Slight preference for residential
                            commercial: 0.75,       // Commercial most affected
                            industrial: 0.85,       // Industrial moderately affected
                            infrastructure: 1.05    // Infrastructure benefits (safe haven)
                          } :
                          { residential: 1.0, commercial: 1.0, industrial: 1.0, infrastructure: 1.0 };
    
    // Aggregate all economic impacts
    const totalEconomicImpact = regimeImpact + gdpImpact + unemploymentImpact + 
                               inflationImpact - creditTightness + interestRateEffect;
    
    // Calculate final appreciation rate with floor constraint
    const dynamicAppreciation = Math.max(-0.25, 
      model.base_appreciation.mean + bubbleComponent + marketShock + totalEconomicImpact);

    return {
      totalValue: model.residential_value + model.commercial_value + 
                  model.industrial_value + model.infrastructure_value,
      appreciation: dynamicAppreciation,
      economicImpact: totalEconomicImpact,
      breakdown: {
        residential: model.residential_value * flightToSafety.residential,
        commercial: model.commercial_value * flightToSafety.commercial,
        industrial: model.industrial_value * flightToSafety.industrial,
        infrastructure: model.infrastructure_value * flightToSafety.infrastructure
      }
    };
  };

  // ===== MONTE CARLO SIMULATION ENGINE =====
  /**
   * Main Monte Carlo simulation engine
   * 
   * Runs the complete stochastic model with:
   * - Multiple simulation paths for uncertainty quantification
   * - Economic regime evolution using Markov chains
   * - Dynamic disaster frequency generation
   * - Property value evolution with economic interactions
   * - Cross-correlation effects between disasters and economics
   * - Adaptation and preparedness feedbacks
   * 
   * The simulation generates comprehensive results including:
   * - Expected annual and cumulative costs
   * - Property value trajectories
   * - Economic regime paths
   * - Disaster-specific cost breakdowns
   * - Risk metrics for each simulation year
   */
  const runMonteCarloSimulation = useMemo(() => {
    // Simulation time horizon (30 years)
    const years = Array.from({ length: 30 }, (_, i) => 2024 + i);
    
    // Climate scenario multipliers for sensitivity analysis
    const scenarios = {
      conservative: { 
        climateMultiplier: 0.8,         // Slower climate change progression
        economicMultiplier: 0.9,        // Lower economic volatility
        adaptationFactor: 0.85          // Conservative adaptation investment
      },
      moderate: { 
        climateMultiplier: 1.0,         // Baseline climate projections
        economicMultiplier: 1.0,        // Baseline economic volatility
        adaptationFactor: 1.0           // Baseline adaptation capacity
      },
      aggressive: { 
        climateMultiplier: 1.3,         // Accelerated climate change
        economicMultiplier: 1.1,        // Higher economic volatility
        adaptationFactor: 1.15          // Enhanced adaptation investment
      }
    };

    const scenario = scenarios[selectedScenario];
    const simulations = [];

    // ===== MAIN SIMULATION LOOP =====
    // Execute Monte Carlo simulations across all specified runs
    for (let sim = 0; sim < numSimulations; sim++) {
      // Create seeded random number generator for this simulation path
      const rng = seededRandom(randomSeed + sim);
      const simulationData = [];
      
      // Initialize simulation state variables
      let cumulativeCost = 0;
      let previousPropertyValue = getDynamicRealEstate(selectedRegion, 2024, rng, 
        getEconomicRegime(2024, rng)).totalValue;
      let economicState = null;

      // ===== TEMPORAL LOOP FOR EACH SIMULATION =====
      // Project forward through each year in the time horizon
      years.forEach((year, yearIdx) => {
        // ===== ECONOMIC REGIME EVOLUTION =====
        // Update economic state using Markov chain transitions
        economicState = getEconomicRegime(year, rng, economicState);
        
        // ===== STOCHASTIC MODEL OUTPUTS =====
        // Generate disaster frequencies with climate and economic interactions
        const disasters = getDynamicDisasterModel(selectedRegion, year, rng);
        // Model urban growth patterns affected by economic conditions
        const urbanGrowth = getDynamicUrbanGrowth(selectedRegion, year, rng, economicState);
        // Calculate property values with economic regime effects
        const realEstate = getDynamicRealEstate(selectedRegion, year, rng, economicState);

        // ===== PROPERTY VALUE EVOLUTION =====
        // Update total property value incorporating economic multipliers
        const propertyGrowthRate = realEstate.appreciation * scenario.economicMultiplier;
        const currentPropertyValue = previousPropertyValue * (1 + propertyGrowthRate);
        
        // ===== VULNERABILITY CALCULATION =====
        // Calculate adaptive vulnerability with economic preparedness effects
        const baseVulnerability = 1 + (urbanGrowth.vulnerability_increase * yearIdx);
        const economicPreparedness = economicState.disasterPreparedness * scenario.adaptationFactor;
        const adaptedVulnerability = baseVulnerability / economicPreparedness;
        
        // ===== ECONOMIC DAMAGE MULTIPLIERS =====
        // Economic regime affects disaster damage severity and recovery capacity
        const economicDamageMultiplier = economicState.regime === 'expansion' ? 0.9 :
                                       economicState.regime === 'recession' ? 1.2 :
                                       economicState.regime === 'depression' ? 1.5 :
                                       economicState.regime === 'financial_crisis' ? 1.4 : 1.1;
        
        // ===== DAMAGE RATE CALIBRATION =====
        // Regional damage rates by disaster type (as fraction of property value)
        const damageRates = {
          gulf_mexico: { 
            hurricanes: 0.045,          // 4.5% of property value per hurricane
            flooding: 0.025,            // 2.5% per flooding event
            storm_surge: 0.035,         // 3.5% per storm surge
            heat_waves: 0.015           // 1.5% per heat wave
          },
          southeast_asia: { 
            typhoons: 0.052,            // Higher damage rates due to exposure
            flooding: 0.032, 
            sea_level_rise: 0.028, 
            landslides: 0.038 
          }
        };

        // ===== DISASTER COST CALCULATION =====
        // Calculate expected cost with all interactions and correlations
        let totalExpectedCost = 0;
        const disasterCosts = {};

        Object.entries(disasters).forEach(([disasterType, frequency]) => {
          // Apply climate scenario multiplier to disaster frequency
          const adjustedFrequency = frequency * scenario.climateMultiplier;
          const baseDamageRate = damageRates[selectedRegion][disasterType];
          
          // Adjust damage rates for economic conditions
          const economicAdjustedDamageRate = baseDamageRate * economicDamageMultiplier;
          
          // ===== CORRELATION EFFECTS =====
          // Model disaster clustering and correlation, especially during economic stress
          const stressCorrelationFactor = economicState.isShockYear ? 1.3 : 1.0;
          const correlationFactor = (1 + 0.1 * normalRandom(0, 1, rng)) * stressCorrelationFactor;
          
          // ===== FINAL COST CALCULATION =====
          // Combine all effects: frequency × property value × damage rate × vulnerability × correlations
          const disasterCost = adjustedFrequency * currentPropertyValue * economicAdjustedDamageRate * 
                              adaptedVulnerability * correlationFactor / 1000; // Convert to billions
          
          // Ensure non-negative costs
          disasterCosts[disasterType] = Math.max(0, disasterCost);
          totalExpectedCost += disasterCosts[disasterType];
        });

        // Update cumulative cost tracker
        cumulativeCost += totalExpectedCost;

        // ===== STORE SIMULATION RESULTS =====
        // Comprehensive data collection for each simulation year
        simulationData.push({
          year,
          expectedCost: totalExpectedCost,                          // Annual expected cost
          cumulativeCost,                                           // Cumulative cost to date
          totalPropertyValue: currentPropertyValue / 1000,         // Property value (billions)
          disasterFrequency: Object.values(disasters).reduce((sum, freq) => sum + freq, 0), // Total disaster count
          vulnerabilityFactor: adaptedVulnerability,               // Vulnerability multiplier
          economicRegime: economicState.regime,                    // Current economic regime
          gdpGrowth: economicState.gdpGrowth,                      // GDP growth rate
          unemployment: economicState.unemployment,                // Unemployment rate
          inflation: economicState.inflation,                      // Inflation rate
          economicImpact: realEstate.economicImpact || 0,         // Economic impact on property
          isShockYear: economicState.isShockYear,                 // External shock indicator
          ...disasterCosts                                         // Individual disaster costs
        });

        // Update property value for next iteration
        previousPropertyValue = currentPropertyValue;
      });

      // Store complete simulation path
      simulations.push(simulationData);
    }

    return simulations;
  }, [selectedRegion, selectedScenario, numSimulations, randomSeed]);

  // ===== STATISTICAL ANALYSIS =====
  /**
   * Calculate comprehensive statistics from Monte Carlo results
   * 
   * Generates statistical summaries including:
   * - Central tendency measures (mean, median)
   * - Dispersion measures (standard deviation, percentiles)
   * - Risk metrics (Value-at-Risk, Conditional Value-at-Risk)
   * - Economic regime statistics
   * - Disaster-specific breakdowns
   * - Confidence intervals based on user selection
   */
  const statisticalResults = useMemo(() => {
    const years = Array.from({ length: 30 }, (_, i) => 2024 + i);
    const confidence = confidenceInterval / 100;
    const lowerPercentile = (1 - confidence) / 2;
    const upperPercentile = confidence + lowerPercentile;

    return years.map((year, yearIdx) => {
      // Extract data for this year across all simulations
      const yearData = runMonteCarloSimulation.map(sim => sim[yearIdx]);
      
      // ===== SORT DATA FOR PERCENTILE CALCULATIONS =====
      const sortedCosts = yearData.map(d => d.expectedCost).sort((a, b) => a - b);
      const sortedCumulative = yearData.map(d => d.cumulativeCost).sort((a, b) => a - b);
      const sortedProperty = yearData.map(d => d.totalPropertyValue).sort((a, b) => a - b);
      
      // ===== CENTRAL TENDENCY AND DISPERSION =====
      const mean = sortedCosts.reduce((sum, val) => sum + val, 0) / sortedCosts.length;
      const variance = sortedCosts.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / sortedCosts.length;
      const stdDev = Math.sqrt(variance);
      
      // ===== RISK METRICS =====
      // Value at Risk (VaR): Loss level not exceeded with given confidence level
      const var95 = sortedCosts[Math.floor(0.95 * sortedCosts.length)];
      // Conditional Value at Risk (CVaR): Expected loss given that VaR is exceeded
      const cvar95 = sortedCosts.slice(Math.floor(0.95 * sortedCosts.length))
                                .reduce((sum, val) => sum + val, 0) / 
                     sortedCosts.slice(Math.floor(0.95 * sortedCosts.length)).length;

      return {
        year,
        // ===== COST STATISTICS =====
        mean,
        median: sortedCosts[Math.floor(0.5 * sortedCosts.length)],
        stdDev,
        lowerBound: sortedCosts[Math.floor(lowerPercentile * sortedCosts.length)],
        upperBound: sortedCosts[Math.floor(upperPercentile * sortedCosts.length)],
        var95,
        cvar95,
        // ===== CUMULATIVE COST STATISTICS =====
        cumulativeMean: sortedCumulative.reduce((sum, val) => sum + val, 0) / sortedCumulative.length,
        cumulativeLower: sortedCumulative[Math.floor(lowerPercentile * sortedCumulative.length)],
        cumulativeUpper: sortedCumulative[Math.floor(upperPercentile * sortedCumulative.length)],
        
        // ===== ECONOMIC REGIME STATISTICS =====
        // Count frequency of each economic regime in this year
        economicRegimeCounts: yearData.reduce((acc, d) => {
          acc[d.economicRegime] = (acc[d.economicRegime] || 0) + 1;
          return acc;
        }, {}),
        avgGdpGrowth: yearData.reduce((sum, d) => sum + d.gdpGrowth, 0) / yearData.length,
        avgUnemployment: yearData.reduce((sum, d) => sum + d.unemployment, 0) / yearData.length,
        avgInflation: yearData.reduce((sum, d) => sum + d.inflation, 0) / yearData.length,
        economicShockProbability: yearData.filter(d => d.isShockYear).length / yearData.length,
        avgEconomicImpact: yearData.reduce((sum, d) => sum + d.economicImpact, 0) / yearData.length,
        propertyValueMean: sortedProperty.reduce((sum, val) => sum + val, 0) / sortedProperty.length,
        
        // ===== DISASTER-SPECIFIC STATISTICS =====
        // Average annual costs by disaster type
        hurricanes: yearData.map(d => d.hurricanes || 0).reduce((sum, val) => sum + val, 0) / yearData.length,
        typhoons: yearData.map(d => d.typhoons || 0).reduce((sum, val) => sum + val, 0) / yearData.length,
        flooding: yearData.map(d => d.flooding || 0).reduce((sum, val) => sum + val, 0) / yearData.length,
        storm_surge: yearData.map(d => d.storm_surge || 0).reduce((sum, val) => sum + val, 0) / yearData.length,
        sea_level_rise: yearData.map(d => d.sea_level_rise || 0).reduce((sum, val) => sum + val, 0) / yearData.length,
        heat_waves: yearData.map(d => d.heat_waves || 0).reduce((sum, val) => sum + val, 0) / yearData.length,
        landslides: yearData.map(d => d.landslides || 0).reduce((sum, val) => sum + val, 0) / yearData.length
      };
    });
  }, [runMonteCarloSimulation, confidenceInterval]);

  // ===== SUMMARY STATISTICS =====
  /**
   * High-level summary statistics for the entire simulation
   * Provides key metrics for quick assessment and comparison
   */
  const summaryStats = useMemo(() => {
    const totalMeanCost = statisticalResults.reduce((sum, data) => sum + data.mean, 0);
    const totalUpperBound = statisticalResults.reduce((sum, data) => sum + data.upperBound, 0);
    const avgAnnualCost = totalMeanCost / 30;
    const peakYear = statisticalResults.reduce((max, data) => data.mean > max.mean ? data : max);
    const finalCumulativeStats = statisticalResults[statisticalResults.length - 1];
    
    return {
      totalMeanCost: totalMeanCost.toFixed(1),
      totalUpperBound: totalUpperBound.toFixed(1),
      avgAnnualCost: avgAnnualCost.toFixed(1),
      peakYear: peakYear.year,
      peakCost: peakYear.mean.toFixed(1),
      peakCostUpper: peakYear.upperBound.toFixed(1),
      cumulativeVar95: finalCumulativeStats.cumulativeUpper.toFixed(1),
      cumulativeCVar: (finalCumulativeStats.cumulativeUpper * 1.2).toFixed(1) // Approximation
    };
  }, [statisticalResults]);

  // ===== DISASTER TYPE VISUALIZATION DATA =====
  /**
   * Prepare aggregated disaster type data for pie chart visualization
   * Calculates average annual costs by disaster type across all simulation years
   */
  const disasterTypeData = useMemo(() => {
    const avgData = statisticalResults.reduce((acc, data) => {
      // Select disaster types based on region
      const types = selectedRegion === 'gulf_mexico' 
        ? ['hurricanes', 'flooding', 'storm_surge', 'heat_waves']
        : ['typhoons', 'flooding', 'sea_level_rise', 'landslides'];
      
      // Accumulate costs across years
      types.forEach(type => {
        acc[type] = (acc[type] || 0) + (data[type] || 0);
      });
      return acc;
    }, {});
    
    // Convert to visualization format with averaged values
    return Object.entries(avgData).map(([name, value]) => ({
      name: name.replace('_', ' ').toUpperCase(),
      value: (value / 30).toFixed(2)
    }));
  }, [statisticalResults, selectedRegion]);

  // Color palette for charts
  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0'];

  // ===== REACT COMPONENT RENDER =====
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* ===== HEADER SECTION ===== */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4 flex items-center justify-center gap-3">
            <AlertTriangle className="text-orange-500" />
            Stochastic Climate Disaster Cost Model
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Advanced probabilistic modeling with Monte Carlo simulation, incorporating dynamic disaster frequencies,
            economic cycles, and market volatility for robust risk assessment.
          </p>
        </div>

        {/* ===== ENHANCED CONTROLS SECTION ===== */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="grid lg:grid-cols-6 gap-4">
            {/* Region Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Region</label>
              <select 
                value={selectedRegion} 
                onChange={(e) => setSelectedRegion(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="gulf_mexico">Gulf of Mexico</option>
                <option value="southeast_asia">Southeast Asia</option>
              </select>
            </div>
            {/* Climate Scenario Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Climate Scenario</label>
              <select 
                value={selectedScenario} 
                onChange={(e) => setSelectedScenario(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="conservative">Conservative</option>
                <option value="moderate">Moderate</option>
                <option value="aggressive">Aggressive</option>
              </select>
            </div>
            {/* Number of Simulations */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Simulations</label>
              <select 
                value={numSimulations} 
                onChange={(e) => setNumSimulations(parseInt(e.target.value))}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value={500}>500</option>
                <option value={1000}>1,000</option>
                <option value={2000}>2,000</option>
                <option value={5000}>5,000</option>
              </select>
            </div>
            {/* Confidence Interval */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Confidence (%)</label>
              <select 
                value={confidenceInterval} 
                onChange={(e) => setConfidenceInterval(parseInt(e.target.value))}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value={90}>90%</option>
                <option value={95}>95%</option>
                <option value={99}>99%</option>
              </select>
            </div>
            {/* Random Seed for Reproducibility */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Random Seed</label>
              <input
                type="number"
                value={randomSeed}
                onChange={(e) => setRandomSeed(parseInt(e.target.value))}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            {/* Toggle Details Button */}
            <div className="flex items-end">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="w-full bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 transition duration-200 flex items-center justify-center gap-2"
              >
                <BarChart3 size={20} />
                {showDetails ? 'Hide' : 'Show'} Details
              </button>
            </div>
          </div>
        </div>

        {/* ===== SUMMARY STATISTICS CARDS ===== */}
        <div className="grid md:grid-cols-6 gap-4 mb-8">
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <DollarSign className="mx-auto text-green-600 mb-2" size={24} />
            <div className="text-2xl font-bold text-gray-900">${summaryStats.totalMeanCost}B</div>
            <div className="text-sm text-gray-600">Mean 30-Year Cost</div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <TrendingUp className="mx-auto text-blue-600 mb-2" size={24} />
            <div className="text-2xl font-bold text-gray-900">${summaryStats.avgAnnualCost}B</div>
            <div className="text-sm text-gray-600">Mean Annual Cost</div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <AlertTriangle className="mx-auto text-red-600 mb-2" size={24} />
            <div className="text-2xl font-bold text-gray-900">${summaryStats.peakCost}B</div>
            <div className="text-sm text-gray-600">Peak Year Mean</div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <Target className="mx-auto text-orange-600 mb-2" size={24} />
            <div className="text-2xl font-bold text-gray-900">${summaryStats.cumulativeVar95}B</div>
            <div className="text-sm text-gray-600">VaR (95%)</div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <Shuffle className="mx-auto text-purple-600 mb-2" size={24} />
            <div className="text-2xl font-bold text-gray-900">{numSimulations.toLocaleString()}</div>
            <div className="text-sm text-gray-600">Simulations</div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <Calendar className="mx-auto text-indigo-600 mb-2" size={24} />
            <div className="text-2xl font-bold text-gray-900">{summaryStats.peakYear}</div>
            <div className="text-sm text-gray-600">Peak Year</div>
          </div>
        </div>

        {/* ===== ECONOMIC REGIME ANALYSIS CARDS ===== */}
        <div className="grid md:grid-cols-6 gap-4 mb-8">
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <div className="text-sm font-medium text-gray-600 mb-1">Most Common Regime</div>
            <div className="text-lg font-bold text-blue-600 capitalize">
              {Object.entries(statisticalResults[0]?.economicRegimeCounts || {})
                .sort(([,a], [,b]) => b - a)[0]?.[0] || 'Expansion'}
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <div className="text-sm font-medium text-gray-600 mb-1">Avg GDP Growth</div>
            <div className="text-lg font-bold text-green-600">
              {(statisticalResults.reduce((sum, data) => sum + data.avgGdpGrowth, 0) / 30 * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <div className="text-sm font-medium text-gray-600 mb-1">Avg Unemployment</div>
            <div className="text-lg font-bold text-orange-600">
              {(statisticalResults.reduce((sum, data) => sum + data.avgUnemployment, 0) / 30 * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <div className="text-sm font-medium text-gray-600 mb-1">Avg Inflation</div>
            <div className="text-lg font-bold text-purple-600">
              {(statisticalResults.reduce((sum, data) => sum + data.avgInflation, 0) / 30 * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <div className="text-sm font-medium text-gray-600 mb-1">Economic Shock Prob</div>
            <div className="text-lg font-bold text-red-600">
              {(statisticalResults.reduce((sum, data) => sum + data.economicShockProbability, 0) / 30 * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <div className="text-sm font-medium text-gray-600 mb-1">Avg Economic Impact</div>
            <div className="text-lg font-bold text-indigo-600">
              {(statisticalResults.reduce((sum, data) => sum + data.avgEconomicImpact, 0) / 30 * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* ===== PROBABILISTIC CHARTS SECTION ===== */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Expected Annual Costs with Confidence Bands */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">Annual Expected Costs (with {confidenceInterval}% CI)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={statisticalResults}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip formatter={(value, name) => [`${parseFloat(value).toFixed(2)}B`, name]} />
                <Area type="monotone" dataKey="upperBound" fill="#8884d8" fillOpacity={0.2} stroke="none" />
                <Area type="monotone" dataKey="lowerBound" fill="white" stroke="none" />
                <Line type="monotone" dataKey="mean" stroke="#8884d8" strokeWidth={2} />
                <Line type="monotone" dataKey="median" stroke="#82ca9d" strokeWidth={2} strokeDasharray="5 5" />
                <Legend />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Risk Metrics Over Time */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">Risk Metrics (VaR & CVaR at 95%)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={statisticalResults}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip formatter={(value) => [`${parseFloat(value).toFixed(2)}B`]} />
                <Line type="monotone" dataKey="var95" stroke="#ff7300" strokeWidth={2} name="VaR (95%)" />
                <Line type="monotone" dataKey="cvar95" stroke="#d084d0" strokeWidth={2} name="CVaR (95%)" />
                <Legend />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Cumulative Cost Distribution */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">Cumulative Costs with Uncertainty</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={statisticalResults}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip formatter={(value) => [`${parseFloat(value).toFixed(2)}B`]} />
                <Area type="monotone" dataKey="cumulativeUpper" fill="#8884d8" fillOpacity={0.3} stroke="none" />
                <Area type="monotone" dataKey="cumulativeLower" fill="white" stroke="none" />
                <Line type="monotone" dataKey="cumulativeMean" stroke="#8884d8" strokeWidth={3} name="Mean" />
                <Legend />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Probabilistic Disaster Type Breakdown */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">Mean Annual Cost by Disaster Type</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={disasterTypeData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {disasterTypeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${value}B`, 'Mean Annual Cost']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ===== VOLATILITY ANALYSIS ===== */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h3 className="text-xl font-semibold mb-4">Risk Volatility Analysis</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={statisticalResults}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip formatter={(value) => [`${parseFloat(value).toFixed(2)}B`]} />
              <Legend />
              <Area type="monotone" dataKey="stdDev" fill="#ffc658" fillOpacity={0.6} name="Standard Deviation" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* ===== DETAILED STOCHASTIC ANALYSIS (CONDITIONAL) ===== */}
        {showDetails && (
          <div className="space-y-8">
            {/* Economic Regime Impact Analysis */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold mb-6">Economic Regime Impact on Disaster Costs</h3>
              <div className="grid lg:grid-cols-2 gap-6">
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={statisticalResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="avgGdpGrowth" fill="#8884d8" fillOpacity={0.6} name="GDP Growth" />
                    <Area type="monotone" dataKey="avgEconomicImpact" fill="#82ca9d" fillOpacity={0.6} name="Economic Impact on Property" />
                  </AreaChart>
                </ResponsiveContainer>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={statisticalResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="avgUnemployment" stroke="#ff7300" name="Unemployment Rate" />
                    <Line type="monotone" dataKey="avgInflation" stroke="#d084d0" name="Inflation Rate" />
                    <Line type="monotone" dataKey="economicShockProbability" stroke="#ffc658" name="Shock Probability" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Stochastic Disaster Cost Breakdown */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold mb-6">Stochastic Disaster Cost Breakdown</h3>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={statisticalResults}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {/* Render different disaster types based on selected region */}
                  {selectedRegion === 'gulf_mexico' ? (
                    <>
                      <Area type="monotone" dataKey="hurricanes" stackId="1" stroke="#8884d8" fill="#8884d8" />
                      <Area type="monotone" dataKey="flooding" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
                      <Area type="monotone" dataKey="storm_surge" stackId="1" stroke="#ffc658" fill="#ffc658" />
                      <Area type="monotone" dataKey="heat_waves" stackId="1" stroke="#ff7300" fill="#ff7300" />
                    </>
                  ) : (
                    <>
                      <Area type="monotone" dataKey="typhoons" stackId="1" stroke="#8884d8" fill="#8884d8" />
                      <Area type="monotone" dataKey="flooding" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
                      <Area type="monotone" dataKey="sea_level_rise" stackId="1" stroke="#ffc658" fill="#ffc658" />
                      <Area type="monotone" dataKey="landslides" stackId="1" stroke="#ff7300" fill="#ff7300" />
                    </>
                  )}
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Statistical Distribution Comparison */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold mb-4">Statistical Distribution Metrics</h3>
              <div className="grid md:grid-cols-3 gap-6">
                {/* Central Tendency Panel */}
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-3">Central Tendency</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={statisticalResults.slice(0, 10)}>
                      <XAxis dataKey="year" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="mean" stroke="#8884d8" name="Mean" />
                      <Line type="monotone" dataKey="median" stroke="#82ca9d" name="Median" strokeDasharray="3 3" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {/* Dispersion Panel */}
                <div className="p-4 bg-yellow-50 rounded-lg">
                  <h4 className="font-medium text-yellow-900 mb-3">Dispersion</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={statisticalResults.slice(0, 10)}>
                      <XAxis dataKey="year" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="stdDev" fill="#ffc658" fillOpacity={0.6} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                {/* Tail Risk Panel */}
                <div className="p-4 bg-red-50 rounded-lg">
                  <h4 className="font-medium text-red-900 mb-3">Tail Risk</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={statisticalResults.slice(0, 10)}>
                      <XAxis dataKey="year" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="var95" stroke="#ff7300" name="VaR 95%" />
                      <Line type="monotone" dataKey="cvar95" stroke="#d084d0" name="CVaR 95%" strokeDasharray="3 3" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ===== COMPREHENSIVE MODEL INFORMATION SECTION ===== */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-semibold mb-6">Stochastic Model Architecture & Methodology</h3>
          
          {/* Model Components Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 text-sm mb-6">
            {/* Economic Regime Modeling */}
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Economic Regime Modeling</h4>
              <ul className="space-y-1 text-gray-600">
                <li>• Markov chain transitions</li>
                <li>• 5 economic states</li>
                <li>• GDP, unemployment, inflation</li>
                <li>• Regime-dependent volatility</li>
                <li>• Black swan shock events</li>
              </ul>
            </div>
            
            {/* Dynamic Disaster Models */}
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Dynamic Disaster Models</h4>
              <ul className="space-y-1 text-gray-600">
                <li>• Gamma-distributed frequencies</li>
                <li>• Climate oscillation cycles</li>
                <li>• Stochastic trend components</li>
                <li>• Economic stress correlations</li>
                <li>• Non-stationarity effects</li>
              </ul>
            </div>
            
            {/* Probabilistic Real Estate */}
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Probabilistic Real Estate</h4>
              <ul className="space-y-1 text-gray-600">
                <li>• Regime-dependent volatility</li>
                <li>• Credit market conditions</li>
                <li>• Flight-to-safety effects</li>
                <li>• Interest rate sensitivity</li>
                <li>• Economic indicator impacts</li>
              </ul>
            </div>
            
            {/* Enhanced Urban Growth */}
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Enhanced Urban Growth</h4>
              <ul className="space-y-1 text-gray-600">
                <li>• Economic cycle effects</li>
                <li>• Policy shock modeling</li>
                <li>• Government spending cycles</li>
                <li>• Development constraints</li>
                <li>• Crisis-dependent volatility</li>
              </ul>
            </div>
          </div>
          
          {/* Detailed Model Explanations */}
          <div className="grid md:grid-cols-3 gap-6 mb-6">
            {/* Economic Regime States */}
            <div className="p-4 bg-blue-50 rounded-lg">
              <h4 className="font-medium text-blue-900 mb-2">Economic Regime States</h4>
              <p className="text-sm text-blue-800">
                <strong>Expansion:</strong> GDP growth ~2.8%, Low unemployment, Strong property markets<br/>
                <strong>Slowdown:</strong> GDP growth ~0.8%, Rising unemployment, Cooling markets<br/>
                <strong>Recession:</strong> GDP decline ~-2.5%, High unemployment, Property decline<br/>
                <strong>Depression:</strong> Severe GDP decline ~-8%, Very high unemployment<br/>
                <strong>Financial Crisis:</strong> Credit crunch, Property market collapse
              </p>
            </div>
            
            {/* Economic-Disaster Interactions */}
            <div className="p-4 bg-green-50 rounded-lg">
              <h4 className="font-medium text-green-900 mb-2">Economic-Disaster Interactions</h4>
              <p className="text-sm text-green-800">
                Economic downturns amplify disaster impacts through: (1) Reduced disaster preparedness funding, 
                (2) Delayed infrastructure maintenance, (3) Slower recovery due to credit constraints, 
                (4) Higher vulnerability from deferred adaptation investments, (5) Correlation clustering during stress periods.
              </p>
            </div>
            
            {/* Advanced Risk Metrics */}
            <div className="p-4 bg-purple-50 rounded-lg">
              <h4 className="font-medium text-purple-900 mb-2">Advanced Risk Metrics</h4>
              <p className="text-sm text-purple-800">
                Comprehensive uncertainty quantification including: (1) Regime-switching volatility, 
                (2) Economic shock probabilities, (3) Cross-correlation effects, (4) Tail dependency modeling, 
                (5) Stress testing across economic scenarios, (6) Multi-dimensional risk assessment.
              </p>
            </div>
          </div>
          
          {/* Model Warning and Disclaimer */}
          <div className="p-4 bg-gradient-to-r from-red-50 to-orange-50 rounded-lg border-l-4 border-red-400">
            <p className="text-sm text-red-800">
              <strong>Enhanced Model Warning:</strong> This advanced stochastic model now incorporates economic regime 
              switching with {numSimulations.toLocaleString()} Monte Carlo simulations. The model captures complex 
              interactions between economic downturns and disaster impacts, including how recessions and financial 
              crises amplify climate risks through reduced preparedness, delayed infrastructure investment, and 
              constrained recovery capacity. Results reflect the full spectrum of economic scenarios from expansion 
              to depression, showing how economic volatility significantly affects climate disaster costs.
            </p>
          </div>

          {/* Technical Implementation Details */}
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-3">Technical Implementation Notes</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-700">
              <div>
                <strong>Random Number Generation:</strong>
                <ul className="mt-1 space-y-1">
                  <li>• Seeded linear congruential generator for reproducibility</li>
                  <li>• Box-Muller transform for normal distributions</li>
                  <li>• Marsaglia-Tsang method for gamma distributions</li>
                  <li>• Ensures consistent results across runs with same seed</li>
                </ul>
              </div>
              <div>
                <strong>Statistical Analysis:</strong>
                <ul className="mt-1 space-y-1">
                  <li>• Percentile-based confidence intervals</li>
                  <li>• Value-at-Risk (VaR) and Conditional VaR calculations</li>
                  <li>• Cross-sectional and time-series analysis</li>
                  <li>• Comprehensive uncertainty propagation</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClimateDisasterCostEstimator;
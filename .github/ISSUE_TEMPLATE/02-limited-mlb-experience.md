---
name: Handle Limited MLB Experience Players Better
about: Improve projections for players with insufficient MLB history through minor league data integration
title: '[ENHANCEMENT] Handle Limited MLB Experience Players Better'
labels: enhancement, model-performance, medium-priority, data-integration
assignees: ''
---

## Problem Description

The current projection system struggles with players who have limited MLB experience, as acknowledged in the project README: "Players with limited MLB experience get wonky projections". This creates significant gaps in the system's utility for:

- Rookie players in their first few seasons
- Players with injury-shortened careers
- International players with limited MLB history
- Players transitioning between levels (AAA callups, etc.)

## Proposed Solution Approach

### Phase 1: Minor League Data Integration
1. **Data Source Research**:
   - Investigate available minor league statistical databases
   - Evaluate data quality and consistency across different leagues/levels
   - Research existing translation factors from MiLB to MLB performance

2. **Translation Model Development**:
   - Create statistical translation factors for different minor league levels
   - Account for age, level, and context when translating performance
   - Develop confidence adjustments based on level and sample size

### Phase 2: Hybrid Modeling Approach
1. **Multi-Source Model Architecture**:
   - Extend LSTM models to handle both MLB and MiLB sequences
   - Implement weighted ensemble approach based on data availability
   - Create confidence scoring based on data quality and quantity

2. **Prospect Integration**:
   - Leverage existing `models/MiLB/` directory work
   - Enhance `matchProspect.py` functionality for better prospect matching
   - Integrate scouting report data if available

### Phase 3: Uncertainty Quantification
1. **Confidence Intervals**:
   - Wider confidence intervals for players with limited experience
   - Dynamic interval sizing based on data availability
   - Clear uncertainty communication in the UI

## Acceptance Criteria

- [ ] Players with <3 years MLB experience show improved projection accuracy
- [ ] System successfully incorporates minor league performance data
- [ ] Confidence intervals appropriately reflect uncertainty for limited-experience players
- [ ] Translation factors validated against historical promotion data
- [ ] API endpoints updated to handle hybrid MLB/MiLB player data
- [ ] UI clearly communicates projection uncertainty for these players

## Technical Considerations

### Implementation Notes
- Build upon existing work in `models/MiLB/` directory
- Enhance `models/MiLB/milb_batter.ipynb` and related notebooks
- Integrate with `models/MiLB/matchProspect.py` for prospect identification
- Update sequence preparation to handle mixed MLB/MiLB data

### Data Requirements
- **Minor League Statistics**: Comprehensive historical MiLB data
- **Player Tracking**: Accurate player progression through minor league levels
- **League Context**: Ballpark factors, league offensive levels by year
- **Age Curves**: Age-adjusted performance expectations by level

### Architecture Changes
```python
# Example data structure for hybrid sequences
player_sequence = {
    'mlb_seasons': [...],  # Standard MLB data
    'milb_seasons': [...], # Translated MiLB data
    'weights': [...],      # Confidence weights per season
    'source_levels': [...] # Level indicators (AAA, AA, etc.)
}
```

### Risk Assessment
- **High Risk**: Minor league data quality may be inconsistent
- **Medium Risk**: Translation factors may not generalize across eras
- **Low Risk**: Can be implemented as optional enhancement to existing models

## Priority Level
**Medium Priority** - Important for system completeness but doesn't affect existing player projections.

## Implementation Phases

### Phase 1 (Research & Data) - 2-3 weeks
- [ ] Minor league data source evaluation
- [ ] Historical translation factor analysis
- [ ] Data pipeline development

### Phase 2 (Model Development) - 3-4 weeks
- [ ] Hybrid LSTM architecture implementation
- [ ] Translation model training and validation
- [ ] Integration with existing projection pipeline

### Phase 3 (Integration & Testing) - 2-3 weeks
- [ ] API endpoint updates
- [ ] UI uncertainty communication features
- [ ] Comprehensive testing with historical data

## Related Issues
- Coordinates with Improve Pitching Projection Accuracy (both involve data scarcity)
- Supports Add Confidence Intervals to Projections (uncertainty quantification)
- May benefit from Integrate Better Data Sources (broader data availability)

## Success Metrics
- [ ] 25% improvement in projection accuracy for players with <500 MLB plate appearances
- [ ] Successful integration of at least 3 minor league levels (AAA, AA, A+)
- [ ] User feedback shows improved confidence in rookie player valuations

## Definition of Done
- [ ] Minor league data successfully integrated into projection pipeline
- [ ] Translation factors validated and documented
- [ ] Hybrid models show improved performance for limited-experience players
- [ ] UI properly communicates uncertainty levels
- [ ] Documentation updated with new data requirements and limitations
---
name: Improve Pitching Projection Accuracy
about: Address acknowledged issues with pitching projections due to smaller sample sizes and pitcher volatility
title: '[ENHANCEMENT] Improve Pitching Projection Accuracy'
labels: enhancement, model-performance, high-priority, pitching
assignees: ''
---

## Problem Description

The current LSTM-based pitching projection model shows acknowledged accuracy issues compared to the hitting projections. As noted in the project README, "The pitching projections... not so much (probably due to smaller sample sizes)". This is likely due to:

- Smaller sample sizes for pitchers compared to position players
- Higher inherent volatility in pitcher performance year-over-year
- Different statistical patterns that may not be well-captured by the current LSTM architecture

## Proposed Solution Approach

### Research Phase
1. **Data Analysis**: Conduct comprehensive analysis of pitcher vs. batter data characteristics
   - Compare sample sizes and data distribution patterns
   - Analyze year-over-year correlation patterns for different pitcher statistics
   - Identify which pitcher metrics are most predictable vs. most volatile

2. **Model Architecture Investigation**:
   - Experiment with different LSTM architectures specifically for pitchers
   - Consider ensemble methods combining multiple models
   - Investigate attention mechanisms for focusing on most relevant historical periods
   - Test different sequence lengths optimized for pitcher career patterns

### Implementation Phase
1. **Feature Engineering**:
   - Incorporate pitcher-specific features (pitch type data, velocity trends, etc.)
   - Add context-aware features (ballpark factors, league adjustments)
   - Consider role-based modeling (starter vs. reliever specific models)

2. **Model Improvements**:
   - Implement separate model architectures for starters vs. relievers
   - Add uncertainty quantification specific to pitcher projections
   - Develop pitcher-specific loss functions that account for volatility

## Acceptance Criteria

- [ ] Pitcher projection accuracy improves by at least 15% as measured by MAE on held-out test data
- [ ] Separate models for starters and relievers show improved performance vs. combined model
- [ ] Model performance analysis document comparing old vs. new approach
- [ ] Updated training pipeline that handles pitcher-specific preprocessing
- [ ] Validation framework showing improved correlation with actual future performance

## Technical Considerations

### Implementation Notes
- Current pitcher model is in `models/pitcher.ipynb`
- Consider splitting into `models/starter_pitcher.ipynb` and `models/reliever_pitcher.ipynb`
- May need to adjust normalization schemes (currently 32 games for starters, 65 innings for relievers)
- Integration with existing backend API in `web-app/backend/app/`

### Data Requirements
- Historical pitcher performance data with role classifications
- Pitch-level data if available for enhanced feature engineering
- Injury history data to better model career interruptions
- Minor league data for pitchers with limited MLB experience

### Risk Assessment
- **Medium Risk**: Changes to projection algorithms may temporarily reduce performance
- **Low Risk**: Can be implemented alongside existing models for A/B testing
- **Consideration**: May require retraining all pitcher models from scratch

## Priority Level
**High Priority** - This directly addresses a known limitation that affects the core product value proposition.

## Related Issues
- Will depend on implementation of Model Validation Framework for proper performance measurement
- Should be coordinated with Handle Limited MLB Experience Players as both involve data scarcity issues

## Definition of Done
- [ ] Research phase completed with documented findings
- [ ] New pitcher projection models implemented and tested
- [ ] Performance benchmarks show measurable improvement
- [ ] Models integrated into production pipeline
- [ ] Documentation updated with new approach and limitations
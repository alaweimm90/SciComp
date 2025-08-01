# Berkeley SciComp Visual Identity Style Guide

**Official UC Berkeley Brand Implementation for Scientific Computing**

*Author: Dr. Meshal Alawein (meshal@berkeley.edu)*  
*Institution: University of California, Berkeley*  
*Created: 2025*

---

## Overview

This style guide implements the official UC Berkeley visual identity for the Berkeley SciComp Framework, ensuring consistent, professional presentation across all computational platforms and scientific outputs.

### 🎯 **Mission Statement**

The Berkeley SciComp visual identity reflects UC Berkeley's commitment to excellence, innovation, and academic rigor in scientific computing, while maintaining accessibility and clarity for diverse audiences.

---

## 🎨 Color Palette

### Primary Brand Colors

#### **Berkeley Blue** `#003262`
- **RGB**: (0, 50, 98)
- **Usage**: Primary text, axes, headers, emphasis
- **Psychology**: Trust, stability, academic authority
- **Implementation**: Main color for all text and structural elements

#### **California Gold** `#FDB515`
- **RGB**: (253, 181, 21)
- **Usage**: Accents, highlights, secondary data series
- **Psychology**: Optimism, innovation, energy
- **Implementation**: Complementary color for contrast and highlights

### Secondary Colors

#### **Founders Rock** `#3B7EA1`
- **RGB**: (59, 126, 161)
- **Usage**: Supporting graphics, tertiary data series
- **Context**: Berkeley's Founders Rock landmark
- **Implementation**: Alternative blue for variety in multi-series plots

#### **Medalist** `#B9975B`
- **RGB**: (185, 151, 91)
- **Usage**: Warm accents, background elements
- **Context**: UC Berkeley's championship legacy
- **Implementation**: Neutral warm tone for backgrounds and supporting elements

### Extended Palette for Data Visualization

#### **Lawrence Blue** `#00B0DA`
- **RGB**: (0, 176, 218)
- **Usage**: Scientific data, technical visualizations
- **Context**: Lawrence Berkeley National Laboratory

#### **Bay Teal** `#00A598`
- **RGB**: (0, 165, 152)
- **Usage**: Environmental data, sustainability themes
- **Context**: San Francisco Bay

#### **Golden Gate Orange** `#ED4E33`
- **RGB**: (237, 78, 51)
- **Usage**: Warnings, error states, critical data
- **Context**: Golden Gate Bridge

#### **Pacific Blue** `#46535E`
- **RGB**: (70, 83, 94)
- **Usage**: Neutral elements, secondary text
- **Context**: Pacific Ocean

### Neutral Colors

#### **Berkeley Grey** `#666666`
- **RGB**: (102, 102, 102)
- **Usage**: Secondary text, grid lines, subtle elements

#### **Light Grey** `#CCCCCC`
- **RGB**: (204, 204, 204)
- **Usage**: Background grids, separators, disabled states

---

## 📝 Typography

### Font Hierarchy

#### **Primary Font**: Arial/DejaVu Sans
- **Rationale**: Excellent readability, cross-platform availability
- **Usage**: All body text, labels, legends
- **Characteristics**: Clean, modern, scientific

#### **Mathematical Font**: Times/DejaVu Sans
- **Usage**: Mathematical expressions, equations, formulas
- **Characteristics**: Traditional academic appearance

#### **Code Font**: Courier/DejaVu Sans Mono
- **Usage**: Code blocks, terminal output, technical specifications
- **Characteristics**: Monospaced, clear distinction

### Font Sizes

| Element | Size (pt) | Usage |
|---------|-----------|-------|
| Title | 16 | Main plot titles, document headers |
| Subtitle | 14 | Section headers, axis titles |
| Body | 12 | Standard text, axis labels |
| Caption | 10 | Legends, annotations |
| Small | 8 | Watermarks, fine print |

### Typography Principles

1. **Hierarchy**: Clear visual hierarchy through size and weight
2. **Consistency**: Same fonts across all platforms
3. **Readability**: High contrast, appropriate sizing
4. **Academic Standards**: Professional scientific appearance

---

## 📊 Data Visualization Guidelines

### Color Application Strategy

#### **Single Series Plots**
- Primary: Berkeley Blue
- Alternative: California Gold (for emphasis)

#### **Multi-Series Plots** (2-4 series)
1. Berkeley Blue
2. California Gold
3. Founders Rock
4. Medalist

#### **Complex Visualizations** (5+ series)
Use full extended palette in order:
1. Berkeley Blue
2. California Gold
3. Founders Rock
4. Medalist
5. Lawrence Blue
6. Bay Teal
7. Golden Gate Orange
8. Pacific Blue

#### **Continuous Data (Heatmaps, Contours)**
- **Recommended**: Custom Berkeley gradient (Blue → Gold)
- **Alternative**: Berkeley Blue → Founders Rock → Bay Teal → California Gold → Golden Gate Orange

### Plot Styling Standards

#### **Axes and Frames**
- **Color**: Berkeley Blue
- **Weight**: 1.5pt for main axes
- **Style**: Clean, minimal, professional

#### **Grid Lines**
- **Color**: Light Grey (#CCCCCC)
- **Opacity**: 70%
- **Style**: Subtle, non-distracting

#### **Line Weights**
- **Data lines**: 2.0-2.5pt
- **Grid lines**: 0.8pt
- **Axes**: 1.5pt

#### **Markers and Symbols**
- **Size**: 6-8pt
- **Style**: Consistent with line colors
- **Shape**: Varied for multi-series distinction

---

## 🖼️ Layout and Composition

### Figure Dimensions

#### **Standard Sizes**
- **Default**: 10" × 6" (landscape)
- **Square**: 8" × 8"
- **Presentation**: 12" × 8"
- **Publication**: As required by journal

#### **Resolution Standards**
- **Screen**: 150 DPI minimum
- **Print**: 300 DPI minimum
- **Publication**: 600 DPI (vector preferred)

### Margins and Spacing

#### **Internal Spacing**
- **Title padding**: 20pt above plot area
- **Axis label padding**: 10pt from axes
- **Legend spacing**: 5pt between items

#### **External Margins**
- **All sides**: 0.1" minimum
- **Presentation**: 0.2" for projection clarity

### Watermarking

#### **Standard Watermark**: "Berkeley SciComp"
- **Position**: Bottom right (default)
- **Opacity**: 30%
- **Font**: Italic, small size
- **Color**: Berkeley Grey

#### **Alternative Positions**
- Top right: For bottom-heavy data
- Bottom left: For right-heavy data
- Top left: For corner emphasis

---

## 🔬 Platform-Specific Implementation

### Python (Matplotlib/Seaborn)

```python
# Import Berkeley styling
from assets.berkeley_style import BerkeleyPlotStyle, BerkeleyColors

# Apply global styling
BerkeleyPlotStyle.apply_style()

# Create Berkeley-styled plot
fig, ax = BerkeleyPlots.setup_physics_plot(
    title="Your Title", 
    xlabel="X Label", 
    ylabel="Y Label"
)

# Add Berkeley watermark
BerkeleyPlots.add_berkeley_watermark(ax)
```

### MATLAB

```matlab
% Apply Berkeley styling
berkeley_style();

% Create Berkeley-styled figure
fig = setup_berkeley_figure('Your Title', 'X Label', 'Y Label');

% Add Berkeley watermark
add_berkeley_watermark(gca);

% Save with Berkeley standards
save_berkeley_figure(fig, 'filename.png');
```

### Mathematica

```mathematica
<< "assets/BerkeleyStyle.wl"

(* Create Berkeley-styled plot *)
Plot[Sin[x], {x, 0, 2π}, 
  Evaluate[SetupBerkeleyPlot["Your Title", "X Label", "Y Label"]]]

(* Add watermark and save *)
AddBerkeleyWatermark[%, "BottomRight"]
SaveBerkeleyFigure[%, "filename.png"]
```

---

## 🎯 Application Examples

### Quantum Physics Visualizations

#### **Wavefunction Plots**
- **Real part**: Berkeley Blue (solid line)
- **Imaginary part**: California Gold (dashed line)
- **Probability density**: Founders Rock (solid line)
- **Potential**: Medalist (thin line)

#### **Energy Level Diagrams**
- **Ground state**: Berkeley Blue
- **Excited states**: California Gold
- **Continuum**: Light grey

### Engineering Applications

#### **Heat Transfer**
- **Temperature field**: Berkeley gradient colormap
- **Isotherms**: Berkeley Blue contour lines
- **Boundaries**: Thick Berkeley Blue lines

#### **Fluid Dynamics**
- **Velocity vectors**: Berkeley Blue
- **Pressure contours**: Berkeley gradient
- **Streamlines**: California Gold

### Machine Learning

#### **Training Curves**
- **Training loss**: Berkeley Blue with circles
- **Validation loss**: California Gold with squares
- **Test accuracy**: Founders Rock with triangles

#### **Neural Network Architectures**
- **Input layer**: Berkeley Blue
- **Hidden layers**: California Gold
- **Output layer**: Founders Rock

---

## ✅ Quality Assurance Checklist

### Pre-Publication Review

#### **Color Compliance**
- [ ] Primary colors used correctly
- [ ] Color blind accessibility considered
- [ ] Sufficient contrast ratios maintained
- [ ] Print compatibility verified

#### **Typography Standards**
- [ ] Consistent font usage
- [ ] Appropriate sizing hierarchy
- [ ] Mathematical notation properly formatted
- [ ] Text legibility confirmed

#### **Layout Standards**
- [ ] Berkeley watermark present
- [ ] Appropriate margins maintained
- [ ] Consistent spacing applied
- [ ] Professional appearance verified

#### **Technical Quality**
- [ ] Sufficient resolution for intended use
- [ ] Vector formats used when possible
- [ ] File sizes optimized
- [ ] Cross-platform compatibility tested

---

## 🚫 Common Mistakes to Avoid

### Color Usage Errors

#### **Don't**
- Use non-Berkeley colors as primary elements
- Combine too many bright colors
- Use red/green combinations (colorblind unfriendly)
- Apply colors inconsistently across related figures

#### **Do**
- Stick to the official palette
- Use Berkeley Blue as the dominant color
- Consider accessibility in color choices
- Maintain consistency across figure series

### Typography Mistakes

#### **Don't**
- Mix multiple font families unnecessarily
- Use decorative fonts for scientific content
- Make text too small to read
- Overcrowd with excessive text

#### **Do**
- Use the designated font hierarchy
- Ensure sufficient contrast
- Test readability at intended viewing size
- Keep text concise and clear

### Layout Issues

#### **Don't**
- Cram too much information in one figure
- Use inconsistent spacing
- Forget the Berkeley watermark
- Ignore margin requirements

#### **Do**
- Plan for clear visual hierarchy
- Use consistent spacing throughout
- Include appropriate attribution
- Test at multiple sizes

---

## 📏 Accessibility Guidelines

### Color Accessibility

#### **Contrast Requirements**
- **Text on background**: Minimum 4.5:1 ratio
- **Large text**: Minimum 3:1 ratio
- **Graphical elements**: Minimum 3:1 ratio

#### **Colorblind Considerations**
- Berkeley Blue and California Gold are distinguishable
- Avoid relying solely on color for information
- Use patterns, shapes, or labels as additional cues
- Test with colorblind simulation tools

### Visual Accessibility

#### **Font Sizing**
- **Minimum text size**: 12pt for body text
- **Presentation minimum**: 18pt for projected content
- **High contrast**: Always on white or light backgrounds

#### **Layout Clarity**
- Clear visual hierarchy
- Adequate spacing between elements
- Logical information flow
- Consistent navigation patterns

---

## 🔄 Version Control and Updates

### Style Guide Maintenance

#### **Update Schedule**
- **Annual review**: Comprehensive style assessment
- **Quarterly updates**: Minor adjustments and additions
- **As-needed**: Critical fixes and new requirements

#### **Change Documentation**
- All style changes documented with rationale
- Version history maintained
- Backward compatibility considered
- Migration guides provided when needed

### Implementation Tracking

#### **Compliance Monitoring**
- Regular audits of published materials
- Feedback collection from users
- Style adherence metrics
- Continuous improvement process

---

## 📞 Support and Resources

### Getting Help

#### **Technical Support**
- **Email**: meshal@berkeley.edu
- **Documentation**: Complete API references available
- **Examples**: Comprehensive example gallery
- **Community**: Berkeley SciComp user group

#### **Style Questions**
- **Brand Guidelines**: UC Berkeley official standards
- **Custom Requirements**: Project-specific adaptations
- **Accessibility**: Section 508 compliance assistance
- **Print Production**: High-quality output optimization

### Additional Resources

#### **UC Berkeley Brand Resources**
- [Official Brand Guidelines](https://brand.berkeley.edu)
- [Color Palette Downloads](https://brand.berkeley.edu/colors)
- [Logo and Trademark Usage](https://brand.berkeley.edu/logos)

#### **Accessibility Resources**
- [Web Accessibility Guidelines](https://accessibility.berkeley.edu)
- [Color Contrast Analyzers](https://webaim.org/resources/contrastchecker)
- [Colorblind Simulation Tools](https://www.color-blindness.com/coblis-color-blindness-simulator)

---

## 📜 Legal and Attribution

### Copyright and Usage

```
Copyright © 2025 Dr. Meshal Alawein — All rights reserved.
University of California, Berkeley

This style guide implements official UC Berkeley brand guidelines
for scientific computing applications. Use in accordance with
UC Berkeley brand standards and trademark policies.
```

### Attribution Requirements

When using Berkeley SciComp styling:
- Include "Berkeley SciComp" watermark on figures
- Credit UC Berkeley in appropriate contexts
- Maintain style guide compliance
- Respect trademark and brand guidelines

---

## 📊 Appendices

### Appendix A: Color Hex Codes Quick Reference

| Color Name | Hex Code | RGB Values |
|------------|----------|------------|
| Berkeley Blue | #003262 | (0, 50, 98) |
| California Gold | #FDB515 | (253, 181, 21) |
| Founders Rock | #3B7EA1 | (59, 126, 161) |
| Medalist | #B9975B | (185, 151, 91) |
| Lawrence Blue | #00B0DA | (0, 176, 218) |
| Bay Teal | #00A598 | (0, 165, 152) |
| Golden Gate Orange | #ED4E33 | (237, 78, 51) |
| Pacific Blue | #46535E | (70, 83, 94) |
| Berkeley Grey | #666666 | (102, 102, 102) |
| Light Grey | #CCCCCC | (204, 204, 204) |

### Appendix B: File Naming Conventions

- **Figures**: `berkeley_[category]_[description].png`
- **Data**: `[project]_[dataset]_[date].csv`
- **Code**: `[module]_berkeley_style.py`
- **Documentation**: `[topic]_style_guide.md`

### Appendix C: Quality Assurance Metrics

- **Color compliance**: 100% for primary elements
- **Typography consistency**: Standardized across platforms
- **Accessibility**: WCAG 2.1 AA compliant
- **Brand alignment**: UC Berkeley standards maintained

---

*This style guide ensures consistent, professional, and accessible scientific visualization across the Berkeley SciComp Framework, maintaining UC Berkeley's standards of excellence in academic and research communications.*

**Last Updated**: 2025  
**Version**: 1.0.0  
**Next Review**: 2026
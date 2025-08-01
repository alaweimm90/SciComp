#!/usr/bin/env python3
"""
Berkeley Visual Identity Style Guide
====================================

Official Berkeley SciComp Framework styling configuration implementing
UC Berkeley brand guidelines for consistent visual identity across all
computational platforms and outputs.

Author: Dr. Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT

Copyright © 2025 Dr. Meshal Alawein — All rights reserved.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# =============================================================================
# UC Berkeley Official Color Palette
# =============================================================================

class BerkeleyColors:
    """Official UC Berkeley color palette with hex, RGB, and normalized values."""
    
    # Primary Brand Colors
    BERKELEY_BLUE = {
        'hex': '#003262',
        'rgb': (0, 50, 98),
        'norm': (0.0, 0.196, 0.384),
        'name': 'Berkeley Blue'
    }
    
    CALIFORNIA_GOLD = {
        'hex': '#FDB515', 
        'rgb': (253, 181, 21),
        'norm': (0.992, 0.710, 0.082),
        'name': 'California Gold'
    }
    
    # Secondary Colors
    FOUNDERS_ROCK = {
        'hex': '#3B7EA1',
        'rgb': (59, 126, 161), 
        'norm': (0.231, 0.494, 0.631),
        'name': 'Founders Rock'
    }
    
    MEDALIST = {
        'hex': '#B9975B',
        'rgb': (185, 151, 91),
        'norm': (0.725, 0.592, 0.357),
        'name': 'Medalist'
    }
    
    # Neutral Colors
    BERKELEY_GREY = {
        'hex': '#666666',
        'rgb': (102, 102, 102),
        'norm': (0.400, 0.400, 0.400),
        'name': 'Berkeley Grey'
    }
    
    LIGHT_GREY = {
        'hex': '#CCCCCC',
        'rgb': (204, 204, 204),
        'norm': (0.800, 0.800, 0.800),
        'name': 'Light Grey'
    }
    
    # Accent Colors for Data Visualization
    PACIFIC_BLUE = {
        'hex': '#46535E',
        'rgb': (70, 83, 94),
        'norm': (0.275, 0.325, 0.369),
        'name': 'Pacific Blue'
    }
    
    LAWRENCE_BLUE = {
        'hex': '#00B0DA',
        'rgb': (0, 176, 218),
        'norm': (0.0, 0.690, 0.855),
        'name': 'Lawrence Blue'
    }
    
    GOLDEN_GATE_ORANGE = {
        'hex': '#ED4E33',
        'rgb': (237, 78, 51),
        'norm': (0.929, 0.306, 0.200),
        'name': 'Golden Gate Orange'
    }
    
    BAY_TEAL = {
        'hex': '#00A598',
        'rgb': (0, 165, 152),
        'norm': (0.0, 0.647, 0.596),
        'name': 'Bay Teal'
    }
    
    @classmethod
    def get_primary_palette(cls) -> List[str]:
        """Get primary color palette as hex codes."""
        return [cls.BERKELEY_BLUE['hex'], cls.CALIFORNIA_GOLD['hex']]
    
    @classmethod
    def get_extended_palette(cls) -> List[str]:
        """Get extended color palette for complex visualizations."""
        return [
            cls.BERKELEY_BLUE['hex'],
            cls.CALIFORNIA_GOLD['hex'], 
            cls.FOUNDERS_ROCK['hex'],
            cls.MEDALIST['hex'],
            cls.LAWRENCE_BLUE['hex'],
            cls.BAY_TEAL['hex'],
            cls.GOLDEN_GATE_ORANGE['hex'],
            cls.PACIFIC_BLUE['hex']
        ]
    
    @classmethod
    def get_gradient_colors(cls, n_colors: int = 10) -> List[str]:
        """Generate gradient colors between Berkeley Blue and California Gold."""
        colors = []
        for i in range(n_colors):
            # Interpolate between Berkeley Blue and California Gold
            t = i / (n_colors - 1) if n_colors > 1 else 0
            r = int(cls.BERKELEY_BLUE['rgb'][0] * (1-t) + cls.CALIFORNIA_GOLD['rgb'][0] * t)
            g = int(cls.BERKELEY_BLUE['rgb'][1] * (1-t) + cls.CALIFORNIA_GOLD['rgb'][1] * t)
            b = int(cls.BERKELEY_BLUE['rgb'][2] * (1-t) + cls.CALIFORNIA_GOLD['rgb'][2] * t)
            colors.append(f'#{r:02x}{g:02x}{b:02x}')
        return colors

# =============================================================================
# Typography and Layout Settings
# =============================================================================

class BerkeleyTypography:
    """Berkeley SciComp typography and layout specifications."""
    
    # Font specifications
    PRIMARY_FONT = 'DejaVu Sans'  # Available across platforms
    MATH_FONT = 'DejaVu Sans'     # For mathematical expressions
    CODE_FONT = 'DejaVu Sans Mono'  # For code blocks
    
    # Font sizes
    TITLE_SIZE = 16
    SUBTITLE_SIZE = 14
    BODY_SIZE = 12
    CAPTION_SIZE = 10
    SMALL_SIZE = 8
    
    # Line spacing
    LINE_SPACING = 1.2
    
    # Margins and spacing
    FIGURE_DPI = 300
    FIGURE_SIZE = (10, 6)  # Default figure size in inches
    
    @staticmethod
    def get_font_config() -> Dict:
        """Get matplotlib font configuration."""
        return {
            'font.family': BerkeleyTypography.PRIMARY_FONT,
            'font.size': BerkeleyTypography.BODY_SIZE,
            'axes.titlesize': BerkeleyTypography.TITLE_SIZE,
            'axes.labelsize': BerkeleyTypography.BODY_SIZE,
            'xtick.labelsize': BerkeleyTypography.CAPTION_SIZE,
            'ytick.labelsize': BerkeleyTypography.CAPTION_SIZE,
            'legend.fontsize': BerkeleyTypography.CAPTION_SIZE,
            'figure.titlesize': BerkeleyTypography.TITLE_SIZE
        }

# =============================================================================
# Berkeley Scientific Plotting Style
# =============================================================================

class BerkeleyPlotStyle:
    """Berkeley SciComp matplotlib styling configuration."""
    
    @staticmethod
    def get_style_dict() -> Dict:
        """Get complete matplotlib style dictionary."""
        colors = BerkeleyColors()
        typography = BerkeleyTypography()
        
        return {
            # Figure settings
            'figure.figsize': typography.FIGURE_SIZE,
            'figure.dpi': typography.FIGURE_DPI,
            'figure.facecolor': 'white',
            'figure.edgecolor': 'none',
            
            # Axes settings
            'axes.facecolor': 'white',
            'axes.edgecolor': colors.BERKELEY_BLUE['hex'],
            'axes.linewidth': 1.5,
            'axes.grid': True,
            'axes.axisbelow': True,
            'axes.prop_cycle': plt.cycler('color', colors.get_extended_palette()),
            'axes.labelcolor': colors.BERKELEY_BLUE['hex'],
            'axes.titlecolor': colors.BERKELEY_BLUE['hex'],
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            
            # Grid settings
            'grid.color': colors.LIGHT_GREY['hex'],
            'grid.linestyle': '-',
            'grid.linewidth': 0.8,
            'grid.alpha': 0.7,
            
            # Ticks
            'xtick.color': colors.BERKELEY_BLUE['hex'],
            'ytick.color': colors.BERKELEY_BLUE['hex'],
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 6,
            'ytick.major.size': 6,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,
            
            # Legend
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': False,
            'legend.framealpha': 0.9,
            'legend.facecolor': 'white',
            'legend.edgecolor': colors.BERKELEY_GREY['hex'],
            
            # Lines
            'lines.linewidth': 2.0,
            'lines.markersize': 8,
            'lines.markeredgewidth': 1.0,
            
            # Patches
            'patch.linewidth': 0.5,
            'patch.facecolor': colors.CALIFORNIA_GOLD['hex'],
            'patch.edgecolor': colors.BERKELEY_BLUE['hex'],
            
            # Images
            'image.cmap': 'viridis',  # Default colormap
            
            # Saving
            'savefig.dpi': typography.FIGURE_DPI,
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # Font settings
            **typography.get_font_config()
        }
    
    @staticmethod
    def apply_style():
        """Apply Berkeley style to matplotlib."""
        style_dict = BerkeleyPlotStyle.get_style_dict()
        plt.rcParams.update(style_dict)
    
    @staticmethod
    def create_berkeley_colormap(name: str = 'berkeley') -> mcolors.LinearSegmentedColormap:
        """Create Berkeley-themed colormap."""
        colors = BerkeleyColors()
        
        # Define color progression
        color_list = [
            colors.BERKELEY_BLUE['norm'],
            colors.FOUNDERS_ROCK['norm'],
            colors.BAY_TEAL['norm'],
            colors.CALIFORNIA_GOLD['norm'],
            colors.GOLDEN_GATE_ORANGE['norm']
        ]
        
        return mcolors.LinearSegmentedColormap.from_list(name, color_list)

# =============================================================================
# Specialized Plot Functions
# =============================================================================

class BerkeleyPlots:
    """Berkeley-styled plotting functions for scientific applications."""
    
    @staticmethod
    def setup_physics_plot(title: str = "", xlabel: str = "", ylabel: str = "") -> Tuple:
        """Set up a physics plot with Berkeley styling."""
        BerkeleyPlotStyle.apply_style()
        
        fig, ax = plt.subplots(figsize=BerkeleyTypography.FIGURE_SIZE)
        
        if title:
            ax.set_title(title, color=BerkeleyColors.BERKELEY_BLUE['hex'], 
                        fontweight='bold', pad=20)
        if xlabel:
            ax.set_xlabel(xlabel, color=BerkeleyColors.BERKELEY_BLUE['hex'])
        if ylabel:
            ax.set_ylabel(ylabel, color=BerkeleyColors.BERKELEY_BLUE['hex'])
        
        return fig, ax
    
    @staticmethod
    def add_berkeley_watermark(ax, position: str = 'bottom_right', alpha: float = 0.3):
        """Add Berkeley SciComp watermark to plot."""
        watermark_text = "Berkeley SciComp"
        
        if position == 'bottom_right':
            x, y = 0.98, 0.02
            ha, va = 'right', 'bottom'
        elif position == 'top_right':
            x, y = 0.98, 0.98
            ha, va = 'right', 'top'
        elif position == 'bottom_left':
            x, y = 0.02, 0.02
            ha, va = 'left', 'bottom'
        else:  # top_left
            x, y = 0.02, 0.98
            ha, va = 'left', 'top'
        
        ax.text(x, y, watermark_text, transform=ax.transAxes,
                fontsize=BerkeleyTypography.SMALL_SIZE,
                color=BerkeleyColors.BERKELEY_GREY['hex'],
                alpha=alpha, ha=ha, va=va,
                style='italic')
    
    @staticmethod
    def quantum_wavefunction_plot(x: np.ndarray, psi: np.ndarray, 
                                 title: str = "Quantum Wavefunction",
                                 potential: Optional[np.ndarray] = None) -> Tuple:
        """Create Berkeley-styled quantum wavefunction plot."""
        fig, ax = BerkeleyPlots.setup_physics_plot(
            title=title, xlabel="Position (x)", ylabel="Wavefunction ψ(x)"
        )
        
        # Plot wavefunction
        ax.plot(x, np.real(psi), color=BerkeleyColors.BERKELEY_BLUE['hex'], 
                linewidth=2.5, label='Re[ψ(x)]')
        
        if np.any(np.iscomplex(psi)):
            ax.plot(x, np.imag(psi), color=BerkeleyColors.CALIFORNIA_GOLD['hex'],
                    linewidth=2.5, label='Im[ψ(x)]', linestyle='--')
        
        # Plot probability density
        ax.plot(x, np.abs(psi)**2, color=BerkeleyColors.FOUNDERS_ROCK['hex'],
                linewidth=2.0, label='|ψ(x)|²', alpha=0.8)
        
        # Add potential if provided
        if potential is not None:
            ax2 = ax.twinx()
            ax2.plot(x, potential, color=BerkeleyColors.MEDALIST['hex'],
                     linewidth=1.5, label='V(x)', alpha=0.7)
            ax2.set_ylabel('Potential V(x)', color=BerkeleyColors.MEDALIST['hex'])
            ax2.tick_params(axis='y', labelcolor=BerkeleyColors.MEDALIST['hex'])
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        BerkeleyPlots.add_berkeley_watermark(ax)
        
        return fig, ax
    
    @staticmethod
    def heat_transfer_plot(x: np.ndarray, y: np.ndarray, temperature: np.ndarray,
                          title: str = "Heat Transfer Analysis") -> Tuple:
        """Create Berkeley-styled heat transfer contour plot."""
        fig, ax = BerkeleyPlots.setup_physics_plot(
            title=title, xlabel="x (m)", ylabel="y (m)"
        )
        
        # Create Berkeley colormap
        berkeley_cmap = BerkeleyPlotStyle.create_berkeley_colormap()
        
        # Contour plot
        contour = ax.contourf(x, y, temperature, levels=20, cmap=berkeley_cmap, alpha=0.8)
        contour_lines = ax.contour(x, y, temperature, levels=10, 
                                  colors=BerkeleyColors.BERKELEY_BLUE['hex'], 
                                  linewidths=0.8, alpha=0.6)
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Temperature (°C)', color=BerkeleyColors.BERKELEY_BLUE['hex'])
        cbar.ax.tick_params(labelcolor=BerkeleyColors.BERKELEY_BLUE['hex'])
        
        # Add contour labels
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        
        ax.set_aspect('equal')
        BerkeleyPlots.add_berkeley_watermark(ax)
        
        return fig, ax
    
    @staticmethod
    def ml_training_plot(epochs: np.ndarray, train_loss: np.ndarray, 
                        val_loss: Optional[np.ndarray] = None,
                        title: str = "ML Physics Training Progress") -> Tuple:
        """Create Berkeley-styled ML training progress plot."""
        fig, ax = BerkeleyPlots.setup_physics_plot(
            title=title, xlabel="Epoch", ylabel="Loss"
        )
        
        # Training loss
        ax.semilogy(epochs, train_loss, color=BerkeleyColors.BERKELEY_BLUE['hex'],
                   linewidth=2.5, label='Training Loss', marker='o', markersize=4)
        
        # Validation loss if provided
        if val_loss is not None:
            ax.semilogy(epochs, val_loss, color=BerkeleyColors.CALIFORNIA_GOLD['hex'],
                       linewidth=2.5, label='Validation Loss', marker='s', markersize=4)
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        BerkeleyPlots.add_berkeley_watermark(ax)
        
        return fig, ax

# =============================================================================
# Export Functions
# =============================================================================

def save_berkeley_figure(fig, filename: str, dpi: int = 300, 
                         bbox_inches: str = 'tight', 
                         facecolor: str = 'white',
                         edgecolor: str = 'none',
                         format: str = 'png'):
    """Save figure with Berkeley formatting standards."""
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                facecolor=facecolor, edgecolor=edgecolor, format=format)

def export_berkeley_style_config(filename: str = 'berkeley_mpl_style.mplstyle'):
    """Export Berkeley matplotlib style as .mplstyle file."""
    style_dict = BerkeleyPlotStyle.get_style_dict()
    
    with open(filename, 'w') as f:
        f.write("# Berkeley SciComp Matplotlib Style\n")
        f.write("# UC Berkeley Visual Identity for Scientific Computing\n")
        f.write("# Author: Dr. Meshal Alawein\n\n")
        
        for key, value in style_dict.items():
            if isinstance(value, str):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")

# =============================================================================
# Demo and Testing Functions
# =============================================================================

def demo_berkeley_style():
    """Demonstrate Berkeley styling with sample plots."""
    # Apply Berkeley style
    BerkeleyPlotStyle.apply_style()
    
    # Create sample data
    x = np.linspace(0, 4*np.pi, 1000)
    y1 = np.exp(-x/4) * np.cos(x)
    y2 = np.exp(-x/4) * np.sin(x)
    
    # Create demo plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top plot: Oscillatory functions
    ax1.plot(x, y1, label='Damped Cosine', linewidth=2.5)
    ax1.plot(x, y2, label='Damped Sine', linewidth=2.5)
    ax1.set_title('Berkeley SciComp Style Demo', fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Heatmap
    X, Y = np.meshgrid(np.linspace(0, 4*np.pi, 50), np.linspace(0, 4*np.pi, 50))
    Z = np.sin(X) * np.cos(Y) * np.exp(-(X**2 + Y**2)/20)
    
    berkeley_cmap = BerkeleyPlotStyle.create_berkeley_colormap()
    im = ax2.imshow(Z, cmap=berkeley_cmap, extent=[0, 4*np.pi, 0, 4*np.pi], 
                    origin='lower', aspect='auto')
    ax2.set_title('Berkeley Colormap Demo')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Amplitude')
    
    # Add watermarks
    BerkeleyPlots.add_berkeley_watermark(ax1)
    BerkeleyPlots.add_berkeley_watermark(ax2)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Run demo
    demo_fig = demo_berkeley_style()
    save_berkeley_figure(demo_fig, 'berkeley_style_demo.png')
    
    # Export style configuration
    export_berkeley_style_config()
    
    print("Berkeley style demo completed!")
    print("Generated files:")
    print("  - berkeley_style_demo.png")
    print("  - berkeley_mpl_style.mplstyle")
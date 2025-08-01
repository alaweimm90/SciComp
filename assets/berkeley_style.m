function berkeley_style()
%BERKELEY_STYLE UC Berkeley Visual Identity for MATLAB
%
% This function sets up Berkeley SciComp visual styling for MATLAB
% figures and plots, implementing official UC Berkeley brand guidelines
% for consistent scientific visualization.
%
% Usage:
%   berkeley_style()  % Apply Berkeley styling to current and future figures
%   berkeley_style('demo')  % Run demonstration plots
%
% Author: Dr. Meshal Alawein (meshal@berkeley.edu)
% Institution: University of California, Berkeley
% Created: 2025
% License: MIT
%
% Copyright © 2025 Dr. Meshal Alawein — All rights reserved.

    if nargin == 0
        apply_berkeley_style();
    elseif nargin == 1 && strcmp(varargin{1}, 'demo')
        demo_berkeley_style();
    else
        error('Invalid arguments. Use berkeley_style() or berkeley_style(''demo'')');
    end
end

function apply_berkeley_style()
%APPLY_BERKELEY_STYLE Set Berkeley visual identity for MATLAB figures

    % UC Berkeley Official Colors
    colors = get_berkeley_colors();
    
    % Typography settings
    typography = get_berkeley_typography();
    
    % Set default figure properties
    set(groot, 'DefaultFigureColor', [1 1 1]);
    set(groot, 'DefaultFigurePaperType', 'usletter');
    set(groot, 'DefaultFigurePaperSize', [11 8.5]);
    set(groot, 'DefaultFigurePosition', [100 100 800 600]);
    set(groot, 'DefaultFigureRenderer', 'painters');
    
    % Set default axes properties
    set(groot, 'DefaultAxesBox', 'off');
    set(groot, 'DefaultAxesColor', [1 1 1]);
    set(groot, 'DefaultAxesXColor', colors.berkeley_blue_rgb);
    set(groot, 'DefaultAxesYColor', colors.berkeley_blue_rgb);
    set(groot, 'DefaultAxesZColor', colors.berkeley_blue_rgb);
    set(groot, 'DefaultAxesLineWidth', 1.5);
    set(groot, 'DefaultAxesFontName', typography.primary_font);
    set(groot, 'DefaultAxesFontSize', typography.body_size);
    set(groot, 'DefaultAxesTitleFontSize', typography.title_size);
    set(groot, 'DefaultAxesLabelFontSize', typography.body_size);
    set(groot, 'DefaultAxesTickLength', [0.01 0.01]);
    
    % Set default line properties
    set(groot, 'DefaultLineLineWidth', 2.0);
    set(groot, 'DefaultLineMarkerSize', 8);
    
    % Set default text properties
    set(groot, 'DefaultTextFontName', typography.primary_font);
    set(groot, 'DefaultTextFontSize', typography.body_size);
    set(groot, 'DefaultTextColor', colors.berkeley_blue_rgb);
    
    % Set color order for multiple plots
    set(groot, 'DefaultAxesColorOrder', colors.extended_palette);
    
    % Grid settings
    set(groot, 'DefaultAxesXGrid', 'on');
    set(groot, 'DefaultAxesYGrid', 'on');
    set(groot, 'DefaultAxesZGrid', 'on');
    set(groot, 'DefaultAxesGridColor', colors.light_grey_rgb);
    set(groot, 'DefaultAxesGridAlpha', 0.7);
    set(groot, 'DefaultAxesMinorGridColor', colors.light_grey_rgb);
    set(groot, 'DefaultAxesMinorGridAlpha', 0.5);
    
    fprintf('Berkeley SciComp visual style applied to MATLAB\n');
    fprintf('Colors: Berkeley Blue, California Gold, and extended palette\n');
    fprintf('Typography: %s font family\n', typography.primary_font);
end

function colors = get_berkeley_colors()
%GET_BERKELEY_COLORS Return UC Berkeley official color palette

    % Primary Brand Colors (RGB normalized 0-1)
    colors.berkeley_blue_rgb = [0.0000, 0.1961, 0.3843];
    colors.california_gold_rgb = [0.9922, 0.7098, 0.0824];
    
    % Secondary Colors
    colors.founders_rock_rgb = [0.2314, 0.4941, 0.6314];
    colors.medalist_rgb = [0.7255, 0.5922, 0.3569];
    
    % Neutral Colors
    colors.berkeley_grey_rgb = [0.4000, 0.4000, 0.4000];
    colors.light_grey_rgb = [0.8000, 0.8000, 0.8000];
    
    % Accent Colors
    colors.pacific_blue_rgb = [0.2745, 0.3255, 0.3686];
    colors.lawrence_blue_rgb = [0.0000, 0.6902, 0.8549];
    colors.golden_gate_orange_rgb = [0.9294, 0.3059, 0.2000];
    colors.bay_teal_rgb = [0.0000, 0.6471, 0.5961];
    
    % Hex codes for reference
    colors.berkeley_blue_hex = '#003262';
    colors.california_gold_hex = '#FDB515';
    colors.founders_rock_hex = '#3B7EA1';
    colors.medalist_hex = '#B9975B';
    
    % Extended palette for multiple series
    colors.extended_palette = [
        colors.berkeley_blue_rgb;
        colors.california_gold_rgb;
        colors.founders_rock_rgb;
        colors.medalist_rgb;
        colors.lawrence_blue_rgb;
        colors.bay_teal_rgb;
        colors.golden_gate_orange_rgb;
        colors.pacific_blue_rgb
    ];
    
    % Primary palette
    colors.primary_palette = [
        colors.berkeley_blue_rgb;
        colors.california_gold_rgb
    ];
end

function typography = get_berkeley_typography()
%GET_BERKELEY_TYPOGRAPHY Return Berkeley typography specifications

    typography.primary_font = 'Arial';  % Available on most systems
    typography.math_font = 'Times';     % For mathematical expressions
    typography.code_font = 'Courier';   % For code blocks
    
    % Font sizes
    typography.title_size = 16;
    typography.subtitle_size = 14;
    typography.body_size = 12;
    typography.caption_size = 10;
    typography.small_size = 8;
    
    % Line spacing
    typography.line_spacing = 1.2;
    
    % Figure settings
    typography.figure_dpi = 300;
    typography.figure_width = 10;   % inches
    typography.figure_height = 6;   % inches
end

function fig = setup_berkeley_figure(title_str, xlabel_str, ylabel_str)
%SETUP_BERKELEY_FIGURE Create Berkeley-styled figure with labels
%
% Inputs:
%   title_str  - Figure title (optional)
%   xlabel_str - X-axis label (optional)  
%   ylabel_str - Y-axis label (optional)
%
% Output:
%   fig - Figure handle

    if nargin < 1, title_str = ''; end
    if nargin < 2, xlabel_str = ''; end
    if nargin < 3, ylabel_str = ''; end
    
    colors = get_berkeley_colors();
    typography = get_berkeley_typography();
    
    fig = figure('Color', [1 1 1]);
    ax = gca;
    
    % Set title
    if ~isempty(title_str)
        title(title_str, 'FontSize', typography.title_size, ...
              'FontWeight', 'bold', 'Color', colors.berkeley_blue_rgb);
    end
    
    % Set axis labels
    if ~isempty(xlabel_str)
        xlabel(xlabel_str, 'FontSize', typography.body_size, ...
               'Color', colors.berkeley_blue_rgb);
    end
    
    if ~isempty(ylabel_str)
        ylabel(ylabel_str, 'FontSize', typography.body_size, ...
               'Color', colors.berkeley_blue_rgb);
    end
    
    % Configure axes
    set(ax, 'Box', 'off');
    set(ax, 'XColor', colors.berkeley_blue_rgb);
    set(ax, 'YColor', colors.berkeley_blue_rgb);
    set(ax, 'LineWidth', 1.5);
    set(ax, 'FontName', typography.primary_font);
    set(ax, 'FontSize', typography.body_size);
    
    % Enable grid
    grid on;
    set(ax, 'GridColor', colors.light_grey_rgb);
    set(ax, 'GridAlpha', 0.7);
    
    % Remove top and right spines
    ax.XAxis.TickDirection = 'out';
    ax.YAxis.TickDirection = 'out';
end

function add_berkeley_watermark(ax, position, alpha)
%ADD_BERKELEY_WATERMARK Add Berkeley SciComp watermark to plot
%
% Inputs:
%   ax       - Axes handle
%   position - 'bottom_right', 'top_right', 'bottom_left', 'top_left'
%   alpha    - Transparency (0-1)

    if nargin < 2, position = 'bottom_right'; end
    if nargin < 3, alpha = 0.3; end
    
    colors = get_berkeley_colors();
    typography = get_berkeley_typography();
    
    watermark_text = 'Berkeley SciComp';
    
    switch position
        case 'bottom_right'
            x = 0.98; y = 0.02;
            ha = 'right'; va = 'bottom';
        case 'top_right'
            x = 0.98; y = 0.98;
            ha = 'right'; va = 'top';
        case 'bottom_left'
            x = 0.02; y = 0.02;
            ha = 'left'; va = 'bottom';
        case 'top_left'
            x = 0.02; y = 0.98;
            ha = 'left'; va = 'top';
        otherwise
            x = 0.98; y = 0.02;
            ha = 'right'; va = 'bottom';
    end
    
    text(ax, x, y, watermark_text, 'Units', 'normalized', ...
         'FontSize', typography.small_size, ...
         'Color', colors.berkeley_grey_rgb, ...
         'HorizontalAlignment', ha, ...
         'VerticalAlignment', va, ...
         'FontStyle', 'italic', ...
         'Alpha', alpha);
end

function fig = quantum_wavefunction_plot(x, psi, title_str, potential)
%QUANTUM_WAVEFUNCTION_PLOT Create Berkeley-styled quantum wavefunction plot
%
% Inputs:
%   x         - Position array
%   psi       - Wavefunction (can be complex)
%   title_str - Plot title (optional)
%   potential - Potential function (optional)

    if nargin < 3, title_str = 'Quantum Wavefunction'; end
    if nargin < 4, potential = []; end
    
    colors = get_berkeley_colors();
    
    fig = setup_berkeley_figure(title_str, 'Position (x)', 'Wavefunction ψ(x)');
    ax = gca;
    
    % Plot real part
    plot(x, real(psi), 'Color', colors.berkeley_blue_rgb, ...
         'LineWidth', 2.5, 'DisplayName', 'Re[ψ(x)]');
    hold on;
    
    % Plot imaginary part if complex
    if any(imag(psi) ~= 0)
        plot(x, imag(psi), 'Color', colors.california_gold_rgb, ...
             'LineWidth', 2.5, 'LineStyle', '--', 'DisplayName', 'Im[ψ(x)]');
    end
    
    % Plot probability density
    plot(x, abs(psi).^2, 'Color', colors.founders_rock_rgb, ...
         'LineWidth', 2.0, 'DisplayName', '|ψ(x)|²');
    
    % Add potential if provided
    if ~isempty(potential)
        yyaxis right;
        plot(x, potential, 'Color', colors.medalist_rgb, ...
             'LineWidth', 1.5, 'DisplayName', 'V(x)');
        ylabel('Potential V(x)', 'Color', colors.medalist_rgb);
        ax.YAxis(2).Color = colors.medalist_rgb;
        yyaxis left;
    end
    
    legend('Location', 'best');
    add_berkeley_watermark(ax);
    hold off;
end

function fig = heat_transfer_plot(x, y, temperature, title_str)
%HEAT_TRANSFER_PLOT Create Berkeley-styled heat transfer contour plot
%
% Inputs:
%   x           - X coordinates
%   y           - Y coordinates  
%   temperature - Temperature field
%   title_str   - Plot title (optional)

    if nargin < 4, title_str = 'Heat Transfer Analysis'; end
    
    colors = get_berkeley_colors();
    
    fig = setup_berkeley_figure(title_str, 'x (m)', 'y (m)');
    ax = gca;
    
    % Create Berkeley colormap
    berkeley_cmap = create_berkeley_colormap();
    
    % Contour plot
    [C, h] = contourf(x, y, temperature, 20);
    colormap(berkeley_cmap);
    
    % Add contour lines
    hold on;
    contour(x, y, temperature, 10, 'Color', colors.berkeley_blue_rgb, ...
            'LineWidth', 0.8);
    hold off;
    
    % Colorbar
    cb = colorbar;
    ylabel(cb, 'Temperature (°C)', 'Color', colors.berkeley_blue_rgb);
    set(cb, 'TickLabelColor', colors.berkeley_blue_rgb);
    
    axis equal;
    add_berkeley_watermark(ax);
end

function fig = ml_training_plot(epochs, train_loss, val_loss, title_str)
%ML_TRAINING_PLOT Create Berkeley-styled ML training progress plot
%
% Inputs:
%   epochs     - Epoch numbers
%   train_loss - Training loss values
%   val_loss   - Validation loss values (optional)
%   title_str  - Plot title (optional)

    if nargin < 3, val_loss = []; end
    if nargin < 4, title_str = 'ML Physics Training Progress'; end
    
    colors = get_berkeley_colors();
    
    fig = setup_berkeley_figure(title_str, 'Epoch', 'Loss');
    ax = gca;
    
    % Training loss
    semilogy(epochs, train_loss, 'o-', 'Color', colors.berkeley_blue_rgb, ...
             'LineWidth', 2.5, 'MarkerSize', 4, 'DisplayName', 'Training Loss');
    hold on;
    
    % Validation loss if provided
    if ~isempty(val_loss)
        semilogy(epochs, val_loss, 's-', 'Color', colors.california_gold_rgb, ...
                 'LineWidth', 2.5, 'MarkerSize', 4, 'DisplayName', 'Validation Loss');
    end
    
    legend('Location', 'best');
    add_berkeley_watermark(ax);
    hold off;
end

function cmap = create_berkeley_colormap(n_colors)
%CREATE_BERKELEY_COLORMAP Create Berkeley-themed colormap
%
% Input:
%   n_colors - Number of colors in colormap (default: 256)
%
% Output:
%   cmap - Colormap matrix

    if nargin < 1, n_colors = 256; end
    
    colors = get_berkeley_colors();
    
    % Define key colors for interpolation
    key_colors = [
        colors.berkeley_blue_rgb;
        colors.founders_rock_rgb;
        colors.bay_teal_rgb;
        colors.california_gold_rgb;
        colors.golden_gate_orange_rgb
    ];
    
    % Interpolate between key colors
    n_keys = size(key_colors, 1);
    x_keys = linspace(1, n_colors, n_keys);
    x_interp = 1:n_colors;
    
    cmap = zeros(n_colors, 3);
    for i = 1:3
        cmap(:, i) = interp1(x_keys, key_colors(:, i), x_interp, 'linear');
    end
    
    % Ensure values are in [0, 1]
    cmap = max(0, min(1, cmap));
end

function save_berkeley_figure(fig, filename, dpi, format)
%SAVE_BERKELEY_FIGURE Save figure with Berkeley formatting standards
%
% Inputs:
%   fig      - Figure handle
%   filename - Output filename
%   dpi      - Resolution (default: 300)
%   format   - File format (default: 'png')

    if nargin < 3, dpi = 300; end
    if nargin < 4, format = 'png'; end
    
    % Set paper properties for consistent output
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperPosition', [0 0 10 6]);
    set(fig, 'PaperSize', [10 6]);
    
    % Save with specified parameters
    print(fig, filename, ['-d' format], ['-r' num2str(dpi)]);
    
    fprintf('Figure saved: %s (DPI: %d, Format: %s)\n', filename, dpi, format);
end

function demo_berkeley_style()
%DEMO_BERKELEY_STYLE Demonstrate Berkeley styling with sample plots

    fprintf('Running Berkeley SciComp style demonstration...\n');
    
    % Apply Berkeley style
    apply_berkeley_style();
    
    colors = get_berkeley_colors();
    
    % Create sample data
    x = linspace(0, 4*pi, 1000);
    y1 = exp(-x/4) .* cos(x);
    y2 = exp(-x/4) .* sin(x);
    
    % Demo 1: Line plot
    fig1 = setup_berkeley_figure('Berkeley SciComp Style Demo', 'x', 'Amplitude');
    plot(x, y1, 'LineWidth', 2.5, 'DisplayName', 'Damped Cosine');
    hold on;
    plot(x, y2, 'LineWidth', 2.5, 'DisplayName', 'Damped Sine');
    legend('Location', 'best');
    add_berkeley_watermark(gca);
    
    % Demo 2: Surface plot with Berkeley colormap
    [X, Y] = meshgrid(linspace(0, 4*pi, 50), linspace(0, 4*pi, 50));
    Z = sin(X) .* cos(Y) .* exp(-(X.^2 + Y.^2)/20);
    
    fig2 = setup_berkeley_figure('Berkeley Colormap Demo', 'x', 'y');
    surf(X, Y, Z, 'EdgeColor', 'none');
    colormap(create_berkeley_colormap());
    cb = colorbar;
    ylabel(cb, 'Amplitude');
    view(45, 30);
    add_berkeley_watermark(gca);
    
    % Demo 3: Quantum wavefunction example
    x_wave = linspace(-5, 5, 1000);
    psi = exp(-x_wave.^2/2) .* exp(1i*x_wave);  % Gaussian wave packet
    potential = 0.1 * x_wave.^2;  % Harmonic potential
    
    fig3 = quantum_wavefunction_plot(x_wave, psi, 'Quantum Wave Packet', potential);
    
    % Demo 4: Heat transfer example
    [X_heat, Y_heat] = meshgrid(linspace(0, 1, 50), linspace(0, 1, 50));
    T = 100 * (1 - X_heat) .* (1 - Y_heat) + 20;  % Simple temperature field
    
    fig4 = heat_transfer_plot(X_heat, Y_heat, T, 'Heat Transfer Simulation');
    
    % Demo 5: ML training plot
    epochs = 1:100;
    train_loss = 1 ./ (1 + 0.1 * epochs) + 0.01 * randn(size(epochs));
    val_loss = 1 ./ (1 + 0.08 * epochs) + 0.02 * randn(size(epochs));
    
    fig5 = ml_training_plot(epochs, train_loss, val_loss, 'PINN Training Progress');
    
    % Save demonstration figures
    save_berkeley_figure(fig1, 'berkeley_demo_1_lineplot.png');
    save_berkeley_figure(fig2, 'berkeley_demo_2_surface.png');
    save_berkeley_figure(fig3, 'berkeley_demo_3_quantum.png');
    save_berkeley_figure(fig4, 'berkeley_demo_4_heattransfer.png');
    save_berkeley_figure(fig5, 'berkeley_demo_5_ml.png');
    
    fprintf('Berkeley style demonstration completed!\n');
    fprintf('Generated demonstration plots:\n');
    fprintf('  - berkeley_demo_1_lineplot.png\n');
    fprintf('  - berkeley_demo_2_surface.png\n');
    fprintf('  - berkeley_demo_3_quantum.png\n');
    fprintf('  - berkeley_demo_4_heattransfer.png\n');
    fprintf('  - berkeley_demo_5_ml.png\n');
end
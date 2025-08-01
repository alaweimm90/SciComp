(* ::Package:: *)

(* ::Title:: *)
(*Berkeley SciComp Visual Identity for Mathematica*)

(* ::Subtitle:: *)
(*UC Berkeley Official Styling Package*)

(* ::Text:: *)
(*Comprehensive visual identity package implementing UC Berkeley brand guidelines for scientific computing and visualization in Mathematica.*)
(**)
(*Author: Dr. Meshal Alawein (meshal@berkeley.edu)*)
(*Institution: University of California, Berkeley*)
(*Created: 2025*)
(*License: MIT*)
(**)
(*Copyright \[Copyright] 2025 Dr. Meshal Alawein \[LongDash] All rights reserved.*)

BeginPackage["BerkeleyStyle`"]

(* ::Section:: *)
(*Package Declarations*)

(* Color Definitions *)
BerkeleyBlue::usage = "UC Berkeley official blue color (#003262)";
CaliforniaGold::usage = "UC Berkeley official gold color (#FDB515)";
FoundersRock::usage = "UC Berkeley secondary blue color (#3B7EA1)";
Medalist::usage = "UC Berkeley warm accent color (#B9975B)";
BerkeleyGrey::usage = "UC Berkeley neutral grey color (#666666)";
PacificBlue::usage = "UC Berkeley dark accent color (#46535E)";
LawrenceBlue::usage = "UC Berkeley bright blue accent (#00B0DA)";
GoldenGateOrange::usage = "UC Berkeley orange accent (#ED4E33)";
BayTeal::usage = "UC Berkeley teal accent (#00A598)";

(* Color Palettes *)
BerkeleyPrimaryPalette::usage = "Primary Berkeley color palette";
BerkeleyExtendedPalette::usage = "Extended Berkeley color palette for complex visualizations";
BerkeleyGradientColors::usage = "Generate gradient colors between Berkeley Blue and California Gold";

(* Styling Functions *)
BerkeleyPlotTheme::usage = "Apply Berkeley visual identity to plots";
BerkeleyColorScheme::usage = "Berkeley color scheme for data visualization";
BerkeleyTypography::usage = "Berkeley typography specifications";

(* Plot Creation Functions *)
SetupBerkeleyPlot::usage = "Create Berkeley-styled plot with proper formatting";
QuantumWavefunctionPlot::usage = "Berkeley-styled quantum wavefunction visualization";
HeatTransferPlot::usage = "Berkeley-styled heat transfer contour plot";
MLTrainingPlot::usage = "Berkeley-styled machine learning training progress plot";

(* Utility Functions *)
AddBerkeleyWatermark::usage = "Add Berkeley SciComp watermark to graphics";
SaveBerkeleyFigure::usage = "Save figure with Berkeley formatting standards";
BerkeleyStyleDemo::usage = "Demonstrate Berkeley styling capabilities";

Begin["`Private`"]

(* ::Section:: *)
(*UC Berkeley Official Color Definitions*)

(* Primary Brand Colors *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];        (* #003262 *)
CaliforniaGold = RGBColor[253/255, 181/255, 21/255]; (* #FDB515 *)

(* Secondary Colors *)
FoundersRock = RGBColor[59/255, 126/255, 161/255];   (* #3B7EA1 *)
Medalist = RGBColor[185/255, 151/255, 91/255];       (* #B9975B *)

(* Neutral Colors *)
BerkeleyGrey = RGBColor[102/255, 102/255, 102/255];  (* #666666 *)
LightGrey = RGBColor[204/255, 204/255, 204/255];     (* #CCCCCC *)

(* Accent Colors *)
PacificBlue = RGBColor[70/255, 83/255, 94/255];      (* #46535E *)
LawrenceBlue = RGBColor[0, 176/255, 218/255];        (* #00B0DA *)
GoldenGateOrange = RGBColor[237/255, 78/255, 51/255]; (* #ED4E33 *)
BayTeal = RGBColor[0, 165/255, 152/255];             (* #00A598 *)

(* ::Section:: *)
(*Color Palettes*)

BerkeleyPrimaryPalette = {BerkeleyBlue, CaliforniaGold};

BerkeleyExtendedPalette = {
  BerkeleyBlue, CaliforniaGold, FoundersRock, Medalist,
  LawrenceBlue, BayTeal, GoldenGateOrange, PacificBlue
};

BerkeleyGradientColors[n_Integer: 10] := 
  Table[Blend[{BerkeleyBlue, CaliforniaGold}, i/(n-1)], {i, 0, n-1}];

(* ::Section:: *)
(*Typography Specifications*)

BerkeleyTypography = <|
  "PrimaryFont" -> "Arial",
  "MathFont" -> "Times",
  "CodeFont" -> "Courier",
  "TitleSize" -> 16,
  "SubtitleSize" -> 14,
  "BodySize" -> 12,
  "CaptionSize" -> 10,
  "SmallSize" -> 8,
  "LineSpacing" -> 1.2,
  "FigureWidth" -> 500,
  "FigureHeight" -> 300
|>;

(* ::Section:: *)
(*Berkeley Plot Theme*)

BerkeleyPlotTheme = {
  (* Background and Frame *)
  Background -> White,
  Frame -> True,
  FrameStyle -> Directive[Thickness[0.002], BerkeleyBlue],
  GridLines -> Automatic,
  GridLinesStyle -> Directive[LightGrey, Opacity[0.7]],
  
  (* Colors *)
  PlotStyle -> BerkeleyExtendedPalette,
  
  (* Typography *)
  BaseStyle -> {
    FontFamily -> BerkeleyTypography["PrimaryFont"],
    FontSize -> BerkeleyTypography["BodySize"],
    FontColor -> BerkeleyBlue
  },
  
  LabelStyle -> {
    FontFamily -> BerkeleyTypography["PrimaryFont"],
    FontSize -> BerkeleyTypography["BodySize"],
    FontColor -> BerkeleyBlue
  },
  
  (* Axes *)
  AxesStyle -> Directive[Thickness[0.002], BerkeleyBlue],
  TicksStyle -> Directive[BerkeleyBlue, FontSize -> BerkeleyTypography["CaptionSize"]],
  
  (* Image size *)
  ImageSize -> {BerkeleyTypography["FigureWidth"], BerkeleyTypography["FigureHeight"]},
  
  (* Plot markers *)
  PlotMarkers -> Automatic
};

(* ::Section:: *)
(*Berkeley Color Scheme*)

BerkeleyColorScheme[data_] := ColorData[{"BlendedColors", BerkeleyExtendedPalette}][data];

(* Create Berkeley colormap *)
BerkeleyColorFunction = Blend[{BerkeleyBlue, FoundersRock, BayTeal, CaliforniaGold, GoldenGateOrange}, #] &;

(* ::Section:: *)
(*Plot Setup Functions*)

SetupBerkeleyPlot[title_String: "", xlabel_String: "", ylabel_String: "", opts___] := 
  Module[{plotOptions},
    plotOptions = {
      PlotLabel -> Style[title, FontSize -> BerkeleyTypography["TitleSize"], 
                        FontWeight -> Bold, FontColor -> BerkeleyBlue],
      FrameLabel -> {{Style[ylabel, FontColor -> BerkeleyBlue], None}, 
                     {Style[xlabel, FontColor -> BerkeleyBlue], None}},
      Sequence@@BerkeleyPlotTheme,
      opts
    };
    plotOptions
  ];

(* ::Section:: *)
(*Specialized Plot Functions*)

QuantumWavefunctionPlot[x_List, psi_List, opts___Rule] := 
  Module[{realPart, imagPart, probDensity, plots, potential, title},
    
    (* Extract options *)
    title = "title" /. {opts} /. "title" -> "Quantum Wavefunction";
    potential = "potential" /. {opts} /. "potential" -> None;
    
    (* Calculate components *)
    realPart = Re[psi];
    imagPart = Im[psi];
    probDensity = Abs[psi]^2;
    
    (* Create plots *)
    plots = {
      ListLinePlot[Transpose[{x, realPart}], 
        PlotStyle -> Directive[BerkeleyBlue, Thickness[0.003]],
        PlotLegends -> {"Re[\[Psi](x)]"}],
      
      If[Max[Abs[imagPart]] > 10^(-10),
        ListLinePlot[Transpose[{x, imagPart}], 
          PlotStyle -> Directive[CaliforniaGold, Thickness[0.003], Dashed],
          PlotLegends -> {"Im[\[Psi](x)]"}],
        Nothing
      ],
      
      ListLinePlot[Transpose[{x, probDensity}], 
        PlotStyle -> Directive[FoundersRock, Thickness[0.002]],
        PlotLegends -> {"|\[Psi](x)|\[TwoSuperscript]"}]
    };
    
    (* Add potential if provided *)
    If[potential =!= None,
      AppendTo[plots, 
        ListLinePlot[Transpose[{x, potential}], 
          PlotStyle -> Directive[Medalist, Thickness[0.002]],
          PlotLegends -> {"V(x)"}]]
    ];
    
    (* Combine plots *)
    Show[plots,
      Sequence@@SetupBerkeleyPlot[title, "Position (x)", "Wavefunction \[Psi](x)"],
      PlotRange -> All,
      PlotLegends -> Automatic
    ]
  ];

HeatTransferPlot[x_List, y_List, temperature_List, opts___Rule] := 
  Module[{title, contourPlot, colorbar},
    
    title = "title" /. {opts} /. "title" -> "Heat Transfer Analysis";
    
    contourPlot = ContourPlot[
      Interpolation[Flatten[Table[{x[[i]], y[[j]], temperature[[j, i]]}, 
        {i, Length[x]}, {j, Length[y]}], 1]][u, v],
      {u, Min[x], Max[x]}, {v, Min[y], Max[y]},
      ColorFunction -> BerkeleyColorFunction,
      Contours -> 20,
      ContourLines -> True,
      ContourStyle -> Directive[BerkeleyBlue, Opacity[0.6], Thickness[0.001]],
      Sequence@@SetupBerkeleyPlot[title, "x (m)", "y (m)"],
      PlotLegends -> Automatic,
      AspectRatio -> Automatic
    ];
    
    contourPlot
  ];

MLTrainingPlot[epochs_List, trainLoss_List, opts___Rule] := 
  Module[{valLoss, title, plots},
    
    valLoss = "validationLoss" /. {opts} /. "validationLoss" -> None;
    title = "title" /. {opts} /. "title" -> "ML Physics Training Progress";
    
    plots = {
      ListLogPlot[Transpose[{epochs, trainLoss}],
        PlotStyle -> Directive[BerkeleyBlue, Thickness[0.003]],
        PlotMarkers -> {Automatic, 8},
        PlotLegends -> {"Training Loss"}]
    };
    
    If[valLoss =!= None,
      AppendTo[plots,
        ListLogPlot[Transpose[{epochs, valLoss}],
          PlotStyle -> Directive[CaliforniaGold, Thickness[0.003]],
          PlotMarkers -> {Automatic, 8},
          PlotLegends -> {"Validation Loss"}]]
    ];
    
    Show[plots,
      Sequence@@SetupBerkeleyPlot[title, "Epoch", "Loss"],
      PlotRange -> All,
      PlotLegends -> Automatic
    ]
  ];

(* ::Section:: *)
(*Utility Functions*)

AddBerkeleyWatermark[graphics_, position_String: "BottomRight", alpha_Real: 0.3] := 
  Module[{watermarkText, textPosition, alignment},
    
    watermarkText = "Berkeley SciComp";
    
    {textPosition, alignment} = Switch[position,
      "BottomRight", {{0.95, 0.05}, {Right, Bottom}},
      "TopRight", {{0.95, 0.95}, {Right, Top}},
      "BottomLeft", {{0.05, 0.05}, {Left, Bottom}},
      "TopLeft", {{0.05, 0.95}, {Left, Top}},
      _, {{0.95, 0.05}, {Right, Bottom}}
    ];
    
    Show[graphics,
      Epilog -> {
        Opacity[alpha],
        BerkeleyGrey,
        Text[Style[watermarkText, 
          FontSize -> BerkeleyTypography["SmallSize"],
          FontSlant -> Italic,
          FontFamily -> BerkeleyTypography["PrimaryFont"]], 
          Scaled[textPosition], alignment]
      }
    ]
  ];

SaveBerkeleyFigure[graphics_, filename_String, opts___Rule] := 
  Module[{dpi, format, exportOptions},
    
    dpi = "DPI" /. {opts} /. "DPI" -> 300;
    format = "Format" /. {opts} /. "Format" -> "PNG";
    
    exportOptions = {
      "Resolution" -> dpi,
      Background -> White,
      ImageSize -> {BerkeleyTypography["FigureWidth"], BerkeleyTypography["FigureHeight"]}
    };
    
    Export[filename, graphics, format, Sequence@@exportOptions];
    Print["Figure saved: " <> filename <> " (DPI: " <> ToString[dpi] <> ", Format: " <> format <> ")"];
  ];

(* ::Section:: *)
(*Demonstration Function*)

BerkeleyStyleDemo[] := 
  Module[{x, y1, y2, X, Y, Z, xWave, psi, potential, XHeat, YHeat, T, 
          epochs, trainLoss, valLoss, fig1, fig2, fig3, fig4, fig5},
    
    Print["Running Berkeley SciComp style demonstration..."];
    
    (* Sample data *)
    x = Range[0, 4*Pi, 4*Pi/999];
    y1 = Exp[-x/4]*Cos[x];
    y2 = Exp[-x/4]*Sin[x];
    
    (* Demo 1: Basic line plot *)
    fig1 = Plot[{Exp[-x/4]*Cos[x], Exp[-x/4]*Sin[x]}, {x, 0, 4*Pi},
      Sequence@@SetupBerkeleyPlot["Berkeley SciComp Style Demo", "x", "Amplitude"],
      PlotLegends -> {"Damped Cosine", "Damped Sine"}
    ];
    fig1 = AddBerkeleyWatermark[fig1];
    
    (* Demo 2: 3D surface plot *)
    fig2 = Plot3D[Sin[x]*Cos[y]*Exp[-(x^2 + y^2)/20], {x, 0, 4*Pi}, {y, 0, 4*Pi},
      ColorFunction -> BerkeleyColorFunction,
      Sequence@@SetupBerkeleyPlot["Berkeley 3D Surface Demo", "x", "y"],
      PlotPoints -> 50,
      Mesh -> None
    ];
    fig2 = AddBerkeleyWatermark[fig2];
    
    (* Demo 3: Quantum wavefunction *)
    xWave = Range[-5, 5, 10/999];
    psi = Exp[-xWave^2/2]*Exp[I*xWave]; (* Gaussian wave packet *)
    potential = 0.1*xWave^2; (* Harmonic potential *)
    
    fig3 = QuantumWavefunctionPlot[xWave, psi, 
      "title" -> "Quantum Wave Packet",
      "potential" -> potential
    ];
    fig3 = AddBerkeleyWatermark[fig3];
    
    (* Demo 4: Heat transfer contour *)
    XHeat = Table[i/49, {i, 0, 49}];
    YHeat = Table[j/49, {j, 0, 49}];
    T = Table[100*(1 - XHeat[[i]])*(1 - YHeat[[j]]) + 20, {j, 50}, {i, 50}];
    
    fig4 = HeatTransferPlot[XHeat, YHeat, T, 
      "title" -> "Heat Transfer Simulation"
    ];
    fig4 = AddBerkeleyWatermark[fig4];
    
    (* Demo 5: ML training plot *)
    epochs = Range[1, 100];
    trainLoss = 1/(1 + 0.1*epochs) + 0.01*RandomReal[NormalDistribution[], 100];
    valLoss = 1/(1 + 0.08*epochs) + 0.02*RandomReal[NormalDistribution[], 100];
    
    fig5 = MLTrainingPlot[epochs, trainLoss, 
      "validationLoss" -> valLoss,
      "title" -> "PINN Training Progress"
    ];
    fig5 = AddBerkeleyWatermark[fig5];
    
    (* Save demonstration figures *)
    SaveBerkeleyFigure[fig1, "berkeley_demo_1_lineplot.png"];
    SaveBerkeleyFigure[fig2, "berkeley_demo_2_surface.png"];
    SaveBerkeleyFigure[fig3, "berkeley_demo_3_quantum.png"];
    SaveBerkeleyFigure[fig4, "berkeley_demo_4_heattransfer.png"];
    SaveBerkeleyFigure[fig5, "berkeley_demo_5_ml.png"];
    
    Print["Berkeley style demonstration completed!"];
    Print["Generated demonstration plots:"];
    Print["  - berkeley_demo_1_lineplot.png"];
    Print["  - berkeley_demo_2_surface.png"];
    Print["  - berkeley_demo_3_quantum.png"];
    Print["  - berkeley_demo_4_heattransfer.png"];
    Print["  - berkeley_demo_5_ml.png"];
    
    (* Return graphics for display *)
    {fig1, fig2, fig3, fig4, fig5}
  ];

(* ::Section:: *)
(*Color Analysis Functions*)

BerkeleyColorAnalysis[] := 
  Module[{colorGrid, gradientBar, paletteComparison},
    
    (* Color grid showing all Berkeley colors *)
    colorGrid = Graphics[{
      (* Primary colors *)
      BerkeleyBlue, Rectangle[{0, 3}, {1, 4}],
      CaliforniaGold, Rectangle[{1, 3}, {2, 4}],
      
      (* Secondary colors *)
      FoundersRock, Rectangle[{0, 2}, {1, 3}],
      Medalist, Rectangle[{1, 2}, {2, 3}],
      
      (* Accent colors *)
      LawrenceBlue, Rectangle[{0, 1}, {1, 2}],
      BayTeal, Rectangle[{1, 1}, {2, 2}],
      GoldenGateOrange, Rectangle[{0, 0}, {1, 1}],
      PacificBlue, Rectangle[{1, 0}, {2, 1}],
      
      (* Labels *)
      Black,
      Text["Berkeley Blue", {0.5, 3.5}],
      Text["California Gold", {1.5, 3.5}],
      Text["Founders Rock", {0.5, 2.5}],
      Text["Medalist", {1.5, 2.5}],
      Text["Lawrence Blue", {0.5, 1.5}],
      Text["Bay Teal", {1.5, 1.5}],
      Text["Golden Gate Orange", {0.5, 0.5}],
      Text["Pacific Blue", {1.5, 0.5}]
    },
    PlotLabel -> Style["UC Berkeley Official Color Palette", 
      FontSize -> BerkeleyTypography["TitleSize"], FontWeight -> Bold],
    ImageSize -> {400, 300}
    ];
    
    (* Gradient demonstration *)
    gradientBar = Graphics[
      Table[{
        Blend[{BerkeleyBlue, CaliforniaGold}, i/99],
        Rectangle[{i, 0}, {i+1, 1}]
      }, {i, 0, 99}],
      PlotLabel -> Style["Berkeley Blue to California Gold Gradient", 
        FontSize -> BerkeleyTypography["SubtitleSize"]],
      ImageSize -> {400, 100}
    ];
    
    {colorGrid, gradientBar}
  ];

End[]
EndPackage[]

(* ::Section:: *)
(*Usage Examples*)

(* ::Text:: *)
(*To use this package, load it with:*)
(*<< "BerkeleyStyle`"*)
(**)
(*Then apply Berkeley styling to plots:*)
(*Plot[Sin[x], {x, 0, 2\[Pi]}, Evaluate[SetupBerkeleyPlot["Sine Function", "x", "sin(x)"]]]*)
(**)
(*Or run the demonstration:*)
(*BerkeleyStyleDemo[]*)

(* ::Text:: *)
(*Color usage examples:*)
(*Graphics[{BerkeleyBlue, Disk[]}]*)
(*Plot[x^2, {x, -2, 2}, PlotStyle -> CaliforniaGold]*)
(**)
(*The package provides comprehensive Berkeley visual identity for all Mathematica visualizations, ensuring consistency with UC Berkeley brand guidelines and professional scientific presentation standards.*)

Print["Berkeley SciComp visual identity package loaded successfully!"];
Print["Use BerkeleyStyleDemo[] to see demonstration plots."];
Print["Colors available: BerkeleyBlue, CaliforniaGold, FoundersRock, Medalist, and more."];
Print["Apply Berkeley styling with SetupBerkeleyPlot[] function."];
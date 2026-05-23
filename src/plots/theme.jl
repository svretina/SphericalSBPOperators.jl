const _APS_PUBLICATION_BG = RGBf(1.0, 1.0, 1.0)
const _APS_PUBLICATION_GRID = RGBAf(0.0, 0.0, 0.0, 0.1)
const _APS_PUBLICATION_ZERO = RGBAf(0.0, 0.0, 0.0, 0.35)

function mytheme_aps()
    return Theme(
                 ;
                 Axis = Attributes(;
                                   spinewidth = 1.1,
                                   xgridvisible = true,
                                   xlabelpadding = -0,
                                   xlabelsize = 12,
                                   xminortickalign = 1,
                                   xminorticks = IntervalsBetween(5, true),
                                   xminorticksize = 3,
                                   xminorticksvisible = true,
                                   xminortickwidth = 0.75,
                                   xtickalign = 1,
                                   xticklabelsize = 8,
                                   xticksize = 5,
                                   xticksmirrored = true,
                                   xtickwidth = 0.8,
                                   ygridvisible = true,
                                   ylabelpadding = 2,
                                   ylabelsize = 12,
                                   yminortickalign = 1,
                                   yminorticks = IntervalsBetween(5, true),
                                   yminorticksize = 3,
                                   yminorticksvisible = true,
                                   yminortickwidth = 0.75,
                                   ytickalign = 1,
                                   yticklabelsize = 10,
                                   yticksize = 5,
                                   yticksmirrored = true,
                                   ytickwidth = 0.8,
                                   xticklabelfont = "cmr10",
                                   yticklabelfont = "cmr10",
                                   xticklabelstyle = Attributes(; italic = false),
                                   yticklabelstyle = Attributes(; italic = false)
                                  ),
                 colgap = 8,
                 figure_padding = 0,
                 rowgap = 8,
                 size = (243, 165),
                 Colorbar = Attributes(;
                                       labelpadding = 2,
                                       labelsize = 10,
                                       minortickalign = 1,
                                       minorticksize = 3,
                                       minorticksvisible = true,
                                       minortickwidth = 0.75,
                                       size = 8,
                                       spinewidth = 1.1,
                                       tickalign = 1,
                                       ticklabelpad = 2,
                                       ticklabelsize = 8,
                                       ticksize = 5,
                                       tickwidth = 0.8),
                 fonts = Attributes(;
                                    bold = "NewComputerModern10 Bold",
                                    bold_italic = "NewComputerModern10 Bold Italic",
                                    italic = "NewComputerModern10 Italic",
                                    regular = "NewComputerModern Math Regular"),
                 Legend = Attributes(;
                                     colgap = 4,
                                     framecolor = (:grey, 0.5),
                                     framevisible = false,
                                     labelsize = 7.5,
                                     margin = (0, 0, 0, 0),
                                     nbanks = 1,
                                     padding = (2, 2, 2, 2),
                                     rowgap = -10),
                 Lines = Attributes(;
                                    cycle = Cycle([[:color] => :color], true)),
                 Scatter = Attributes(;
                                      cycle = Cycle([[:color] => :color, [:marker] => :marker], true),
                                      markersize = 7,
                                      strokewidth = 0),
                 markersize = 7,
                 palette = Attributes(;
                                      color = [RGBAf(0.298039, 0.447059, 0.690196, 1.0),
                                               RGBAf(0.866667, 0.517647, 0.321569, 1.0),
                                               RGBAf(0.333333, 0.658824, 0.407843, 1.0),
                                               RGBAf(0.768627, 0.305882, 0.321569, 1.0),
                                               RGBAf(0.505882, 0.447059, 0.701961, 1.0),
                                               RGBAf(0.576471, 0.470588, 0.376471, 1.0),
                                               RGBAf(0.854902, 0.545098, 0.764706, 1.0),
                                               RGBAf(0.54902, 0.54902, 0.54902, 1.0),
                                               RGBAf(0.8, 0.72549, 0.454902, 1.0),
                                               RGBAf(0.392157, 0.709804, 0.803922, 1.0)],
                                      linestyle = [nothing, :dash, :dot, :dashdot, :dashdotdot],
                                      marker = [:circle, :rect, :dtriangle, :utriangle, :cross,
                                                :diamond, :ltriangle, :rtriangle, :pentagon,
                                                :xcross, :hexagon],
                                      markercolor = [RGBAf(0.298039, 0.447059, 0.690196, 1.0),
                                                     RGBAf(0.866667, 0.517647, 0.321569, 1.0),
                                                     RGBAf(0.333333, 0.658824, 0.407843, 1.0),
                                                     RGBAf(0.768627, 0.305882, 0.321569, 1.0),
                                                     RGBAf(0.505882, 0.447059, 0.701961, 1.0),
                                                     RGBAf(0.576471, 0.470588, 0.376471, 1.0),
                                                     RGBAf(0.854902, 0.545098, 0.764706, 1.0),
                                                     RGBAf(0.54902, 0.54902, 0.54902, 1.0),
                                                     RGBAf(0.8, 0.72549, 0.454902, 1.0),
                                                     RGBAf(0.392157, 0.709804, 0.803922, 1.0)],
                                      patchcolor = [RGBAf(0.298039, 0.447059, 0.690196, 1.0),
                                                    RGBAf(0.866667, 0.517647, 0.321569, 1.0),
                                                    RGBAf(0.333333, 0.658824, 0.407843, 1.0),
                                                    RGBAf(0.768627, 0.305882, 0.321569, 1.0),
                                                    RGBAf(0.505882, 0.447059, 0.701961, 1.0),
                                                    RGBAf(0.576471, 0.470588, 0.376471, 1.0),
                                                    RGBAf(0.854902, 0.545098, 0.764706, 1.0),
                                                    RGBAf(0.54902, 0.54902, 0.54902, 1.0),
                                                    RGBAf(0.8, 0.72549, 0.454902, 1.0),
                                                    RGBAf(0.392157, 0.709804, 0.803922, 1.0)]
                                     ),
                 PolarAxis = Attributes(; spinewidth = 1.1))
end

function mytheme_aps_publication(;
                                 axis_labelsize::Real = 13,
                                 tick_labelsize::Real = 12,
                                 title_size::Real = 16,
                                 legend_labelsize::Real = 10,
                                 figure_padding = (8, 8, 6, 6),
                                 backgroundcolor = _APS_PUBLICATION_BG,
                                 gridcolor = _APS_PUBLICATION_GRID)
    _ = axis_labelsize
    _ = tick_labelsize
    _ = title_size
    _ = legend_labelsize
    _ = figure_padding
    _ = backgroundcolor
    _ = gridcolor
    return mytheme_aps()
end

function mytheme_aps_spectrum(;
                              axis_labelsize::Real = 13,
                              tick_labelsize::Real = 12,
                              title_size::Real = 16,
                              legend_labelsize::Real = 10,
                              patchsize = (30, 16),
                              legend_rowgap::Real = 2,
                              legend_colgap::Real = 12,
                              legend_margin = (0, 0, 0, 0),
                              legend_padding = (2, 2, 2, 2),
                              legend_framevisible::Bool = false,
                              figure_padding = (8, 8, 6, 6),
                              backgroundcolor = _APS_PUBLICATION_BG,
                              gridcolor = _APS_PUBLICATION_GRID)
    return Theme(
                 ;
                 figure_padding = figure_padding,
                 Figure = Attributes(; backgroundcolor = backgroundcolor),
                 fonts = Attributes(;
                                    bold = "NewComputerModern10 Bold",
                                    bold_italic = "NewComputerModern10 Bold Italic",
                                    italic = "NewComputerModern10 Italic",
                                    regular = "NewComputerModern Math Regular"),
                 Axis = Attributes(;
                                   backgroundcolor = backgroundcolor,
                                   spinewidth = 1.8,
                                   xgridvisible = true,
                                   ygridvisible = true,
                                   xgridcolor = gridcolor,
                                   ygridcolor = gridcolor,
                                   xgridwidth = 1.0,
                                   ygridwidth = 1.0,
                                   xlabelsize = axis_labelsize,
                                   ylabelsize = axis_labelsize,
                                   xticklabelsize = tick_labelsize,
                                   yticklabelsize = tick_labelsize,
                                   xticklabelfont = "cmr10",
                                   yticklabelfont = "cmr10",
                                   xticklabelstyle = Attributes(; italic = false),
                                   yticklabelstyle = Attributes(; italic = false),
                                   titlesize = title_size,
                                   xminorticksvisible = true,
                                   yminorticksvisible = true,
                                   xminorticks = IntervalsBetween(5),
                                   yminorticks = IntervalsBetween(5),
                                   xminortickalign = 1.0,
                                   yminortickalign = 1.0,
                                   xminorticksize = 4,
                                   yminorticksize = 4,
                                   xminortickwidth = 1.1,
                                   yminortickwidth = 1.1,
                                   xtickalign = 1.0,
                                   ytickalign = 1.0,
                                   xticksmirrored = true,
                                   yticksmirrored = true,
                                   xticksize = 8,
                                   yticksize = 8,
                                   xtickwidth = 1.3,
                                   ytickwidth = 1.3),
                 Legend = Attributes(;
                                     framevisible = legend_framevisible,
                                     labelsize = legend_labelsize,
                                     patchsize = patchsize,
                                     rowgap = legend_rowgap,
                                     colgap = legend_colgap,
                                     margin = legend_margin,
                                     padding = legend_padding),
                 Scatter = Attributes(;
                                      cycle = Cycle([[:color] => :color, [:marker] => :marker], true),
                                      markersize = 7,
                                      strokewidth = 0),
                 markersize = 7,
                 palette = Attributes(;
                                      color = [RGBAf(0.298039, 0.447059, 0.690196, 1.0),
                                               RGBAf(0.866667, 0.517647, 0.321569, 1.0),
                                               RGBAf(0.333333, 0.658824, 0.407843, 1.0),
                                               RGBAf(0.768627, 0.305882, 0.321569, 1.0),
                                               RGBAf(0.505882, 0.447059, 0.701961, 1.0),
                                               RGBAf(0.576471, 0.470588, 0.376471, 1.0),
                                               RGBAf(0.854902, 0.545098, 0.764706, 1.0),
                                               RGBAf(0.54902, 0.54902, 0.54902, 1.0),
                                               RGBAf(0.8, 0.72549, 0.454902, 1.0),
                                               RGBAf(0.392157, 0.709804, 0.803922, 1.0)],
                                      marker = [:circle, :rect, :dtriangle, :utriangle, :cross,
                                                :diamond, :ltriangle, :rtriangle, :pentagon,
                                                :xcross, :hexagon],
                                      markercolor = [RGBAf(0.298039, 0.447059, 0.690196, 1.0),
                                                     RGBAf(0.866667, 0.517647, 0.321569, 1.0),
                                                     RGBAf(0.333333, 0.658824, 0.407843, 1.0),
                                                     RGBAf(0.768627, 0.305882, 0.321569, 1.0),
                                                     RGBAf(0.505882, 0.447059, 0.701961, 1.0),
                                                     RGBAf(0.576471, 0.470588, 0.376471, 1.0),
                                                     RGBAf(0.854902, 0.545098, 0.764706, 1.0),
                                                     RGBAf(0.54902, 0.54902, 0.54902, 1.0),
                                                     RGBAf(0.8, 0.72549, 0.454902, 1.0),
                                                     RGBAf(0.392157, 0.709804, 0.803922, 1.0)],
                                      patchcolor = [RGBAf(0.298039, 0.447059, 0.690196, 1.0),
                                                    RGBAf(0.866667, 0.517647, 0.321569, 1.0),
                                                    RGBAf(0.333333, 0.658824, 0.407843, 1.0),
                                                    RGBAf(0.768627, 0.305882, 0.321569, 1.0),
                                                    RGBAf(0.505882, 0.447059, 0.701961, 1.0),
                                                    RGBAf(0.576471, 0.470588, 0.376471, 1.0),
                                                    RGBAf(0.854902, 0.545098, 0.764706, 1.0),
                                                    RGBAf(0.54902, 0.54902, 0.54902, 1.0),
                                                    RGBAf(0.8, 0.72549, 0.454902, 1.0),
                                                    RGBAf(0.392157, 0.709804, 0.803922, 1.0)]))
end

@inline function _activate_cairo_backend()
    CairoMakie.activate!()
    return nothing
end

@inline function _with_aps_theme(f::Function)
    _activate_cairo_backend()
    return with_theme(mytheme_aps()) do
        f()
    end
end

@inline function _with_publication_theme(f::Function; kwargs...)
    _activate_cairo_backend()
    return with_theme(mytheme_aps_publication(; kwargs...)) do
        f()
    end
end

@inline function _with_spectrum_theme(f::Function; kwargs...)
    _activate_cairo_backend()
    return with_theme(mytheme_aps()) do
        with_theme(mytheme_aps_spectrum(; kwargs...)) do
            f()
        end
    end
end

import Pkg
Pkg.activate(".")
Pkg.status()

import JuMP

# import Alpine
import EAGO
import HiGHS
import Ipopt
import Juniper
# import MAiNGO

import CSV
import DataFrames
import Random

using Plots

function build_model(
    nx::Int,
    cx::AbstractVecOrMat,
    xlb::AbstractVector,
    xub::AbstractVector,
    ny::Int,
    cy::AbstractVecOrMat,
    cr::Real,
    ns::Int,
    xi::AbstractMatrix,
    demand::Real;
    optimizer=EAGO.Optimizer,
)
    ps = ones(ns) ./ ns
    @assert(isapprox(sum(ps), 1.0))

    m = JuMP.Model(optimizer)
    # JuMP.@variable(m, x[1:nx] >= 0)
    JuMP.@variable(m, xlb[i] <= x[i=1:nx] <= xub[i])
    # JuMP.@variable(m, y[i=1:ny, s=1:ns], lower_bound=0.0, upper_bound=1.0)
    JuMP.@variable(m, y[i=1:ny, s=1:ns], lower_bound=0.0, upper_bound=1.0, binary=true)

    obj_fs = JuMP.@expression(
        m, obj_first_stage,
        sum(cx[i, 1] * x[i] + cx[i, 2] * x[i] * x[i] for i in 1:nx)
    )

    obj_ss = JuMP.@expression(
        m, obj_second_stage,
        sum(ps[s] * (cy[i, 1]*y[i, s] + cy[i, 2]*y[i, s]^2) * xi[i, s] for i in 1:ny, s in 1:ns)
        +
        sum(ps[s] * (cr * y[i, s]) * (1 - xi[i, s]) for i in 1:ny, s in 1:ns)
    )

    JuMP.@objective(m, Min, obj_fs + obj_ss)

    # JuMP.@objective(m, Min,
    #     sum(cx[i, 1] * x[i] + cx[i, 2] * x[i] * x[i] for i in 1:nx)
    #     + sum(ps[s] * (cy[i, 1]*y[i, s] + cy[i, 2]*y[i, s]^2) * xi[i, s] for i in 1:ny, s in 1:ns)
    #     + sum(ps[s] * (cr * y[i, s]) * (1 - xi[i, s]) for i in 1:ny, s in 1:ns)
    # )
    # JuMP.@objective(m, Min, 
    #     sum(cx[i,1] * x[i] for i in 1:nx)
    #     + sum(ps[s] * cy[i,1] * y[i,s] * xi[i,s] for i in 1:ny, s in 1:ns)
    #     + sum(ps[s] * cr * y[i,s] * (1 - xi[i,s]) for i in 1:ny, s in 1:ns)
    # )

    for s in 1:ns
        JuMP.@constraint(m, sum(x[i] for i in 1:nx) + sum(y[i, s] for i in 1:ny) == demand)
    end

    return m

end

"""
    second_stage_surface(demand, x_values, nx, cx, ny, cy, cr, ns, xi; optimizer, x_step)

For a fixed `demand`, sweep `x` over `x_values`, fix the first-stage variable at
each value, optimise the second-stage (y) variables, and return a DataFrame with
columns `x`, `first_stage`, `second_stage`, and `obj`.

Fixing x is achieved by passing `xlb = xub = [x_val, ...]` to `build_model`.
"""
function second_stage_surface(
    demand::Real,
    x_values::AbstractVector,
    nx::Int,
    cx::AbstractVecOrMat,
    ny::Int,
    cy::AbstractVecOrMat,
    cr::Real,
    ns::Int,
    xi::AbstractMatrix;
    optimizer=nothing,
)
    rows = DataFrames.DataFrame(
        :x => Float64[],
        :first_stage => Float64[],
        :second_stage => Float64[],
        :obj => Float64[],
        :status => String[],
    )

    for x_val in x_values
        xlb_fixed = fill(x_val, nx)
        xub_fixed = fill(x_val, nx)
        m = build_model(nx, cx, xlb_fixed, xub_fixed, ny, cy, cr, ns, xi, demand;
            optimizer=optimizer)
        JuMP.fix.(m[:x], x_val; force=true)
        JuMP.optimize!(m)
        term = string(JuMP.termination_status(m))
        if JuMP.has_values(m)
            fs = JuMP.value(m[:obj_first_stage])
            ss = JuMP.value(m[:obj_second_stage])
            obj = JuMP.objective_value(m)
        else
            fs = ss = obj = NaN
        end
        push!(rows, (x_val, fs, ss, obj, term))
    end

    return rows
end

function plot_second_stage_surface(
    demand::Real,
    x_values::AbstractVector,
    nx::Int,
    cx::AbstractVecOrMat,
    ny::Int,
    cy::AbstractVecOrMat,
    cr::Real,
    ns::Int,
    xi::AbstractMatrix;
    optimizer=nothing,
)
    df = second_stage_surface(demand, x_values, nx, cx, ny, cy, cr, ns, xi;
        optimizer=optimizer)

    p = plot(df.x, df.second_stage, marker=:circle, label="Second-stage \$\\phi(x)\$",
        xlabel="Gas commitment \$x\$", ylabel="Cost",
        title="Second-stage surface (demand = $demand)")
    plot!(p, df.x, df.obj, marker=:square, linestyle=:dash, label="Total \$o(x)\$")
    display(p)

    return df, p
end

function main()

    # d = 8
    demand_values = 1:12

    cx = [
        4.0 0.0;
        # 5.0 1e-2;
    ]
    nx = size(cx, 1)
    xlb = zeros(nx)
    # xub = fill(6.0, nx)
    xub = fill(Inf, nx)

    ny = 4
    cy1 = range(2.0, 3.0, ny)
    cy2 = fill(1e-3, ny)
    cy = hcat(cy1, cy2)

    cr = 10.0

    ns = 2^ny
    xi = [((s-1) >> (ny-i)) & 1 for i in 1:ny, s in 1:ns]

    #### Alpine ####
    # ipopt = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    # highs = JuMP.optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
    # solver = JuMP.optimizer_with_attributes(Alpine.Optimizer, "nlp_solver" => ipopt, "mip_solver" => highs)
    #### EAGO ####
    # solver = EAGO.Optimizer
    #### HiGHS ####
    # solver = HiGHS.Optimizer
    #### Ipopt ####
    # solver = Ipopt.Optimizer
    #### Juniper ####
    ipopt = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)
    solver = JuMP.optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt)
    #### MAiNGO ####
    # solver = JuMP.optimizer_with_attributes(MAiNGO.Optimizer, "epsilonA"=> 1e-8)

    x_vals = 4.0:1.0:8.0   # feasible range = [d-ny, d] = [4, 8]
    df_surf, p_surf = plot_second_stage_surface(8.0,
        x_vals, nx, cx,
        ny, cy, cr,
        ns, xi;
        optimizer=solver)
    png(p_surf, joinpath(@__DIR__, "objective_surface_d8"))

    results = DataFrames.DataFrame(
        :demand=>Float64[],
        :status=>String[],
        :obj=>Float64[],
        :first_stage=>Float64[],
        :second_stage=>Float64[],
        :max_violation=>Float64[],
        :n_fractional=>Int[],
        :xsol=>Float64[],
        # :xsol => Vector{Vector{Float64}},
    )

    for d in demand_values

        println("*"^16, " d = ", d, " ", "*"^16)

        m = build_model(nx, cx, xlb, xub, ny, cy, cr, ns, xi, d; optimizer=solver)
        JuMP.optimize!(m)
        JuMP.solution_summary(m)

        xsol = JuMP.value(m[:x])
        ysol = JuMP.value(m[:y])
        idx = abs.(JuMP.value(m[:y]) .- Int.(round.(JuMP.value(m[:y])))) .> 1e-6
        println("Number of fractional values: ", sum(idx))

        term_status = string(JuMP.termination_status(m))
        @show term_status
        obj = JuMP.objective_value(m)
        report = JuMP.primal_feasibility_report(m)
        max_viol = isempty(report) ? 0.0 : maximum(values(report))

        @show obj
        @show JuMP.value(m[:obj_first_stage])
        @show JuMP.value(m[:obj_second_stage])
        @show max_viol

        push!(results, (
            d,
            term_status,
            obj,
            JuMP.value(m[:obj_first_stage]),
            JuMP.value(m[:obj_second_stage]),
            max_viol,
            sum(idx),
            # string(round.(xsol)),
            round(xsol[1]),
        ))

    end

    display(results)
    CSV.write(joinpath(@__DIR__, "poster_example_jump_results.csv"), results)

    display(df_surf)
    CSV.write(joinpath(@__DIR__, "objective_surface_d8.csv"), df_surf)

    return

end

main()

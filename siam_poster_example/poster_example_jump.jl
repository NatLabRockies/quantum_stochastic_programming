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

function main()

    # d = 8
    demand_values = 1:12

    cx = [
        4.0 1e-2;
        # 5.0 1e-2;
    ]
    nx = size(cx, 1)
    xlb = zeros(nx)
    # xub = fill(6.0, nx)
    xub = fill(12.0, nx)

    ny = 4
    cy1 = range(0.1, 1.0, ny)
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

    results = DataFrames.DataFrame(
        :demand=>Int[],
        :status=>String[],
        :obj=>Float64[],
        :first_stage=>Float64[],
        :second_stage=>Float64[],
        :max_violation=>Float64[],
        :n_fractional=>Int[],
        :xsol=>String[],
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
            string(round.(xsol)),
        ))

    end

    display(results)
    CSV.write("poster_example_jump_results.csv", results)

    return

end

main()

using LinearAlgebra, Printf, Statistics

BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "booster_batched_uq.jl"))
using .BoosterBatchedUQ
const UQ = BoosterBatchedUQ

const PAIRS = map(split(get(ENV, "BOOSTER_CPU_PAIRS", "1:1"), ',')) do pair
    experiments, chains = parse.(Int, split(pair, ':'))
    experiments > 0 && chains > 0 && experiments * chains <= 32768 ||
        error("invalid experiments:chains pair $pair")
    (experiments, chains)
end
const SAMPLES = parse(Int, get(ENV, "BOOSTER_CPU_SAMPLES", "3"))
const CHUNKSIZE = parse(Int, get(ENV, "BOOSTER_CPU_CHUNK", "4"))
const CONFIG = get(ENV, "BOOSTER_CPU_CONFIG", "threads")
const OUTPUT = abspath(get(ENV, "BOOSTER_CPU_OUTPUT", "cpu_inference.csv"))
const PROCESS_MODE = get(ENV, "BOOSTER_CPU_PROCESS_MODE", "0") == "1"
const RANK = parse(Int, get(ENV, "SLURM_PROCID", "0"))
const NRANKS = parse(Int, get(ENV, "SLURM_NTASKS", "1"))
const BARRIER_DIRECTORY = get(ENV, "BOOSTER_CPU_BARRIER_DIRECTORY", "")
const BARRIER_TIMEOUT = parse(Float64, get(ENV, "BOOSTER_CPU_BARRIER_TIMEOUT", "1800"))
SAMPLES >= 2 || error("at least two timed samples are required")
1 <= CHUNKSIZE <= UQ.N_QUADS || error("invalid ForwardDiff width")
PROCESS_MODE && isempty(BARRIER_DIRECTORY) && error("process mode requires a barrier directory")

function operating_points(n)
    base = UQ._BoosterLatticeTemplate.default_booster_operating_point()
    [merge(base, (;
        idipo=base.idipo * (1 + 0.01sin(0.031i)),
        iqhc=base.iqhc * (1 + 0.01cos(0.037i)),
        iqvc=base.iqvc * (1 + 0.01sin(0.041i)),
        corrector_currents=ntuple(q -> 0.1sin(0.013i + 0.07q), UQ.N_QUADS),
    )) for i in 1:n]
end

factor_sets(n, sample) = [
    1.0 + 0.002sin(0.071f + 0.113q) + 0.00005sin(0.3sample + 0.19f - 0.07q)
    for f in 1:n, q in 1:UQ.N_QUADS
]

sensitivity() = UQ.BatchedForwardSensitivity(
    chunksize=CHUNKSIZE, warm_start=true, maxiter=40, threaded=false)
evaluate(problem, factors, ::Val{G}) where {G} = UQ._evaluate_core!(problem, factors, Val(G))

function make_chain_problems(prepared, nworkers)
    first_problem = UQ.BoosterBatchProblem(prepared; sensitivity=sensitivity())
    problems = Vector{typeof(first_problem)}(undef, nworkers)
    problems[1] = first_problem
    Threads.@threads :static for worker in 2:nworkers
        problems[worker] = UQ.BoosterBatchProblem(prepared; sensitivity=sensitivity())
    end
    problems
end

function run_chain_set!(problems, factors, mode)
    nchains = size(factors, 1)
    nworkers = length(problems)
    values = zeros(nchains)
    gradient_norms2 = zeros(nchains)
    Threads.@threads :static for worker in 1:nworkers
        problem = problems[worker]
        for chain in worker:nworkers:nchains
            result = evaluate(problem, @view(factors[chain, :]), mode)
            if mode isa Val{true}
                values[chain] = result[1]
                gradient_norms2[chain] = sum(abs2, result[2])
            else
                values[chain] = result
            end
        end
    end
    all(isfinite, values) || error("nonfinite likelihood")
    all(isfinite, gradient_norms2) || error("nonfinite gradient")
    sum(values), sqrt(sum(gradient_norms2))
end

function benchmark_threads()
    mkpath(dirname(OUTPUT))
    capacity = Threads.nthreads()
    println("CPU config=$CONFIG threads=$capacity Float64 width=$CHUNKSIZE pairs=$PAIRS")
    open(OUTPUT, "w") do io
        println(io, "config,workers,active_workers,smt,experiments,chains,points,mode,sample,seconds,value_sum,gradient_norm,status")
        for (nexperiments, nchains) in PAIRS
            active = min(nchains, capacity)
            setup_start = time_ns()
            prepared = UQ.prepare_booster_batch(
                operating_points(nexperiments), zeros(nexperiments, UQ.N_BPMS);
                noise_std_mm=0.2)
            problems = make_chain_problems(prepared, active)
            setup_seconds = (time_ns() - setup_start) / 1e9
            prefix = "$CONFIG,$capacity,$active,$(capacity > 128),$nexperiments,$nchains,$(nexperiments*nchains)"
            @printf(io, "%s,setup,0,%.9f,NaN,NaN,ok\n", prefix, setup_seconds)
            # Give the CPU Newton path a nominal closed orbit before perturbing
            # factors, matching inference chains initialized at their prior center.
            run_chain_set!(problems, ones(nchains, UQ.N_QUADS), Val(true))
            for (name, mode, offset) in (("value_gradient", Val(true), 0), ("likelihood", Val(false), 1000))
                warm = factor_sets(nchains, offset)
                warm_start = time_ns()
                run_chain_set!(problems, warm, mode)
                warm_seconds = (time_ns() - warm_start) / 1e9
                @printf(io, "%s,%s_warmup,0,%.9f,NaN,NaN,ok\n", prefix, name, warm_seconds)
                elapsed = Float64[]
                for sample in 1:SAMPLES
                    factors = factor_sets(nchains, sample + offset)
                    start = time_ns()
                    value_sum, gradient_norm = run_chain_set!(problems, factors, mode)
                    seconds = (time_ns() - start) / 1e9
                    push!(elapsed, seconds)
                    @printf(io, "%s,%s,%d,%.9f,%.17g,%.17g,ok\n",
                        prefix, name, sample, seconds, value_sum, gradient_norm)
                    flush(io)
                end
                @printf("experiments=%d chains=%d mode=%s median=%.4fs chains/s=%.2f points/s=%.2f\n",
                    nexperiments, nchains, name, median(elapsed),
                    nchains / median(elapsed), nexperiments * nchains / median(elapsed))
            end
            problems = nothing
            GC.gc(true)
        end
    end
end

function barrier(label)
    mkpath(BARRIER_DIRECTORY)
    open(joinpath(BARRIER_DIRECTORY, "$label.rank$RANK"), "w") do io
        print(io, RANK)
    end
    release = joinpath(BARRIER_DIRECTORY, "$label.release")
    if RANK == 0
        prefix = "$label.rank"
        deadline = time() + BARRIER_TIMEOUT
        while count(name -> startswith(name, prefix), readdir(BARRIER_DIRECTORY)) < NRANKS
            time() < deadline || error("barrier timeout: $label")
            sleep(0.05)
        end
        open(release, "w") do io
            print(io, "release")
        end
    else
        deadline = time() + BARRIER_TIMEOUT
        while !isfile(release)
            time() < deadline || error("barrier timeout: $label")
            sleep(0.05)
        end
    end
    nothing
end

function process_rank_path(rank=RANK)
    stem, extension = splitext(OUTPUT)
    "$stem.rank$rank$extension"
end

function benchmark_process_rank()
    mkpath(dirname(OUTPUT))
    rank_path = process_rank_path()
    open(rank_path, "w") do io
        println(io, "config,workers,active_workers,smt,rank,active,experiments,chains,points,mode,sample,seconds,value_sum,gradient_norm,status")
        for (pair_index, (nexperiments, nchains)) in enumerate(PAIRS)
            active_workers = min(nchains, NRANKS)
            active = RANK < active_workers
            setup_start = time_ns()
            problem = if active
                prepared = UQ.prepare_booster_batch(
                    operating_points(nexperiments), zeros(nexperiments, UQ.N_BPMS);
                    noise_std_mm=0.2)
                UQ.BoosterBatchProblem(prepared; sensitivity=sensitivity())
            else
                nothing
            end
            setup_seconds = (time_ns() - setup_start) / 1e9
            prefix = "$CONFIG,$NRANKS,$active_workers,$(NRANKS > 128),$RANK,$active,$nexperiments,$nchains,$(nexperiments*nchains)"
            @printf(io, "%s,setup,0,%.9f,NaN,NaN,ok\n", prefix, setup_seconds)
            active && evaluate(problem, ones(UQ.N_QUADS), Val(true))
            barrier("p$(pair_index)_nominal")
            for (name, mode, offset) in (("value_gradient", Val(true), 0), ("likelihood", Val(false), 1000))
                if active
                    first_chain = RANK + 1
                    warm_factors = factor_sets(nchains, offset)
                    warm = @view warm_factors[first_chain, :]
                    evaluate(problem, warm, mode)
                end
                barrier("p$(pair_index)_$(name)_warm")
                for sample in 1:SAMPLES
                    factors = active ? factor_sets(nchains, sample + offset) : nothing
                    barrier("p$(pair_index)_$(name)_s$(sample)_start")
                    start = time_ns()
                    value_sum = 0.0
                    gradient_norm2 = 0.0
                    if active
                        for chain in (RANK + 1):active_workers:nchains
                            result = evaluate(problem, @view(factors[chain, :]), mode)
                            if mode isa Val{true}
                                value_sum += result[1]
                                gradient_norm2 += sum(abs2, result[2])
                            else
                                value_sum += result
                            end
                        end
                    end
                    seconds = (time_ns() - start) / 1e9
                    @printf(io, "%s,%s,%d,%.9f,%.17g,%.17g,ok\n",
                        prefix, name, sample, seconds, value_sum, sqrt(gradient_norm2))
                    flush(io)
                    barrier("p$(pair_index)_$(name)_s$(sample)_done")
                end
            end
            problem = nothing
            GC.gc(true)
            barrier("p$(pair_index)_finished")
        end
    end
    barrier("rank_files_closed")
    RANK == 0 && summarize_process_ranks()
end

function summarize_process_ranks()
    rows = [split(line, ',') for rank in 0:(NRANKS - 1)
            for line in Iterators.drop(eachline(process_rank_path(rank)), 1)]
    groups = Dict{Tuple{String,String,String,String},Vector{Vector{SubString{String}}}}()
    for row in rows
        key = (row[7], row[8], row[10], row[11])
        push!(get!(groups, key, Vector{Vector{SubString{String}}}()), row)
    end
    open(OUTPUT, "w") do io
        println(io, "config,workers,active_workers,smt,experiments,chains,points,mode,sample,seconds,value_sum,gradient_norm,status")
        for (_, group) in sort!(collect(groups); by=x ->
                (parse(Int, x[1][1]), parse(Int, x[1][2]), x[1][3], parse(Int, x[1][4])))
            firstrow = first(group)
            seconds = maximum(parse(Float64, row[12]) for row in group)
            values = [parse(Float64, row[13]) for row in group if row[13] != "NaN"]
            gradients = [parse(Float64, row[14]) for row in group if row[14] != "NaN"]
            value_sum = isempty(values) ? NaN : sum(values)
            gradient_norm = isempty(gradients) ? NaN : sqrt(sum(abs2, gradients))
            @printf(io, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%.9f,%.17g,%.17g,ok\n",
                firstrow[1], firstrow[2], firstrow[3], firstrow[4],
                firstrow[7], firstrow[8], firstrow[9], firstrow[10], firstrow[11],
                seconds, value_sum, gradient_norm)
        end
    end
end

PROCESS_MODE ? benchmark_process_rank() : benchmark_threads()

# Error paths. Every line here used to print something that read as an answer.
#
# stderr is golden too, so the wording of a diagnostic is part of the contract: a
# message that stops naming what went wrong is a change a reviewer should see.
sym_eval("x*y", "x=3")
bfgs("(x-3)^2", [0])
bisection("x^2 - 2", 0, 2)
sym_limit("1/x", "x", 0)
sym_mellin("exp(-2*t)", "t", "s")
bigint("495.0")
A = [1, 2; 3, 4]
A(1, 1) = 9
stats_one_way_anova([1, 1, 1; 2, 2, 2])
rle_encode_vec([0; 300])
ode_rk45("cos(1000*t)", 0, 0, 100, 1e-6, 1e-9)
numthy_prime_pi(200000000)

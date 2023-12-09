# Introduction of the package

using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("JuMP")
Pkg.add("Gurobi")
#Pkg.add("NLopt")

using CSV, DataFrames, JuMP, Gurobi

############### Constant of the model 

eta = 0.8; # Efficiency of the battery
CO2_price = 100; # price of the CO2 per ton
H = 2;

################ Time to consider

T = 50
T_start = 987
time_ls_opt = T_start-24:T_start+24


hrs = 4000
hre = 5000
sel_hrs = hrs:hre
println(sel_hrs)
techs = 5
n_eu = 4
# pad year by 1 day
add_h = 24


println("Timespan: ", T)
#time_ls = 1+add_h:T-add_h
time_ls = T_start-24+1:T_start+24-add_h
vre_ls = 1:2
fossil_ls = 3:4
batt_ind = 5



######## Need to create a loop on that ========

name_demand_flex_input = ["demand_flex0_vre0", "demand_flex50_vre0", "demand_flex50_vre50", 
"demand_flex50_vre100", "demand_flex100_vre0", "demand_flex100_vre50", "demand_flex100_vre100"]

name_cost_ls = ["mean", "max"]


for name in name_demand_flex_input

    for cost_name in name_cost_ls
    
        # Read the CSV
    SCALE_MWH = 1000
    SCALE_DOLLAR = 1000

    fixed_cost = CSV.read("fixed_cost_" *cost_name*".csv",DataFrame) ./ SCALE_DOLLAR

    variable_cost = CSV.read("variable_cost" *cost_name*".csv",DataFrame) ./ SCALE_DOLLAR

    co2_intensity = CSV.read("co2_intensity.csv",DataFrame)

    P_max_s = 95897829.43 / SCALE_MWH
    P_max_w = 11158008 / SCALE_MWH;
    P_max_s, P_max_w

    Availability_matrix = CSV.read("Availability.csv", DataFrame);
    println(size(Availability_matrix))

    curtailment_costs = [0, 2, 0, 0, 0, 0] ./ SCALE_DOLLAR



    file_to_read = name*".csv"

    demand = Matrix(CSV.read(file_to_read, DataFrame)) ./ SCALE_MWH;
    println(size(demand))
    first(demand, 5)

    ramp_limits = CSV.read("ramp_limits.csv", DataFrame)

    #######################################



    println("Cost")
    C_FC = fixed_cost[1:techs,1]
    C_FOM = fixed_cost[1:techs,2]
    E_CO2 = co2_intensity[1:techs,1]
    C_VOM = variable_cost[1:techs,1]
    C_ramp = variable_cost[1:techs,2]
    C_start = variable_cost[1:techs,3]
    C_curt = curtailment_costs[1:techs]

    println("Demand")
    D_b = demand[sel_hrs] # Baseline demand with no flex
    println(size(D_b))
    D_b = vcat(reverse(reverse(demand)[1:add_h]), D_b, D_b[1:24])
    println(size(D_b))
    total_demand = D_b
    X_D = D_b

    A_r = ramp_limits[1:techs, 1]
    println("Availability")
    A_s = Availability_matrix[sel_hrs,1:techs]
    println("Try reverse")
    A_s = vcat(reverse(reverse(Availability_matrix)[1:add_h, 1:techs]), A_s, A_s[1:24, 1:techs])
    println(size(A_s))
    # over time, opt each year with worst day for availability and highest demand?



    # Model


    model = Model(Gurobi.Optimizer) # TODO: options? other optimizers?
    set_optimizer_attribute(model, "NonConvex", 2)
    #set_optimizer_attribute(model, "IterationLimit", 10000000)
    set_optimizer_attribute(model, "TimeLimit", 120)
    println("Model: initialized!\n")

    n = size(C_FC, 1) # number of technologies
    println("Number of technologies: ", n)
    #T = size(D_b, 1) # hours in simulation

    # Non-negativity constraints
    # The installed power capacity of the mix
    @variable(model, P[1:n] >= 0)

    # The hourly power generation of each technology
    @variable(model, X_gen[time_ls_opt, 1:n] >= 0); # hourly production/use for each technology

    # Curtailment (only renewables?)
    @variable(model, X_cur[time_ls_opt, 1:n] >= 0); # hourly production/use for each technology

    ###### CONSTRAINTS ######

    # Demand constraint - meet demand per hour
    println("Demand constraint")
    @constraint(model, Con_dh[t = time_ls_opt], sum(X_gen[t, i] for i in 1:n) >= X_D[t]); # TODO: add curtailment
    # Meet total demand
    # @constraint(model, Con_d, sum(X_gen[t, i] for i in 1:n for t in time_ls_opt) >= sum(total_demand));

    # Installed capacity renewables < max capacity
    println("Renewable maximum installed capacity constraint")
    @constraint(model, Con_sgen, P[1] <= P_max_s);
    @constraint(model, Con_wgen, P[2] <= P_max_w);

    println("Renewable minimum installed capacity constraint")
    #@constraint(model, Con_solar_lb, P[1] >= 50);
    #@constraint(model, Con_wind_lb, P[2] >= 600);

    # Storage variable
    println("Storage variable")
    @variable(model, 0<= X_soc[time_ls_opt] <=1)
    @variable(model, X_ch[time_ls_opt]>=0)

    # Cannot produce more than the installed power
    # println("Installed power constraint")
    # @constraint(model, Con_p[i = 1:n, t = time_ls_opt], X_gen[t, i] <= P[i])

    # For each tech, at each hour, usage cannot exceed available capacity
    println("Availability & installed power constraint")
    @constraint(model, [i = 1:n, t = time_ls_opt], X_gen[t, i] <= P[i]*A_s[t, i])
    # Renewables produce maximum possible capacity
    @constraint(model, [i = vre_ls, t = time_ls_opt], X_gen[t, i] >= 0.7*P[i]*A_s[t, i]) 


    println("Availability & installed power constraint")
    @constraint(model, [i = 1:n, t = time_ls_opt], X_gen[t, i] <= P[i]*A_s[t, i])

    println("Curtailment definition")
    @constraint(model, [t = time_ls_opt], sum(X_cur[t, i] for i in 1:n) == sum(X_gen[t, i] for i in 1:n) - X_D[t]) 
    @constraint(model, [i = 1:n, t = time_ls_opt], X_cur[t, i] <= X_gen[t, i])

    # Cannot produce less than the minimum stable generation: X_gen must be zero or min to max
    # min_usage = [0, 0.1, 0, 0] #TODO - cant be more than ramp
    # @constraint(model, [i = 1:n, t = time_ls_opt], X_gen[t, i] >= P[i]*min_usage[i])

    # For each tech, at each hour, change in production cannot exceed ramp rate (TODO might need to convert this to time)
    println("Ramp constraint")
    # Dummy rate of change variable
    # @variable(model, X_roc[time_ls, 1:n])
    # @constraint(model, [i = 1:4, t = time_ls], X_roc[t, i] == X_gen[t, i]-X_gen[limit_bound_max(t+1, T), i])

    # Auxiliary variables to linearize the rate of change constraint
    @constraint(model, Con_rdwn[i = 1:n, t = time_ls], (X_gen[t+1, i] - X_gen[t, i]) >= -A_r[i]*X_gen[t, i]) 
    @constraint(model, Con_rup[i = 1:n, t = time_ls], (X_gen[t+1, i] - X_gen[t, i]) <= A_r[i]*X_gen[t, i]);


    #TODO RAMP AND STARTUP COSTS

    @objective(model, Min, 
        (
            sum((C_FC[i] + C_FOM[i])*P[i] for i in 1:n) +
            sum(sum((C_VOM[i] + CO2_price* E_CO2[i])* X_gen[t, i] for i in 1:n) for t in time_ls_opt) +
            sum(sum(C_curt[i] * X_cur[t, i] for i in 1:n) for t in time_ls)
    ));

    # Storage model
    M = 10000000


    # Need to put this conditions above
    @variable(model, Z_X_gen_greater_X_D[time_ls_opt], Bin)

    @constraint(model, Z_X_gen_greater_X_D_1[t in time_ls_opt], sum(X_gen[t,i] for i in 1:4) - X_D[t] <= M*Z_X_gen_greater_X_D[t])
    @constraint(model, Z_X_gen_greater_X_D_2[t in time_ls_opt], -sum(X_gen[t,i] for i in 1:4) + X_D[t] <= M*(1-Z_X_gen_greater_X_D[t]))


    # X_1, X_2, X_3 = [eta*(sum(X_gen[t,i] for i in Power_generation_ls)- X_D[t]) , eta*P3], (1- X_soc[t])*P[5]* H]
    @variable(model, z1[t in time_ls_opt], Bin)
    @variable(model, z2[t in time_ls_opt], Bin)

    # z_1 equal 1 if eta*(x_g_t -x_d) is is greatter than eta*E_max*delta t
    println("Z_1 expression")
    @constraint(model, z_1_1[t in time_ls_opt], eta*P[5] - eta*(sum(X_gen[t,i] for i in 1:4)- X_D[t])   <=  M*z1[t])
    @constraint(model, z_1_2[t in time_ls_opt], -eta*P[5] + eta*(sum(X_gen[t,i] for i in 1:4)- X_D[t]) <= M*(1-z1[t]))


    #Minimum between X_1 and X_2
    @variable(model, min_X_1_X_2[time_ls_opt])

    println("Min X_1, X_2 expression")
    @constraint(model, X_1_X_2_lower_1[t in time_ls_opt], min_X_1_X_2[t] <= eta*(sum(X_gen[t,i] for i in 1:4)- X_D[t]))
    @constraint(model, X_1_X_2_lower_2[t in time_ls_opt], min_X_1_X_2[t] <= eta*P[5])
    @constraint(model, X_1_X_2_higher_1[t in time_ls_opt], min_X_1_X_2[t] >= eta*(sum(X_gen[t,i] for i in 1:4)- X_D[t]) - M*(1-z1[t]))
    @constraint(model, X_1_X_2_higher_2[t in time_ls_opt], min_X_1_X_2[t] >= eta*P[5] - M*z1[t])


    # z_2 equal 1 if min(X_1, X_2) is is greater than (1- x_soc[t])*h*E_max
    println(" Take the min of min(X_1, X_2) and X_3")
    @constraint(model, z_2_1[t in time_ls_opt], (1- X_soc[t])* P[5] * H - min_X_1_X_2[t] <=  M*z2[t])
    @constraint(model, z_2_2[t in time_ls_opt], - (1- X_soc[t]) * P[5] * H + min_X_1_X_2[t] <= M*(1-z2[t]))


    # Constraint for the variable X_ch
    println(" Take the overall min")
    @variable(model, min_X_1_X_2_X_3[time_ls_opt])

    @constraint(model, X_1_X_2_X_3_lower_1[t in time_ls_opt], min_X_1_X_2_X_3[t] <= min_X_1_X_2[t])
    @constraint(model, X_1_X_2_X_3_lower_2[t in time_ls_opt], min_X_1_X_2_X_3[t] <= (1- X_soc[t])*P[3]* H)
    @constraint(model, X_1_X_2_X_3_higher_1[t in time_ls_opt], min_X_1_X_2_X_3[t] >= min_X_1_X_2[t] - M*(1-z2[t]))
    @constraint(model, X_1_X_2_X_3_higher_2[t in time_ls_opt], min_X_1_X_2_X_3[t] >= (1- X_soc[t])*P[3]* H - M*z2[t]);


    #Expression of X_ch
    @constraint(model, X_ch_expression[t in time_ls_opt], X_ch[t]  == Z_X_gen_greater_X_D[t]*min_X_1_X_2_X_3[t]);



    @variable(model, y2[time_ls_opt], Bin)

    # z_2 equal 1 if min(X_1, X_2) is is greater than (1- x_soc[t])*h*E_max
    println(" Take the min of min(X_1, X_2) and X_3")
    @constraint(model, y_2_1[t in time_ls_opt], X_soc[t]* P[5] * H - P[5] <=  M*y2[t])
    @constraint(model, y_2_2[t in time_ls_opt], - X_soc[t] * P[5] * H + P[5] <= M*(1-y2[t]))


    # Constraint for the variable X_ch
    println(" Take the overall min")
    @variable(model, min_extra_prod_P5_soc[time_ls_opt])

    @constraint(model, min_extra_prod_P5_soc_lower_1[t in time_ls_opt], min_extra_prod_P5_soc[t] <= P[5])
    @constraint(model, min_extra_prod_P5_soc_lower_2[t in time_ls_opt], min_extra_prod_P5_soc[t] <= X_soc[t]*P[5]* H)
    @constraint(model, min_extra_prod_P5_soc_higher_1[t in time_ls_opt], min_extra_prod_P5_soc[t] >= P[5] - M*(1-y2[t]))
    @constraint(model, min_extra_prod_P5_soc_higher_2[t in time_ls_opt], min_extra_prod_P5_soc[t] >= X_soc[t]*P[5]* H - M*y2[t]);

    @variable(model, batt_dispatch[time_ls_opt], Bin)

    #Expression of X_dch
    @constraint(model, X_dch_expression[t in time_ls_opt], X_gen[t,5]  <= batt_dispatch[t]*min_extra_prod_P5_soc[t]);


    # First hour need to be equal to one
    # Start with a charged battery
    @constraint(model, SOC_t_1, X_soc[time_ls_opt[1]] == 0)

    #
    @constraint(model, SOC_t[t in time_ls_opt[2]:time_ls_opt[end]], (X_soc[t]-X_soc[t-1])*H*P[5] == (X_ch[t-1]- X_gen[t-1,5]));

    #Optimize
    optimize!(model)



    using CSV

    # Save the results for the generation
    # Assuming your matrix is named `my_matrix`
    my_matrix = [value.(X_gen) value.(X_soc) X_D[time_ls_opt]]  # Replace this with your actual matrix
    println(my_matrix)
    # Column names

    column_names = ["Solar", "Wind", "CCGT", "Coal", "Battery", "State_of_Charge", "Demand"]

    # Create a DataFrame with vectors for each column
    df = DataFrame([Symbol(name) => my_matrix[:, i] for (i, name) in enumerate(column_names)])

    # Specify the file path for your CSV file
    csv_file_path = "Results/Final_results/"*cost_name* "/Generation" * name* ".csv"

    # Export the DataFrame to CSV
    CSV.write(csv_file_path, df)

    println("CSV file exported to: $csv_file_path")

    # Save the results for the power installed power
    # Assuming your matrix is named `my_matrix`
    Power = value.(P)   # Replace this with your actual matrix
    # Column names
    println(Power)
    column_names_power = "Installed Power"

    # Create a DataFrame with vectors for each column
    df_Power = DataFrame(Symbol(column_names_power) => Power)

    # Specify the file path for your CSV file
    csv_file_path_power = "Results/Final_results/"*cost_name* "/Power" * name* ".csv"

    # Export the DataFrame to CSV
    CSV.write(csv_file_path_power, df_Power)

    println("CSV file exported to: $csv_file_path_power")

end
end

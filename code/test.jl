using CSV
using DataFrames
using SparseArrays

#%% read in data
# alt-shift-enter to run the cell
constraints_ind = CSV.read("data/toy_constraints_ind.csv")
constraints_bg = CSV.read("data/toy_constraints_bg.csv")
constraints_trt = CSV.read("data/toy_constraints_trt.csv")
#%%

endswith(names(constraints_bg)[1], 's')

se_cols = [endswith(i, 's') for i in names(constraints_bg)]
constraints_bg[:,se_cols]

geo_lookup = DataFrame(bg = string.(collect(constraints_bg[:GEOID]), trt = string.(collect(constraints_bg[:GEOID])))

# use `collect` to get column values
collect(constraints_bg.GEOID)

# or if we wanted to use numerical indices
collect(constraints_bg[!,1])

# # this subsets rows
# constraints_bg[1:2,:]

#%% build geo lookup
# apply string conversion to bg GEOIDs
bg_id = string.(collect(constraints_bg[!,1]))
trt_id = [s[1:2] for s in bg_id]
geo_lookup = DataFrame(bg = bg_id, trt = trt_id)
#%%

#%% PUMS response ids
serial = collect(constraints_ind.SERIAL)
#%%

#%% sample weights
wt = collect(constraints_ind.PERWT)
#%%

#%% population and sample size
# use semicolons to supress printing
N = sum(constraints_bg.POP);
n = nrow(constraints_ind);
#%%

#%% individual (PUMS) constraints
# we need to use the ∉ symbol to represent a logical "not in"
# type it as \notin + TAB
constraint_cols = [i ∉ blah for i in names(constraints_ind)];
pX = constraints_ind[!,constraint_cols];
pX = convert(Matrix, pX);
px = sparse(pX);
#%%

#%% geographic constraints
est_cols = [!endswith(i, 's') && i != "GEOID" for i in names(constraints_bg)]
Y1 = convert(Matrix, constraints_trt[!,est_cols])
Y2 = convert(Matrix, constraints_bg[!,est_cols])
#%%

#%% error variances
se_cols = [endsWith(i, 's') for i in names(constraints_bg)]
square(x) = x^2
V1 = square.(convert(Matrix, constraints_trt[!,se_cols]))
V2 = square.(convert(Matrix, constraints_bg[!,se_cols]))
#%%

#### MISC ####
# applying functions
square(x) = x^2
blah = [1,2,3]
square.(blah)

# subsetting
constraints_bg[["GEOID", "CONST1"]] # columns
constraints_bg[!,[1,4]] # columns by index
constraints_bg[1:2,] # rows

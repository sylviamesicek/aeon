##########################
## Exports ###############
##########################

export AnalyticUnknown

##########################
## Unknown ###############
##########################

struct AnalyticUnknown
end

Base.:(*)(scale, ::AnalyticUnknown) = ScaleOperator(scale)



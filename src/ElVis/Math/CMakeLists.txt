SET(ProjectName "Math")


SET(Headers
    AdaptiveQuadrature.hpp
    GaussLegendreQuadrature.hpp
    Gregory.hpp
    Integration.hpp
    Jacobi.hpp
    Polynomial.hpp
    PolynomialProjection.hpp
    RiemannIntegration.hpp
    RombergIntegration.hpp
    SimpsonsIntegration.hpp
    TrapezoidalIntegration.hpp
    
) 

SET(Sources 
)


ADD_ELVIS_STATIC_LIBRARY(${ProjectName} 
    ${Headers} 
    ${Sources} 
)

TARGET_LINK_LIBRARIES(${ProjectName} 
)

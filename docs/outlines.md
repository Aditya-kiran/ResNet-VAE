# Link to the paper (overleaf/sharelatex)
https://v2.overleaf.com/9818261169mwrpqjjgnbzq


# Abstract

# Introduction
## Contribution of this work

# Background
 * introducing about normilized flow -- we can recycle your writing here
 * We view the computation of the posterior as deformation of the prior
 * introducing 2-3 most similar invertible flows
   - Planar
   - auto-regressive flow
   - [any suggestion?]
   
 

# Proposed flow
**NOTE** we need to have a better title up there. Unfortunately, the Diffeomophic flow is claimed by that Arxiv paper. ResNet flow is one option but people had used ResNet for ODE before, may "Reversible RNN Flow" ? I am open to suggestion

## Background of Diffeomophic function
 * Introduce the notion of Diffeomophism, velocity field etc -- we can recycle what I wrote in the draft
 * Stationary and non-stationary velocity field
 * We may need a theorem to show that this construction results in invertible function (References are needed)
   

## Stationary Velocity Field
* We implement the ODE as RNN 
* explain the architecture
* explain how this is more general than planar
* computing the log(det J) is challenging -- setting the stage for the next subsection  
### computation of log(det J)
 * Talk about the challenges (ie computing the log(det J) -- because we can write the transformation as composition of small transformation, we can approximate the log(det J) as sum of the small deformations and for the small deformations we can use Taylor expansion

### Inversion of the flow
 * Explain how the flow can be inverted
 * We can mention that because of numerical issue the inversion may not work perfetly but we can encourage inverse consistency by adding a regularization (see issue 16)
 
## Extension to the Stationary Velocity Field
 * Modeling the non-stationary velocity field as concatenaton of the stationary velocity field
 * **Regularization terms:**
    (1) Incorporating inverse consistency term to the encourage inverse consistency
    (2) Adding the integral of the velocity that ecourage the flow to be geodesic distance
    
## Theoritical work -- not sure we get to do this but if we add this, it will be very strong
- [Suggestion for Theorem:] Theoriticall characterize the set of transformations that plannar cannot acheive
- [Suggestion for Theorem:] Stationary velocity field cannot model all kinds of transformation  -- this motivate the work that we need non-stationary velocity field by concatenation of the statinary velocity field
- [Suggestion for Theorem:] Adding the integral of the L2 norm of the velocity field reduces the variance

# Related works
TODO

# Experiments
TODO 

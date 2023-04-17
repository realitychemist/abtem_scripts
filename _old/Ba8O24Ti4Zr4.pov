#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -13.71*x up 13.71*y
  direction 1.00*z
  location <0,0,50.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7}
#declare pale = finish {ambient 0.5 diffuse 0.85 roughness 0.001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 specular 0.5 }
#declare jmol = finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.05 diffuse 0.3 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.01 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.070;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

cylinder {< -3.13,  -3.13,  -8.17>, < -3.13,  -3.13,   0.00>, Rcell pigment {Black}}
cylinder {<  5.04,  -3.13,  -8.17>, <  5.04,  -3.13,   0.00>, Rcell pigment {Black}}
cylinder {<  5.04,   5.04,  -8.17>, <  5.04,   5.04,   0.00>, Rcell pigment {Black}}
cylinder {< -3.13,   5.04,  -8.17>, < -3.13,   5.04,   0.00>, Rcell pigment {Black}}
cylinder {< -3.13,  -3.13,  -8.17>, <  5.04,  -3.13,  -8.17>, Rcell pigment {Black}}
cylinder {< -3.13,  -3.13,   0.00>, <  5.04,  -3.13,   0.00>, Rcell pigment {Black}}
cylinder {< -3.13,   5.04,   0.00>, <  5.04,   5.04,   0.00>, Rcell pigment {Black}}
cylinder {< -3.13,   5.04,  -8.17>, <  5.04,   5.04,  -8.17>, Rcell pigment {Black}}
cylinder {< -3.13,  -3.13,  -8.17>, < -3.13,   5.04,  -8.17>, Rcell pigment {Black}}
cylinder {< -3.13,  -3.13,   0.00>, < -3.13,   5.04,   0.00>, Rcell pigment {Black}}
cylinder {<  5.04,  -3.13,   0.00>, <  5.04,   5.04,   0.00>, Rcell pigment {Black}}
cylinder {<  5.04,  -3.13,  -8.17>, <  5.04,   5.04,  -8.17>, Rcell pigment {Black}}
atom(< -3.13,  -3.13,  -8.17>, 1.91, rgb <0.00, 0.78, 0.00>, 0.0, ase2) // #0
atom(< -1.08,  -1.08,  -6.12>, 1.56, rgb <0.58, 0.87, 0.87>, 0.0, ase2) // #1
atom(< -1.08,  -3.13,  -6.12>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #2
atom(< -3.13,  -1.08,  -6.12>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #3
atom(< -1.08,  -1.08,  -8.17>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #4
atom(< -3.13,   0.96,  -8.17>, 1.91, rgb <0.00, 0.78, 0.00>, 0.0, ase2) // #5
atom(< -1.08,   3.00,  -6.12>, 1.42, rgb <0.75, 0.76, 0.78>, 0.0, ase2) // #6
atom(< -1.08,   0.96,  -6.12>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #7
atom(< -3.13,   3.00,  -6.12>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #8
atom(< -1.08,   3.00,  -8.17>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #9
atom(<  0.96,  -3.13,  -8.17>, 1.91, rgb <0.00, 0.78, 0.00>, 0.0, ase2) // #10
atom(<  3.00,  -1.08,  -6.12>, 1.56, rgb <0.58, 0.87, 0.87>, 0.0, ase2) // #11
atom(<  3.00,  -3.13,  -6.12>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #12
atom(<  0.96,  -1.08,  -6.12>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #13
atom(<  3.00,  -1.08,  -8.17>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #14
atom(<  0.96,   0.96,  -8.17>, 1.91, rgb <0.00, 0.78, 0.00>, 0.0, ase2) // #15
atom(<  3.00,   3.00,  -6.12>, 1.42, rgb <0.75, 0.76, 0.78>, 0.0, ase2) // #16
atom(<  3.00,   0.96,  -6.12>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #17
atom(<  0.96,   3.00,  -6.12>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #18
atom(<  3.00,   3.00,  -8.17>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #19
atom(< -3.13,  -3.13,  -4.08>, 1.91, rgb <0.00, 0.78, 0.00>, 0.0, ase2) // #20
atom(< -1.08,  -1.08,  -2.04>, 1.56, rgb <0.58, 0.87, 0.87>, 0.0, ase2) // #21
atom(< -1.08,  -3.13,  -2.04>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #22
atom(< -3.13,  -1.08,  -2.04>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #23
atom(< -1.08,  -1.08,  -4.08>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #24
atom(< -3.13,   0.96,  -4.08>, 1.91, rgb <0.00, 0.78, 0.00>, 0.0, ase2) // #25
atom(< -1.08,   3.00,  -2.04>, 1.42, rgb <0.75, 0.76, 0.78>, 0.0, ase2) // #26
atom(< -1.08,   0.96,  -2.04>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #27
atom(< -3.13,   3.00,  -2.04>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #28
atom(< -1.08,   3.00,  -4.08>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #29
atom(<  0.96,  -3.13,  -4.08>, 1.91, rgb <0.00, 0.78, 0.00>, 0.0, ase2) // #30
atom(<  3.00,  -1.08,  -2.04>, 1.56, rgb <0.58, 0.87, 0.87>, 0.0, ase2) // #31
atom(<  3.00,  -3.13,  -2.04>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #32
atom(<  0.96,  -1.08,  -2.04>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #33
atom(<  3.00,  -1.08,  -4.08>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #34
atom(<  0.96,   0.96,  -4.08>, 1.91, rgb <0.00, 0.78, 0.00>, 0.0, ase2) // #35
atom(<  3.00,   3.00,  -2.04>, 1.42, rgb <0.75, 0.76, 0.78>, 0.0, ase2) // #36
atom(<  3.00,   0.96,  -2.04>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #37
atom(<  0.96,   3.00,  -2.04>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #38
atom(<  3.00,   3.00,  -4.08>, 0.59, rgb <1.00, 0.05, 0.05>, 0.0, ase2) // #39

// no constraints

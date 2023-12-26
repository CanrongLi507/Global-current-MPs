dio:

# Seawater density calculation

The calculation of seawater density refers to the UNESCO 1983 (EOS 80) empirical formula
- In this chapter, EOS 80 seawater equation of state is used to calculate the density of seawater layer by layer, and the approximate density of seawater is finally obtained according to the three variable data of temperature, salinity and pressure extracted from the remote sensing data of ocean.
- EOS 80 is a relatively complex empirical formula that is widely used in oceanographic research.

# Usage overview

Although TEOS-10 (Thermodynamic Equation of Seawater 2010) provides more precise equations for seawater state calculations, most parameters of seabed water flows in nature are already encompassed by EOS 80. In areas not involving extreme seawater conditions, this EOS 80 equation can also be utilized.The terms of use typically include:
- The temperature is between -2°C and 42°C.
- The salinity is between 0—40 PSU (actual salinity unit)
- The pressure can reach up to 100 MPa at its maximum
- Assuming that seawater is a completely mixed solution, mainly composed of water and salt, without considering the influence of other dissolved substances

## Citation

If you use this seawater density formula for research applications and publish papers, please cite:
[1]. Fofonoff, N. P., & Millard Jr, R. C. (1983). Algorithms for the computation of fundamental properties of seawater. 
[1]. Millero, F. J., Chen, C. T., Bradshaw, A., & Schleicher, K. (1980). A new high pressure equation of state for seawater. Deep Sea Research Part A. Oceanographic Research Papers, 27(3-4), 255-264.
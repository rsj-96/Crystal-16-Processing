# Crystal16 Data Processor
Streamlit based tool for processing and analysing Crystal16 solubility experiments.

Application automatically generates Van't Hoff plots, solubility curves and calculates theoretical crystallisation yields from Crystal16.csv files.

Link to [Crystal16 data processor](https://crystal-16-processing.streamlit.app/)

## Features
### Van't Hoff Analysis
Automatically generates separate Van't Hoff plots for clear and cloud points upon uploading .csv file.

Users can select which cycles (from 1-3) to include in further analysis.

Compare individual cyles vs combined cycles.

User can choose to omit cloud points from further analysis.

### Solubility Curve Generator
Solubility curve plotted using the Van't Hoff Equation:

$\ln C = m (\frac{1}T) + c \$

where:  
C = Solubility  
T = Temperature (K)  
m = Slope  
c = Intercept  

Solubility curve also plotted with Experimental values from Crystal16 experiments.


### Theoretical Yield Calculator
Theoretical yield is calculated from the expected concentration at the two defined temperature limits (upper and lower):

$\ Yield\$ (%) $\= \frac{C_{upper} - C_{lower}} {C_{upper}} * 100\$

Where:  
$C_{upper}$ = solubility at higher temperature.  
$C_{lower}$ = solubility at lower temperature.

Theoretical yield calculator will output:
1. Upper and lower solubility value.
2. Solvent Volumes at the upper temperature.
3. Theoretical yield.

## How To Use
Below is the general procedure on how to use the Crystal16 data processing application:

1. Upload a Crystal16 .csv file.
2. Van’t Hoff plots are automatically generated.
3. Select cycles to include in the analysis.
4. Define the temperature range for solubility curves.
5. Define upper and lower temperature limits for theoretical yield calculation.
6. Click _Generate Solubility Curves and Calculate Theoretical Yield_.
7. Solubilty Curve and Theoretical Yield will be calculated.

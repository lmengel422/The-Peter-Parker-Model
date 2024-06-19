# The-Peter-Parker-Model
<img src="https://github.com/lmengel422/The-Peter-Parker-Model/blob/main/PPM_logo.png" width="75%" height="75%">

Total Exchange Flow Box Model + Nutrient Phytoplankton Zooplankton Detritus Box Model inspired by Peter JS Franks and Parker MacCready. Source code for efflux/reflux box model with Chatwin Solution from Parker MacCready. Modifications to add sinking tracer and NPZD by Lily Engel. 

To recreate results for sinking tracer as in Engel and Stacey 2024, use files traceronly.py and rflx_fun_tracer. Sample run: run traceronly.py -exp Tracer_Sink

For The Peter-Parker Model and to rerun detritus experiment as in Engel and Stacey 2024 or other experiments in Lily Engel's Dissertation, use NPZDonly.py and rflx_fun_NPZDonly.py. Sample run: run NPZDonly.py -exp D_Sink 

Relevant citations:

Engel and Stacey 2024

Engel, L. (2023). Into the Plankter-Verse: Physical-Biological Interactions in Estuaries (Doctoral dissertation). University of California, Berkeley, Department of Civil and Environmental Engineering.

MacCready, P., McCabe, R. M., Siedlecki, S. A., Lorenz, M., Giddings, S. N., Bos, J., Albertson, S., Banas, N. S. & Garnier, S. (2021), ‘Estuarine Circulation, Mixing, and Residence Times in the Salish Sea’, Journal of Geophysical Research: Oceans 126(2), e2020JC016738 

Banas, N. S., Lessard, E. J., Kudela, R. M., MacCready, P., Peterson, T. D., Hickey, B. M. & Frame, E. (2009), ‘Planktonic growth and grazing in the Columbia River plume region: A biophysical model study (vol 114, 2009)’, Journal of Geophysical Research-Oceans 114(Journal Article), C00B13

Franks, P., Wroblewski, J. S. & Flierl, G. R. (1986), ‘Behavior of a Simple Plankton Model with Food-Level Acclimation by Herbivores’, Marine Biology 91(1), 121–129.

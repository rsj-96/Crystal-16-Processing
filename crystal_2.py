import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


st.title("Crystal16 Processing")
with st.expander("How to Use 📝"):
    st.markdown("""
                1. Upload .csv file to file uploader.
                2. Dropdown boxes with Van't Hoff Plots will be automatically generated.
                3. Select cycles to be used for solubility curve.
                4. Define temperature range.
                5. Add curve title (if required).
                6. Define Temperatures for Theoretical Yield Calculation
                7. Click "Generate Solubility Curve and Calculate Theoretical Yield" and a Solubility Curve and Theoretical Yield will be generated for you.
                + Any questions speak with RJ""")


file = st.file_uploader("Choose a '.csv' File", type = ['.csv']) # streamlit file uploader where the excel type is specified
if file:
    og = pd.read_csv(file)  # reads the file into the dataframe using pandas
    
    #st.write(f'Preview of .csv file')
    #st.write(og.head()) # displays dataframe in the streamlit application
    
    
    # Split clear/cloud dataframes
    clear_df = og[[col for col in og.columns if col.endswith('_clear')]]
    clear_df.columns = [col.replace('_clear', '') for col in clear_df.columns]

    cloud_df = og[[col for col in og.columns if col.endswith('_cloud')]]
    cloud_df.columns = [col.replace('_cloud', '') for col in clear_df.columns]

    # Get unique solvents for CLEAR RESULTS
    solvents = clear_df["Solvent"].unique()

    #print(solvents)

    colours = ["#118ab2", "#06bc8b", "#ef476f"]
    colour = "#f48c06" # can't be a list or it wont work


    all_cloud =[]
    all_clear = []
    
    filtered_clear_all = []
    filtered_cloud_all = []
    
    
    # Loop through each solvent group
    for solvent in solvents:
        with st.expander(f"Van't Hoff Plots for {solvent}"):
            
            #--- RESULTS FOR CLEAR POINT ---
            
            solvent_df = clear_df[clear_df["Solvent"] == solvent].copy()
            solvent_df = solvent_df[solvent_df["Cycle"].isin([1, 2, 3])] # making sure it's just for cycles 1 to 3
            solvent_df["LnC"] = np.log(solvent_df["Concentration"])
            solvent_df["1/T"] = 1 / (solvent_df["Temps"] + 273)
            
            #print(solvent_df)
            
            st.subheader(f"Results for Clear Point for {solvent}")
            
            # Selecting for all cycles
            
            available_cycles_clear = solvent_df["Cycle"].unique().tolist()
            selected_cycles_clear = st.multiselect(f"Select cycles to include for {solvent} Clear Point:", options=available_cycles_clear, default=available_cycles_clear)
            filtered_df_clear = solvent_df[solvent_df["Cycle"].isin(selected_cycles_clear)]
            
            # Linear regression for all cycles combined in this solvent ###here
            if not filtered_df_clear.empty:
                x_all = filtered_df_clear["1/T"]
                y_all = filtered_df_clear["LnC"]
                slope, intercept, r_value, p_value, std_err = linregress(x_all, y_all)
                all_results = pd.DataFrame([{"Solvent": solvent, "Cycle": "All", "Slope (m)": slope, "Intercept (c)": intercept, "R²": r_value**2}])

            # Linear regression for each cycle separately in specific solvent
            vant_results = []
            for cycle, group in solvent_df.groupby("Cycle"):
                slope, intercept, r_value, p_value, std_err = linregress(group["1/T"], group["LnC"])
                vant_results.append({"Solvent": solvent, "Cycle": cycle, "Slope (m)": slope,"Intercept (c)": intercept, "R²": r_value**2})

            vant_res_clear_df = pd.DataFrame(vant_results)
            vant_clear_df = pd.concat([all_results, vant_res_clear_df], ignore_index=True)
            
            all_clear.append(all_results) # saves selected results to table outside loop for later
            filtered_clear_all.append(filtered_df_clear.copy())
            
            st.write(vant_clear_df)  
            
            #---START PLOTTING FOR CLEAR POINT---

            # Plot Van't Hoff plot by Cycle for this solvent
            g = sns.lmplot(data=solvent_df, x="1/T", y="LnC", col="Cycle", hue="Cycle", palette=colours, line_kws={"linewidth":1, "linestyle":"--"}, ci=None)
            
            g.figure.suptitle(f"Van't Hoff Plot for Cycles 1-3 in {solvent}", y=1.02)
            g.figure.subplots_adjust(top=0.85)
            g.figure.set_size_inches(10,4)

            
            #plt.show()
            st.pyplot(plt.gcf())

            # Plot all cycles combined in one plot for each solvent solvent ## here
            b=sns.lmplot(data=filtered_df_clear, x="1/T", y="LnC", line_kws={"linewidth":1, "linestyle":"--", "color":colour}, scatter_kws={"color": colour}, ci=None)
            b.figure.set_size_inches(5.5, 3.6)
            
            plt.xlabel("1/T",fontsize=8)
            plt.xticks(fontsize=8)
            plt.ylabel("LnC",fontsize=8)
            plt.yticks(fontsize=8)
            plt.title(f"Van't Hoff Plot for Selected Cycles in {solvent}", fontsize=10)
            #plt.show()
            
            st.pyplot(plt.gcf())
            
            # --- RESULTS FOR CLOUD POINT ---
            st.subheader(f"Results for Cloud Point for {solvent}")
            
            solvents_cloud = cloud_df["Solvent"].unique()
            
            solvent_cloud_df = cloud_df[cloud_df["Solvent"] == solvent].copy()
            solvent_cloud_df = solvent_cloud_df[solvent_cloud_df["Cycle"].isin([1, 2, 3])] # making sure it's just for cycles 1 to 3
            solvent_cloud_df["LnC"] = np.log(solvent_cloud_df["Concentration"])
            solvent_cloud_df["1/T"] = 1 / (solvent_cloud_df["Temps"] + 273)
            
            #print(solvent_df)
            
            #selecting data for inclusion
            
            available_cycles_cloud = solvent_cloud_df["Cycle"].unique().tolist()
            selected_cycles_cloud = st.multiselect(f"Select cycles to include for {solvent} Cloud Point:", options=available_cycles_cloud, default=available_cycles_cloud)
            filtered_df_cloud = solvent_cloud_df[solvent_cloud_df["Cycle"].isin(selected_cycles_cloud)]
            
            # Linear regression for all cycles combined in this solvent ###here
            if not filtered_df_cloud.empty:
                x_all = filtered_df_cloud["1/T"]
                y_all = filtered_df_cloud["LnC"]
                slope, intercept, r_value, p_value, std_err = linregress(x_all, y_all)
                all_results_cloud = pd.DataFrame([{"Solvent": solvent, "Cycle": "All", "Slope (m)": slope, "Intercept (c)": intercept, "R²": r_value**2}])

            # Linear regression for each cycle separately in specific solvent
            vant_results_cloud = []
            for cycle, group in solvent_cloud_df.groupby("Cycle"):
                slope, intercept, r_value, p_value, std_err = linregress(group["1/T"], group["LnC"])
                vant_results_cloud.append({"Solvent": solvent, "Cycle": cycle, "Slope (m)": slope,"Intercept (c)": intercept, "R²": r_value**2})

            vant_res_cloud_df = pd.DataFrame(vant_results_cloud)
            vant_cloud_df = pd.concat([all_results_cloud, vant_res_cloud_df], ignore_index=True)
            
            
            
            all_cloud.append(all_results_cloud) # save all selected values
            filtered_cloud_all.append(filtered_df_cloud.copy())

            st.write(vant_cloud_df)  
            
            #---START PLOTTING FOR CLOUD POINT---

            # Plot Van't Hoff plot by Cycle for this solvent
            g = sns.lmplot(data=solvent_cloud_df, x="1/T", y="LnC", col="Cycle", hue="Cycle", palette=colours, line_kws={"linewidth":1, "linestyle":"--"}, ci=None)
            
            g.figure.suptitle(f"Van't Hoff Plot for Cycles 1-3 in {solvent}", y=1.02)
            g.figure.subplots_adjust(top=0.85)
            g.figure.set_size_inches(10,4)

            
            #plt.show()
            st.pyplot(plt.gcf())

            # Plot all cycles combined in one plot for each solvent solvent
            b=sns.lmplot(data=filtered_df_cloud, x="1/T", y="LnC", line_kws={"linewidth":1, "linestyle":"--", "color":colour}, scatter_kws={"color": colour}, ci=None, palette=["red"])
            b.figure.set_size_inches(5.5, 3.6)
            
            plt.xlabel("1/T",fontsize=8)
            plt.xticks(fontsize=8)
            plt.ylabel("LnC",fontsize=8)
            plt.yticks(fontsize=8)
            plt.title(f"Van't Hoff Plot for Selected Cycles in {solvent}", fontsize=10)
            #plt.show()
            
            st.pyplot(plt.gcf())
            

            
            
    st.subheader("Results for Plotting Solubility Curve")
        
    final_clear = pd.concat(all_clear, ignore_index=True)
    final_cloud = pd.concat(all_cloud, ignore_index=True)
    numeric_cols = ["Slope (m)", "Intercept (c)", "R²"]
    final_clear[numeric_cols] = final_clear[numeric_cols].apply(pd.to_numeric, errors="coerce")
    final_cloud[numeric_cols] = final_cloud[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    # Make column of tick boxes for selecting what data to include
    
    st.write("Clear Point Data")
    st.write(final_clear)
    st.write("Cloud Point Data")
    st.write(final_cloud)
    
    clear_all = pd.concat(filtered_clear_all, ignore_index=True) # transform list to dataframe
    cloud_all = pd.concat(filtered_cloud_all, ignore_index=True)
    

    
    #--- Define Temperature range ---
    
    st.subheader("Define Temperature Range for Solubility Curve")
    
    col1, col2 = st.columns([1,1])
    with col1:
        lower = st.number_input("Input lower temperature",min_value=-100, max_value=300, value=-5)
    with col2:
        upper = st.number_input("Input upper temperature",min_value=-100, max_value=300, value=100,)
        
    #Title:
    
    title = st.text_input("Set Solubility Curve Title - if required", "")
    
    #--- Input temperatures for theoretical yield calculation ---
    st.subheader("Define Temperature Range for Theoretical Yield Calculation")
    
    col1,col2 = st.columns([1,1])
    with col1:
        low = st.number_input("Input lower temperature", min_value=-100, value=0, max_value=200)
    with col2:
        high = st.number_input("Input upper temperature", min_value=-100, value=50, max_value=200)
    

    #-- Plotting Solubility Curves --# 
        
    if st.button("Generate Solubility Curves and Calculate Theoretical Yield"):
        with st.spinner():
             
            st.subheader("Solubility Curves")           
            #--- Create dataframe for solubility curves ---
            
            values = list(range(lower, (upper+1) , 5))

            clear_sol = pd.DataFrame({"Temperature": values})
            clear_sol["x"] = 1/(clear_sol["Temperature"]+273)
            cloud_sol = pd.DataFrame({"Temperature": values})
            cloud_sol["x"] = 1/(cloud_sol["Temperature"]+273)
            
            
                        
            # --- Iterating over Clear points ---
            #st.subheader("Clear points")
            for _, row in final_clear.iterrows():
                solvent = row["Solvent"]
                cycle = row["Cycle"]
                slope = row["Slope (m)"]
                intercept = row["Intercept (c)"]

                # Calculate solubility at each temperature
                col_name = f"{solvent} Cycle {cycle} - Clear"
                clear_sol[col_name] = np.exp(slope * clear_sol["x"] + intercept)

            #st.write("Generated Solubility Curve Data:")
            #st.dataframe(clear_sol)
            
        
            # --- Iterating over Cloud points ---
            
            #st.subheader("Cloud points")
            for _, row in final_cloud.iterrows():
                solvent = row["Solvent"]
                cycle = row["Cycle"]
                slope = row["Slope (m)"]
                intercept = row["Intercept (c)"]

                # Calculate solubility at each temperature
                col_name = f"{solvent} Cycle {cycle} - Cloud"
                cloud_sol[col_name] = np.exp(slope * cloud_sol["x"] + intercept)

            #st.write("Generated Solubility Curve Data:")
            #st.dataframe(cloud_sol)
            
            # --- Plotting Solubility Curves ---
            
            
            plt.figure(figsize=(10, 8))
            
            colours = ["#118ab2", "#ef476f", "#f48c06", "#ba8cfc","#5cfab0"]
            

            all_solvents = pd.concat([clear_all[["Solvent"]], cloud_all[["Solvent"]]])
            solvents_unique = all_solvents["Solvent"].unique().tolist()

            # --- Assign a color to each solvent ---
            colour_map = {solvent: colours[i % len(colours)] for i, solvent in enumerate(solvents_unique)}

            for col in [c for c in clear_sol.columns if c not in ["Temperature", "x"]]:
                solvent_name = col.split(" Cycle")[0]  # e.g. "Methanol"
                plt.plot(clear_sol["Temperature"], clear_sol[col], label=col, color=colour_map[solvent_name])

            for col in [c for c in cloud_sol.columns if c not in ["Temperature", "x"]]:
                solvent_name = col.split(" Cycle")[0]
                plt.plot(cloud_sol["Temperature"], cloud_sol[col], label=col, linestyle="--", color=colour_map[solvent_name])

            for solvent in clear_all["Solvent"].unique():
                sub_df = clear_all[clear_all["Solvent"] == solvent]
                plt.scatter(sub_df["Temps"], sub_df["Concentration"], marker='o', color=colour_map[solvent], label=f"{solvent} - Clear")

            for solvent in cloud_all["Solvent"].unique():
                sub_df = cloud_all[cloud_all["Solvent"] == solvent]
                plt.scatter(sub_df["Temps"], sub_df["Concentration"], marker='x', color=colour_map[solvent], label=f"{solvent} - Cloud")    
            
            plt.xlabel("Temperature (°C)", fontsize = 12)
            plt.ylabel("Solubility (mg/ml)", fontsize = 12)
            plt.title(title, fontsize = 15)
            plt.legend(loc='upper left', bbox_to_anchor=(1,1))
            st.pyplot(plt.gcf())
            
            # ----- Calculating Yield --------
            
            st.subheader("Theoretical Yield")
            
                                    
            #---Add new columns to existing dataframes ---
                            
            #-- Clear Point data ---
            final_clear["Solubility lower T (mg/ml)"] = round((np.exp((final_clear["Slope (m)"]*(1/(low + 273)))+final_clear["Intercept (c)"])),2)
            final_clear["Solubility upper T (mg/ml)"] = round((np.exp((final_clear["Slope (m)"]*(1/(high + 273)))+final_clear["Intercept (c)"])),2)
            final_clear["Solvent Vols upper T"] = round(((1*1000)/final_clear["Solubility upper T (mg/ml)"]),2)
            final_clear["Theoretical Yield"] = round((((final_clear["Solubility upper T (mg/ml)"]-final_clear["Solubility lower T (mg/ml)"])/(final_clear["Solubility upper T (mg/ml)"]))*100),2)
            
            #-- Cloud Point data ---
            final_cloud["Solubility lower T (mg/ml)"] = round((np.exp((final_cloud["Slope (m)"]*(1/(low + 273)))+final_cloud["Intercept (c)"])), 2)
            final_cloud["Solubility upper T (mg/ml)"] = round((np.exp((final_cloud["Slope (m)"]*(1/(high + 273)))+final_cloud["Intercept (c)"])), 2)
            final_cloud["Solvent Vols upper T"] = round(((1*1000)/final_cloud["Solubility upper T (mg/ml)"]),2)
            final_cloud["Theoretical Yield"] = round((((final_cloud["Solubility upper T (mg/ml)"]-final_cloud["Solubility lower T (mg/ml)"])/(final_cloud["Solubility upper T (mg/ml)"]))*100),2)
            
            #--- Print dataframe ---

            cols = [
                "Solvent",
                "Solubility lower T (mg/ml)",
                "Solubility upper T (mg/ml)",
                "Solvent Vols upper T",
                "Theoretical Yield"
            ]
            
            st.write("Clear Point Results")
            st.dataframe(final_clear[cols], use_container_width=True, hide_index=True)
            
            st.write("Cloud Point Results")
            st.dataframe(final_cloud[cols], use_container_width=True, hide_index=True)
            
            
            
                    
            
            
    
        
            

        

            
            
            
            

            


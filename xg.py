import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
import plotly.express as px
import json

# Load recepten from JSON with fallback to hardcoded for robustness
try:
    with open('recepten.json', 'r') as f:
        recepten = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    st.error("Error loading recepten.json. Falling back to hardcoded data.")
    recepten = {
        "Zero Waste Solution A": {
            "Hoeveelheid": 1100.0,
            "Dichtheid": 1.307,
            "Max_Volume": None,
            "Data": {
                'Grondstof': ['Fosforzuur 75%', 'Magnesiumchloride', 'Yzerchloride', 'Cobaltsulfaat oplossing', 'Nikkelsulfaat oplossing', 'Demiwater'],
                'Element': ['P2O5', 'MgO', 'Fe', 'Co', 'Ni', 'Geen'],
                'Gemeten_Concentratie': [23.7, 2.7, 1.17, 105, 200, 0],
                'Eenheid': ['wt%', 'wt%', 'wt%', 'mg/kg', 'mg/kg', 'wt%'],
                'Spec': [25, 3, 1.175, 110, 220, 0],
                'Min': [24.2, 2.7, 1.1, 90, 190, 0],
                'Max': [25.8, 3.3, 1.25, 130, 250, 0],
                'Concentratie_Grondstof': [74.9, 19.7, 13.7, 8.2, 6.4, 0],
                'Ratio_Element': [0.5424258, 0.197, 0.137, 0.082, 0.064, 0]
            }
        },
        "DKP 0-20-25": {
            "Hoeveelheid": 14100.0,
            "Dichtheid": 1.503,
            "Max_Volume": 27.33,
            "Data": {
                'Grondstof': ['Fosforzuur 75%', 'Kaliloog 50%', 'Demiwater'],
                'Element': ['P2O5', 'K2O', 'Geen'],
                'Gemeten_Concentratie': [18.2, 25.4, 0],
                'Eenheid': ['wt%', 'wt%', 'wt%'],
                'Spec': [20.25, 25.25, 0],
                'Min': [19.5, 24.5, 0],
                'Max': [21, 26, 0],
                'Concentratie_Grondstof': [75, 50.3, 0],
                'Ratio_Element': [0.54315, 0.4222685, 0]
            }
        },
        "PB Bravo": {
            "Hoeveelheid": 24000.0,
            "Dichtheid": 1.295,
            "Max_Volume": 18.0,
            "Data": {
                'Grondstof': ['Ureum prills', 'Zwavelzuur 96%', 'Demiwater'],
                'Element': ['N-Ureum', 'SO3', 'Geen'],
                'Gemeten_Concentratie': [19.4, 13.9, 0],
                'Eenheid': ['wt%', 'wt%', 'wt%'],
                'Spec': [22, 10, 0],
                'Min': [21.4, 9.1, 0],
                'Max': [22.6, 10.9, 0],
                'Concentratie_Grondstof': [46, 78.36, 0],
                'Ratio_Element': [0.46, 0.7836, 0]
            }
        }
    }

# Load tanks from JSON (no fallback)
try:
    with open('tanks.json', 'r') as f:
        tanks = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    st.error(f"Error loading tanks.json: {str(e)}. Please ensure the file exists and is valid JSON.")
    st.stop()

# Titel van de app
st.title("Correctie Recept Rekenmodule")

# Recept- en tankselectie
col_recept, col_tank = st.columns(2)
with col_recept:
    recept = st.selectbox("Kies een recept", list(recepten.keys()), help="Selecteer het recept om te bewerken.")
with col_tank:
    tank = st.selectbox("Kies een tank", list(tanks.keys()), help="Selecteer de tank/mixer voor de berekening.")

# Algemene gegevens
with st.expander("Algemene Gegevens", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        hoeveelheid = st.number_input("Hoeveelheid (liter)", value=recepten[recept]["Hoeveelheid"], step=100.0, min_value=0.0, help="De totale hoeveelheid product in liters.")
    with col2:
        dichtheid = st.number_input("Dichtheid (kg/l)", value=recepten[recept]["Dichtheid"], step=0.01, min_value=0.01, help="De dichtheid van het product in kg per liter.", format="%.3f")
    massa_product = hoeveelheid * dichtheid
    st.info(f"Originele massa product: **{massa_product:.2f} kg**")

# Toggle voor Max_Volume constraint
col_max_vol1, col_max_vol2 = st.columns([1, 2])
with col_max_vol1:
    use_max_volume = st.checkbox("Gebruik Max. Volume beperking", value=True, help="Schakel uit om de maximale volumebeperking te negeren.")
with col_max_vol2:
    default_max_volume_m3 = tanks[tank]["Max_Volume_m3"]
    max_volume_m3 = st.number_input(
        "Max Volume (m³)",
        min_value=0.0,
        value=default_max_volume_m3,
        step=0.001,
        disabled=not use_max_volume,
        help=f"Maximale volumebeperking voor {tanks[tank]['Name']} in kubieke meters."
    )
    max_volume_liters = max_volume_m3 * 1000.0

# Gegevens uit het geselecteerde recept
data = recepten[recept]["Data"]
df = pd.DataFrame(data)

# Initialize step sizes in session state with default 0
if ('stapgroottes' not in st.session_state or
    st.session_state.get('current_recept') != recept or
    st.session_state.get('current_tank') != tank):
    st.session_state.stapgroottes = {
        row['Grondstof']: 0
        for _, row in df.iterrows()
    }
    st.session_state.current_recept = recept
    st.session_state.current_tank = tank

# Input voor gemeten concentraties en stapgroottes
with st.expander("Gemeten Concentratie en Stapgroottes per Element", expanded=True):
    st.info("Voer de gemeten concentraties en stapgroottes in (wt% of mg/kg, afhankelijk van de eenheid). Stapgrootte 0 betekent geen stappen.")
    measured_concentrations = {}
    cols = st.columns(2)
    for idx, row in df.iterrows():
        grondstof = row['Grondstof']
        element = row['Element']
        eenheid = row['Eenheid']
        default_conc = row['Gemeten_Concentratie']
        with cols[idx % 2]:
            if element != 'Geen':
                col_conc, col_step = st.columns([2, 1])
                with col_conc:
                    conc = st.number_input(
                        f"{element} ({eenheid}) voor {grondstof}",
                        value=float(default_conc),
                        step=0.01 if eenheid == 'wt%' else 1.0,
                        min_value=0.0,
                        format="%.2f" if eenheid == 'wt%' else "%.0f",
                        key=f"conc_{recept}_{tank}_{element}",
                        help=f"Voer de gemeten concentratie van {element} in {eenheid}."
                    )
                    measured_concentrations[element] = conc
                with col_step:
                    default_step = tanks[tank]["Stapgrootte"].get(grondstof, 0)
                    step_value = st.number_input(
                        "Stap (kg)",
                        value=st.session_state.stapgroottes.get(grondstof, default_step),
                        step=1,
                        min_value=0,
                        key=f"step_{recept}_{tank}_{grondstof}",
                        help=f"Stapgrootte voor {grondstof} in kg (0 voor geen stappen, standaard: {default_step} kg)."
                    )
                    st.session_state.stapgroottes[grondstof] = int(step_value)
            else:
                measured_concentrations[element] = 0.0
                col_conc, col_step = st.columns([2, 1])
                with col_conc:
                    st.number_input(
                        f"{element} ({eenheid}) voor {grondstof}",
                        value=0.0,
                        disabled=True,
                        key=f"conc_{recept}_{tank}_{element}"
                    )
                with col_step:
                    default_step = tanks[tank]["Stapgrootte"].get(grondstof, 0)
                    step_value = st.number_input(
                        "Stap (kg)",
                        value=st.session_state.stapgroottes.get(grondstof, default_step),
                        step=1,
                        min_value=0,
                        key=f"step_{recept}_{tank}_{grondstof}",
                        help=f"Stapgrootte voor {grondstof} in kg (0 voor geen stappen, standaard: {default_step} kg)."
                    )
                    st.session_state.stapgroottes[grondstof] = int(step_value)

# Section for Prefer Water Option
with st.expander("Correctie Instellingen", expanded=True):
    prefer_water = st.checkbox(
        "Prefereer Demiwater voor verdunning (goedkoopste optie)",
        value=True,
        help="Minimaliseert toevoeging van duurdere grondstoffen door water te prioriteren."
    )

# Update DataFrame met ingevoerde concentraties
df['Gemeten_Concentratie'] = df['Element'].map(measured_concentrations)

# Berekeningen zoals in Excel
def bereken_actuele_hoeveelheid_element(row, massa_product):
    """Berekening kolom F: Actuele hoeveelheid element in kg."""
    conversie_factor = 100 if row['Eenheid'] == 'wt%' else 1000000
    return (massa_product * row['Gemeten_Concentratie']) / conversie_factor

def bereken_actuele_hoeveelheid_grondstof(row):
    """Berekening kolom G: Actuele hoeveelheid grondstof in kg."""
    if row['Ratio_Element'] == 0:
        return 0.0
    return row['Actuele_Hoeveelheid_Element'] / row['Ratio_Element']

def bereken_toegevoegd_element(row):
    """Berekening kolom I: Toegevoegd element door grondstoftoevoeging."""
    return row['Toevoegen Grondstof'] * row['Ratio_Element']

def bereken_gecorrigeerde_concentratie(row, totale_massa_correctie):
    """Berekening kolom J: Gecorrigeerde concentratie na toevoeging."""
    conversie_factor = 100 if row['Eenheid'] == 'wt%' else 1000000
    totale_element = row['Actuele_Hoeveelheid_Element'] + row['Geoptimaliseerde_Toegevoegd_Element']
    return (totale_element / totale_massa_correctie) * conversie_factor

# Bereken actuele hoeveelheid element (kolom F)
df['Actuele_Hoeveelheid_Element'] = df.apply(bereken_actuele_hoeveelheid_element, axis=1, massa_product=massa_product)

# Bereken actuele hoeveelheid grondstof (kolom G)
df['Actuele_Hoeveelheid_Grondstof'] = df.apply(bereken_actuele_hoeveelheid_grondstof, axis=1)

# Optimized Function (Simplified, No Buffer Factor)
def optimize_toevoegingen(massa_product, actuele_hoeveelheden, ratios, eenheden, min_specs, max_specs, specs, use_max_volume, max_volume, stapgroottes, prefer_water, asymmetric_above=False):
    if massa_product <= 0:
        return np.zeros(len(ratios), dtype=int), massa_product, "Geen massa; geen optimalisatie nodig."

    prob = LpProblem("Minimize_Additions_And_Deviations", LpMinimize)

    # Define variables for additions
    toevoegingen = {}
    for i in range(len(ratios)):
        stapgrootte = stapgroottes[i]
        if stapgrootte > 0:
            aantal_stappen = LpVariable(f"steps_{i}", lowBound=0, cat='Integer')
            toevoegingen[i] = aantal_stappen * stapgrootte
        else:
            toevoegingen[i] = LpVariable(f"add_{i}", lowBound=0, cat='Integer')

    # Total mass
    totale_massa = massa_product + lpSum(toevoegingen.values())

    # Objective: Minimize additions (weighted) + deviations from target
    costs = [0.01 if prefer_water and df['Grondstof'][i] == 'Demiwater' else 1 for i in range(len(ratios))]
    objective = lpSum(costs[i] * toevoegingen[i] for i in range(len(ratios)))

    # Constraints and deviations
    diagnostics = []
    deviation_weight = 10000  # High weight to prioritize exact target
    for i in range(len(ratios)):
        if ratios[i] == 0:
            continue
        conversie_factor = 100 if eenheden[i] == 'wt%' else 1000000
        totale_element = actuele_hoeveelheden[i] + toevoegingen[i] * ratios[i]
        current_conc = (actuele_hoeveelheden[i] / massa_product) * conversie_factor if massa_product > 0 else 0

        # Define target based on consensus, with asymmetric option
        if current_conc < min_specs[i]:
            target = (min_specs[i] + specs[i]) / 2
            diag_msg = f"midpoint between Min {min_specs[i]:.2f} and Spec {specs[i]:.2f}"
        elif current_conc > max_specs[i]:
            if asymmetric_above:
                target = (min_specs[i] + specs[i]) / 2
                diag_msg = f"asymmetric midpoint between Min {min_specs[i]:.2f} and Spec {specs[i]:.2f}"
            else:
                target = (specs[i] + max_specs[i]) / 2
                diag_msg = f"midpoint between Spec {specs[i]:.2f} and Max {max_specs[i]:.2f}"
        else:
            target = specs[i]
            diag_msg = f"Spec {specs[i]:.2f}"

        diagnostics.append(f"{df['Element'][i]} target: {target:.2f} ({diag_msg}, current: {current_conc:.2f})")

        # Convert concentration constraints to linear form
        target_element_mass = target * totale_massa / conversie_factor
        min_element_mass = min_specs[i] * totale_massa / conversie_factor
        max_element_mass = max_specs[i] * totale_massa / conversie_factor

        # Deviation variables for target
        dev_plus = LpVariable(f"dev_plus_{i}", lowBound=0)
        dev_minus = LpVariable(f"dev_minus_{i}", lowBound=0)
        prob += totale_element == target_element_mass + dev_plus - dev_minus, f"target_dev_{i}"
        objective += deviation_weight * (dev_plus + dev_minus)

        # Hard min/max constraints
        prob += totale_element >= min_element_mass, f"Min_{i}"
        prob += totale_element <= max_element_mass, f"Max_{i}"

    prob += objective  # Set full objective

    # Volume constraint
    if use_max_volume and max_volume is not None and max_volume > 0:
        prob += totale_massa / dichtheid <= max_volume, "Max_Volume_Constraint"

    # Solve
    prob.solve(PULP_CBC_CMD(timeLimit=60, msg=1))

    if LpStatus[prob.status] == 'Optimal':
        optimized_toevoegingen = []
        for i in range(len(ratios)):
            if stapgroottes[i] > 0:
                aantal_stappen = prob.variablesDict().get(f"steps_{i}")
                value = int(round(aantal_stappen.varValue * stapgroottes[i])) if aantal_stappen and aantal_stappen.varValue is not None else 0
            else:
                value = int(toevoegingen[i].varValue) if toevoegingen[i].varValue is not None else 0
            optimized_toevoegingen.append(value)
            print(f"Grondstof {df['Grondstof'][i]}: Raw addition = {value} kg")
        optimized_toevoegingen = np.array(optimized_toevoegingen, dtype=int)
        optimized_totale_massa = massa_product + sum(optimized_toevoegingen)
        for diag in diagnostics:
            print(diag)
        return optimized_toevoegingen, optimized_totale_massa, "Succes"
    else:
        fail_msg = f"Optimalisatie mislukt: {LpStatus[prob.status]}. Targets:\n" + "\n".join(diagnostics)
        return None, None, fail_msg

# Knop voor optimalisatie
st.header("Geoptimaliseerde Toevoegingen")
if st.button("Bereken Geoptimaliseerde Toevoegingen", help="Klik om de optimalisatie te starten op basis van de ingevoerde concentraties."):
    with st.spinner("Optimalisatie wordt uitgevoerd..."):
        stapgroottes = [st.session_state.stapgroottes[grondstof] for grondstof in df['Grondstof']]
        
        optimized_toevoegingen, optimized_totale_massa, status = optimize_toevoegingen(
            massa_product, df['Actuele_Hoeveelheid_Element'].values, df['Ratio_Element'].values, 
            df['Eenheid'].values, df['Min'].values, df['Max'].values, df['Spec'].values, 
            use_max_volume, max_volume_liters, stapgroottes, prefer_water
        )

    if status == "Succes":
        df['Toevoegen Grondstof'] = optimized_toevoegingen
        df['Geoptimaliseerde_Toegevoegd_Element'] = df.apply(bereken_toegevoegd_element, axis=1)
        df['Na Optimalisatie'] = df.apply(bereken_gecorrigeerde_concentratie, axis=1, totale_massa_correctie=optimized_totale_massa)
        df['Geoptimaliseerde_Binnen_Specs'] = (df['Na Optimalisatie'] >= df['Min']) & (df['Na Optimalisatie'] <= df['Max'])

        totale_toevoeging = optimized_totale_massa - massa_product
        percentage_toename = (totale_toevoeging / massa_product) * 100 if massa_product > 0 else 0
        st.success(f"Optimalisatie succesvol! Totale massa na correctie: **{optimized_totale_massa:.2f} kg** (toevoeging: **{totale_toevoeging:.2f} kg**, {percentage_toename:.2f}% toename)")

        def highlight_specs(val):
            color = 'green' if val else 'red'
            return f'background-color: {color}'

        def format_number(val, eenheid):
            if eenheid == 'mg/kg' or val == 0:
                return '{:.0f}'.format(val)  # No decimals for mg/kg or zero
            return '{:.3f}'.format(val).rstrip('0').rstrip('.')  # Up to 3 decimals for wt%, strip trailing zeros

        # Create format dictionary for columns
        format_dict = {
            'Toevoegen Grondstof': '{:.0f}',
            'Gemeten_Concentratie': lambda x: '{:.2f}'.format(x),
            'Na Optimalisatie': lambda x: '{:.2f}'.format(x),
            'Spec': lambda x: format_number(x, df.loc[df['Spec'] == x, 'Eenheid'].iloc[0] if not df.loc[df['Spec'] == x, 'Eenheid'].empty else 'wt%'),
            'Min': lambda x: format_number(x, df.loc[df['Min'] == x, 'Eenheid'].iloc[0] if not df.loc[df['Min'] == x, 'Eenheid'].empty else 'wt%'),
            'Max': lambda x: format_number(x, df.loc[df['Max'] == x, 'Eenheid'].iloc[0] if not df.loc[df['Max'] == x, 'Eenheid'].empty else 'wt%')
        }

        optimized_resultaten = df[['Grondstof', 'Element', 'Gemeten_Concentratie', 'Toevoegen Grondstof', 'Na Optimalisatie', 'Spec', 'Min', 'Max', 'Geoptimaliseerde_Binnen_Specs']]
        st.dataframe(optimized_resultaten.style.applymap(highlight_specs, subset=['Geoptimaliseerde_Binnen_Specs']).format(format_dict))

        st.subheader("Waarschuwingen")
        warnings_present = False
        for index, row in df.iterrows():
            if not row['Geoptimaliseerde_Binnen_Specs'] and row['Element'] != 'Geen':
                st.warning(f"{row['Element']} ({row['Na Optimalisatie']:.2f} {row['Eenheid']}) ligt buiten specificaties ({row['Min']} - {row['Max']})")
                warnings_present = True
        if not warnings_present:
            st.success("Alle concentraties liggen binnen de specificaties!")

        st.subheader("Visualisatie van Concentratievergelijking")
        viz_df = df.melt(id_vars=['Element', 'Eenheid', 'Min', 'Max'], value_vars=['Gemeten_Concentratie', 'Na Optimalisatie', 'Spec'], var_name='Type', value_name='Waarde')
        viz_df['Error_Low'] = viz_df['Waarde'] - viz_df['Min']
        viz_df['Error_High'] = viz_df['Max'] - viz_df['Waarde']
        fig = px.bar(viz_df, x='Element', y='Waarde', color='Type', barmode='group', facet_col='Eenheid',
                     error_y='Error_High', error_y_minus='Error_Low',
                     title='Vergelijking van Concentratie: Gemeten vs Geoptimaliseerd vs Specificaties (Gescheiden per Eenheid)',
                     labels={'Waarde': 'Concentratie', 'Element': 'Element'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        csv = optimized_resultaten.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Resultaten als CSV", data=csv, file_name=f'geoptimaliseerde_correctie_{recept}_{tank}.csv', mime='text/csv', help="Download de geoptimaliseerde resultaten als CSV-bestand.")
    else:
        st.error(status)
else:
    st.info("Klik op de knop om de optimalisatie te starten.")
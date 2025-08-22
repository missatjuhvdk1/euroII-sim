import io
import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
import plotly.express as px
import json

# Set page configuration for wide mode at the start
st.set_page_config(layout="wide")

# Load recepten from JSON with fallback to hardcoded for robustness
try:
    with open('recepten.json', 'r') as f:
        recepten = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    st.error("Error loading recepten.json. Falling back to hardcoded data.")
    st.stop

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

# Initialize step sizes and costs in session state
if ('stapgroottes' not in st.session_state or
    'kosten' not in st.session_state or
    st.session_state.get('current_recept') != recept or
    st.session_state.get('current_tank') != tank):
    st.session_state.stapgroottes = {
        row['Grondstof']: 0  # Default step size is 0 for all grondstoffen
        for _, row in df.iterrows()
    }
    st.session_state.kosten = {
        row['Grondstof']: 0.01 if row['Grondstof'] == 'Demiwater' else 1.0
        for _, row in df.iterrows()
    }
    st.session_state.current_recept = recept
    st.session_state.current_tank = tank

# Input for measured concentrations
with st.expander("Gemeten Concentratie per Element", expanded=True):
    st.info("Voer de gemeten concentraties in (wt% of mg/kg, afhankelijk van de eenheid).")
    measured_concentrations = {}
    for idx, row in df.iterrows():
        element = row['Element']
        eenheid = row['Eenheid']
        default_conc = row['Gemeten_Concentratie']  # From recepten data
        if element != 'Geen':
            conc = st.number_input(
                f"{element} ({eenheid}) voor {row['Grondstof']}",
                value=float(default_conc),
                step=0.01 if eenheid == 'wt%' else 1.0,
                min_value=0.0,
                format="%.2f" if eenheid == 'wt%' else "%.0f",
                key=f"conc_{recept}_{tank}_{element}",
                help=f"Voer de gemeten concentratie van {element} in {eenheid}."
            )
            measured_concentrations[element] = conc
        else:
            measured_concentrations[element] = 0.0
            st.number_input(
                f"{element} ({eenheid}) voor {row['Grondstof']}",
                value=0.0,
                disabled=True,
                key=f"conc_{recept}_{tank}_{element}"
            )

# Optimalisatie Penalties
with st.expander("Optimalisatie Penalties", expanded=True):
    st.info("Stel de verhouding in tussen de penalties voor massaminimalisatie en specificatie afwijkingen. De slider bepaalt de verhouding. Standaard is 1:10")

    # Initialize session state if not already set
    if 'ratio_log' not in st.session_state:
        st.session_state.ratio_log = 1.0  # Default to 1:1 ratio
    if 'mass_penalty' not in st.session_state:
        st.session_state.mass_penalty = 10000.0
    if 'deviation_weight' not in st.session_state:
        st.session_state.deviation_weight = 10000.0

    # Reset button for penalties
    if st.button("Reset Penalties naar Standaard", help="Herstel de standaard penalties (verhouding 1:1)"):
        st.session_state.ratio_log = 1.0
        st.session_state.mass_penalty = 10000.0
        st.session_state.deviation_weight = 10000.0

    # Ratio slider with explicit callback to update session state
    def update_ratio():
        st.session_state.ratio_log = st.session_state.ratio_slider

    ratio_log = st.slider(
        "Straf Verhouding (Massaminimalisatie : Specificatie Afwijkingen)",
        min_value=0.0,  # Represents 1:1 (log10(1))
        max_value=2.0,  # Represents 100:1 (log10(100))
        value=st.session_state.ratio_log,  # Use session state for slider value
        step=0.1,
        format="%.1f",
        key="ratio_slider",
        on_change=update_ratio,  # Update session state on slider change
        help="Verplaatst de slider om de verhouding tussen massaminimalisatie en specificatie afwijkingen aan te passen. 0.0 = 1:1, 1.0 = 10:1, 2.0 = 100:1."
    )

    # Calculate penalties based on the ratio
    ratio = 10 ** st.session_state.ratio_log  # Use session state directly
    base_penalty = 10000.0

    # Mass penalty is base_penalty, deviation penalty is base_penalty * ratio
    mass_penalty = base_penalty
    deviation_weight = base_penalty * ratio

    # Ensure penalties stay within reasonable bounds
    mass_penalty = max(0.0, min(mass_penalty, 1000000.0))
    deviation_weight = max(0.0, min(deviation_weight, 1000000.0))

    # Update session state
    st.session_state.mass_penalty = mass_penalty
    st.session_state.deviation_weight = deviation_weight

# Reset session state bij wijziging van recept of tank
if ('current_recept' not in st.session_state or st.session_state.current_recept != recept or
    'current_tank' not in st.session_state or st.session_state.current_tank != tank):
    st.session_state.stapgroottes = {
        row['Grondstof']: 0 for _, row in df.iterrows()
    }
    st.session_state.kosten = {
        row['Grondstof']: 0.01 if row['Grondstof'] == 'Demiwater' else 1.0
        for _, row in df.iterrows()
    }
    st.session_state.excluded_indices = []
    st.session_state.current_recept = recept
    st.session_state.current_tank = tank

# Stapgroottes sectie
with st.expander("Stapgroottes", expanded=False):
    st.info("Voer de stapgroottes (in kg) in. Stapgrootte 0 betekent geen stappen.")
    for idx, row in df.iterrows():
        grondstof = row['Grondstof']
        step_value = st.number_input(
            f"Stapgrootte (kg) voor {grondstof}",
            value=st.session_state.stapgroottes.get(grondstof, 0),  # Default to 0
            step=1,
            min_value=0,
            key=f"step_{recept}_{tank}_{grondstof}_{idx}",  # Unique key with idx
            help=f"Stapgrootte voor {grondstof} in kg (0 voor geen stappen)."
        )
        st.session_state.stapgroottes[grondstof] = int(step_value)

# Update DataFrame met ingevoerde concentraties
df['Gemeten Concentratie'] = df['Element'].map(measured_concentrations)

# Berekeningen zoals in Excel
def bereken_actuele_hoeveelheid_element(row, massa_product):
    """Berekening kolom F: Actuele hoeveelheid element in kg."""
    conversie_factor = 100 if row['Eenheid'] == 'wt%' else 1000000
    return (massa_product * row['Gemeten Concentratie']) / conversie_factor

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

def optimize_toevoegingen(massa_product, actuele_hoeveelheden, ratios, eenheden, min_specs, max_specs, specs, use_max_volume, max_volume, stapgroottes, mass_penalty, deviation_weight, excluded_indices=None, fixed_values=None, refine=False):
    if excluded_indices is None:
        excluded_indices = []
    if fixed_values is None:
        fixed_values = {}
    if massa_product <= 0:
        return np.zeros(len(ratios), dtype=float), massa_product, "Geen massa; geen optimalisatie nodig."

    prob = LpProblem("Minimize_Additions_And_Deviations", LpMinimize)

    # Define toevoegingen: variables or constants
    toevoegingen = {}
    for i in range(len(ratios)):
        if i in excluded_indices:
            toevoegingen[i] = 0.0
        elif i in fixed_values:
            toevoegingen[i] = fixed_values[i]
        else:
            if refine:
                # Continuous for refinement
                toevoegingen[i] = LpVariable(f"add_{i}", lowBound=0, cat='Continuous')
            else:
                stapgrootte = stapgroottes[i]
                if stapgrootte > 0:
                    aantal_stappen = LpVariable(f"steps_{i}", lowBound=0, cat='Integer')
                    toevoegingen[i] = aantal_stappen * stapgrootte
                else:
                    toevoegingen[i] = LpVariable(f"add_{i}", lowBound=0, cat='Integer')

    # Identify variables and fixed additions
    var_toevoegingen = {i: t for i, t in toevoegingen.items() if isinstance(t, LpVariable)}
    fixed_add_mass = sum([t for i, t in toevoegingen.items() if not isinstance(t, LpVariable)])
    fixed_cost = sum([st.session_state.kosten[df['Grondstof'][i]] * toevoegingen[i] for i in toevoegingen if not isinstance(toevoegingen[i], LpVariable)])

    # Total mass
    totale_massa = massa_product + fixed_add_mass + lpSum(var_toevoegingen.values())

    # Objective
    costs = [st.session_state.kosten[df['Grondstof'][i]] for i in range(len(ratios))]
    objective = fixed_cost + lpSum(costs[i] * var_toevoegingen[i] for i in var_toevoegingen)
    objective += mass_penalty * (fixed_add_mass + lpSum(var_toevoegingen.values()))
    
    # Effective initial for diagnostics
    effective_massa = massa_product + fixed_add_mass
    effective_actuele = [actuele_hoeveelheden[i] + (fixed_values.get(i, 0) if i in fixed_values else 0) * ratios[i] for i in range(len(ratios))]

    # Diagnostics and targets
    diagnostics = []
    target_set = []
    targets = {}
    for i in range(len(ratios)):
        if ratios[i] == 0 or i in excluded_indices:
            diagnostics.append(f"{df['Element'][i]}: Excluded or no ratio, skipped")
            continue
        conversie_factor = 100 if eenheden[i] == 'wt%' else 1000000
        current_conc = (effective_actuele[i] / effective_massa) * conversie_factor if effective_massa > 0 else 0
        if current_conc < specs[i]:
            target_set.append(i)
            target = (min_specs[i] + specs[i]) / 2
            targets[i] = target
            diag_msg = f"midpoint between Min {min_specs[i]:.2f} and Spec {specs[i]:.2f}"
        elif current_conc > specs[i]:
            target_set.append(i)
            target = (specs[i] + max_specs[i]) / 2
            targets[i] = target
            diag_msg = f"midpoint between Spec {specs[i]:.2f} and Max {max_specs[i]:.2f}"
        else:
            diag_msg = f"At or equal to Spec, no target set (current: {current_conc:.2f})"
        target_value = targets.get(i, None)
        target_str = f"{target_value:.2f}" if isinstance(target_value, (int, float)) else "None"
        diagnostics.append(f"{df['Element'][i]} target: {target_str} ({diag_msg}, current: {current_conc:.2f})")

    # Constraints and deviations
    for i in target_set:
        conversie_factor = 100 if eenheden[i] == 'wt%' else 1000000
        totale_element = effective_actuele[i] + toevoegingen[i] * ratios[i] - (fixed_values.get(i, 0) * ratios[i] if i in fixed_values else 0)
        target_element_mass = targets[i] * totale_massa / conversie_factor
        dev_plus = LpVariable(f"dev_plus_{i}", lowBound=0)
        dev_minus = LpVariable(f"dev_minus_{i}", lowBound=0)
        prob += totale_element == target_element_mass + dev_plus - dev_minus, f"target_dev_{i}"
        objective += deviation_weight * (dev_plus + dev_minus)

    # Hard min/max constraints
    for i in range(len(ratios)):
        if ratios[i] == 0 or i in excluded_indices:
            continue
        conversie_factor = 100 if eenheden[i] == 'wt%' else 1000000
        totale_element = effective_actuele[i] + toevoegingen[i] * ratios[i] - (fixed_values.get(i, 0) * ratios[i] if i in fixed_values else 0)
        min_element_mass = min_specs[i] * totale_massa / conversie_factor
        max_element_mass = max_specs[i] * totale_massa / conversie_factor
        prob += totale_element >= min_element_mass, f"Min_{i}"
        prob += totale_element <= max_element_mass, f"Max_{i}"

    prob += objective

    # Volume constraint
    if use_max_volume and max_volume is not None and max_volume > 0:
        effective_initial_volume = effective_massa / dichtheid
        if effective_initial_volume > max_volume:
            diagnostics.append(
                f"Effectieve initiële massa ({effective_massa:.2f} kg) resulteert in een volume van {effective_initial_volume:.2f} liter, "
                f"wat de maximale volumebeperking van {max_volume:.2f} liter overschrijdt."
            )
        prob += totale_massa / dichtheid <= max_volume, "Max_Volume_Constraint"

    # Solve
    prob.solve(PULP_CBC_CMD(timeLimit=180, msg=1))

    if LpStatus[prob.status] == 'Optimal':
        optimized_toevoegingen = []
        for i in range(len(ratios)):
            t = toevoegingen[i]
            if isinstance(t, LpVariable):
                value = t.varValue if t.varValue is not None else 0.0
                if not refine and stapgroottes[i] > 0:
                    value = round(value)
            else:
                value = t
            optimized_toevoegingen.append(value)
        optimized_toevoegingen = np.array(optimized_toevoegingen, dtype=float)
        optimized_totale_massa = massa_product + sum(optimized_toevoegingen)
        for diag in diagnostics:
            print(diag)
        print("\nMaterial Additions:")
        for i, value in enumerate(optimized_toevoegingen):
            if value > 0:
                print(f"{df['Grondstof'][i]}: {value} kg (Cost: {costs[i]:.2f})")
        return optimized_toevoegingen, optimized_totale_massa, "Succes"
    else:
        fail_msg = f"Optimalisatie mislukt: {LpStatus[prob.status]}. Targets:\n" + "\n".join(diagnostics)
        if use_max_volume and max_volume is not None and max_volume > 0:
            initial_volume = massa_product / dichtheid
            fail_msg += f"\nControleer de maximale volumebeperking: huidige massa ({massa_product:.2f} kg) heeft een volume van {initial_volume:.2f} liter, terwijl het maximum {max_volume:.2f} liter is."
        return None, None, fail_msg
    
# Initialisatie van session state
if 'excluded_indices' not in st.session_state:
    st.session_state.excluded_indices = []
if 'optimized_resultaten' not in st.session_state:
    st.session_state.optimized_resultaten = None
if 'optimized_status' not in st.session_state:
    st.session_state.optimized_status = None
if 'optimized_totale_massa' not in st.session_state:
    st.session_state.optimized_totale_massa = None
if 'optimized_toevoegingen' not in st.session_state:
    st.session_state.optimized_toevoegingen = None

# Knop voor optimalisatie
st.header("Geoptimaliseerde Toevoegingen")
if st.button("Bereken Geoptimaliseerde Toevoegingen", help="Klik om de optimalisatie te starten op basis van de ingevoerde concentraties."):
    with st.spinner("Optimalisatie wordt uitgevoerd..."):
        # Clear previous optimization results to avoid stale data
        st.session_state.optimized_resultaten = None
        st.session_state.optimized_status = None
        st.session_state.optimized_totale_massa = None
        st.session_state.optimized_toevoegingen = None
        
        stapgroottes = [st.session_state.stapgroottes[grondstof] for grondstof in df['Grondstof']]
        optimized_toevoegingen, optimized_totale_massa, status = optimize_toevoegingen(
            massa_product, 
            df['Actuele_Hoeveelheid_Element'].values, 
            df['Ratio_Element'].values,
            df['Eenheid'].values, 
            df['Min'].values, 
            df['Max'].values, 
            df['Spec'].values,
            use_max_volume, 
            max_volume_liters, 
            stapgroottes, 
            st.session_state.mass_penalty, 
            st.session_state.deviation_weight,
            excluded_indices=[]  # Geen exclusies voor originele optimalisatie
        )
        if status == "Succes":
            # Check for small additions to refine
            refine_indices = [i for i in range(len(optimized_toevoegingen)) if 0 < optimized_toevoegingen[i] <= 9]
            if refine_indices:
                fixed_values = {i: optimized_toevoegingen[i] for i in range(len(optimized_toevoegingen)) if i not in refine_indices}
                optimized_toevoegingen, optimized_totale_massa, status = optimize_toevoegingen(
                    massa_product, 
                    df['Actuele_Hoeveelheid_Element'].values, 
                    df['Ratio_Element'].values,
                    df['Eenheid'].values, 
                    df['Min'].values, 
                    df['Max'].values, 
                    df['Spec'].values,
                    use_max_volume, 
                    max_volume_liters, 
                    stapgroottes, 
                    st.session_state.mass_penalty, 
                    st.session_state.deviation_weight,
                    excluded_indices=[],
                    fixed_values=fixed_values,
                    refine=True
                )
            if status == "Succes":
                df['Toevoegen Grondstof'] = optimized_toevoegingen
                df['Geoptimaliseerde_Toegevoegd_Element'] = df.apply(bereken_toegevoegd_element, axis=1)
                df['Na Optimalisatie'] = df.apply(bereken_gecorrigeerde_concentratie, axis=1, totale_massa_correctie=optimized_totale_massa)
                df['Binnen Specs'] = (df['Na Optimalisatie'] >= df['Min']) & (df['Na Optimalisatie'] <= df['Max'])
                st.session_state.optimized_resultaten = df[['Grondstof', 'Element', 'Gemeten Concentratie', 'Toevoegen Grondstof', 'Na Optimalisatie', 'Spec', 'Min', 'Max', 'Binnen Specs', 'Eenheid']].copy()
                st.session_state.optimized_status = status
                st.session_state.optimized_totale_massa = optimized_totale_massa
                st.session_state.optimized_toevoegingen = optimized_toevoegingen
        if status != "Succes":
            st.error(f"Optimalisatie mislukt: {status}")
            if "Infeasible" in status and use_max_volume:
                st.warning(
                    f"De maximale volumebeperking ({max_volume_liters:.2f} liter) is waarschijnlijk te streng. "
                    "Probeer het maximale volume te verhogen of schakel de volumebeperking uit."
                )
            else:
                st.warning(
                    "De optimalisatie kon geen oplossing vinden. Controleer de invoerparameters (bijv. gemeten concentraties, specificaties, of stapgroottes) en probeer opnieuw."
                )

# Toon originele resultaten als ze beschikbaar zijn
if st.session_state.optimized_status == "Succes" and st.session_state.optimized_resultaten is not None:
    def highlight_specs(val):
        color = 'green' if val else 'red'
        return f'background-color: {color}'
    
    def format_number(val, eenheid):
        if eenheid == 'mg/kg' or val == 0:
            return '{:.0f}'.format(val)
        return '{:.3f}'.format(val).rstrip('0').rstrip('.')
    
    def format_toevoegen_grondstof(val):
        if val < 10:
            return '{:.1f}'.format(val)
        return '{:.0f}'.format(val)
    
    format_dict = {
        'Toevoegen Grondstof': format_toevoegen_grondstof,
        'Gemeten Concentratie': lambda x: '{:.2f}'.format(x),
        'Na Optimalisatie': lambda x: '{:.2f}'.format(x),
        'Spec': lambda x: format_number(x, st.session_state.optimized_resultaten.loc[st.session_state.optimized_resultaten['Spec'] == x, 'Eenheid'].iloc[0] if not st.session_state.optimized_resultaten.loc[st.session_state.optimized_resultaten['Spec'] == x, 'Eenheid'].empty else 'wt%'),
        'Min': lambda x: format_number(x, st.session_state.optimized_resultaten.loc[st.session_state.optimized_resultaten['Min'] == x, 'Eenheid'].iloc[0] if not st.session_state.optimized_resultaten.loc[st.session_state.optimized_resultaten['Min'] == x, 'Eenheid'].empty else 'wt%'),
        'Max': lambda x: format_number(x, st.session_state.optimized_resultaten.loc[st.session_state.optimized_resultaten['Max'] == x, 'Eenheid'].iloc[0] if not st.session_state.optimized_resultaten.loc[st.session_state.optimized_resultaten['Max'] == x, 'Eenheid'].empty else 'wt%')
    }
    
    totale_toevoeging = st.session_state.optimized_totale_massa - massa_product
    percentage_toename = (totale_toevoeging / massa_product) * 100 if massa_product > 0 else 0
    st.success(f"Optimalisatie succesvol! Totale massa na correctie: **{st.session_state.optimized_totale_massa:.2f} kg** (toevoeging: **{totale_toevoeging:.2f} kg**, {percentage_toename:.2f}% toename)")
    
    st.dataframe(st.session_state.optimized_resultaten.style.applymap(highlight_specs, subset=['Binnen Specs']).format(format_dict))
    
    st.subheader("Waarschuwingen")
    warnings_present = False
    for index, row in st.session_state.optimized_resultaten.iterrows():
        if not row['Binnen Specs'] and row['Element'] != 'Geen':
            st.warning(f"{row['Element']} ({row['Na Optimalisatie']:.2f} {row['Eenheid']}) ligt buiten specificaties ({row['Min']} - {row['Max']})")
            warnings_present = True
    if not warnings_present:
        st.success("Alle concentraties liggen binnen de specificaties!")
    
    # Export button for original results (Excel only)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        st.session_state.optimized_resultaten.to_excel(writer, index=False, sheet_name='Resultaten')
    excel_data = output.getvalue()
    st.download_button(
        label="Download Resultaten als Excel",
        data=excel_data,
        file_name=f'geoptimaliseerde_correctie_{recept}_{tank}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        help="Download de geoptimaliseerde resultaten als Excel-bestand."
    )

# Aangepaste Correctie Sectie
if st.session_state.optimized_toevoegingen is not None:
    excluded_indices = []
    st.info("Selecteer de grondstoffen die je wilt toevoegen. Vink uit om te excluderen bij heroptimalisatie.")
    for i, grondstof in enumerate(df['Grondstof']):
        default_value = st.session_state.get(f"exclude_{recept}_{tank}_{grondstof}_{i}", True)
        if not st.checkbox(
            f"{grondstof} toevoegen",
            value=default_value,
            key=f"exclude_{recept}_{tank}_{grondstof}_{i}"
        ):
            excluded_indices.append(i)
    st.session_state.excluded_indices = excluded_indices

    if st.button("Herbereken Zonder Geselecteerde Grondstoffen", help="Herbereken optimalisatie zonder de uitgeschakelde grondstoffen"):
        with st.spinner("Aangepaste optimalisatie wordt uitgevoerd..."):
            adj_df = df.copy()
            stapgroottes = [st.session_state.stapgroottes[grondstof] for grondstof in adj_df['Grondstof']]
            adj_optimized_toevoegingen, adj_optimized_totale_massa, adj_status = optimize_toevoegingen(
                massa_product, 
                adj_df['Actuele_Hoeveelheid_Element'].values, 
                adj_df['Ratio_Element'].values,
                adj_df['Eenheid'].values, 
                adj_df['Min'].values, 
                adj_df['Max'].values, 
                adj_df['Spec'].values,
                use_max_volume, 
                max_volume_liters, 
                stapgroottes, 
                st.session_state.mass_penalty, 
                st.session_state.deviation_weight,
                excluded_indices=st.session_state.excluded_indices
            )
            if adj_status == "Succes":
                # Check for small additions to refine
                refine_indices = [i for i in range(len(adj_optimized_toevoegingen)) if 0 < adj_optimized_toevoegingen[i] <= 9]
                if refine_indices:
                    fixed_values = {i: adj_optimized_toevoegingen[i] for i in range(len(adj_optimized_toevoegingen)) if i not in refine_indices}
                    adj_optimized_toevoegingen, adj_optimized_totale_massa, adj_status = optimize_toevoegingen(
                        massa_product, 
                        adj_df['Actuele_Hoeveelheid_Element'].values, 
                        adj_df['Ratio_Element'].values,
                        adj_df['Eenheid'].values, 
                        adj_df['Min'].values, 
                        adj_df['Max'].values, 
                        adj_df['Spec'].values,
                        use_max_volume, 
                        max_volume_liters, 
                        stapgroottes, 
                        st.session_state.mass_penalty, 
                        st.session_state.deviation_weight,
                        excluded_indices=st.session_state.excluded_indices,
                        fixed_values=fixed_values,
                        refine=True
                    )
            if adj_status == "Succes":
                adj_df['Toevoegen Grondstof'] = adj_optimized_toevoegingen
                adj_df['Geoptimaliseerde_Toegevoegd_Element'] = adj_df.apply(bereken_toegevoegd_element, axis=1)
                adj_df['Na Optimalisatie'] = adj_df.apply(bereken_gecorrigeerde_concentratie, axis=1, totale_massa_correctie=adj_optimized_totale_massa)
                adj_df['Binnen Specs'] = (adj_df['Na Optimalisatie'] >= adj_df['Min']) & (adj_df['Na Optimalisatie'] <= adj_df['Max'])
                adj_totale_toevoeging = adj_optimized_totale_massa - massa_product
                adj_percentage_toename = (adj_totale_toevoeging / massa_product) * 100 if massa_product > 0 else 0
                st.success(f"Aangepaste optimalisatie succesvol! Totale massa na correctie: **{adj_optimized_totale_massa:.2f} kg** (toevoeging: **{adj_totale_toevoeging:.2f} kg**, {adj_percentage_toename:.2f}% toename)")
                
                adj_optimized_resultaten = adj_df[['Grondstof', 'Element', 'Gemeten Concentratie', 'Toevoegen Grondstof', 'Na Optimalisatie', 'Spec', 'Min', 'Max', 'Binnen Specs', 'Eenheid']]
                st.dataframe(
                    adj_optimized_resultaten.style.applymap(highlight_specs, subset=['Binnen Specs']).format(format_dict)
                )
                
                st.subheader("Aangepaste Waarschuwingen")
                warnings_present = False
                for index, row in adj_optimized_resultaten.iterrows():
                    if not row['Binnen Specs'] and row['Element'] != 'Geen':
                        st.warning(f"{row['Element']} ({row['Na Optimalisatie']:.2f} {row['Eenheid']}) ligt buiten specificaties ({row['Min']} - {row['Max']})")
                        warnings_present = True
                if not warnings_present:
                    st.success("Alle concentraties liggen binnen de specificaties!")
                
                # Export button for adjusted results (Excel only)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    adj_optimized_resultaten.to_excel(writer, index=False, sheet_name='Aangepaste_Resultaten')
                excel_data = output.getvalue()
                st.download_button(
                    label="Download Aangepaste Resultaten als Excel",
                    data=excel_data,
                    file_name=f'aangepaste_correctie_{recept}_{tank}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    help="Download de aangepaste geoptimaliseerde resultaten als Excel-bestand."
                )
            else:
                st.error(f"Aangepaste optimalisatie mislukt: {adj_status}")
else:
    st.info("Geen optimalisatieresultaten beschikbaar. Voer eerst de originele optimalisatie uit.")
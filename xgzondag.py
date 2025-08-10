import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import plotly.express as px
import json

# Load recepten from JSON with fallback to hardcoded for robustness
try:
    with open('recepten.json', 'r') as f:
        recepten = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    st.warning("Error loading recepten.json. Falling back to hardcoded data.")
    recepten = {
        "Zero Waste Solution A": {
            "Hoeveelheid": 1100.0,
            "Dichtheid": 1.307,
            "Max_Volume": None,
            "Data": {
                'Grondstof': ['Fosforzuur 75%', 'Magnesiumchloride', 'Yzerchloride', 'Cobaltsulfaat oplossing', 'Nikkelsulfaat oplossing'],
                'Element': ['P2O5', 'MgO', 'Fe', 'Co', 'Ni'],
                'Gemeten_Concentratie': [23.7, 2.7, 1.17, 105, 200],
                'Eenheid': ['wt%', 'wt%', 'wt%', 'mg/kg', 'mg/kg'],
                'Spec': [25, 3, 1.175, 110, 220],
                'Min': [24.2, 2.7, 1.1, 90, 190],
                'Max': [25.8, 3.3, 1.25, 130, 250],
                'Concentratie_Grondstof': [74.9, 19.7, 13.7, 8.2, 6.4],
                'Ratio_Element': [0.5424258, 0.197, 0.137, 0.082, 0.064]
            }
        },
        "DKP 0-20-25": {
            "Hoeveelheid": 14100.0,
            "Dichtheid": 1.503,
            "Max_Volume": 140000.0,
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

# Titel van de app
st.title("Correctie Recept Rekenmodule")

# Receptselectie
recept = st.selectbox("Kies een recept", list(recepten.keys()), help="Selecteer het recept om te bewerken.")

# Inleiding en uitleg
#st.markdown("""
#Deze app berekent geoptimaliseerde toevoegingen van grondstoffen in **gehele kilogrammen** om concentraties binnen specificaties te brengen, 
#met minimale totale massa-toevoeging. Voer algemene gegevens en initiële schattingen in, en start de optimalisatie.
#""")

# Toevoegen aan de "Algemene Gegevens" sectie of een nieuwe expander
with st.expander("Algemene Gegevens", expanded=True):
    col1, col2, col3 = st.columns(3)  # Extra kolom voor stapgrootte
    with col1:
        hoeveelheid = st.number_input("Hoeveelheid (liter)", value=recepten[recept]["Hoeveelheid"], step=100.0, min_value=0.0, help="De totale hoeveelheid product in liters.")
    with col2:
        dichtheid = st.number_input("Dichtheid (kg/l)", value=recepten[recept]["Dichtheid"], step=0.01, min_value=0.01, help="De dichtheid van het product in kg per liter.", format="%.3f")
    with col3:
        stapgrootte = st.number_input("Stapgrootte (kg)", value=0, step=1, min_value=0, help="Geef de stapgrootte in kg (bijv. 5 voor veelvouden van 5 kg). Gebruik 0 voor geen stappen.")
    massa_product = hoeveelheid * dichtheid
    st.info(f"Originele massa product: **{massa_product:.2f} kg**")

# Toggle voor Max_Volume constraint with input
col_max_vol1, col_max_vol2 = st.columns([1, 2])
with col_max_vol1:
    use_max_volume = st.checkbox("Gebruik Max_Volume beperking", value=True, help="Schakel uit om de maximale volumebeperking te negeren (kan helpen bij infeasible problemen).")
with col_max_vol2:
    default_max_volume = recepten[recept]["Max_Volume"] if recepten[recept]["Max_Volume"] is not None else 0.0
    max_volume_m3 = st.number_input(
        "Max Volume (m³)",
        min_value=0.0,
        value=default_max_volume / 1000.0 if default_max_volume != 0.0 else 0.0,
        step=0.001,
        disabled=not use_max_volume,
        help="Voer de maximale volumebeperking in kubieke meters in (alleen actief als de checkbox is aangevinkt)."
    )
    max_volume_liters = max_volume_m3 * 1000.0

# Gegevens uit het geselecteerde recept
data = recepten[recept]["Data"]

# Maak een DataFrame
df = pd.DataFrame(data)

# Functies voor berekeningen
def bereken_actuele_hoeveelheid_element(row, massa_product):
    conversie_factor = 100 if row['Eenheid'] == 'wt%' else 1000000
    return (massa_product * row['Gemeten_Concentratie']) / conversie_factor

def bereken_actuele_hoeveelheid_grondstof(row):
    if row['Ratio_Element'] == 0:
        return 0.0
    return row['Actuele_Hoeveelheid_Element'] / row['Ratio_Element']

def bereken_toegevoegd_element(row):
    return row['Geoptimaliseerde_Toevoegen_Grondstof'] * row['Ratio_Element']

def bereken_gecorrigeerde_concentratie(row, totale_massa_correctie):
    conversie_factor = 100 if row['Eenheid'] == 'wt%' else 1000000
    totale_element = row['Actuele_Hoeveelheid_Element'] + row['Geoptimaliseerde_Toegevoegd_Element']
    return (totale_element / totale_massa_correctie) * conversie_factor

# Bereken actuele hoeveelheid element (kolom F)
df['Actuele_Hoeveelheid_Element'] = df.apply(bereken_actuele_hoeveelheid_element, axis=1, massa_product=massa_product)

# Bereken actuele hoeveelheid grondstof (kolom G)
df['Actuele_Hoeveelheid_Grondstof'] = df.apply(bereken_actuele_hoeveelheid_grondstof, axis=1)

# State voor initiële schattingen
if 'initiele_schattingen' not in st.session_state or st.session_state.get('current_recept') != recept:
    st.session_state.initiele_schattingen = {row['Grondstof']: 0 for _, row in df.iterrows()}
    st.session_state.current_recept = recept

# Heuristische suggestie voor initiële schattingen (reverted to original)
def suggest_initial_guesses(df, massa_product, dichtheid, ratios, eenheden, min_specs, max_specs, use_max_volume, max_volume):
    prob = LpProblem("Initial_Guess_Relaxation", LpMinimize)
    
    toevoegingen = {i: LpVariable(f"add_{i}", lowBound=0, cat='Continuous') for i in range(len(ratios))}
    
    prob += lpSum(toevoegingen.values())
    
    totale_massa = massa_product + lpSum(toevoegingen.values())
    for i in range(len(ratios)):
        if ratios[i] == 0:
            continue
        conversie_factor = 100 if eenheden[i] == 'wt%' else 1000000
        totale_element = df['Actuele_Hoeveelheid_Element'].iloc[i] + toevoegingen[i] * ratios[i]
        prob += totale_element * conversie_factor >= min_specs[i] * totale_massa, f"Min_Constraint_{i}"
        prob += totale_element * conversie_factor <= max_specs[i] * totale_massa, f"Max_Constraint_{i}"
    
    if use_max_volume and max_volume is not None and max_volume > 0:
        prob += totale_massa / dichtheid <= max_volume, "Max_Volume_Constraint"
    
    prob.solve()
    
    suggestions = {}
    if LpStatus[prob.status] == 'Optimal':
        for i, grondstof in enumerate(df['Grondstof']):
            suggestions[grondstof] = int(round(toevoegingen[i].varValue)) if toevoegingen[i].varValue is not None else 0
    else:
        for grondstof in df['Grondstof']:
            suggestions[grondstof] = 0
        st.warning(f"LP relaxation voor initiële schattingen mislukt: {LpStatus[prob.status]}. Gebruikte nulwaarden.")
    
    return suggestions

# Interactieve invoer voor initiële schattingen
with st.expander("Initiële Schattingen voor Optimalisatie (kg)", expanded=True):
    st.info("Voer initiële schattingen in (in gehele kilogrammen). Deze worden getoond als referentie, maar de optimalisatie kan lagere waarden kiezen om een oplossing te vinden.")
    cols = st.columns(3)
    for idx, (grondstof, value) in enumerate(st.session_state.initiele_schattingen.items()):
        with cols[idx % 3]:
            input_value = st.number_input(
                f"{grondstof}", 
                value=value, 
                step=1, 
                min_value=0, 
                key=f"init_{recept}_{grondstof}",
                help=f"Schatting voor {grondstof} in gehele kg (referentie)."
            )
            st.session_state.initiele_schattingen[grondstof] = int(input_value)

# Knoppen voor suggestie en reset
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("Suggesteer Initiële Schattingen", help="Bereken eenvoudige startwaarden gebaseerd op minimale specs."):
        suggestions = suggest_initial_guesses(
            df=df,
            massa_product=massa_product,
            dichtheid=dichtheid,
            ratios=df['Ratio_Element'].values,
            eenheden=df['Eenheid'].values,
            min_specs=df['Min'].values,
            max_specs=df['Max'].values,
            use_max_volume=use_max_volume,
            max_volume=max_volume_liters
        )
        st.session_state.initiele_schattingen.update(suggestions)
        st.rerun()
with col_btn2:
    if st.button("Reset Initiële Schattingen", help="Zet alle initiële schattingen terug naar 0."):
        for key in st.session_state.initiele_schattingen:
            st.session_state.initiele_schattingen[key] = 0
        st.rerun()

# PuLP Optimalisatie
st.header("Geoptimaliseerde Toevoegingen")

# Reverted to original optimization logic
def optimize_toevoegingen(massa_product, actuele_hoeveelheden, ratios, eenheden, min_specs, max_specs, use_max_volume, max_volume, stapgrootte):
    prob = LpProblem("Minimize_Additions", LpMinimize)
    
    # Definieer variabelen
    if stapgrootte > 0:
        # Gebruik integer variabelen voor het aantal stappen
        aantal_stappen = {i: LpVariable(f"steps_{i}", lowBound=0, cat='Integer') for i in range(len(ratios))}
        # Toevoegingen zijn veelvouden van stapgrootte
        toevoegingen = {i: aantal_stappen[i] * stapgrootte for i in range(len(ratios))}
    else:
        # Geen stappen, gebruik gewone integer variabelen
        toevoegingen = {i: LpVariable(f"add_{i}", lowBound=0, cat='Integer') for i in range(len(ratios))}
    
    # Doelfunctie: minimaliseer totale toevoeging
    prob += lpSum(toevoegingen.values())
    
    # Totale massa na toevoegingen
    totale_massa = massa_product + lpSum(toevoegingen.values())
    
    # Concentratiebeperkingen
    for i in range(len(ratios)):
        if ratios[i] == 0:
            continue
        conversie_factor = 100 if eenheden[i] == 'wt%' else 1000000
        totale_element = actuele_hoeveelheden[i] + toevoegingen[i] * ratios[i]
        prob += totale_element * conversie_factor >= min_specs[i] * totale_massa, f"Min_Constraint_{i}"
        prob += totale_element * conversie_factor <= max_specs[i] * totale_massa, f"Max_Constraint_{i}"
    
    # Volumebeperking (indien actief)
    if use_max_volume and max_volume is not None and max_volume > 0:
        prob += totale_massa / dichtheid <= max_volume, "Max_Volume_Constraint"
    
    # Los het probleem op
    prob.solve()
    
    if LpStatus[prob.status] == 'Optimal':
        if stapgrootte > 0:
            optimized_toevoegingen = np.array([aantal_stappen[i].varValue * stapgrootte for i in range(len(ratios))], dtype=int)
        else:
            optimized_toevoegingen = np.array([toevoegingen[i].varValue for i in range(len(ratios))], dtype=int)
        optimized_totale_massa = massa_product + sum(optimized_toevoegingen)
        return optimized_toevoegingen, optimized_totale_massa, "Succes"
    else:
        diagnostics = []
        for i in range(len(ratios)):
            if ratios[i] == 0:
                continue
            conversie_factor = 100 if eenheden[i] == 'wt%' else 1000000
            current_conc = (actuele_hoeveelheden[i] / massa_product) * conversie_factor if massa_product > 0 else 0
            if current_conc < min_specs[i]:
                diagnostics.append(f"{df['Element'][i]}: Huidige concentratie ({current_conc:.2f} {eenheden[i]}) is onder Min ({min_specs[i]}).")
            if current_conc > max_specs[i]:
                diagnostics.append(f"{df['Element'][i]}: Huidige concentratie ({current_conc:.2f} {eenheden[i]}) is boven Max ({max_specs[i]}).")
        if use_max_volume and max_volume is not None and max_volume > 0 and massa_product / dichtheid > max_volume:
            diagnostics.append(f"Huidige volume ({massa_product / dichtheid:.2f} L) overschrijdt Max_Volume ({max_volume} L).")
        return None, None, f"Optimalisatie mislukt: {LpStatus[prob.status]}. Mogelijke oorzaken:\n" + "\n".join(diagnostics) if diagnostics else f"Optimalisatie mislukt: {LpStatus[prob.status]}."
    
# Knop om optimalisatie uit te voeren
if st.button("Bereken Geoptimaliseerde Toevoegingen", help="Klik om de optimalisatie te starten op basis van de ingevoerde gegevens."):
    with st.spinner("Optimalisatie wordt uitgevoerd..."):
        initiele_schattingen = st.session_state.initiele_schattingen
        
        optimized_toevoegingen, optimized_totale_massa, status = optimize_toevoegingen(
            massa_product, df['Actuele_Hoeveelheid_Element'].values, df['Ratio_Element'].values, 
            df['Eenheid'].values, df['Min'].values, df['Max'].values, use_max_volume, max_volume_liters, stapgrootte
        )

    if status == "Succes":
        df['Geoptimaliseerde_Toevoegen_Grondstof'] = optimized_toevoegingen
        df['Geoptimaliseerde_Toegevoegd_Element'] = df.apply(bereken_toegevoegd_element, axis=1)
        df['Geoptimaliseerde_Concentratie'] = df.apply(bereken_gecorrigeerde_concentratie, axis=1, totale_massa_correctie=optimized_totale_massa)
        df['Geoptimaliseerde_Binnen_Specs'] = (df['Geoptimaliseerde_Concentratie'] >= df['Min']) & (df['Geoptimaliseerde_Concentratie'] <= df['Max'])

        totale_toevoeging = optimized_totale_massa - massa_product
        percentage_toename = (totale_toevoeging / massa_product) * 100 if massa_product > 0 else 0
        st.success(f"Optimalisatie succesvol! Totale massa na correctie: **{optimized_totale_massa:.2f} kg** (toevoeging: **{totale_toevoeging:.2f} kg**, {percentage_toename:.2f}% toename)")

        def highlight_specs(val):
            color = 'green' if val else 'red'
            return f'background-color: {color}'

        optimized_resultaten = df[['Grondstof', 'Element', 'Gemeten_Concentratie', 'Geoptimaliseerde_Toevoegen_Grondstof', 'Geoptimaliseerde_Concentratie', 'Spec', 'Min', 'Max', 'Geoptimaliseerde_Binnen_Specs']]
        st.dataframe(optimized_resultaten.style.applymap(highlight_specs, subset=['Geoptimaliseerde_Binnen_Specs']).format({'Geoptimaliseerde_Toevoegen_Grondstof': '{:.0f}'}))

        st.subheader("Waarschuwingen")
        warnings_present = False
        for index, row in df.iterrows():
            if not row['Geoptimaliseerde_Binnen_Specs']:
                st.warning(f"{row['Element']} ({row['Geoptimaliseerde_Concentratie']:.2f} {row['Eenheid']}) ligt buiten specificaties ({row['Min']} - {row['Max']})")
                warnings_present = True
        if not warnings_present:
            st.success("Alle concentraties liggen binnen de specificaties!")

        st.subheader("Visualisatie van Concentratievergelijking")
        viz_df = df.melt(id_vars=['Element', 'Eenheid', 'Min', 'Max'], value_vars=['Gemeten_Concentratie', 'Geoptimaliseerde_Concentratie', 'Spec'], var_name='Type', value_name='Waarde')
        viz_df['Error_Low'] = viz_df['Waarde'] - viz_df['Min']
        viz_df['Error_High'] = viz_df['Max'] - viz_df['Waarde']
        fig = px.bar(viz_df, x='Element', y='Waarde', color='Type', barmode='group', facet_col='Eenheid',
                     error_y='Error_High', error_y_minus='Error_Low',
                     title='Vergelijking van Concentratie: Gemeten vs Geoptimaliseerd vs Specificaties (Gescheiden per Eenheid)',
                     labels={'Waarde': 'Concentratie', 'Element': 'Element'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        csv = optimized_resultaten.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Resultaten als CSV", data=csv, file_name=f'geoptimaliseerde_correctie_{recept}.csv', mime='text/csv', help="Download de geoptimaliseerde resultaten als CSV-bestand.")
    else:
        st.error(status)
else:
    st.info("Klik op de knop om de optimalisatie te starten.")
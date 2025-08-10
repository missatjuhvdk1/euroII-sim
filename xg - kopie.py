import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px

# Titel van de app
st.title("Correctie Recept Calculator (Zero Waste Solution A)")

# Inleiding en uitleg
st.markdown("""
Deze app berekent geoptimaliseerde toevoegingen van grondstoffen om concentraties binnen specificaties te brengen, 
met minimale totale massa-toevoeging. Voer algemene gegevens en initiële schattingen in, en start de optimalisatie.
Later kan dit worden uitgebreid met historische data voor automatische schattingen.
""")

# Algemene gegevens invoeren
with st.expander("Algemene Gegevens", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        hoeveelheid = st.number_input("Hoeveelheid (liter)", value=1100.0, step=100.0, min_value=0.0, help="De totale hoeveelheid product in liters.")
    with col2:
        dichtheid = st.number_input("Dichtheid (kg/l)", value=1.307, step=0.01, min_value=0.0, help="De dichtheid van het product in kg per liter.")
    massa_product = hoeveelheid * dichtheid
    st.info(f"Originele massa product: **{massa_product:.2f} kg**")

# Gegevens uit het werkblad "Zero Waste Solution A"
data = {
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

# Maak een DataFrame
df = pd.DataFrame(data)

# Functies voor berekeningen
def bereken_actuele_hoeveelheid_element(row, massa_product):
    conversie_factor = 100 if row['Eenheid'] == 'wt%' else 1000000
    return (massa_product * row['Gemeten_Concentratie']) / conversie_factor

def bereken_actuele_hoeveelheid_grondstof(row):
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
if 'initiele_schattingen' not in st.session_state:
    st.session_state.initiele_schattingen = {row['Grondstof']: float(0.0) for _, row in df.iterrows()}

# Heuristische suggestie voor initiële schattingen
def suggest_initial_guesses(df, massa_product):
    suggestions = {}
    for _, row in df.iterrows():
        conv_factor = 100 if row['Eenheid'] == 'wt%' else 1000000
        deficit_conc = max(0, row['Min'] - row['Gemeten_Concentratie'])
        deficit_element = (deficit_conc * massa_product) / conv_factor
        addition_grondstof = deficit_element / row['Ratio_Element']
        suggestions[row['Grondstof']] = float(max(0, addition_grondstof))  # Ensure float
    return suggestions

# Interactieve invoer voor initiële schattingen (in een expander)
with st.expander("Initiële Schattingen voor Optimalisatie (kg)", expanded=True):
    st.info("Voer initiële schattingen in voor de optimalisatie. Deze worden gebruikt als startpunt. Gebruik de suggestie-knop voor een eenvoudige heuristische schatting.")
    cols = st.columns(3)
    for idx, (grondstof, value) in enumerate(st.session_state.initiele_schattingen.items()):
        with cols[idx % 3]:
            st.session_state.initiele_schattingen[grondstof] = st.number_input(
                f"{grondstof}", 
                value=float(value),  # Ensure float
                step=1.0, 
                min_value=0.0, 
                key=f"init_{grondstof}", 
                help=f"Startschatting voor {grondstof} in kg."
            )

# Knoppen voor suggestie en reset
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("Suggesteer Initiële Schattingen", help="Bereken eenvoudige startwaarden gebaseerd op minimale specs (negeert interacties)."):
        suggestions = suggest_initial_guesses(df, massa_product)
        st.session_state.initiele_schattingen.update(suggestions)
        st.rerun()
with col_btn2:
    if st.button("Reset Initiële Schattingen", help="Zet alle initiële schattingen terug naar 0."):
        for key in st.session_state.initiele_schattingen:
            st.session_state.initiele_schattingen[key] = 0.0
        st.rerun()

# SciPy Optimalisatie
st.header("Geoptimaliseerde Toevoegingen (Minimale Massa)")

def objective_function(toevoegingen, massa_product, actuele_hoeveelheden, ratios, eenheden, min_specs, max_specs, initiele_schattingen, penalty_factor=0.1):
    total_mass = sum(toevoegingen)
    penalty = penalty_factor * sum((toevoegingen[i] - initiele_schattingen[i])**2 for i in range(len(toevoegingen)))
    return total_mass + penalty

def constraint_function(toevoegingen, massa_product, actuele_hoeveelheden, ratios, eenheden, min_specs, max_specs, index):
    totale_massa = massa_product + sum(toevoegingen)
    conversie_factor = 100 if eenheden[index] == 'wt%' else 1000000
    totale_element = actuele_hoeveelheden[index] + toevoegingen[index] * ratios[index]
    concentratie = (totale_element / totale_massa) * conversie_factor
    return concentratie

def optimize_toevoegingen(massa_product, actuele_hoeveelheden, ratios, eenheden, min_specs, max_specs, initiele_schattingen, max_iterations=50, tolerance=1e-8):
    toevoegingen = np.array(list(initiele_schattingen.values()))
    totale_massa = massa_product + sum(toevoegingen)
    for _ in range(max_iterations):
        constraints = []
        for i in range(len(ratios)):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_function(x, massa_product, actuele_hoeveelheden, ratios, eenheden, min_specs, max_specs, i) - min_specs[i]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: max_specs[i] - constraint_function(x, massa_product, actuele_hoeveelheden, ratios, eenheden, min_specs, max_specs, i)})
        bounds = [(0, None)] * len(ratios)
        result = minimize(objective_function, toevoegingen, args=(massa_product, actuele_hoeveelheden, ratios, eenheden, min_specs, max_specs, list(initiele_schattingen.values())),
                          constraints=constraints, bounds=bounds, method='SLSQP', options={'ftol': 1e-10, 'maxiter': 1000, 'disp': False})
        if not result.success:
            return None, None, f"Optimalisatie mislukt: {result.message}"
        new_toevoegingen = result.x
        new_totale_massa = massa_product + sum(new_toevoegingen)
        concentraties = [(actuele_hoeveelheden[i] + new_toevoegingen[i] * ratios[i]) / new_totale_massa * (100 if eenheden[i] == 'wt%' else 1000000) for i in range(len(ratios))]
        if all(min_specs[i] <= concentraties[i] <= max_specs[i] for i in range(len(ratios))):
            if abs(new_totale_massa - totale_massa) < tolerance:
                return new_toevoegingen, new_totale_massa, "Succes"
        toevoegingen = new_toevoegingen
        totale_massa = new_totale_massa
    return None, None, "Max iteraties bereikt zonder convergentie"

# Knop om optimalisatie uit te voeren
if st.button("Bereken Geoptimaliseerde Toevoegingen", help="Klik om de optimalisatie te starten op basis van de ingevoerde gegevens."):
    with st.spinner("Optimalisatie wordt uitgevoerd..."):
        initiele_schattingen = st.session_state.initiele_schattingen
        optimized_toevoegingen, optimized_totale_massa, status = optimize_toevoegingen(
            massa_product, df['Actuele_Hoeveelheid_Element'].values, df['Ratio_Element'].values, df['Eenheid'].values, df['Min'].values, df['Max'].values, initiele_schattingen
        )

    if status == "Succes":
        df['Geoptimaliseerde_Toevoegen_Grondstof'] = optimized_toevoegingen
        df['Geoptimaliseerde_Toegevoegd_Element'] = df.apply(bereken_toegevoegd_element, axis=1)
        df['Geoptimaliseerde_Concentratie'] = df.apply(bereken_gecorrigeerde_concentratie, axis=1, totale_massa_correctie=optimized_totale_massa)
        df['Geoptimaliseerde_Binnen_Specs'] = (df['Geoptimaliseerde_Concentratie'] >= df['Min']) & (df['Geoptimaliseerde_Concentratie'] <= df['Max'])

        totale_toevoeging = optimized_totale_massa - massa_product
        percentage_toename = (totale_toevoeging / massa_product) * 100 if massa_product > 0 else 0
        st.success(f"Optimalisatie succesvol! Totale massa na correctie: **{optimized_totale_massa:.2f} kg** (toevoeging: **{totale_toevoeging:.2f} kg**, {percentage_toename:.2f}% toename)")

        # Resultatentabel met kleurcodering
        def highlight_specs(val):
            color = 'green' if val else 'red'
            return f'background-color: {color}'

        optimized_resultaten = df[['Grondstof', 'Element', 'Gemeten_Concentratie', 'Geoptimaliseerde_Toevoegen_Grondstof', 'Geoptimaliseerde_Concentratie', 'Spec', 'Min', 'Max', 'Geoptimaliseerde_Binnen_Specs']]
        st.dataframe(optimized_resultaten.style.applymap(highlight_specs, subset=['Geoptimaliseerde_Binnen_Specs']))

        # Waarschuwingen
        st.subheader("Waarschuwingen")
        warnings_present = False
        for index, row in df.iterrows():
            if not row['Geoptimaliseerde_Binnen_Specs']:
                st.warning(f"{row['Element']} ({row['Geoptimaliseerde_Concentratie']:.2f} {row['Eenheid']}) ligt buiten specificaties ({row['Min']} - {row['Max']})")
                warnings_present = True
        if not warnings_present:
            st.success("Alle concentraties liggen binnen de specificaties!")

        # Visualisatie (met facets voor units en gecorrigeerde error bars)
        st.subheader("Visualisatie van Concentratievergelijking")
        viz_df = df.melt(id_vars=['Element', 'Eenheid', 'Min', 'Max'], value_vars=['Gemeten_Concentratie', 'Geoptimaliseerde_Concentratie', 'Spec'], var_name='Type', value_name='Waarde')
        # Bereken error bars relatief
        viz_df['Error_Low'] = viz_df['Waarde'] - viz_df['Min']
        viz_df['Error_High'] = viz_df['Max'] - viz_df['Waarde']
        fig = px.bar(viz_df, x='Element', y='Waarde', color='Type', barmode='group', facet_col='Eenheid',
                     error_y='Error_High', error_y_minus='Error_Low',
                     title='Vergelijking van Concentratie: Gemeten vs Geoptimaliseerd vs Specificaties (Gescheiden per Eenheid)',
                     labels={'Waarde': 'Concentratie', 'Element': 'Element'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Export naar CSV
        csv = optimized_resultaten.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Resultaten als CSV", data=csv, file_name='geoptimaliseerde_correctie.csv', mime='text/csv', help="Download de geoptimaliseerde resultaten als CSV-bestand.")
    else:
        st.error(f"Optimalisatie mislukt: {status}")
else:
    st.info("Klik op de knop om de optimalisatie te starten.")
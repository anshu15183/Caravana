import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass

# Define the ComponentScore dataclass for easy data management
@dataclass
class ComponentScore:
    raw_value: float
    normalized_score: float
    weight: float
    contribution: float
    status: str
    emoji: str
    recommendation: str

class VehicleComponentIndex:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.weights = {
            'mechanical_index': 0.25,
            'age_mileage_index': 0.20,
            'efficiency_index': 0.15,
            'maintenance_history_index': 0.20,
            'tire_brake_index': 0.20
        }
        self.thresholds = {
            'critical': 80,
            'warning': 60,
            'moderate': 40,
            'good': 20
        }

    def calculate_mechanical_index(self, engine_health, transmission_health, spark_plug_health):
        raw_value = np.mean([engine_health, transmission_health, spark_plug_health])
        normalized_score = self._normalize_score(raw_value)
        weight = self.weights['mechanical_index']
        contribution = normalized_score * weight
        status, emoji, recommendation = self._get_status_and_recommendation('mechanical', normalized_score)
        return ComponentScore(raw_value, normalized_score, weight, contribution, status, emoji, recommendation)

    def calculate_tire_brake_index(self, tire_condition, brake_pad_health, brake_fluid_health, brake_rotor_health):
        brake_health = np.mean([brake_pad_health, brake_fluid_health, brake_rotor_health])
        raw_value = np.mean([tire_condition, brake_health])
        normalized_score = self._normalize_score(raw_value)
        weight = self.weights['tire_brake_index']
        contribution = normalized_score * weight
        status, emoji, recommendation = self._get_status_and_recommendation('tire_brake', normalized_score)
        return ComponentScore(raw_value, normalized_score, weight, contribution, status, emoji, recommendation)

    def calculate_age_mileage_index(self, age_years, mileage, vehicle_traveled):
        age_score = max(0, 100 - (age_years * 7))
        mileage_score = max(0, 100 - (mileage / 2000))
        travel_score = max(0, 100 - (vehicle_traveled / 100) * 100)
        raw_value = np.mean([age_score, mileage_score, travel_score])
        normalized_score = self._normalize_score(raw_value)
        weight = self.weights['age_mileage_index']
        contribution = normalized_score * weight
        status, emoji, recommendation = self._get_status_and_recommendation('age_mileage', normalized_score)
        return ComponentScore(raw_value, normalized_score, weight, contribution, status, emoji, recommendation)

    def calculate_efficiency_index(self, fuel_efficiency, oil_consumption, coolant_health, spark_plug_efficiency):
        fuel_score = min(100, (fuel_efficiency / 50) * 100)
        oil_score = max(0, 100 - (oil_consumption * 20))
        raw_value = np.mean([fuel_score, oil_score, coolant_health, spark_plug_efficiency])
        normalized_score = self._normalize_score(raw_value)
        weight = self.weights['efficiency_index']
        contribution = normalized_score * weight
        status, emoji, recommendation = self._get_status_and_recommendation('efficiency', normalized_score)
        return ComponentScore(raw_value, normalized_score, weight, contribution, status, emoji, recommendation)

    def calculate_maintenance_history_index(self, service_interval_adherence, last_service_age_months, repair_frequency, tire_rotation_adherence, brake_service_adherence):
        service_age_score = max(0, 100 - (last_service_age_months * 8.33))
        repair_score = max(0, 100 - (repair_frequency * 25))
        raw_value = np.mean([service_interval_adherence, service_age_score, repair_score, tire_rotation_adherence, brake_service_adherence])
        normalized_score = self._normalize_score(raw_value)
        weight = self.weights['maintenance_history_index']
        contribution = normalized_score * weight
        status, emoji, recommendation = self._get_status_and_recommendation('maintenance_history', normalized_score)
        return ComponentScore(raw_value, normalized_score, weight, contribution, status, emoji, recommendation)

    def _normalize_score(self, score):
        return max(0, min(100, score))

    def _get_status_and_recommendation(self, component_type, score):
        status_recommendations = {
            'mechanical': {'critical': ('Critical', '🚨', 'Immediate inspection required'), 'warning': ('Warning', '⚠️', 'Service within 7 days'), 'moderate': ('Moderate', '🛠️', 'Check within 30 days'), 'good': ('Good', '✅', 'Routine checkup')},
            'tire_brake': {'critical': ('Critical', '🚨', 'Immediate brake or tire service'), 'warning': ('Warning', '⚠️', 'Schedule inspection within 7 days'), 'moderate': ('Moderate', '🛠️', 'Check tire & brake wear within 30 days'), 'good': ('Good', '✅', 'Normal maintenance')},
            'age_mileage': {'critical': ('High Risk', '🚨', 'Consider replacement or major repairs'), 'warning': ('Aging', '⚠️', 'Schedule major maintenance within 7 days'), 'moderate': ('Moderate', '🛠️', 'Routine check within 30 days'), 'good': ('Good', '✅', 'No major concerns')},
            'efficiency': {'critical': ('Poor', '🚨', 'Immediate diagnostics'), 'warning': ('Declining', '⚠️', 'Inspect within 7 days'), 'moderate': ('Moderate', '🛠️', 'Monitor efficiency'), 'good': ('Efficient', '✅', 'Maintain current efficiency')},
            'maintenance_history': {'critical': ('Poor', '🚨', 'Immediate maintenance check'), 'warning': ('Irregular', '⚠️', 'Improve adherence soon'), 'moderate': ('Moderate', '🛠️', 'Maintain schedule'), 'good': ('Good', '✅', 'Regular maintenance')}
        }
        if score >= self.thresholds['critical']:
            return status_recommendations[component_type]['critical']
        elif score >= self.thresholds['warning']:
            return status_recommendations[component_type]['warning']
        elif score >= self.thresholds['moderate']:
            return status_recommendations[component_type]['moderate']
        else:
            return status_recommendations[component_type]['good']

    def calculate_total_index(self, component_scores):
        return sum(score.contribution for score in component_scores.values())

    def plot_component_breakdown(self, component_scores):
        components = list(component_scores.keys())
        contributions = [score.contribution for score in component_scores.values()]
        fig, ax = plt.subplots()
        ax.bar(components, contributions)
        ax.set_title('Component Contributions to Total Index')
        ax.set_ylabel('Weighted Contribution')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Initialize indexer
indexer = VehicleComponentIndex()

st.title("Vehicle Maintenance Index Calculator")

# Check if results have been submitted
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Collect inputs from user with input boxes
with st.form("maintenance_form"):
    engine_health = st.number_input("Engine Health", min_value=0, max_value=100, value=85)
    transmission_health = st.number_input("Transmission Health", min_value=0, max_value=100, value=88)
    spark_plug_health = st.number_input("Spark Plug Health", min_value=0, max_value=100, value=90)

    tire_condition = st.number_input("Tire Condition", min_value=0, max_value=100, value=85)
    brake_pad_health = st.number_input("Brake Pad Health", min_value=0, max_value=100, value=90)
    brake_fluid_health = st.number_input("Brake Fluid Health", min_value=0, max_value=100, value=95)
    brake_rotor_health = st.number_input("Brake Rotor Health", min_value=0, max_value=100, value=88)

    age_years = st.number_input("Vehicle Age (Years)", min_value=0, max_value=20, value=5)
    mileage = st.number_input("Mileage", min_value=0, max_value=200000, value=60000)
    vehicle_traveled = st.number_input("Average Daily Miles", min_value=0, max_value=100, value=45)

    fuel_efficiency = st.number_input("Fuel Efficiency (MPG)", min_value=0, max_value=50, value=28)
    oil_consumption = st.number_input("Oil Consumption (quarts/1000 miles)", min_value=0.0, max_value=5.0, value=0.5)
    coolant_health = st.number_input("Coolant Health", min_value=0, max_value=100, value=95)
    spark_plug_efficiency = st.number_input("Spark Plug Efficiency", min_value=0, max_value=100, value=90)

    service_interval_adherence = st.number_input("Service Interval Adherence", min_value=0, max_value=100, value=95)
    last_service_age_months = st.number_input("Last Service Age (Months)", min_value=0, max_value=12, value=3)
    repair_frequency = st.number_input("Repair Frequency (repairs/year)", min_value=0.0, max_value=4.0, value=0.5)
    tire_rotation_adherence = st.number_input("Tire Rotation Adherence", min_value=0, max_value=100, value=90)
    brake_service_adherence = st.number_input("Brake Service Adherence", min_value=0, max_value=100, value=92)

    # Submit button
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    # Calculate component scores
    component_scores = {
        'Mechanical Components': indexer.calculate_mechanical_index(engine_health, transmission_health, spark_plug_health),
        'Tire & Brake': indexer.calculate_tire_brake_index(tire_condition, brake_pad_health, brake_fluid_health, brake_rotor_health),
        'Age & Mileage': indexer.calculate_age_mileage_index(age_years, mileage, vehicle_traveled),
        'Liquid Efficiency': indexer.calculate_efficiency_index(fuel_efficiency, oil_consumption, coolant_health, spark_plug_efficiency),
        'Maintenance History': indexer.calculate_maintenance_history_index(service_interval_adherence, last_service_age_months, repair_frequency, tire_rotation_adherence, brake_service_adherence)
    }

    # Calculate and display the total maintenance index
    total_index = indexer.calculate_total_index(component_scores)

    # Store results in session state
    st.session_state.component_scores = component_scores
    st.session_state.total_index = round(total_index, 2)
    st.session_state.submitted = True

# Display Results if submitted
if st.session_state.submitted:
    st.title("Vehicle Maintenance Index Results")
    
    # Display the total maintenance index
    st.write("### Total Maintenance Index:", st.session_state.total_index)

    # Display component breakdown in a table
    breakdown_data = {
        'Component': list(st.session_state.component_scores.keys()),
        'Raw Value': [score.raw_value for score in st.session_state.component_scores.values()],
        'Normalized Score': [score.normalized_score for score in st.session_state.component_scores.values()],
        'Weight': [score.weight for score in st.session_state.component_scores.values()],
        'Contribution': [score.contribution for score in st.session_state.component_scores.values()],
        'Status': [score.status for score in st.session_state.component_scores.values()],
        'Emoji': [score.emoji for score in st.session_state.component_scores.values()],
        'Recommendation': [score.recommendation for score in st.session_state.component_scores.values()]
    }
    breakdown_df = pd.DataFrame(breakdown_data)
    st.table(breakdown_df)

    # Plot component breakdown
    indexer.plot_component_breakdown(st.session_state.component_scores)

    # Option to go back to the input form
    if st.button("Go Back to Input"):
        st.session_state.submitted = False
        st.experimental_rerun()  # This will refresh the page and clear session state

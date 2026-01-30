#!/bin/bash

################################################################################
# Script 1: Train All 5 Scenarios (64 APs)
#
# Purpose: Train PPO agents for all 5 optimized scenarios
# Output: experiments/YYYYMMDD_HHMMSS_scenario_X/
#
# Usage: bash 1_train_all_scenarios.sh
################################################################################

set -e  # Exit on error


# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_BASE="./experiments"
LOGS_DIR="./logs/training_${TIMESTAMP}"

mkdir -p "${EXPERIMENTS_BASE}"
mkdir -p "${LOGS_DIR}"

################################################################################
# Scenario Definitions (5 NEW SCENARIOS - All 64 APs)
################################################################################

declare -a SCENARIOS=(
    "1:1_balanced_baseline_64ap.yaml:Balanced_Baseline"
    "2:2_low_density_shutdown_64ap.yaml:Low_Density_Shutdown"
    "3:3_high_density_interference_64ap.yaml:High_Density_Interference"
    "4:4_qos_priority_64ap.yaml:QoS_Priority"
    "5:5_green_network_64ap.yaml:Green_Network"
)

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}\n"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${YELLOW}ℹ️  $1${NC}"; }

################################################################################
# Training Function
################################################################################

train_scenario() {
    local scenario_num=$1
    local config_file=$2
    local scenario_name=$3

    print_header "TRAINING SCENARIO ${scenario_num}: ${scenario_name}"

    local config_path="configs/ppo_scenarios/${config_file}"
    local experiment_dir="${EXPERIMENTS_BASE}/${TIMESTAMP}_scenario_${scenario_num}_${scenario_name}"
    local train_log="${LOGS_DIR}/scenario_${scenario_num}_training.log"

    # Check config exists
    if [ ! -f "${config_path}" ]; then
        print_error "Config not found: ${config_path}"
        return 1
    fi

    print_info "Config: ${config_path}"
    print_info "Experiment Dir: ${experiment_dir}"
    print_info "Log: ${train_log}"

    # Create experiment directory
    mkdir -p "${experiment_dir}"

    # Run training
    echo "$(date): Starting training..." >> "${train_log}"

    if python src/train_agent.py \
        --config "${config_path}" \
        --agent ppo \
        >> "${train_log}" 2>&1; then

        print_success "Training completed for Scenario ${scenario_num}"

        # Move the latest experiment directory content
        local latest_exp=$(ls -td experiments/exp_* 2>/dev/null | head -1)

        if [ -n "$latest_exp" ] && [ -d "$latest_exp" ]; then
            print_info "Moving model from ${latest_exp} to ${experiment_dir}"

            # Move models and results
            mv "${latest_exp}"/* "${experiment_dir}/" 2>/dev/null || true
            rmdir "${latest_exp}" 2>/dev/null || true

            print_success "Model saved to: ${experiment_dir}"
        else
            print_warning "No experiment directory found (training may have saved elsewhere)"
        fi

        # Save metadata
        echo "${experiment_dir}" > "${LOGS_DIR}/scenario_${scenario_num}_path.txt"

        return 0
    else
        print_error "Training failed for Scenario ${scenario_num}"
        print_error "Check log: ${train_log}"
        return 1
    fi
}

################################################################################
# Main Pipeline
################################################################################

main() {
    print_header "🚀 TRAINING ALL 5 SCENARIOS (64 APs)"

    print_info "Timestamp: ${TIMESTAMP}"
    print_info "Experiments Base: ${EXPERIMENTS_BASE}"
    print_info "Logs Directory: ${LOGS_DIR}"

    local total_scenarios=5
    local success_count=0
    local failed_scenarios=""

    # Train each scenario
    for scenario_data in "${SCENARIOS[@]}"; do
        IFS=':' read -r num config name <<< "$scenario_data"

        if train_scenario "$num" "$config" "$name"; then
            success_count=$((success_count + 1))
        else
            failed_scenarios="${failed_scenarios} ${num}"
        fi

        # Small delay between trainings
        sleep 2
    done

    # Summary
    print_header "📊 TRAINING SUMMARY"

    echo -e "${GREEN}Successful:${NC} ${success_count}/${total_scenarios}"

    if [ -n "$failed_scenarios" ]; then
        echo -e "${RED}Failed Scenarios:${NC}${failed_scenarios}"
    fi

    echo -e "\n${BLUE}Trained models saved in:${NC} ${EXPERIMENTS_BASE}/"
    echo -e "${BLUE}Training logs saved in:${NC} ${LOGS_DIR}/"

    # Generate summary file
    {
        echo "Training Summary - ${TIMESTAMP}"
        echo "======================================"
        echo "Total Scenarios: ${total_scenarios}"
        echo "Successful: ${success_count}"
        echo "Failed:${failed_scenarios}"
        echo ""
        echo "Experiments Directory: ${EXPERIMENTS_BASE}/"
        echo "Logs Directory: ${LOGS_DIR}/"
    } > "${LOGS_DIR}/TRAINING_SUMMARY.txt"

    print_success "Summary saved: ${LOGS_DIR}/TRAINING_SUMMARY.txt"

    if [ ${success_count} -eq ${total_scenarios} ]; then
        print_header "🎉 ALL SCENARIOS TRAINED SUCCESSFULLY!"
        return 0
    else
        print_header "⚠️  TRAINING COMPLETED WITH SOME FAILURES"
        return 1
    fi
}

# Run main
main

exit $?
